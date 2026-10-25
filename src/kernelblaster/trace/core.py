# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The trace store: content-addressed blobs + a JSONL index (docs/trace.md).

Layout under the store root (``<out>/trace``)::

    index.jsonl          one JSON record per line (append-only, several writers safe)
    blobs/ab/<sha256>.gz the stored texts, gzip, named by the sha256 of the stored bytes

A record is ``{"t": kind, "id": stable id, "ts": epoch s, "pid", ...context, ...fields}``.
Texts go in as *refs*: ``{"sha", "n"}`` (blob; ``n`` = stored bytes, ``trunc`` = bytes
cut from the middle) or ``{"text"}`` inline when short; at level ``meta`` only
``{"sha", "n", "meta": true}`` (nothing is written to ``blobs/``).

Context (``scope`` / ``set_attrs``) lives in a :mod:`contextvars` variable like the
telemetry spans: concurrent asyncio tasks (rollouts, problems, unit compiles)
each see their own. Writes are serialised by a lock; blob files are written to a
temporary name and renamed, so two processes sharing a store (never needed, but
harmless) cannot tear a blob.

Stdlib only: ModelBlaster loads this package straight from the checkout
(``pipeline/kb_trace.py``).
"""
from __future__ import annotations

import atexit
import contextvars
import gzip
import hashlib
import json
import os
import re
import socket
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

LEVELS = ("off", "meta", "full")
DEFAULT_BLOB_MAX = 4 << 20            # bytes of one stored text (head + tail kept beyond it)
DEFAULT_TOTAL_MAX_MB = 2048           # compressed blob bytes per store; beyond it: meta only (+ a warning)
INLINE_MAX = 200                      # texts up to this many bytes go inline in the record

_ctx: contextvars.ContextVar = contextvars.ContextVar("kb_trace_ctx", default=None)


def _env_level(default: str = "off") -> str:
    v = (os.getenv("KB_TRACE") or default).strip().lower()
    return {"0": "off", "false": "off", "no": "off", "1": "full", "true": "full", "on": "full"}.get(v, v)


# ---------------------------------------------------------------- secrets

_SECRET_KEY = re.compile(r"(KEY|TOKEN|SECRET|PASSW|BEARER|CREDENTIAL|AUTH|SESSION)", re.I)
_SECRET_PATTERNS = [
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                                   # AWS access key ids
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),                              # OpenAI / Anthropic style keys
    re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/=\-]{20,}"),              # Authorization: Bearer ...
    re.compile(r"\bABSK[A-Za-z0-9+/=]{20,}"),                              # Bedrock API keys
]
REDACTED = "***"


class Scrubber:
    """Replaces the values of secret-looking environment variables (and a few
    key formats) in every text the store keeps. The environment is re-read
    when it changes (a key exported after ``configure``)."""

    def __init__(self, extra: tuple = ()):
        self.extra = tuple(x for x in extra if x and len(x) >= 8)
        self._sig = None
        self.values: list[str] = []
        self.refresh()

    def refresh(self) -> None:
        sig = hash(tuple(sorted((k, v) for k, v in os.environ.items() if _SECRET_KEY.search(k))))
        if sig == self._sig:
            return
        self._sig = sig
        vals = set(self.extra)
        for k, v in os.environ.items():
            if _SECRET_KEY.search(k) and v and len(v) >= 8 and not v.isdigit() and v.lower() not in (
                    "true", "false", "none"):
                vals.add(v)
        tf = os.getenv("KB_LLM_SERVER_TOKEN_FILE")
        if tf:
            try:
                t = Path(tf).read_text().strip()
                if len(t) >= 8:
                    vals.add(t)
            except OSError:
                pass
        # longest first: a secret containing another is replaced whole
        self.values = sorted(vals, key=len, reverse=True)

    def __call__(self, text: str) -> str:
        if not text:
            return text
        self.refresh()
        for v in self.values:
            if v in text:
                text = text.replace(v, REDACTED)
        for rx in _SECRET_PATTERNS:
            if rx.search(text):
                text = rx.sub(lambda m: (m.group(1) if m.groups() else "") + REDACTED, text)
        return text

    def env(self, env: Optional[dict]) -> Optional[dict]:
        """An environment for a record: secret-named keys' values redacted."""
        if not env:
            return env
        return {k: (REDACTED if _SECRET_KEY.search(k) else self(str(v))) for k, v in env.items()}


# ---------------------------------------------------------------- store

def _truncate(data: bytes, cap: int) -> tuple[bytes, int]:
    """Head (2/3) + tail (1/3) of ``data`` within ``cap`` bytes -> (bytes, bytes cut)."""
    if cap <= 0 or len(data) <= cap:
        return data, 0
    head, tail = (2 * cap) // 3, cap // 3
    cut = len(data) - head - tail
    mark = f"\n...[trace: {cut} bytes cut]...\n".encode()
    return data[:head] + mark + data[len(data) - tail:], cut


class Store:
    """One trace store (see the module doc). ``level``: off | meta | full."""

    def __init__(self, root, *, level: str = "full", run_id: Optional[str] = None,
                 blob_max: Optional[int] = None, total_max_mb: Optional[float] = None,
                 flush_every_s: float = 1.0):
        if level not in LEVELS:
            raise ValueError(f"trace level must be one of {LEVELS}, not {level!r}")
        self.root = Path(root)
        self.level = level
        self.run_id = run_id or f"{os.getpid()}-{int(time.time())}"
        self.blob_max = int(os.getenv("KB_TRACE_BLOB_MAX") or blob_max or DEFAULT_BLOB_MAX)
        self.total_max = int(float(os.getenv("KB_TRACE_MAX_MB") or total_max_mb or DEFAULT_TOTAL_MAX_MB) * (1 << 20))
        self.scrub = Scrubber()
        self._lock = threading.Lock()
        self._known: set[str] = set()
        self._seq: dict[str, int] = {}
        self._buf: list[str] = []
        self._last_flush = time.monotonic()
        self.flush_every_s = flush_every_s
        self.stats = {"records": 0, "blobs": 0, "blob_hits": 0, "bytes_in": 0, "bytes_stored": 0,
                      "truncated": 0, "dropped": 0, "seconds": 0.0}
        self.capped = False
        self._f = None
        if level == "off":
            return
        (self.root / "blobs").mkdir(parents=True, exist_ok=True)
        self._f = open(self.root / "index.jsonl", "a", encoding="utf-8")
        self.total = self._existing_bytes()

    def _existing_bytes(self) -> int:
        tot = 0
        try:
            for d in os.scandir(self.root / "blobs"):
                if d.is_dir():
                    for f in os.scandir(d.path):
                        tot += f.stat().st_size
        except OSError:
            pass
        return tot

    # ---- blobs
    def blob_path(self, sha: str) -> Path:
        return self.root / "blobs" / sha[:2] / f"{sha}.gz"

    def put(self, data: Any, *, scrub: bool = True, cap: Optional[int] = None) -> Optional[dict]:
        """Store a text (or bytes) -> its ref (see the module doc); None for None."""
        if data is None or self.level == "off":
            return None
        t0 = time.perf_counter()
        if not isinstance(data, (bytes, bytearray)):
            if not isinstance(data, str):
                data = json.dumps(data, default=str, sort_keys=True) if isinstance(data, (dict, list)) else str(data)
            if scrub:
                data = self.scrub(data)
            raw = data.encode("utf-8", "replace")
        else:
            raw = bytes(data)
        n_in = len(raw)
        raw, cut = _truncate(raw, cap if cap is not None else self.blob_max)
        sha = hashlib.sha256(raw).hexdigest()
        ref: dict = {"sha": sha, "n": len(raw)}
        if cut:
            ref["trunc"] = cut
        with self._lock:
            self.stats["bytes_in"] += n_in
            if cut:
                self.stats["truncated"] += 1
        if len(raw) <= INLINE_MAX and self.level == "full":
            try:
                ref = {"text": raw.decode("utf-8")} | ({"trunc": cut} if cut else {})
                return ref
            except UnicodeDecodeError:
                pass
        if self.level == "meta":
            ref["meta"] = True
            return ref
        with self._lock:
            if sha in self._known:
                self.stats["blob_hits"] += 1
                self.stats["seconds"] += time.perf_counter() - t0
                return ref
        p = self.blob_path(sha)
        if p.exists():
            with self._lock:
                self._known.add(sha)
                self.stats["blob_hits"] += 1
            return ref
        z = gzip.compress(raw, compresslevel=6, mtime=0)
        with self._lock:
            if self.total + len(z) > self.total_max:
                self.stats["dropped"] += 1
                if not self.capped:
                    self.capped = True
                    _warn(f"[trace] {self.root}: blob store reached {self.total_max / (1 << 20):.3g} MB "
                          "(KB_TRACE_MAX_MB); keeping hashes and sizes only from now on")
                ref["dropped"] = True
                return ref
            self.total += len(z)
            self.stats["blobs"] += 1
            self.stats["bytes_stored"] += len(z)
            self._known.add(sha)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex[:8]}")
            tmp.write_bytes(z)
            os.replace(tmp, p)
        except OSError as e:
            _warn(f"[trace] blob write failed: {e}")
            ref["dropped"] = True
        with self._lock:
            self.stats["seconds"] += time.perf_counter() - t0
        return ref

    def read(self, ref: Optional[dict]) -> Optional[str]:
        return read_ref(self.root, ref)

    # ---- records
    def seq(self, key: str) -> int:
        """1, 2, ... per key (children of one evaluation, retries of one call)."""
        with self._lock:
            n = self._seq.get(key, 0) + 1
            self._seq[key] = n
            return n

    def write(self, rec: dict) -> None:
        if self._f is None:
            return
        line = json.dumps(rec, default=str, separators=(",", ":"))
        with self._lock:
            self._buf.append(line)
            self.stats["records"] += 1
            if len(self._buf) >= 64 or time.monotonic() - self._last_flush > self.flush_every_s:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if self._buf and self._f is not None and not self._f.closed:
            self._f.write("\n".join(self._buf) + "\n")
            self._f.flush()
        self._buf.clear()
        self._last_flush = time.monotonic()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        with self._lock:
            self._flush_locked()
            if self._f is not None:
                self._f.close()
                self._f = None

    def summary(self) -> dict:
        with self._lock:
            return {"level": self.level, "root": str(self.root), **self.stats,
                    "seconds": round(self.stats["seconds"], 4), "total_bytes": getattr(self, "total", 0),
                    "capped": self.capped}


def read_ref(root, ref: Optional[dict]) -> Optional[str]:
    """The text of a ref (None if it is missing / meta-only / dropped)."""
    if not ref:
        return None
    if "text" in ref:
        return ref["text"]
    sha = ref.get("sha")
    if not sha:
        return None
    p = Path(root) / "blobs" / sha[:2] / f"{sha}.gz"
    try:
        return gzip.decompress(p.read_bytes()).decode("utf-8", "replace")
    except OSError:
        return None


def _warn(msg: str) -> None:
    try:
        from loguru import logger
        logger.warning(msg)
    except Exception:  # noqa: BLE001
        print(msg, file=sys.stderr)


# ---------------------------------------------------------------- the process's store

_store: Optional[Store] = None


def configure(root=None, *, level: Optional[str] = None, run_id: Optional[str] = None, **kw) -> Optional[Store]:
    """Open the process's store at ``root`` (``KB_TRACE_DIR`` overrides) at ``level``
    (``KB_TRACE`` when not given; default off). Returns None when off."""
    global _store
    close()
    level = (level or _env_level()).lower()
    level = {"0": "off", "1": "full"}.get(level, level)
    root = os.getenv("KB_TRACE_DIR") or root
    if level == "off" or root is None:
        return None
    _store = Store(root, level=level, run_id=run_id, **kw)
    _store.write({"t": "run", "id": f"run/{_store.run_id}/{os.getpid()}", "ts": round(time.time(), 6),
                  "pid": os.getpid(), "run": _store.run_id, "level": level, "host": socket.gethostname(),
                  "argv": [_store.scrub(a) for a in sys.argv], "cwd": os.getcwd(),
                  "blob_max": _store.blob_max, "total_max": _store.total_max})
    _store.flush()
    return _store


def store() -> Optional[Store]:
    return _store


def enabled() -> bool:
    return _store is not None


def level() -> str:
    return _store.level if _store is not None else "off"


def flush() -> None:
    if _store is not None:
        _store.flush()


def close() -> Optional[dict]:
    """Flush and close the store -> its summary (sizes, time spent)."""
    global _store
    s = _store
    if s is None:
        return None
    s.write({"t": "close", "id": f"close/{s.run_id}/{os.getpid()}", "ts": round(time.time(), 6),
             "pid": os.getpid(), "stats": s.summary()})
    s.close()
    _store = None
    return s.summary()


atexit.register(flush)


def put(data: Any, **kw) -> Optional[dict]:
    s = _store
    return s.put(data, **kw) if s is not None else None


def put_json(obj: Any) -> Optional[dict]:
    return put(json.dumps(obj, indent=1, default=str, sort_keys=True))


# ---------------------------------------------------------------- context

class scope:
    """``with scope(problem=p, rollout=r): ...`` -- attributes for the records below.

    ``step=`` scopes also record a ``step_error`` when the block raises. The
    entered value is the scope's dict (``set_attrs`` writes into the innermost one)."""
    __slots__ = ("kw", "tok", "d")

    def __init__(self, **kw):
        self.kw, self.tok, self.d = kw, None, None

    def __enter__(self) -> dict:
        if _store is None:
            self.d = {}
            return self.d
        parent = _ctx.get() or {}
        self.d = {**parent, **{k: v for k, v in self.kw.items() if v is not None}}
        self.tok = _ctx.set(self.d)
        return self.d

    def __exit__(self, et, ev, tb):
        if self.tok is not None:
            if et is not None and "step" in self.kw and _store is not None and not issubclass(et, GeneratorExit):
                import traceback
                record("step_error", rid=key("step_error"),
                       error=f"{et.__name__}: {ev}"[:2000],
                       traceback=put("".join(traceback.format_exception(et, ev, tb))))
            try:
                _ctx.reset(self.tok)
            except ValueError:
                pass
        return False


def current() -> dict:
    return _ctx.get() or {}


def set_attrs(**kw) -> None:
    """Update the innermost scope's attributes in place (seen by later records of the scope)."""
    if _store is None:
        return
    d = _ctx.get()
    if d is None:
        d = {}
        _ctx.set(d)
    d.update({k: v for k, v in kw.items()})


CTX_FIELDS = ("run", "problem", "rollout", "step", "slot", "technique", "attempt", "eval", "eval_occ", "purpose",
              "role", "mb_round", "mb_stage", "hint", "ir")


def base_key(ctx: Optional[dict] = None) -> str:
    """The stable id prefix of the current place: ``<run>/<problem>/r<R>/s<S>`` (or shorter)."""
    c = current() if ctx is None else ctx
    parts = [str(c.get("run") or (_store.run_id if _store is not None else "run"))]
    if c.get("problem") is not None:
        parts.append(str(c["problem"]))
    if c.get("rollout") is not None:
        parts.append(f"r{c['rollout']}")
    if c.get("step") is not None:
        parts.append(f"s{c['step']}")
    return "/".join(parts)


def key(*suffix, ctx: Optional[dict] = None) -> str:
    return "/".join([base_key(ctx)] + [str(s) for s in suffix if s is not None and s != ""])


def record(kind: str, *, rid: Optional[str] = None, ctx: Optional[dict] = None, **fields) -> Optional[dict]:
    """Append one record (context fields from the current scope)."""
    s = _store
    if s is None:
        return None
    c = current() if ctx is None else ctx
    rec = {"t": kind, "id": rid or key(kind, uuid.uuid4().hex[:8], ctx=c), "ts": round(time.time(), 6),
           "pid": os.getpid()}
    for k in CTX_FIELDS:
        if c.get(k) is not None:
            rec[k] = c[k]
    if "mb_round" not in rec and telemetry_attr("round") is not None:     # ModelBlaster's mb.round span
        rec["mb_round"] = telemetry_attr("round")
    if "tel_span" not in fields:
        fields["tel_span"] = telemetry_span_id()
    for k, v in fields.items():
        if v is not None:
            rec[k] = s.scrub(v) if isinstance(v, str) else v
    s.write(rec)
    return rec


# the telemetry package as KernelBlaster imports it, and as ModelBlaster loads it (pipeline/kb_telemetry.py)
_TEL_MODULES = ("kernelblaster.telemetry.core", "_mb_kb_telemetry.core")


def _tel_span():
    for n in _TEL_MODULES:
        mod = sys.modules.get(n)
        if mod is not None:
            try:
                return mod.current()
            except Exception:  # noqa: BLE001
                return None
    return None


def telemetry_span_id() -> Optional[int]:
    """The current telemetry span's id (the records' link to telemetry.jsonl), if telemetry is loaded."""
    sp = _tel_span()
    return sp.id if sp is not None and getattr(sp, "real", False) else None


def telemetry_attr(name: str):
    sp = _tel_span()
    try:
        return sp.attrs.get(name) if sp is not None else None
    except Exception:  # noqa: BLE001
        return None
