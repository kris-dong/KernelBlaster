# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spans, collectors and the JSONL event sink (docs/telemetry.md).

A *span* times one region (``with span("build"): ...``). The current span
lives in a :mod:`contextvars` variable, so concurrent asyncio tasks (parallel
problems, rollouts, unit compiles) each nest their own spans: a task started
by ``asyncio.gather`` / ``create_task`` / ``to_thread`` inherits its parent
span at creation and never sees its siblings'.

A span opened with ``collect=True`` (a run, a problem, a step) owns a
:class:`Collector`: every span closing below it, every LLM call and every
counter is added to it, and :meth:`Collector.summary` gives

* ``time_s`` -- wall-clock attribution: each instant of the collector's
  wall time goes to the innermost span(s) open at that instant, split
  evenly between concurrent ones. Sums to ``wall_s`` exactly (the
  collector's own name takes the time no child span covers);
* ``busy_s`` -- exclusive time per span name (duration minus the union of
  its children), summed over concurrent tasks: "agent-seconds";
* ``llm`` -- tokens / latency / cost / truncations, total and per role+model;
* ``counters`` and ``evals`` (evaluate() calls per purpose).

Spans with ``name=None`` only set attributes (role, model, purpose, ...)
inherited by the spans below them; they are not recorded.

Laps: ``lap(name)`` closes the current lap of the nearest span opened with
``laps=True`` and opens a new child span ``name`` -- sequential phases of a
long function marked with one line each, without re-indenting it.

Disabled with ``KB_TELEMETRY=0`` (or :func:`set_enabled`): ``span`` then
returns a shared no-op and nothing is recorded or written.
"""
from __future__ import annotations

import atexit
import contextvars
import functools
import inspect
import itertools
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

_OFF = ("0", "false", "no", "off")
_enabled = os.getenv("KB_TELEMETRY", "1").strip().lower() not in _OFF
_cur: contextvars.ContextVar = contextvars.ContextVar("kb_telemetry_span", default=None)
_ids = itertools.count(1)
_now = time.perf_counter
# perf_counter -> epoch seconds (events carry wall timestamps for cross-process reports)
_EPOCH0 = time.time() - time.perf_counter()


def enabled() -> bool:
    return _enabled


def set_enabled(on: bool) -> None:
    global _enabled
    _enabled = bool(on)


def epoch(t: float) -> float:
    """perf_counter value -> epoch seconds."""
    return _EPOCH0 + t


# ---------------------------------------------------------------- sink

class Sink:
    """Thread-safe buffered JSONL writer (one file per run)."""

    def __init__(self, path: Path, run_id: str, flush_every_s: float = 2.0, flush_lines: int = 256):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._f = open(self.path, "a", encoding="utf-8")
        self._buf: list[str] = []
        self._lock = threading.Lock()
        self._last = time.monotonic()
        self.flush_every_s, self.flush_lines = flush_every_s, flush_lines
        self.events = 0

    def write(self, ev: dict) -> None:
        ev.setdefault("run", self.run_id)
        line = json.dumps(ev, default=str, separators=(",", ":"))
        with self._lock:
            self._buf.append(line)
            self.events += 1
            if len(self._buf) >= self.flush_lines or time.monotonic() - self._last > self.flush_every_s:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if self._buf and not self._f.closed:
            self._f.write("\n".join(self._buf) + "\n")
            self._f.flush()
        self._buf.clear()
        self._last = time.monotonic()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        with self._lock:
            self._flush_locked()
            self._f.close()


_sink: Optional[Sink] = None
_run_id = os.getenv("KB_TELEMETRY_RUN_ID") or f"{os.getpid()}-{int(time.time())}"


def configure(path=None, *, run_id: Optional[str] = None) -> Optional[Sink]:
    """Write events to ``path`` (JSONL, appended). ``None``: no file (in-memory
    collectors only). ``KB_TELEMETRY_FILE`` overrides ``path``."""
    global _sink, _run_id
    if run_id:
        _run_id = run_id
    path = os.getenv("KB_TELEMETRY_FILE") or path
    if _sink is not None:
        _sink.close()
        _sink = None
    if path and _enabled:
        _sink = Sink(Path(path), _run_id)
        _sink.write({"t": "meta", "ts": time.time(), "pid": os.getpid(), "epoch0": _EPOCH0})
    return _sink


def run_id() -> str:
    return _run_id


def sink() -> Optional[Sink]:
    return _sink


def emit(ev: dict) -> None:
    s = _sink
    if s is not None:
        s.write(ev)


def flush() -> None:
    if _sink is not None:
        _sink.flush()


def close() -> None:
    global _sink
    if _sink is not None:
        _sink.close()
        _sink = None


atexit.register(flush)


# ---------------------------------------------------------------- LLM aggregation

LLM_FIELDS = ("calls", "errors", "input_tokens", "uncached_input_tokens", "cache_read_input_tokens",
              "cache_write_input_tokens", "output_tokens", "cost_usd", "est_usd", "truncated", "latency_s",
              "server_latency_s", "ttft_s", "provider_retries", "backoff_s", "backoffs", "relay_queue_s",
              "relay_network_s")


def _llm_zero() -> dict:
    return {k: 0 for k in LLM_FIELDS} | {"latency_max_s": 0.0}


class LLMStats:
    __slots__ = ("total", "by")

    def __init__(self):
        self.total = _llm_zero()
        self.by: dict[str, dict] = {}

    def add(self, rec: dict) -> None:
        key = f"{rec.get('role') or 'other'} {rec.get('model') or '?'}"
        b = self.by.get(key)
        if b is None:
            b = self.by[key] = {"role": rec.get("role") or "other", "model": rec.get("model") or "?", **_llm_zero()}
        for d in (self.total, b):
            for k in LLM_FIELDS:
                v = rec.get(k)
                if v:
                    d[k] += v
            d["latency_max_s"] = max(d["latency_max_s"], rec.get("latency_s") or 0.0)

    def merge(self, other: "LLMStats") -> None:
        for d_src, d_dst in [(other.total, self.total)]:
            for k in LLM_FIELDS:
                d_dst[k] += d_src[k]
            d_dst["latency_max_s"] = max(d_dst["latency_max_s"], d_src["latency_max_s"])
        for key, b in other.by.items():
            mine = self.by.setdefault(key, {"role": b["role"], "model": b["model"], **_llm_zero()})
            for k in LLM_FIELDS:
                mine[k] += b[k]
            mine["latency_max_s"] = max(mine["latency_max_s"], b["latency_max_s"])

    def to_json(self, nd: int = 4) -> dict:
        def r(d):
            return {k: (round(v, nd) if isinstance(v, float) else v) for k, v in d.items()}
        return {"total": r(self.total), "by_role_model": [r(b) for b in sorted(self.by.values(),
                                                                                key=lambda x: (x["role"], x["model"]))]}


# ---------------------------------------------------------------- collector

GPU_SPANS = ("run.exec", "ncu.exec")


class Collector:
    """Aggregates everything below one span (see the module doc)."""

    __slots__ = ("owner", "records", "busy", "llm", "counters", "evals", "gpu", "_lock")

    def __init__(self, owner: "Span"):
        self.owner = owner
        self.records: list[tuple] = []       # (t0, t1, name, id, parent id) of closed spans below
        self.busy: dict[str, float] = {}
        self.llm = LLMStats()
        self.counters: dict[str, float] = {}
        self.evals: dict[str, list] = {}      # purpose -> [n, seconds, failed]
        self.gpu: list[tuple] = []           # (t0, t1, device) of GPU-holding spans
        self._lock = threading.Lock()        # spans may close on worker threads (to_thread)

    def add_span(self, sp: "Span", self_s: float) -> None:
        with self._lock:
            self.records.append((sp.t0, sp.t1, sp.name, sp.id, sp.parent_id))
            self.busy[sp.name] = self.busy.get(sp.name, 0.0) + self_s
            if sp.name in GPU_SPANS:
                self.gpu.append((sp.t0, sp.t1, sp.attrs.get("device") or "default"))
            if sp.name == "eval":
                e = self.evals.setdefault(sp.attrs.get("purpose") or "other", [0, 0.0, 0])
                e[0] += 1
                e[1] += sp.t1 - sp.t0
                if sp.attrs.get("ok") is False:
                    e[2] += 1

    def add_count(self, name: str, n: float) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + n

    def add_llm(self, rec: dict) -> None:
        with self._lock:
            self.llm.add(rec)

    # -- summaries
    def attribution(self, t_end: Optional[float] = None) -> dict[str, float]:
        """Wall-clock attribution (sweep line over the recorded spans + the owner)."""
        own = self.owner
        t_end = t_end if t_end is not None else (own.t1 if own.t1 is not None else _now())
        recs = [(own.t0, t_end, own.name, own.id, None)] + [
            (max(a, own.t0), min(b, t_end), n, i, p) for a, b, n, i, p in self.records]
        ids = {r[3] for r in recs}
        parent = {r[3]: (r[4] if r[4] in ids else own.id) for r in recs}
        parent[own.id] = None
        depth: dict[int, int] = {own.id: 0}

        def d(i):
            stack = []
            while i not in depth:
                stack.append(i)
                i = parent[i]
            base = depth[i]
            for c in reversed(stack):
                base += 1
                depth[c] = base
            return depth[stack[0]] if stack else base
        evs = []
        names = {}
        for a, b, n, i, p in recs:
            if b <= a and i != own.id:
                continue
            names[i] = n
            evs.append((a, 1, d(i), i))
            evs.append((b, 0, -d(i), i))
        evs.sort()
        out: dict[str, float] = {}
        active_children: dict[int, int] = {}
        leaves: set = set()
        active: set = set()
        last = None
        for t, kind, _, i in evs:
            if last is not None and t > last and leaves:
                share = (t - last) / len(leaves)
                for l in leaves:
                    out[names[l]] = out.get(names[l], 0.0) + share
            last = t
            p = parent.get(i)
            if kind == 1:
                active.add(i)
                active_children.setdefault(i, 0)
                if active_children[i] == 0:
                    leaves.add(i)
                if p is not None and p in active:
                    active_children[p] += 1
                    leaves.discard(p)
            else:
                active.discard(i)
                leaves.discard(i)
                if p is not None and p in active:
                    active_children[p] -= 1
                    if active_children[p] == 0:
                        leaves.add(p)
        return out

    def gpu_busy(self, t_end: Optional[float] = None) -> dict:
        """{device: seconds} union of GPU-holding spans, clipped to the owner."""
        own = self.owner
        t_end = t_end if t_end is not None else (own.t1 if own.t1 is not None else _now())
        by: dict[str, list] = {}
        for a, b, dev in self.gpu:
            by.setdefault(dev, []).append((max(a, own.t0), min(b, t_end)))
        return {dev: union_length(iv) for dev, iv in by.items()}

    def summary(self, nd: int = 3) -> dict:
        own = self.owner
        t_end = own.t1 if own.t1 is not None else _now()
        wall = max(t_end - own.t0, 0.0)
        with self._lock:
            att = self.attribution(t_end)
            busy = dict(self.busy)
            llm = self.llm.to_json()
            counters = dict(self.counters)
            evals = {k: {"n": v[0], "s": round(v[1], nd), "failed": v[2]} for k, v in self.evals.items()}
            gpu = self.gpu_busy(t_end)
        tot = sum(att.values()) or 1.0
        order = sorted(att, key=lambda k: -att[k])
        return {
            "wall_s": round(wall, nd),
            "time_s": {k: round(att[k], nd) for k in order},
            "time_pct": {k: round(100.0 * att[k] / tot, 1) for k in order},
            "busy_s": {k: round(v, nd) for k, v in sorted(busy.items(), key=lambda kv: -kv[1])},
            "llm": llm,
            "counters": {k: (round(v, nd) if isinstance(v, float) else v) for k, v in sorted(counters.items())},
            "evals": evals,
            "gpu_busy_s": {k: round(v, nd) for k, v in gpu.items()},
        }


def union_length(iv: list[tuple]) -> float:
    tot, cur_a, cur_b = 0.0, None, None
    for a, b in sorted(iv):
        if b <= a:
            continue
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                tot += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        tot += cur_b - cur_a
    return tot


# ---------------------------------------------------------------- spans

class Span:
    __slots__ = ("id", "name", "parent", "parent_id", "attrs", "t0", "t1", "children", "collector",
                 "collectors", "laps", "lap", "real", "_token", "_child_lock", "emit")

    def __init__(self, name: Optional[str], parent: Optional["Span"], attrs: dict, collect: bool,
                 laps: bool, emit: bool = True):
        self.real = name is not None
        self.name = name
        self.parent = parent
        rp = parent                                     # nearest recorded span
        while rp is not None and not rp.real:
            rp = rp.parent
        self.parent_id = rp.id if rp is not None else None
        self.id = next(_ids) if self.real else (rp.id if rp is not None else 0)
        base = parent.attrs if parent is not None else {}
        self.attrs = {**base, **attrs} if attrs else base
        self.t0 = _now()
        self.t1 = None
        self.children: list = []
        self.collector = Collector(self) if collect and self.real else None
        pc = parent.collectors if parent is not None else ()
        self.collectors = (pc + (self.collector,)) if self.collector is not None else pc
        self.laps = laps
        self.lap: Optional[Span] = None
        self._token = None
        self._child_lock = None
        self.emit = emit

    # attribute helpers
    def set(self, **attrs) -> None:
        if self.attrs is (self.parent.attrs if self.parent is not None else None) or not self.real:
            self.attrs = dict(self.attrs)
        self.attrs.update(attrs)

    def count(self, name: str, n: float = 1) -> None:
        for c in self.collectors:
            c.add_count(name, n)

    def _record_child(self, a: float, b: float) -> None:
        self.children.append((a, b))

    def close(self, t1: Optional[float] = None) -> None:
        if self.t1 is not None:
            return
        if self.lap is not None:
            self.lap.close()
            self.lap = None
        self.t1 = t1 if t1 is not None else _now()
        if not self.real:
            return
        dur = self.t1 - self.t0
        self_s = max(dur - union_length(self.children), 0.0)
        rp = self.parent
        while rp is not None and not rp.real:
            rp = rp.parent
        if rp is not None:
            rp._record_child(self.t0, self.t1)
        for c in self.collectors:
            if c.owner is not self:
                c.add_span(self, self_s)
        if self.emit and _sink is not None:
            ev = {"t": "span", "id": self.id, "parent": self.parent_id, "name": self.name,
                  "ts": round(epoch(self.t0), 6), "dur": round(dur, 6), "self": round(self_s, 6)}
            a = {k: v for k, v in self.attrs.items() if v is not None and not k.startswith("_")}
            if a:
                ev["attrs"] = a
            _sink.write(ev)


class _NoSpan:
    """Shared no-op returned when telemetry is disabled."""
    attrs: dict = {}
    collector = None
    collectors = ()
    real = False
    name = None
    id = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def set(self, **kw):
        pass

    def count(self, *a, **kw):
        pass

    def summary(self):
        return None


NOSPAN = _NoSpan()


class _SpanCM:
    __slots__ = ("name", "attrs", "collect", "laps", "span", "emit")

    def __init__(self, name, attrs, collect, laps, emit):
        self.name, self.attrs, self.collect, self.laps, self.emit = name, attrs, collect, laps, emit
        self.span = None

    def __enter__(self) -> Span:
        sp = Span(self.name, _cur.get(), self.attrs, self.collect, self.laps, self.emit)
        sp._token = _cur.set(sp)
        self.span = sp
        return sp

    def __exit__(self, et, ev, tb):
        sp = self.span
        if et is not None and sp.real:
            sp.set(error=et.__name__)
        sp.close()
        try:
            _cur.reset(sp._token)
        except ValueError:                       # closed in another context (should not happen)
            _cur.set(sp.parent)
        return False


def span(name: Optional[str] = None, *, collect: bool = False, laps: bool = False, emit: bool = True, **attrs):
    """``with span("build", slot=s) as sp: ...`` (see the module doc)."""
    if not _enabled:
        return NOSPAN
    return _SpanCM(name, attrs, collect, laps, emit)


def attrs(**kw):
    """Attributes only (inherited by the spans below), no recorded span."""
    return span(None, **kw)


def current() -> Optional[Span]:
    return _cur.get() if _enabled else None


def count(name: str, n: float = 1) -> None:
    """Add ``n`` to counter ``name`` in every collector above the current span."""
    if not _enabled:
        return
    sp = _cur.get()
    if sp is not None:
        sp.count(name, n)
    if _sink is not None:
        ev = {"t": "count", "name": name, "n": n}
        if sp is not None and sp.attrs.get("problem") is not None:
            ev["problem"] = sp.attrs["problem"]
        _sink.write(ev)


def lap(name: str, **set_attrs) -> None:
    """Close the current lap of the nearest ``laps=True`` span and open ``name``.
    ``set_attrs`` are set on that span (and so inherited by later laps)."""
    if not _enabled:
        return
    sp = _cur.get()
    while sp is not None and not sp.laps:
        sp = sp.parent
    if sp is None:
        return
    if set_attrs:
        sp.set(**set_attrs)
    if sp.lap is not None:
        sp.lap.close()
    child = Span(name, sp, {}, False, False)
    sp.lap = child
    _cur.set(child)


def end_laps(sp) -> None:
    """Close ``sp``'s open lap and make ``sp`` current again (the ``laps`` owner's exit)."""
    if not _enabled or not isinstance(sp, Span):
        return
    if sp.lap is not None:
        sp.lap.close()
        sp.lap = None
    _cur.set(sp)


def record_span(name: str, t0: float, t1: float, parent: Optional[Span] = None, **attrs) -> None:
    """A span measured elsewhere (e.g. a driver phase), as a child of ``parent``."""
    if not _enabled:
        return
    parent = parent if parent is not None else _cur.get()
    sp = Span(name, parent, attrs, False, False)
    sp.t0 = t0
    sp.close(t1)


class within:
    """``with within(sp): ...`` -- make ``sp`` the current span (e.g. in a
    worker thread, whose context does not inherit the submitter's)."""
    __slots__ = ("sp", "tok")

    def __init__(self, sp):
        self.sp, self.tok = sp, None

    def __enter__(self):
        if _enabled and isinstance(self.sp, Span):
            self.tok = _cur.set(self.sp)
        return self.sp

    def __exit__(self, *exc):
        if self.tok is not None:
            _cur.reset(self.tok)
        return False


def traced(name: str, **span_attrs):
    """Decorator: run the (sync or async) function inside ``span(name, **span_attrs)``."""
    def deco(fn: Callable):
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def aw(*a, **kw):
                if not _enabled:
                    return await fn(*a, **kw)
                with span(name, **span_attrs):
                    return await fn(*a, **kw)
            aw.__kb_traced__ = name
            return aw

        @functools.wraps(fn)
        def w(*a, **kw):
            if not _enabled:
                return fn(*a, **kw)
            with span(name, **span_attrs):
                return fn(*a, **kw)
        w.__kb_traced__ = name
        return w
    return deco


def summary_of(sp) -> Optional[dict]:
    """The collector summary of a ``collect=True`` span (None when disabled)."""
    c = getattr(sp, "collector", None)
    return c.summary() if c is not None else None


def ctx_summary(sp) -> Optional[dict]:
    """Compact per-step form of :func:`summary_of` (for StepLog.context)."""
    s = summary_of(sp)
    if s is None:
        return None
    t = s["llm"]["total"]
    out = {"wall_s": round(s["wall_s"], 2), "time_s": {k: round(v, 2) for k, v in s["time_s"].items() if v >= 0.005},
           "llm": {"calls": t["calls"], "in": t["input_tokens"], "out": t["output_tokens"],
                   "cache_read": t["cache_read_input_tokens"], "truncated": t["truncated"],
                   "latency_s": round(t["latency_s"], 2), "usd": round(t["cost_usd"] or t["est_usd"], 5)},
           "evals": {k: v["n"] for k, v in s["evals"].items()}}
    hits, miss = s["counters"].get("build.cache_hit", 0), s["counters"].get("build.cache_miss", 0)
    if hits or miss:
        out["build_cache"] = {"hit": hits, "miss": miss}
    return out


def merge_into(sp, data: Any) -> None:
    """Annotate a span with extra attributes after the fact."""
    if isinstance(sp, Span):
        sp.set(**data)
