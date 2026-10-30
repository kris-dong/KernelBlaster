# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Content-addressed cache for build products (objects, linked binaries).

A key is the SHA-256 of everything that determines the output: the exact
compile / link command (with the per-job paths replaced by placeholders), the
source text, the text of every header it includes from the job directory, and
the identity (path, size, mtime) of absolute-path inputs such as a driver's
``.incbin`` data file. Same key -> byte-identical inputs to the same tool, so
the cached file is reused instead of rebuilt.

Concurrency: entries are written to a unique temp name in the cache directory
and ``os.replace``d into place, so readers only ever see complete files, and
two builders racing on one key (other processes sharing the directory) both
produce a valid entry. Within one process an ``asyncio.Lock`` per key makes the
second requester wait for the first build instead of duplicating it.

Shared mode (compile pool, docs ``l3_farm/analysis/compile_pool.md``): when
``KERNELBLASTER_BUILD_CACHE=<dir>`` is set (or a ``BuildCache`` is made with
``shared=True``) the directory is shared by every problem and process of a
driver:

* *single flight across processes*: a ``flock`` per key, so N controllers that
  ask for the same unit at the same moment build it once;
* *negative entries*: a deterministic compile error (a :class:`BuildError`
  raised by the build) is stored with its text and served again for
  ``neg_ttl`` seconds (``KERNELBLASTER_BUILD_CACHE_NEG_TTL``, default 3600; 0
  = off). Timeouts, killed compilers and resource errors are never stored;
* *size cap with LRU eviction* (``KERNELBLASTER_BUILD_CACHE_MAX_GB``, default
  unlimited): a hit touches the entry, eviction removes the least recently used
  entries first, never one younger than ``min_age`` seconds (a reader that just
  got a path can still link it);
* *metrics*: hits / misses / negative hits / bytes and compile seconds saved,
  per process in ``<stats dir>/*.json`` (``KERNELBLASTER_BUILD_CACHE_STATS_DIR``,
  default ``<root>/.stats``), summarised by :func:`summarize_stats` and by the
  telemetry counters ``build.cache_*``.

With the variable unset nothing of this runs: the cache is the per-problem
``<work>/.build_cache`` it always was.
"""
from __future__ import annotations

import asyncio
import atexit
import fcntl
import hashlib
import json
import os
import random
import re
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Optional

from ...telemetry.core import count as _tel_count

# absolute paths a TU pulls in that are not part of its text
_ABS_INPUT_RE = re.compile(r'(?:\.incbin\s+\\?"|#\s*include\s+")(/[^"\\]+)')
_ABS_INCLUDE_RE = re.compile(r'#\s*include\s+"(/[^"\\]+)')

_OFF = ("", "0", "off", "false", "no")
_CONTENT_SHA: dict = {}            # (path, size, mtime_ns) -> sha256 of the content


class BuildError(Exception):
    """A build step failed. ``text`` is the compiler output; ``cacheable`` is False for failures that say
    nothing about the source (timeouts, a killed compiler, no memory / disk): those are never stored."""

    def __init__(self, name: str = "", text: str = "", cacheable: bool = True):
        super().__init__(name or text[:200])
        self.name, self.text, self.cacheable = name, text, cacheable


class CachedBuildError(BuildError):
    """A stored compile error served again (no compiler run)."""


# output that says the compiler ran out of something or was killed, not that the source is wrong
_INFRA_RE = re.compile(r"No space left|Cannot allocate memory|out of memory|Killed|Resource temporarily unavailable|"
                       r"Segmentation fault|internal error|Input/output error|Transport endpoint|Connection (?:reset|refused)|"
                       r"Stale file handle|Too many open files|fork:|timed out", re.I)


def deterministic_failure(rc: int, text: str) -> bool:
    """Is a failed compiler run (``rc`` != 0) a verdict on the source, worth caching?"""
    return 0 < rc < 128 and not _INFRA_RE.search(text or "")


def shared_dir() -> Optional[Path]:
    """The shared cache directory (``KERNELBLASTER_BUILD_CACHE``), None when off. ``~`` is expanded."""
    v = os.getenv("KERNELBLASTER_BUILD_CACHE", "").strip()
    return None if v.lower() in _OFF else Path(os.path.expanduser(v))


def _content_sha(path: str) -> str:
    """sha256 of a file's content. Small files (headers) are hashed every time -- a rewrite with the same size and
    mtime must not hit; only files over 16 MiB are remembered per (path, size, mtime)."""
    try:
        st = os.stat(path)
    except OSError:
        return "missing"
    k = (path, st.st_size, st.st_mtime_ns)
    if st.st_size > (16 << 20) and k in _CONTENT_SHA:
        return _CONTENT_SHA[k]
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
    except OSError:
        return "missing"
    if st.st_size > (16 << 20):
        _CONTENT_SHA[k] = h.hexdigest()
    return h.hexdigest()


def external_inputs(text: str, content: bool = False) -> list[str]:
    """``path:size:mtime_ns`` for every absolute ``.incbin`` / ``#include "..."``
    target in ``text`` (missing files are keyed as missing; the build fails
    on its own). ``content=True`` (shared cache: the same header on another
    node has another mtime) keys an ``#include``d header by its sha256 instead; a ``.incbin``
    data file (possibly GBs) stays keyed by size and mtime."""
    out = []
    inc = set(_ABS_INCLUDE_RE.findall(text)) if content else set()
    for p in sorted(set(_ABS_INPUT_RE.findall(text))):
        if p in inc:
            out.append(f"{p}:sha256:{_content_sha(p)}")
            continue
        try:
            st = os.stat(p)
            out.append(f"{p}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            out.append(f"{p}:missing")
    return out


def cache_key(*parts: str | bytes | Iterable[str]) -> str:
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, (list, tuple)):
            part = "\n".join(part)
        b = part if isinstance(part, bytes) else str(part).encode()
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- toolchain identity

_TOOL_ID: dict = {}
# environment that changes what nvcc / the host compiler produce
_TOOL_ENV = ("NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "CUDAFLAGS")


def toolchain_text(nvcc_out: str, cxx_out: str = "") -> str:
    """The identity string of a toolchain from the output of ``nvcc --version`` and ``c++ --version``
    (the release / build lines and the first line). The local probe and a remote host's probe build it the
    same way, so two machines with the same toolchain give the same string."""
    nv = " | ".join(l.strip() for l in nvcc_out.splitlines() if "release" in l or l.strip().startswith("Build"))
    cx = (cxx_out.splitlines() or [""])[0].strip()
    return f"nvcc[{nv or nvcc_out.strip()[-200:]}] cxx[{cx}]"


def tool_env_text(env: Optional[dict] = None) -> str:
    """The compiler-relevant environment (flags / include paths nvcc and the host compiler pick up)."""
    e = os.environ if env is None else env
    return ";".join(f"{k}={e.get(k, '')}" for k in _TOOL_ENV)


def tool_core(nvcc: str = "nvcc") -> Optional[str]:
    """``toolchain_text`` of the local nvcc (probed once per path), None when nvcc cannot be run."""
    if nvcc not in _TOOL_ID:
        def run(cmd):
            try:
                cp = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                return cp.stdout + cp.stderr
            except (OSError, subprocess.SubprocessError):
                return ""
        out = run([nvcc, "--version"])
        _TOOL_ID[nvcc] = toolchain_text(out, run(["c++", "--version"])) if out else None
    return _TOOL_ID[nvcc]


def tool_id(nvcc: str = "nvcc") -> str:
    """Identity of the local toolchain (nvcc release + build, host compiler, compiler env); part of every key of a
    shared cache."""
    core = tool_core(nvcc)
    return f"{core} env[{tool_env_text()}]" if core else f"nvcc-unavailable:{nvcc}"


# ---------------------------------------------------------------- the cache

class BuildCache:
    """``get_or_build(key, suffix, build)`` -> path of the cached product.

    ``max_bytes`` caps the directory (LRU eviction), ``neg_ttl`` > 0 stores deterministic compile errors,
    ``shared`` adds the cross-process single flight. All three default from the environment only when
    ``shared`` (see the module doc); a plain ``BuildCache(root)`` behaves as before."""

    def __init__(self, root: Path, *, shared: bool = False, max_bytes: Optional[int] = None,
                 neg_ttl: Optional[float] = None, min_age: float = 600.0, stats_dir: Optional[Path] = None,
                 lock_wait_s: float = 1800.0):
        self.root = Path(root)
        self.shared = shared
        if shared:
            gb = os.getenv("KERNELBLASTER_BUILD_CACHE_MAX_GB", "").strip()
            self.max_bytes = max_bytes if max_bytes is not None else (int(float(gb) * 2**30) if gb else None)
            ttl = os.getenv("KERNELBLASTER_BUILD_CACHE_NEG_TTL", "").strip()
            self.neg_ttl = neg_ttl if neg_ttl is not None else (float(ttl) if ttl else 3600.0)
            sd = os.getenv("KERNELBLASTER_BUILD_CACHE_STATS_DIR", "").strip()
            self.stats_dir = Path(stats_dir) if stats_dir else (Path(os.path.expanduser(sd)) if sd else self.root / ".stats")
        else:
            self.max_bytes, self.neg_ttl, self.stats_dir = max_bytes, (neg_ttl or 0.0), (Path(stats_dir) if stats_dir else None)
        self.min_age, self.lock_wait_s = float(min_age), float(lock_wait_s)
        self._locks: dict[str, asyncio.Lock] = {}
        self.hits = 0
        self.misses = 0
        self.neg_hits = 0
        self.st = {"hits": 0, "misses": 0, "neg_hits": 0, "neg_stores": 0, "bytes_saved": 0, "seconds_saved": 0.0,
                   "bytes_built": 0, "seconds_built": 0.0, "lock_waits": 0, "lock_wait_s": 0.0, "evicted": 0,
                   "evicted_bytes": 0, "failed_uncached": 0}
        self._added = 0
        self._last_scan = time.time()
        self._last_flush = 0.0
        self._sid = f"{re.sub(r'[^A-Za-z0-9_.-]', '_', socket.gethostname())}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        if self.stats_dir is not None:
            atexit.register(self.flush_stats)

    # ---- paths
    def path(self, key: str, suffix: str) -> Path:
        return self.root / key[:2] / f"{key}{suffix}"

    def _err_path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.err"

    @staticmethod
    def _meta_path(dst: Path) -> Path:
        return dst.with_name(dst.name + ".meta")

    # ---- stats
    def stats(self) -> dict:
        return {**self.st, "root": str(self.root), "shared": self.shared, "max_bytes": self.max_bytes,
                "neg_ttl": self.neg_ttl, "t": round(time.time(), 1), "pid": os.getpid()}

    def flush_stats(self) -> None:
        """Write this process's counters to ``<stats dir>/<id>.json`` (atomic)."""
        if self.stats_dir is None:
            return
        try:
            self.stats_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.stats_dir / f".{self._sid}.tmp"
            tmp.write_text(json.dumps(self.stats(), sort_keys=True))
            os.replace(tmp, self.stats_dir / f"{self._sid}.json")
            self._last_flush = time.time()
        except OSError:
            pass

    def _maybe_flush(self) -> None:
        if self.stats_dir is not None and time.time() - self._last_flush > 5.0:
            self.flush_stats()

    def _hit(self, dst: Path) -> None:
        self.hits += 1
        self.st["hits"] += 1
        size, secs = 0, 0.0
        try:
            size = dst.stat().st_size
        except OSError:
            pass
        try:
            secs = float(json.loads(self._meta_path(dst).read_text()).get("seconds", 0.0))
        except (OSError, ValueError):
            pass
        self.st["bytes_saved"] += size
        self.st["seconds_saved"] = round(self.st["seconds_saved"] + secs, 4)
        _tel_count("build.cache_hit")
        if self.shared:
            _tel_count("build.cache_ms_saved", round(secs * 1000))
            _tel_count("build.cache_bytes_saved", size)
            try:                                  # LRU: a use refreshes the entry (at most once a minute)
                if time.time() - dst.stat().st_mtime > 60:
                    os.utime(dst)
            except OSError:
                pass
        self._maybe_flush()

    # ---- negative entries
    def _neg_get(self, key: str) -> Optional[str]:
        if self.neg_ttl <= 0:
            return None
        p = self._err_path(key)
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            return None
        if time.time() - float(d.get("t", 0)) > self.neg_ttl:
            return None
        return str(d.get("text", ""))

    def _neg_put(self, key: str, text: str, seconds: float) -> None:
        p = self._err_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(json.dumps({"text": text[-12000:], "t": time.time(), "seconds": round(seconds, 4)}))
            os.replace(tmp, p)
            self.st["neg_stores"] += 1
        finally:
            if tmp.exists():
                tmp.unlink()

    def neg_lookup(self, key: str) -> Optional[str]:
        """A stored compile error for ``key`` (counted as a negative hit), else None."""
        t = self._neg_get(key)
        if t is not None:
            self._neg_hit()
        return t

    def neg_store(self, key: str, text: str, seconds: float = 0.0) -> None:
        if self.neg_ttl > 0 and text:
            self._neg_put(key, text, seconds)

    # ---- cross-process single flight
    async def _flock(self, key: str) -> Optional[int]:
        d = self.root / ".locks"
        d.mkdir(parents=True, exist_ok=True)
        fd = os.open(d / key, os.O_CREAT | os.O_RDWR, 0o644)
        t0 = time.time()
        waited = False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if waited:
                    self.st["lock_waits"] += 1
                    self.st["lock_wait_s"] = round(self.st["lock_wait_s"] + time.time() - t0, 3)
                return fd
            except BlockingIOError:
                waited = True
                if time.time() - t0 > self.lock_wait_s:
                    os.close(fd)
                    return None                  # a stuck builder: build it ourselves
                await asyncio.sleep(0.03 + random.random() * 0.05)

    @staticmethod
    def _unlock(fd: Optional[int]) -> None:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    async def get_or_build(
        self, key: str, suffix: str, build: Callable[[Path], Awaitable[None]]
    ) -> tuple[Path, bool]:
        """Return ``(path, hit)``. ``build(tmp)`` must write the product to
        ``tmp``; it raises on failure and nothing is cached -- except that a
        :class:`BuildError` (a compile error) is stored as a negative entry
        when ``neg_ttl`` > 0 and its ``cacheable`` flag is set; later requests
        for the key raise :class:`CachedBuildError` with the stored text."""
        dst = self.path(key, suffix)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if dst.exists():
                self._hit(dst)
                return dst, True
            if (neg := self._neg_get(key)) is not None:
                self._neg_hit()
                raise CachedBuildError(key[:12], neg)
            dst.parent.mkdir(parents=True, exist_ok=True)
            fd = await self._flock(key) if self.shared else None
            try:
                if fd is not None:               # another process may have built it while we waited
                    if dst.exists():
                        self._hit(dst)
                        return dst, True
                    if (neg := self._neg_get(key)) is not None:
                        self._neg_hit()
                        raise CachedBuildError(key[:12], neg)
                tmp = dst.with_name(f".{dst.name}.{uuid.uuid4().hex[:8]}.tmp")
                t0 = time.time()
                try:
                    await build(tmp)
                    if not tmp.exists():
                        raise FileNotFoundError(f"build did not produce {tmp}")
                    secs = time.time() - t0
                    size = tmp.stat().st_size
                    os.replace(tmp, dst)
                except BuildError as e:
                    if self.neg_ttl > 0 and e.cacheable and e.text:
                        self._neg_put(key, e.text, time.time() - t0)
                    else:
                        self.st["failed_uncached"] += 1
                    raise
                finally:
                    if tmp.exists():
                        tmp.unlink()
                if self.shared:                  # what a later hit saves (seconds / bytes)
                    try:
                        self._meta_path(dst).write_text(json.dumps({"seconds": round(secs, 4), "bytes": size}))
                    except OSError:
                        pass
            finally:
                self._unlock(fd)
            self.misses += 1
            self.st["misses"] += 1
            self.st["bytes_built"] += size
            self.st["seconds_built"] = round(self.st["seconds_built"] + secs, 4)
            _tel_count("build.cache_miss")
            self._added += size
            self._maybe_flush()
            if self.max_bytes and (self._added > self.max_bytes // 20 or time.time() - self._last_scan > 300):
                self._added, self._last_scan = 0, time.time()
                await asyncio.to_thread(self.evict)
            return dst, False

    def _neg_hit(self) -> None:
        self.neg_hits += 1
        self.st["neg_hits"] += 1
        _tel_count("build.cache_neg_hit")
        self._maybe_flush()

    # ---- size cap
    def scan(self) -> list[tuple[float, int, Path]]:
        """``(mtime, size, path)`` of every entry file (not temp, stats or lock files)."""
        out = []
        try:
            subs = list(os.scandir(self.root))
        except OSError:
            return out
        for d in subs:
            if not d.is_dir() or d.name.startswith("."):
                continue
            try:
                for f in os.scandir(d.path):
                    if f.name.startswith(".") or not f.is_file():
                        continue
                    st = f.stat()
                    out.append((st.st_mtime, st.st_size, Path(f.path)))
            except OSError:
                continue
        return out

    def size_bytes(self) -> int:
        return sum(s for _, s, _ in self.scan())

    def evict(self, max_bytes: Optional[int] = None) -> dict:
        """Remove least recently used entries until the cache is under 80% of the cap (never one younger
        than ``min_age``), expired negative entries, and stale temp / lock files. One evictor at a time per
        directory (flock); -> {"files", "bytes", "kept_bytes"}."""
        cap = max_bytes if max_bytes is not None else self.max_bytes
        rep = {"files": 0, "bytes": 0, "kept_bytes": 0}
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            lk = os.open(self.root / ".evict.lock", os.O_CREAT | os.O_RDWR, 0o644)
        except OSError:
            return rep
        try:
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return rep
            now = time.time()
            ents = self.scan()
            for m, s, p in list(ents):
                if p.suffix == ".err" and now - m > max(self.neg_ttl, 1.0):
                    self._rm(p, rep, s)
                    ents.remove((m, s, p))
            # sidecars follow their entry; count them with it
            data = [(m, s, p) for m, s, p in ents if p.suffix != ".meta"]
            total = sum(s for _, s, _ in ents)
            if cap and total > cap:
                goal = int(cap * 0.8)
                for m, s, p in sorted(data):
                    if total <= goal:
                        break
                    if now - m < self.min_age:
                        continue
                    meta = self._meta_path(p)
                    ms = meta.stat().st_size if meta.exists() else 0
                    self._rm(p, rep, s)
                    if ms:
                        self._rm(meta, rep, ms)
                    total -= s + ms
            rep["kept_bytes"] = total
            # stale temp files (a killed builder) and lock files
            for sub in (self.root, self.root / ".locks"):
                try:
                    for f in os.scandir(sub):
                        if f.is_file() and ((f.name.endswith(".tmp") and now - f.stat().st_mtime > 6 * 3600)
                                            or (sub.name == ".locks" and now - f.stat().st_mtime > 3600)):
                            os.unlink(f.path)
                except OSError:
                    pass
            for d in list(os.scandir(self.root)):
                if d.is_dir() and not d.name.startswith("."):
                    try:
                        for f in os.scandir(d.path):
                            if f.name.endswith(".tmp") and now - f.stat().st_mtime > 6 * 3600:
                                os.unlink(f.path)
                    except OSError:
                        pass
        finally:
            os.close(lk)
        self.st["evicted"] += rep["files"]
        self.st["evicted_bytes"] += rep["bytes"]
        return rep

    @staticmethod
    def _rm(p: Path, rep: dict, size: int) -> None:
        try:
            p.unlink()
            rep["files"] += 1
            rep["bytes"] += size
        except OSError:
            pass


def summarize_stats(root: Path, stats_dir: Optional[Path] = None) -> dict:
    """Sum the per-process stats files of a cache directory -> totals, hit rate, files, size."""
    sd = Path(stats_dir) if stats_dir else Path(root) / ".stats"
    tot: dict = {}
    n = 0
    for f in sorted(sd.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        n += 1
        for k, v in d.items():
            if isinstance(v, (int, float)) and k not in ("t", "pid", "max_bytes", "neg_ttl"):
                tot[k] = round(tot.get(k, 0) + v, 4)
    req = tot.get("hits", 0) + tot.get("misses", 0) + tot.get("neg_hits", 0)
    tot["processes"] = n
    tot["requests"] = req
    tot["hit_rate"] = round((tot.get("hits", 0) + tot.get("neg_hits", 0)) / req, 4) if req else None
    c = BuildCache(root)
    ents = [e for e in c.scan() if e[2].suffix not in (".meta", ".err")]
    tot["entries"], tot["cache_bytes"] = len(ents), sum(s for _, s, _ in c.scan())
    return tot


def link_or_copy(src: Path, dst: Path) -> None:
    """Materialise a cached product at ``dst`` (hard link when possible, so a
    100 MB binary with linked-in data costs nothing; copy across devices)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        os.link(src, tmp)
    except OSError:
        import shutil

        shutil.copyfile(src, tmp)
        os.chmod(tmp, 0o755)
    os.replace(tmp, dst)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="shared build cache: stats / evict / clear")
    ap.add_argument("cmd", choices=["stats", "evict", "clear"])
    ap.add_argument("root", nargs="?", default=str(shared_dir() or ""))
    ap.add_argument("--max-gb", type=float, default=None)
    a = ap.parse_args(argv)
    if not a.root:
        ap.error("no cache directory (set KERNELBLASTER_BUILD_CACHE or pass one)")
    root = Path(a.root)
    if a.cmd == "stats":
        print(json.dumps(summarize_stats(root), indent=1, sort_keys=True))
    elif a.cmd == "evict":
        print(json.dumps(BuildCache(root, min_age=0).evict(int((a.max_gb or 0) * 2**30))))
    else:
        import shutil

        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
