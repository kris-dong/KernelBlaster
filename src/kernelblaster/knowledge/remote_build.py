# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distributed builds on the pool GPU hosts (compile pool, ``l3_farm/analysis/compile_pool.md``).

Off by default; ``KERNELBLASTER_REMOTE_BUILD=1`` (or ``SSHEvaluator(remote_build=True)``) makes the
:class:`SSHEvaluator` compile its slot / tail translation units (the ``nvcc -c`` steps, the CPU cost of a
candidate) on the pool hosts instead of on the driver:

* the unit's source goes to the chosen host over the existing ssh connection (stdin), is compiled there with
  ``nice -n 19`` / ``ionice -c3`` under a per-host slot limit (``KERNELBLASTER_REMOTE_BUILD_JOBS``, default 3, an
  flock slot on the host, so the bound holds across every controller process of the driver), and the object
  (~100 KB - 1 MB) comes back with scp into the driver's build cache; the (cheap) link and everything after it
  are unchanged, so the cache, the shared objects, the timed run and the rsync of the shared object work exactly
  as before. The host keeps a content-addressed object cache (``<remote_dir>/.kbcc``) of its own;
* the scheduler (:func:`rank_hosts`) sends each unit to the eligible host with the fewest builds in flight
  (this process's own count, then the load the hosts last reported); a host with no free slot answers ``FULL``
  at once and the next one is tried; when every host stays full for ``KERNELBLASTER_REMOTE_BUILD_SPILL_S``
  seconds (default 20; 0 = wait) the unit builds on the driver;
* a host is eligible only if its nvcc release / build and host compiler equal the driver's (so a cached object
  is the same bytes wherever it was built) and it answered the probe;
* **infrastructure failures are not candidate failures**: ssh down, a truncated reply, a failed scp, a killed
  compiler or a host out of disk / memory never reach the caller as a compile error -- the unit is built on the
  driver (counted in ``remote_build`` stats) and the host is struck / quarantined like a failing GPU host. Only
  a compile error that the remote nvcc reported with the compiler's own exit status is a
  :class:`BuildError` (and may be negatively cached);
* ``KERNELBLASTER_REMOTE_BUILD_POLICY``: ``overlap`` (default) builds on a host whatever it is timing;
  ``avoid`` does not start a build on a host while one of its GPU flocks is held (a timed run is in progress),
  trying the next host and finally the driver.

Not remote: the driver object (``-I`` / ``.incbin`` of data.bin: absolute inputs only the driver has), the host
binary, the links, the one-shot assembled build, ncu.
"""
from __future__ import annotations

import asyncio
import os
import random
import re
import shlex
import time
import uuid
from pathlib import Path
from typing import Iterable, Optional

from .. import trace

_FALSE = ("0", "false", "no", "off", "")


def env_on(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() not in _FALSE


class RemoteInfra(Exception):
    """The remote build could not be done (transport, host, killed compiler): build on the driver instead."""


def rank_hosts(hosts: Iterable[str], inflight: dict, load: dict, *, cap: int, quarantined: Iterable[str] = (),
               busy: Iterable[str] = (), avoid: Iterable[str] = (), rr: int = 0) -> list[str]:
    """The order in which to try the hosts for the next build: not quarantined (all of them if every one is),
    not in ``avoid`` (hosts that just answered FULL / BUSY, unless that leaves none), then fewest builds in flight
    from this process, then the lowest load a host last reported, then round robin from ``rr``."""
    hosts = list(hosts)
    q = set(quarantined)
    ok = [h for h in hosts if h not in q] or hosts
    av = set(avoid) | set(busy)
    pre = [h for h in ok if h not in av]
    ok = pre or ok
    n = max(1, len(hosts))
    idx = {h: (i - rr) % n for i, h in enumerate(hosts)}
    return sorted(ok, key=lambda h: (inflight.get(h, 0), load.get(h, 0), idx[h]))


def remote_script(*, cache_dir: str, work_dir: str, name: str, key: str, args: list[str], cuda_bin: str,
                  env: dict, timeout_s: int, slots: int, avoid_busy: bool, keep_objs: int = 4000) -> str:
    """The bash program one remote build runs (``bash -c``; the unit's source is on stdin). Replies with
    ``KB_RB_*`` marker lines; ``KB_RB_END`` last (its absence = a truncated reply = infrastructure)."""
    q = shlex.quote
    exports = "".join(f"export {k}={q(v)}; " for k, v in env.items() if v)
    argv = " ".join(q(a) for a in args)
    lock_probe = (
        'busy=0; for f in /dev/shm/kernelblaster_gpu_*.lock; do [ -e "$f" ] || continue; '
        'flock -n "$f" true 2>/dev/null || busy=1; done; '
        'if [ $busy = 1 ]; then echo KB_RB_BUSY=1; echo KB_RB_END; exit 0; fi; ') if avoid_busy else ""
    return (
        f'D={q(cache_dir)}; W={q(work_dir)}; N={q(name)}; K={q(key)}; '
        'mkdir -p "$D" "$W" || { echo KB_RB_INFRA=mkdir; echo KB_RB_END; exit 0; }; '
        'cat > "$W/$N" || { echo KB_RB_INFRA=stdin; echo KB_RB_END; exit 0; }; '
        'if [ -s "$D/$K.o" ]; then touch "$D/$K.o"; echo KB_RB_CACHED=1; echo KB_RB_RC=0; echo KB_RB_END; '
        'rm -f "$W/$N"; exit 0; fi; '
        + lock_probe +
        # a build slot: one of N flock files, non-blocking; none free -> FULL (the client tries another host)
        f'got=0; i=0; while [ $i -lt {int(slots)} ]; do exec 9>"$D/slot.$i.lock" 2>/dev/null; '
        'if flock -n 9; then got=1; break; fi; i=$((i+1)); done; '
        'if [ $got = 0 ]; then rm -f "$W/$N"; echo KB_RB_FULL=1; echo KB_RB_END; exit 0; fi; '
        f'{exports}export PATH={q(cuda_bin)}:$PATH; '
        'P="nice -n 19"; command -v ionice >/dev/null 2>&1 && P="$P ionice -c3"; '
        f'T=$(command -v timeout >/dev/null 2>&1 && echo "timeout -k 5 {max(1, int(timeout_s))}"); '
        'S=$(date +%s.%N); '
        f'$P $T nvcc {argv} -c "$W/$N" -o "$W/$N.o" > "$W/$N.log" 2>&1; rc=$?; '
        'E=$(date +%s.%N); echo KB_RB_RC=$rc; x=$(awk "BEGIN{print $E-$S}" 2>/dev/null); echo KB_RB_SECS=${x:-0}; '
        'if [ $rc = 0 ]; then '
        'cp "$W/$N.o" "$D/.t.$K.$$" && mv -f "$D/.t.$K.$$" "$D/$K.o" || echo KB_RB_INFRA=store; '
        f'if [ $((RANDOM % 25)) = 0 ]; then ls -1t "$D" 2>/dev/null | grep -v "^slot\\." | tail -n +{int(keep_objs) + 1} '
        '| head -400 | while read f; do rm -f "$D/$f"; done; fi; '
        'else echo KB_RB_LOG_BEGIN; tail -c 12000 "$W/$N.log"; echo; echo KB_RB_LOG_END; fi; '
        'rm -f "$W/$N" "$W/$N.o" "$W/$N.log"; echo KB_RB_END')


class RemoteReply:
    """A parsed ``remote_script`` reply."""

    def __init__(self, text: str):
        self.text = text
        self.end = "KB_RB_END" in text
        m = re.search(r"^KB_RB_RC=(-?\d+)", text, re.M)
        self.rc: Optional[int] = int(m.group(1)) if m else None
        m = re.search(r"^KB_RB_SECS=([0-9.]+)", text, re.M)
        self.seconds = float(m.group(1)) if m else 0.0
        self.full = bool(re.search(r"^KB_RB_FULL=", text, re.M))
        self.busy = bool(re.search(r"^KB_RB_BUSY=", text, re.M))
        self.cached = bool(re.search(r"^KB_RB_CACHED=", text, re.M))
        m = re.search(r"^KB_RB_INFRA=(.*)$", text, re.M)
        self.infra: Optional[str] = m.group(1) if m else None
        m = re.search(r"^KB_RB_LOG_BEGIN\n(.*?)\n?KB_RB_LOG_END", text, re.M | re.S)
        self.log = m.group(1) if m else ""


class RemoteBuildMixin:
    """Mixed into :class:`SSHEvaluator` before :class:`LocalEvaluator`: ``_compile_obj`` goes to a pool host."""

    def _rb_init(self, remote_build: Optional[bool], jobs_per_host: Optional[int]) -> None:
        self.remote_build = env_on("KERNELBLASTER_REMOTE_BUILD") if remote_build is None else bool(remote_build)
        self.rb_jobs = int(jobs_per_host or os.getenv("KERNELBLASTER_REMOTE_BUILD_JOBS") or 3)
        self.rb_spill_s = float(os.getenv("KERNELBLASTER_REMOTE_BUILD_SPILL_S", "20"))
        self.rb_policy = os.getenv("KERNELBLASTER_REMOTE_BUILD_POLICY", "overlap").strip().lower()
        self._rb_inflight: dict = {}
        self._rb_load: dict = {}
        self._rb_state: dict = {}                 # host -> {"ok": bool, "t": probe time}
        self._rb_probe_lock: Optional[asyncio.Lock] = None
        self._rb_strikes: dict = {}
        self._rb_quarantine: dict = {}            # host -> monotonic time until which it is skipped
        self._rb_rr = 0
        self.rb_stats = {"remote": 0, "remote_cached": 0, "local": 0, "spilled": 0, "infra": 0, "full": 0, "busy": 0,
                         "compile_errors": 0, "remote_s": 0.0, "queue_s": 0.0, "transfer_s": 0.0, "bytes_back": 0,
                         "per_host": {}}

    # ---- eligibility
    def _rb_hosts_all(self) -> list[str]:
        return list(self.hosts)

    async def _rb_probe(self, host: str) -> bool:
        """Does ``host`` answer and have the driver's toolchain?"""
        from ..servers.utils.build_cache import tool_core, toolchain_text

        local = tool_core(self.nvcc)
        if local is None:
            return False
        q = shlex.quote
        script = (f"mkdir -p {q(self.remote_dirs[host])}/.kbcc {q(self._stage_dir(host))}/rb && "
                  f"export PATH={q(self.remote_path)}:$PATH; nvcc --version; echo @@KB@@; c++ --version 2>&1 | head -1; "
                  "echo @@KB@@; command -v flock >/dev/null && echo KB_HAS_FLOCK")
        rc, out = await self._sh(self.ssh, *self._ssh_base(), host, "bash -c " + q(script), timeout=90)
        if rc != 0 or "@@KB@@" not in out or "KB_HAS_FLOCK" not in out:
            _rblog().warning(f"remote build: {host} not usable (rc {rc}): {out[-200:]!r}")
            return False
        parts = out.split("@@KB@@")
        remote = toolchain_text(parts[0], parts[1].strip() if len(parts) > 1 else "")
        if remote != local:
            _rblog().warning(f"remote build: {host} has a different toolchain ({remote} vs the driver's {local}); "
                             f"not used for builds")
            return False
        self._stage_made.add(host)
        return True

    async def _rb_eligible(self) -> list[str]:
        from ..servers.utils.resources import loop_local

        self._rb_probe_lock = loop_local(self._rb_probe_lock, asyncio.Lock)
        async with self._rb_probe_lock:
            now = time.monotonic()
            for h in self._rb_hosts_all():
                st = self._rb_state.get(h)
                if st is None or (not st["ok"] and now - st["t"] > 120):
                    self._rb_state[h] = {"ok": await self._rb_probe(h), "t": time.monotonic()}
        return [h for h in self._rb_hosts_all() if self._rb_state.get(h, {}).get("ok")]

    def _rb_strike(self, host: str) -> None:
        self._rb_strikes[host] = self._rb_strikes.get(host, 0) + 1
        if self._rb_strikes[host] >= 2:
            self._rb_strikes[host] = 0
            self._rb_quarantine[host] = time.monotonic() + 120
            _rblog().warning(f"remote build: {host} skipped for 120 s after repeated infrastructure failures")

    # ---- one build
    def _rb_applicable(self, src: Path, args: list[str], text: str) -> bool:
        """Only units that are self-contained text: no absolute inputs (the driver's .incbin / -I of the driver
        host), no path-bearing flags."""
        from ..servers.utils.build_cache import external_inputs

        return not external_inputs(text) and not any(a.startswith(("-I", "-L", "-isystem")) for a in args)

    async def _compile_obj(self, src: Path, args: list[str], tmp: Path, text: str = "") -> None:
        if not getattr(self, "remote_build", False) or not self._rb_applicable(src, args, text):
            return await super()._compile_obj(src, args, tmp, text)
        from .. import telemetry as tel
        from ..servers.utils.build_cache import BuildError, cache_key, tool_core

        t0 = time.monotonic()
        key = cache_key("rb-obj", tool_core(self.nvcc) or "", *args, text)
        deadline = t0 + self.rb_spill_s if self.rb_spill_s > 0 else None
        backoff = 0.05
        avoid: set = set()
        counted_full = False
        while True:
            elig = await self._rb_eligible()
            if not elig:
                break
            now = time.monotonic()
            quar = [h for h, t in self._rb_quarantine.items() if t > now]
            for h in rank_hosts(elig, self._rb_inflight, self._rb_load, cap=self.rb_jobs, quarantined=quar,
                                avoid=avoid, rr=self._rb_rr):
                # a driver-side slot first (a local flock shared by every controller process of this driver): waiting
                # for a busy host costs no ssh round trip; the host's own flock stays the bound across drivers
                slot = self._rb_slot(h)
                if slot is None:
                    if not counted_full:
                        counted_full = True
                        self.rb_stats["full"] += 1
                    avoid.add(h)
                    continue
                self._rb_rr += 1
                self._rb_inflight[h] = self._rb_inflight.get(h, 0) + 1
                try:
                    rep = await self._rb_run(h, src, args, key, text)
                except RemoteInfra as e:
                    self.rb_stats["infra"] += 1
                    self._ph(h)["infra"] = self._ph(h).get("infra", 0) + 1
                    self._rb_strike(h)
                    avoid.add(h)
                    _rblog().warning(f"remote build on {h} failed (infrastructure, built on the driver instead if no "
                                     f"other host takes it): {e}")
                    continue
                finally:
                    self._rb_inflight[h] -= 1
                    os.close(slot)                           # releases the driver-side slot
                if rep is None:                              # FULL / BUSY: next host
                    avoid.add(h)
                    continue
                self._rb_strikes[h] = 0
                self.rb_stats["queue_s"] = round(self.rb_stats["queue_s"] + (time.monotonic() - t0), 3)
                tel.count("build.remote")
                try:
                    await self._rb_fetch(h, key, tmp)
                except RemoteInfra as e:
                    self.rb_stats["infra"] += 1
                    self._rb_strike(h)
                    _rblog().warning(f"remote build on {h}: fetching the object failed ({e}); building on the driver")
                    tel.count("build.remote_fallback")
                    return await super()._compile_obj(src, args, tmp, text)
                return None
            if len(avoid) >= len(elig):
                avoid = set()                                 # every host said FULL / BUSY / failed: look again
                if deadline is not None and time.monotonic() > deadline:
                    self.rb_stats["spilled"] += 1
                    break
                await asyncio.sleep(backoff + random.random() * 0.05)
                backoff = min(backoff * 1.5, 0.4)
        self.rb_stats["local"] += 1
        tel.count("build.remote_fallback")
        return await super()._compile_obj(src, args, tmp, text)

    def _rb_slot(self, host: str) -> Optional[int]:
        """A non-blocking flock on one of ``rb_jobs`` per-host slot files of this driver (``KERNELBLASTER_REMOTE_BUILD_LOCK_DIR``,
        default ``$TMPDIR/kb_rb_slots_<uid>``): the fd (closing it releases the slot) or None when every slot is taken."""
        import fcntl
        import tempfile

        d = Path(os.getenv("KERNELBLASTER_REMOTE_BUILD_LOCK_DIR") or Path(tempfile.gettempdir()) / f"kb_rb_slots_{os.getuid()}")
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            return os.open(os.devnull, os.O_RDONLY)          # no driver-side gate; the host's flock still bounds
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", host)
        for i in range(self.rb_jobs):
            try:
                fd = os.open(d / f"{safe}.{i}.lock", os.O_CREAT | os.O_RDWR, 0o666)
            except OSError:
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except OSError:
                os.close(fd)
        return None

    def _ph(self, host: str) -> dict:
        return self.rb_stats["per_host"].setdefault(host, {"builds": 0, "seconds": 0.0})

    async def _sh_in(self, argv: list[str], data: bytes, timeout: float) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(data), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return -1, f"timed out after {timeout}s"
        except (OSError, BrokenPipeError) as e:
            return 255, str(e)
        return proc.returncode, out.decode(errors="replace")

    async def _rb_run(self, host: str, src: Path, args: list[str], key: str, text: str):
        """Compile ``text`` on ``host``. Returns the reply (object stored in the host's cache under ``key``), None
        when the host has no free slot (or is timing, policy ``avoid``); raises :class:`BuildError` for a
        compile error and :class:`RemoteInfra` for anything that is not the source's fault."""
        from ..servers.utils.build_cache import BuildError, deterministic_failure, tool_env_text, _TOOL_ENV

        name = f"{re.sub(r'[^A-Za-z0-9_.-]', '_', src.stem)}_{uuid.uuid4().hex[:8]}.cu"
        env = {k: os.environ.get(k, "") for k in _TOOL_ENV}
        script = remote_script(
            cache_dir=f"{self.remote_dirs[host]}/.kbcc", work_dir=f"{self._stage_dir(host)}/rb", name=name, key=key,
            args=args, cuda_bin=self.remote_path, env=env, timeout_s=int(self.build_timeout_s), slots=self.rb_jobs,
            avoid_busy=(self.rb_policy == "avoid"))
        self._note_remote(host, "file", name="rb")
        t0 = time.monotonic()
        rc, out = await self._sh_in([self.ssh, *self._ssh_base(), host, "bash -c " + shlex.quote(script)],
                                    text.encode(), timeout=self.build_timeout_s + 120)
        wall = time.monotonic() - t0
        rep = RemoteReply(out)
        if rc != 0 or not rep.end:
            raise RemoteInfra(f"ssh rc {rc}, reply {out[-200:]!r}")
        if rep.infra:
            raise RemoteInfra(f"remote step {rep.infra}")
        if rep.full:
            self.rb_stats["full"] += 1
            self._rb_load[host] = self.rb_jobs
            return None
        if rep.busy:
            self.rb_stats["busy"] += 1
            return None
        self._rb_load[host] = max(0, self._rb_load.get(host, 0) - 1)
        if rep.rc is None:
            raise RemoteInfra(f"no exit status in reply {out[-200:]!r}")
        cmd = ["ssh", host, "nvcc", *args, "-c", name]
        if rep.rc == 0:
            self.rb_stats["remote"] += 1
            self.rb_stats["remote_cached"] += int(rep.cached)
            self.rb_stats["remote_s"] = round(self.rb_stats["remote_s"] + rep.seconds, 3)
            ph = self._ph(host)
            ph["builds"] += 1
            ph["seconds"] = round(ph["seconds"] + rep.seconds, 3)
            trace.tool("nvcc", cmd, 0, "" if not rep.cached else "(host cache)", rep.seconds or wall)
            return rep
        trace.tool("nvcc", cmd, rep.rc, rep.log, rep.seconds or wall)
        if rep.rc == 124 or rep.rc == 137 or rep.rc >= 128:      # timeout / killed
            raise RemoteInfra(f"nvcc exit {rep.rc} on {host}: {rep.log[-200:]!r}")
        if deterministic_failure(rep.rc, rep.log):
            self.rb_stats["compile_errors"] += 1
            raise BuildError(src.name, rep.log, cacheable=True)
        raise RemoteInfra(f"nvcc exit {rep.rc} on {host} with a non-source error: {rep.log[-200:]!r}")

    async def _rb_fetch(self, host: str, key: str, tmp: Path) -> None:
        """scp the object from the host's cache to ``tmp``."""
        t0 = time.monotonic()
        tmp.parent.mkdir(parents=True, exist_ok=True)
        rc, out = await self._sh(self.scp, "-q", *self._ssh_base(), f"{host}:{self.remote_dirs[host]}/.kbcc/{key}.o",
                                 str(tmp), timeout=300)
        if rc != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            if tmp.exists():
                tmp.unlink()
            raise RemoteInfra(f"scp rc {rc}: {out[-200:]!r}")
        self.rb_stats["transfer_s"] = round(self.rb_stats["transfer_s"] + time.monotonic() - t0, 3)
        self.rb_stats["bytes_back"] += tmp.stat().st_size

    def remote_build_stats(self) -> dict:
        return {"enabled": self.remote_build, "jobs_per_host": self.rb_jobs, "policy": self.rb_policy,
                "spill_s": self.rb_spill_s, **{k: (dict(v) if isinstance(v, dict) else v) for k, v in self.rb_stats.items()},
                "eligible": [h for h, s in self._rb_state.items() if s.get("ok")]}


def _rblog():
    import logging
    return logging.getLogger("kernelblaster.remote_build")
