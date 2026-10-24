# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent ModelBlaster driver hosts (the efficient multi-slot runtime).

A ModelBlaster problem's ``driver.cpp`` built with ``-DMB_HOST`` is a
long-lived process that keeps the CUDA context, the device buffers and the
reference data, and evaluates one candidate per request: the candidate's
kernels are a shared object (the slot units + ``walker_tail.cu`` built
``-shared -fPIC`` with the per-thread default stream) that the host dlopens
for the request and dlcloses after. The checks and the timing are the
one-process driver's own code, so a request prints the same ``MB_*`` lines a
driver run with the same variables prints (ModelBlaster
``docs/gpu_harness_perf.md``). Protocol (one line per request, tab-separated)::

    EVAL <nonce> so=<path> [ref=<path>] [lock=<path>] [lock_wait=<s>] [MB_*=<value> ...]
    -> driver output lines, then  MB_HOST_DONE <nonce> rc=<0|1> poisoned=<0|1> recycle=<0|1>

``ref=`` loads the incumbent's object too and times both, interleaved rep by
rep, so the end-to-end comparison is paired (``MB_E2E_RATIO``). ``lock=`` is
the GPU's cross-process flock (``resources.gpu_lock_file``), taken by the host
for the request only, so an idle host never blocks other runs on its GPU.

Failure handling: a candidate that crashes the process, hangs past the
timeout, or leaves the context unusable (``poisoned=1``) is reported as a
runtime error, and the host is gone: the next request starts a new one, so a
bad candidate never affects the next candidate's result. The host also exits
after ``MB_HOST_MAX_EVALS`` requests or once it has leaked
``MB_HOST_LEAK_MB`` of device memory (``recycle=1``), and on its own when its
stdin closes (the client died or the ssh channel dropped).
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

HOST_MARKER = "MB_HOST_READY"
DONE_PREFIX = "MB_HOST_DONE"


def supports_host(driver_text: str) -> bool:
    """The driver has the persistent host (exported after it was added)."""
    return HOST_MARKER in driver_text


@dataclass
class HostReply:
    text: str                      # the driver lines of this request
    status: str                    # "done" | "crash" | "timeout" | "start_failed"
    rc: Optional[int] = None
    poisoned: bool = False
    recycle: bool = False
    wall_s: float = 0.0
    started: bool = False          # this request started the host (paid its start-up)
    start_s: float = 0.0

    @property
    def crashed(self) -> bool:
        return self.status != "done" or self.poisoned


class HostClient:
    """One host process (local, or ``ssh host cmd``): ``request()`` sends an
    EVAL and reads the reply; one request at a time."""

    def __init__(self, argv: list[str], *, env: Optional[dict] = None, label: str = "host",
                 start_timeout: float = 600.0, device: Optional[str] = None):
        self.argv, self.env, self.label, self.start_timeout = list(argv), env, label, start_timeout
        self.device = device             # telemetry: the GPU the host's requests run on
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.ready = ""
        self.starts = 0
        self.requests = 0
        self.last_used = 0.0
        self._lock = asyncio.Lock()
        self.loop = None                 # the event loop its process and pipes belong to

    def abandon(self):
        """Kill the process without awaiting it (it belongs to another, finished event loop)."""
        p, self.proc = self.proc, None
        if p is not None and p.returncode is None:
            try:
                os.kill(p.pid, 9)
            except (ProcessLookupError, PermissionError):
                pass

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def start(self) -> Optional[str]:
        """Start the process and wait for MB_HOST_READY; None, or why not."""
        await self.kill()
        self.loop = asyncio.get_running_loop()
        self.proc = await asyncio.create_subprocess_exec(
            *self.argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, env=self.env, limit=1 << 24)
        self.starts += 1
        out, deadline = [], time.monotonic() + self.start_timeout
        while True:
            line = await self._readline(deadline)
            if not line:
                await self.kill()
                return "".join(out)[-3000:] or ("timed out" if line is None else "exited")
            out.append(line)
            if line.startswith(HOST_MARKER):
                self.ready = line.strip()
                return None

    async def _readline(self, deadline: float) -> Optional[str]:
        """Next line; "" at EOF, None at the deadline."""
        left = deadline - time.monotonic()
        if left <= 0:
            return None
        try:
            b = await asyncio.wait_for(self.proc.stdout.readline(), left)
        except asyncio.TimeoutError:
            return None
        except (ValueError, asyncio.LimitOverrunError):      # absurdly long line: treat as garbage
            return "\n"
        return b.decode(errors="replace")

    async def request(self, so: str, *, ref: Optional[str] = None, env: Optional[dict] = None,
                      lock: Optional[str] = None, lock_wait: float = 3600.0, timeout: float = 900.0) -> HostReply:
        async with self._lock:
            from ..telemetry import core as tel
            from ..telemetry.evaluator import add_driver_phases, driver_phases

            started, start_s = False, 0.0
            if not self.alive:
                t0 = time.monotonic()
                restart = self.starts > 0
                with tel.span("driver.host_start", restart=restart, device=self.device):
                    if restart:
                        tel.count("host.restart")
                    tel.count("host.start")
                    why = await self.start()
                start_s = time.monotonic() - t0
                if why is not None:
                    return HostReply(why, "start_failed", started=True, start_s=start_s)
                started = True
            with tel.span("run.exec", device=self.device, runtime="host") as sp:
                rep = await self._request(so, ref=ref, env=env, lock=lock, lock_wait=lock_wait, timeout=timeout,
                                          started=started, start_s=start_s)
                add_driver_phases(sp, driver_phases(rep.text))
                sp.set(status=rep.status)
                if rep.crashed:
                    tel.count(f"host.{'poisoned' if rep.poisoned else rep.status}")
                return rep

    async def _request(self, so, *, ref, env, lock, lock_wait, timeout, started, start_s) -> HostReply:
        nonce = uuid.uuid4().hex[:12]
        for v in [so, ref or "", *(f"{k}={v}" for k, v in (env or {}).items())]:
            if "\t" in v or "\n" in v:
                raise ValueError(f"host request field {v!r} has a tab or newline")
        fields = ["EVAL", nonce, f"so={so}"] + ([f"ref={ref}"] if ref else []) + \
                 ([f"lock={lock}", f"lock_wait={int(lock_wait)}"] if lock else []) + \
                 [f"{k}={v}" for k, v in (env or {}).items()]
        t0 = time.monotonic()
        self.requests += 1
        self.last_used = time.time()
        try:
            self.proc.stdin.write(("\t".join(fields) + "\n").encode())
            await self.proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass                                       # the read below sees the EOF
        out, deadline = [], t0 + timeout + (lock_wait if lock else 0)
        while True:
            line = await self._readline(deadline)
            if not line:
                status = "timeout" if line is None else "crash"
                rc = None
                if status == "crash":
                    try:
                        rc = await asyncio.wait_for(self.proc.wait(), 10)
                    except asyncio.TimeoutError:
                        pass
                await self.kill()
                return HostReply("".join(out), status, rc=rc, wall_s=time.monotonic() - t0,
                                 started=started, start_s=start_s)
            if line.startswith(f"{DONE_PREFIX} {nonce} "):
                break
            out.append(line)
        rep = HostReply("".join(out), "done", rc=0 if " rc=0" in line else 1, poisoned="poisoned=1" in line,
                        recycle="recycle=1" in line, wall_s=time.monotonic() - t0, started=started,
                        start_s=start_s)
        if rep.poisoned or rep.recycle:            # the host exits by itself
            try:
                await asyncio.wait_for(self.proc.wait(), 30)
            except asyncio.TimeoutError:
                pass
            await self.kill()
        return rep

    async def kill(self):
        p, self.proc = self.proc, None
        if p is not None and p.returncode is None:
            try:
                p.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(p.wait(), 30)
            except asyncio.TimeoutError:
                pass

    async def close(self):
        if self.alive:
            try:
                self.proc.stdin.write(b"QUIT\n")
                await self.proc.stdin.drain()
                await asyncio.wait_for(self.proc.wait(), 20)
            except Exception:
                pass
        await self.kill()


@dataclass
class HostStats:
    requests: int = 0
    starts: int = 0
    crashes: int = 0
    timeouts: int = 0
    poisoned: int = 0
    recycles: int = 0
    start_s: float = 0.0
    request_s: float = 0.0
    by_device: dict = field(default_factory=dict)

    def note(self, rep: HostReply, device_label: str):
        self.requests += 1
        self.starts += int(rep.started)
        self.start_s += rep.start_s
        self.request_s += rep.wall_s
        self.crashes += int(rep.status == "crash")
        self.timeouts += int(rep.status == "timeout")
        self.poisoned += int(rep.poisoned)
        self.recycles += int(rep.recycle)
        self.by_device[device_label] = self.by_device.get(device_label, 0) + 1

    def to_json(self) -> dict:
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class HostPool:
    """Hosts by (problem, device); at most ``per_device`` live hosts per device
    (least recently used ones are closed: each holds a CUDA context and the
    problem's buffers on that GPU)."""

    def __init__(self, per_device: int = 3):
        self.per_device = per_device
        self.hosts: "OrderedDict[tuple, HostClient]" = OrderedDict()
        self.stats = HostStats()

    async def get(self, key: tuple, device_key, factory) -> HostClient:
        h = self.hosts.get(key)
        if h is not None and h.loop is not None and h.loop is not asyncio.get_running_loop():
            h.abandon()                  # made under another asyncio.run(): start over in this loop
            h = self.hosts.pop(key)
            h = None
        if h is None:
            h = self.hosts[key] = factory()
            same = [k for k in self.hosts if k[1] == device_key and k != key]
            while len(same) >= self.per_device:
                old = self.hosts.pop(same.pop(0))
                await old.close()
        self.hosts.move_to_end(key)
        return h

    async def drop(self, key: tuple):
        h = self.hosts.pop(key, None)
        if h is not None:
            await h.close()

    async def close(self):
        hosts, self.hosts = list(self.hosts.values()), OrderedDict()
        loop = asyncio.get_running_loop()
        for h in hosts:
            if h.loop is not None and h.loop is not loop:
                h.abandon()
        await asyncio.gather(*[h.close() for h in hosts if h.loop is None or h.loop is loop],
                             return_exceptions=True)


def host_env(extra: Optional[dict] = None) -> dict:
    return {**os.environ, **(extra or {})}
