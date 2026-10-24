# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host resources for load balancing: CPUs, memory, GPUs.

*CPUs and memory* (build concurrency):

- :func:`usable_cpus`: CPUs this process may run on, ``sched_getaffinity``
  capped by the cgroup CPU quota (v2 ``cpu.max``, v1 ``cfs_quota_us``).
- :func:`free_cpus`: usable CPUs minus other users' load, i.e. the 1-minute
  load average minus the jobs this process is running itself.
- :func:`mem_jobs`: ``MemAvailable`` / a per-compiler-process estimate.
- :class:`AdaptiveLimiter`: an async semaphore whose capacity is re-evaluated
  every few seconds from the above, so builds back off when a shared box gets
  busy and scale up when it is idle. A fixed capacity (explicit argument or env
  var) turns the adaptation off.

*GPUs* (timed runs):

- :func:`gpu_inventory` / :func:`gpu_inventory_async`: ``nvidia-smi`` locally
  or over ssh -> :class:`GPUInfo` (index, uuid, name, memory, utilisation,
  compute processes).
- :func:`select_gpus`: ``"auto"`` = idle GPUs (no foreign compute process,
  low utilisation); ``KERNELBLASTER_ALLOW_SHARED_GPU=1`` adds busy ones; an
  explicit list always wins; an externally set ``CUDA_VISIBLE_DEVICES`` limits
  what auto may pick on this host.
- :class:`GPUPool`: exclusive leases on (host, gpu) devices -- one timed run or
  NCU pass per device at a time -- picking the least recently used free device,
  pinning each group (problem) to one GPU model so its timings stay
  comparable, and quarantining a device after repeated infrastructure failures.

Nothing here ever signals or otherwise touches processes it did not start.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import shlex
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

log = logging.getLogger(__name__)

GiB = 1024 ** 3

# ---------------------------------------------------------------- env helpers


def env_int(name: str) -> Optional[int]:
    """``int(os.environ[name])``, or None when unset / empty / "auto"."""
    v = os.getenv(name, "").strip()
    if not v or v.lower() == "auto":
        return None
    return int(v)


def env_float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    return float(v) if v else default


def truthy(v: Optional[str]) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def allow_shared_gpu() -> bool:
    return truthy(os.getenv("KERNELBLASTER_ALLOW_SHARED_GPU"))


def loop_local(obj, factory):
    """``obj`` if it was made on the running event loop, else a new ``factory()``
    (asyncio primitives bind to the loop they first wait on; an evaluator may
    outlive one ``asyncio.run``)."""
    loop = asyncio.get_running_loop()
    if obj is not None and getattr(obj, "_kb_loop", None) is loop:
        return obj
    obj = factory()
    obj._kb_loop = loop
    return obj


# ---------------------------------------------------------------- CPUs


def parse_cpu_max(text: str) -> Optional[float]:
    """cgroup v2 ``cpu.max`` ("<quota> <period>" or "max <period>") -> CPUs, None if unlimited."""
    parts = text.split()
    if not parts or parts[0] == "max":
        return None
    quota = float(parts[0])
    period = float(parts[1]) if len(parts) > 1 else 100000.0
    return quota / period if quota > 0 and period > 0 else None


def parse_cfs(quota_text: str, period_text: str) -> Optional[float]:
    """cgroup v1 ``cpu.cfs_quota_us`` / ``cpu.cfs_period_us`` -> CPUs (quota -1 = unlimited)."""
    q, p = float(quota_text.strip()), float(period_text.strip())
    return q / p if q > 0 and p > 0 else None


def cgroup_cpu_limit(proc_cgroup: str = "/proc/self/cgroup", root: str = "/sys/fs/cgroup") -> Optional[float]:
    """The tightest CPU quota on this process's cgroup path (v2, else v1); None if unlimited."""
    try:
        lines = Path(proc_cgroup).read_text().splitlines()
    except OSError:
        return None
    limits = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, ctrls, path = parts
        rel = path.lstrip("/")
        if ctrls == "":                                 # v2 unified hierarchy: check every ancestor
            p = Path(root) / rel
            while True:
                try:
                    lim = parse_cpu_max((p / "cpu.max").read_text())
                    if lim:
                        limits.append(lim)
                except (OSError, ValueError, IndexError):
                    pass
                if p == Path(root) or p.parent == p:
                    break
                p = p.parent
        elif "cpu" in ctrls.split(","):
            for base in (Path(root) / "cpu,cpuacct", Path(root) / "cpu"):
                try:
                    lim = parse_cfs((base / rel / "cpu.cfs_quota_us").read_text(),
                                    (base / rel / "cpu.cfs_period_us").read_text())
                    if lim:
                        limits.append(lim)
                    break
                except (OSError, ValueError):
                    continue
    return min(limits) if limits else None


def affinity_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


_PROBE = object()


def usable_cpus(*, affinity: Optional[int] = None, quota=_PROBE) -> float:
    """CPUs this process may use: affinity mask capped by the cgroup quota
    (``quota=None``: no quota; default: read it from the cgroup)."""
    n = float(affinity if affinity is not None else affinity_cpus())
    q = cgroup_cpu_limit() if quota is _PROBE else quota
    return min(n, q) if q else n


def load_average() -> float:
    try:
        return os.getloadavg()[0]
    except OSError:
        return 0.0


# Jobs this process is running right now, across every AdaptiveLimiter: the
# load average counts them too, so they are subtracted to estimate other users'
# load.
_OWN = {"running": 0}


def own_running() -> int:
    return _OWN["running"]


def free_cpus(*, usable: Optional[float] = None, load: Optional[float] = None,
              own: Optional[int] = None) -> float:
    """Usable CPUs minus other users' load (load average minus our own running jobs)."""
    usable = usable_cpus() if usable is None else usable
    load = load_average() if load is None else load
    own = own_running() if own is None else own
    other = max(0.0, load - own)
    return max(0.0, usable - other)


def mem_available() -> Optional[int]:
    """``MemAvailable`` in bytes (None if unknown)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def nvcc_mem_estimate(default_gib: float = 1.5) -> int:
    """Bytes one compiler process needs (``KERNELBLASTER_NVCC_MEM_MB`` overrides)."""
    mb = env_int("KERNELBLASTER_NVCC_MEM_MB")
    return int(mb * 1024 * 1024) if mb else int(default_gib * GiB)


def mem_jobs(per_job: Optional[int] = None, *, available: Optional[int] = None,
             reserve: int = 2 * GiB, own: int = 0) -> Optional[int]:
    """How many more compiler processes memory allows, plus the ``own`` already
    running (their memory is already out of ``MemAvailable``)."""
    per_job = per_job or nvcc_mem_estimate()
    available = mem_available() if available is None else available
    if available is None:
        return None
    return own + max(0, int((available - reserve) // per_job))


def auto_jobs(*, max_jobs: int = 16, min_jobs: int = 1, share: Optional[float] = None,
              per_job_mem: Optional[int] = None, own: Optional[int] = None,
              usable: Optional[float] = None, load: Optional[float] = None,
              available: Optional[int] = None) -> int:
    """Concurrent compiler processes for now: ``share`` of the free CPUs (other
    users' load subtracted), at most what memory allows, in [min_jobs, max_jobs].

    ``share`` (``KERNELBLASTER_CPU_SHARE``, default 0.75) leaves headroom on a
    shared box."""
    share = env_float("KERNELBLASTER_CPU_SHARE", 0.75) if share is None else share
    own = own_running() if own is None else own
    usable = usable_cpus() if usable is None else usable
    free = free_cpus(usable=usable, load=load, own=own)
    n = int(math.floor(free * share))
    m = mem_jobs(per_job_mem, available=available, own=own)
    if m is not None:
        n = min(n, m)
    return max(min_jobs, min(max_jobs, n, max(1, int(usable))))


class AdaptiveLimiter:
    """Async semaphore whose capacity follows the host's free CPUs / memory.

    ``capacity_fn(in_flight) -> int`` is called at most every ``interval``
    seconds (by waiters, and on acquire); ``fixed`` pins the capacity. Running
    jobs count towards :func:`own_running` (``counts_as_load``) so they are not
    mistaken for other users' load. Shrinking never interrupts running jobs; it
    only delays new ones."""

    def __init__(self, capacity_fn: Optional[Callable[[int], int]] = None, *, fixed: Optional[int] = None,
                 interval: float = 5.0, name: str = "jobs", counts_as_load: bool = True,
                 clock: Callable[[], float] = time.monotonic):
        if fixed is None and capacity_fn is None:
            capacity_fn = lambda own: auto_jobs()   # noqa: E731
        self.fixed = fixed
        self.capacity_fn = capacity_fn
        self.interval, self.name, self.counts = interval, name, counts_as_load
        self.clock = clock
        self.in_flight = 0
        self.peak = 0
        self._cap = max(1, fixed) if fixed else None
        self._checked = -1e18
        self._cond: Optional[asyncio.Condition] = None
        self.history: list[tuple[float, int]] = []          # (time, capacity) changes

    @property
    def adaptive(self) -> bool:
        return self.fixed is None

    def capacity(self, *, force: bool = False) -> int:
        if self.fixed:
            return self._cap
        now = self.clock()
        if force or self._cap is None or now - self._checked >= self.interval:
            self._checked = now
            try:
                cap = max(1, int(self.capacity_fn(self.in_flight)))
            except Exception as e:                       # probing must never block builds
                log.warning(f"[{self.name}] capacity probe failed: {e!r}")
                cap = self._cap or 1
            if cap != self._cap:
                if self._cap is not None:
                    log.info(f"[{self.name}] capacity {self._cap} -> {cap} "
                             f"(load {load_average():.1f}, ours {own_running()})")
                self.history.append((now, cap))
                self._cap = cap
        return self._cap

    def _condition(self) -> asyncio.Condition:
        self._cond = loop_local(self._cond, asyncio.Condition)
        return self._cond

    async def acquire(self) -> None:
        cond = self._condition()
        async with cond:
            while self.in_flight >= self.capacity():
                try:   # re-probe periodically even when nothing is released
                    await asyncio.wait_for(cond.wait(), timeout=max(0.05, self.interval))
                except asyncio.TimeoutError:
                    pass
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            if self.counts:
                _OWN["running"] += 1

    async def release(self) -> None:
        cond = self._condition()
        async with cond:
            self.in_flight -= 1
            if self.counts:
                _OWN["running"] -= 1
            cond.notify_all()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *exc):
        await self.release()

    def describe(self) -> str:
        return f"{self._cap if self._cap else '?'}" + ("" if self.fixed else " (adaptive)")


def build_limiter(explicit: Optional[int] = None, *, env: Optional[str] = None, max_jobs: int = 16,
                  per_job_mem: Optional[int] = None, name: str = "jobs", interval: float = 5.0,
                  counts_as_load: bool = True) -> AdaptiveLimiter:
    """A fixed limiter when ``explicit`` or the env var ``env`` gives a number,
    else an adaptive one (``KERNELBLASTER_MAX_BUILD_JOBS`` caps it)."""
    n = explicit if explicit not in (None, 0, "auto") else (env_int(env) if env else None)
    if n:
        return AdaptiveLimiter(fixed=int(n), name=name, counts_as_load=counts_as_load)
    cap = env_int("KERNELBLASTER_MAX_BUILD_JOBS") or max_jobs
    return AdaptiveLimiter(lambda _in_flight: auto_jobs(max_jobs=cap, per_job_mem=per_job_mem),
                           name=name, interval=interval, counts_as_load=counts_as_load)


# ---------------------------------------------------------------- GPUs

QUERY_GPUS = ("nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu "
              "--format=csv,noheader,nounits")
QUERY_APPS = "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits"
_SEP = "@@KB_APPS@@"


@dataclass
class GPUInfo:
    index: int
    uuid: str
    name: str
    mem_used_mb: float = 0.0
    mem_total_mb: float = 0.0
    util: float = 0.0
    procs: list = field(default_factory=list)        # [(pid, process name, used MiB)] of foreign processes
    host: Optional[str] = None

    def idle(self, util_max: Optional[float] = None) -> bool:
        util_max = env_float("KERNELBLASTER_GPU_IDLE_UTIL", 10.0) if util_max is None else util_max
        return not self.procs and self.util <= util_max

    def short(self) -> str:
        busy = f", {len(self.procs)} foreign proc(s)" if self.procs else ""
        return f"{self.host + ':' if self.host else ''}{self.index} {self.name} util={self.util:.0f}%{busy}"


def _num(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return 0.0                                   # "[N/A]", "[Not Supported]"


def parse_gpu_query(text: str) -> list[GPUInfo]:
    out = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        out.append(GPUInfo(int(parts[0]), parts[1], parts[2],
                           _num(parts[3]) if len(parts) > 3 else 0.0,
                           _num(parts[4]) if len(parts) > 4 else 0.0,
                           _num(parts[5]) if len(parts) > 5 else 0.0))
    return out


def parse_apps_query(text: str) -> dict[str, list]:
    """uuid -> [(pid, name, used MiB)]; nvidia-smi's "[Not Found]" names are kept
    (the process exists in another pid namespace: still someone else's)."""
    out: dict[str, list] = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].startswith("GPU-"):
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        out.setdefault(parts[0], []).append((pid, parts[2] if len(parts) > 2 else "",
                                             _num(parts[3]) if len(parts) > 3 else 0.0))
    return out


def parse_inventory(text: str, *, host: Optional[str] = None,
                    own_pids: Iterable[int] = ()) -> list[GPUInfo]:
    """Output of ``QUERY_GPUS; echo _SEP; QUERY_APPS`` -> GPUs with their foreign processes."""
    gpus_txt, _, apps_txt = text.partition(_SEP)
    gpus = parse_gpu_query(gpus_txt)
    apps = parse_apps_query(apps_txt)
    own = set(own_pids)
    for g in gpus:
        g.host = host
        g.procs = [p for p in apps.get(g.uuid, []) if p[0] not in own]
    return gpus


def inventory_command() -> str:
    return f"{QUERY_GPUS}; echo {_SEP}; {QUERY_APPS}"


def _own_pids() -> set[int]:
    """This process and its descendants (our own runs are not foreign)."""
    pids = {os.getpid()}
    try:
        import psutil
        pids |= {c.pid for c in psutil.Process().children(recursive=True)}
    except Exception:
        pass
    return pids


def ssh_argv(host: str, ssh_key: Optional[str] = None, ssh: str = "ssh") -> list[str]:
    return [ssh, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"] + (["-i", ssh_key] if ssh_key else []) + [host]


def gpu_inventory(host: Optional[str] = None, *, ssh_key: Optional[str] = None, ssh: str = "ssh",
                  timeout: float = 30) -> list[GPUInfo]:
    """GPUs on this host (``host=None``) or on ``host`` over ssh; [] when nvidia-smi fails."""
    cmd = inventory_command()
    argv = ssh_argv(host, ssh_key, ssh) + [cmd] if host else ["bash", "-c", cmd]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning(f"GPU inventory on {host or 'localhost'} failed: {e!r}")
        return []
    return parse_inventory(r.stdout, host=host, own_pids=() if host else _own_pids())


async def gpu_inventory_async(host: Optional[str] = None, *, ssh_key: Optional[str] = None, ssh: str = "ssh",
                              timeout: float = 30) -> list[GPUInfo]:
    cmd = inventory_command()
    argv = ssh_argv(host, ssh_key, ssh) + [cmd] if host else ["bash", "-c", cmd]
    try:
        proc = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except (OSError, asyncio.TimeoutError) as e:
        log.warning(f"GPU inventory on {host or 'localhost'} failed: {e!r}")
        return []
    return parse_inventory(out.decode(errors="replace"), host=host, own_pids=() if host else _own_pids())


def parse_gpu_spec(spec) -> Optional[list[str]]:
    """None / "" / "auto" -> None (auto); "0,2" / [0, 2] / "GPU-..." -> explicit ids."""
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        ids = [str(s).strip() for s in spec if str(s).strip()]
        return ids or None
    s = str(spec).strip()
    if not s or s.lower() == "auto":
        return None
    if s.lower() == "all":
        return ["all"]
    return [p.strip() for p in s.split(",") if p.strip()]


def _matches(g: GPUInfo, ident: str) -> bool:
    return ident == str(g.index) or (ident.startswith("GPU-") and g.uuid.startswith(ident))


def select_gpus(spec, inventory: Sequence[GPUInfo], *, allow_shared: Optional[bool] = None,
                visible: Optional[str] = "env", util_max: Optional[float] = None) -> tuple[list[GPUInfo], str]:
    """-> (GPUs to use, why).

    - explicit ids (nvidia-smi indices or UUIDs): exactly those, idle or not;
    - ``"all"``: every GPU (visible ones, see below);
    - auto (None / "auto"): the idle GPUs; with ``allow_shared`` every GPU;
      when none is idle, the least busy one (the pre-auto behaviour: one GPU,
      shared), flagged in ``why``.

    ``visible`` (default: this process's ``CUDA_VISIBLE_DEVICES``; pass None
    for a remote host) limits auto/all to the GPUs it names."""
    allow_shared = allow_shared_gpu() if allow_shared is None else allow_shared
    ids = parse_gpu_spec(spec)
    inv = list(inventory)
    if ids and ids != ["all"]:
        chosen = [g for i in ids for g in inv if _matches(g, i)]
        missing = [i for i in ids if not any(_matches(g, i) for g in inv)]
        why = "explicit" + (f" (not found: {missing})" if missing else "")
        return chosen, why
    if visible == "env":
        visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() != "":
        vis = [v.strip() for v in visible.split(",") if v.strip()]
        inv = [g for g in inv if any(_matches(g, v) for v in vis)]
    if ids == ["all"]:
        return inv, "all"
    idle = [g for g in inv if g.idle(util_max)]
    if allow_shared:
        return inv, "auto, shared GPUs allowed"
    if idle:
        return idle, "auto: idle"
    if not inv:
        return [], "auto: no GPU found"
    least = min(inv, key=lambda g: (len(g.procs), g.util, g.mem_used_mb, g.index))
    return [least], "auto: no idle GPU, using the least busy one (shared; set KERNELBLASTER_ALLOW_SHARED_GPU=1 to use all)"


# ---------------------------------------------------------------- pool


@dataclass(eq=False)
class Device:
    """One schedulable GPU: ``host`` None = this host. ``ident`` is what goes in
    the run's ``CUDA_VISIBLE_DEVICES`` (a UUID when known, so CUDA's device
    ordering cannot remap it); None = leave the environment alone."""
    host: Optional[str]
    ident: Optional[str]
    name: str = "gpu"
    index: Optional[int] = None
    runs: int = 0
    busy_s: float = 0.0
    last_used: float = 0.0
    infra_fail: int = 0                              # consecutive infrastructure failures
    quarantined_until: float = 0.0
    quarantines: int = 0

    @property
    def key(self) -> tuple:
        return (self.host, self.ident)

    def env(self) -> dict:
        return {"CUDA_VISIBLE_DEVICES": self.ident} if self.ident is not None else {}

    def label(self) -> str:
        i = self.index if self.index is not None else (self.ident or "default")
        return f"{self.host + ':' if self.host else ''}{i}"

    @classmethod
    def from_info(cls, g: GPUInfo) -> "Device":
        return cls(g.host, g.uuid or str(g.index), g.name, g.index)


_ANY = object()


class NoDevice(RuntimeError):
    pass


class GPUPool:
    """Exclusive leases on devices, across hosts.

    ``acquire(group)`` waits for a free device and returns it; ``release`` gives
    it back. The free device used least recently wins (ties: fewest runs), so
    load spreads over every device. The first lease of a ``group`` (a problem)
    pins the group to that device's model: later leases of the group only get
    devices of the same model, so timings within a problem stay comparable
    when the pool mixes GPU models. ``release(infra_error=True)`` counts an
    infrastructure failure (scp/ssh down, no CUDA device, ...); after
    ``quarantine_after`` consecutive ones the device sits out ``quarantine_s``
    seconds. Candidate failures (compile errors, wrong output, crashes) must be
    released with ``infra_error=False``."""

    def __init__(self, devices: Sequence[Device], *, quarantine_after: int = 3, quarantine_s: float = 300.0,
                 clock: Callable[[], float] = time.monotonic, name: str = "gpus"):
        if not devices:
            raise NoDevice("GPU pool has no devices")
        self.devices = list(devices)
        self.quarantine_after, self.quarantine_s = quarantine_after, quarantine_s
        self.clock, self.name = clock, name
        self._busy: set = set()
        self._started: dict = {}
        self._group_model: dict = {}
        self._pending: dict = {}                     # host -> runs staged there, not yet leased
        self._cond: Optional[asyncio.Condition] = None
        self.peak_busy = 0

    def __len__(self) -> int:
        return len(self.devices)

    @property
    def models(self) -> list[str]:
        return sorted({d.name for d in self.devices})

    def homogeneous(self) -> bool:
        return len(self.models) <= 1

    def _condition(self) -> asyncio.Condition:
        self._cond = loop_local(self._cond, asyncio.Condition)
        return self._cond

    def _candidates(self, group, avoid=(), host=_ANY) -> list[Device]:
        model = self._group_model.get(group) if group is not None else None
        cands = [d for d in self.devices if (model is None or d.name == model) and (host is _ANY or d.host == host)]
        rest = [d for d in cands if d.key not in avoid]
        return rest or cands                         # avoid is a preference, never a dead end

    def _pick(self, group, avoid=(), host=_ANY) -> tuple[Optional[Device], float]:
        """A free healthy device, or (None, seconds until a quarantine ends)."""
        now = self.clock()
        cands = self._candidates(group, avoid, host)
        healthy = [d for d in cands if d.quarantined_until <= now]
        free = [d for d in healthy if d.key not in self._busy]
        if free:
            return min(free, key=lambda d: (d.last_used, d.runs, d.busy_s)), 0.0
        if not healthy:                              # everything quarantined: wait for the first to return
            return None, max(0.05, min(d.quarantined_until for d in cands) - now)
        return None, 0.0

    def pick_host(self, group=None, avoid_hosts: Iterable = (), reserve: bool = True):
        """The host to send the next run of ``group`` to: the most free healthy
        devices (minus runs already on their way there), then the least
        recently used one (None: no device). ``reserve`` counts this run as on
        its way until ``acquire(host=...)`` or ``unreserve``."""
        now = self.clock()
        cands = [d for d in self._candidates(group) if d.quarantined_until <= now] or self._candidates(group)
        rest = [d for d in cands if d.host not in set(avoid_hosts)] or cands
        if not rest:
            return None
        score: dict = {}
        for d in rest:
            free, lru = score.get(d.host, (0, float("inf")))
            score[d.host] = (free + (d.key not in self._busy), min(lru, d.last_used))
        host = max(score, key=lambda h: (score[h][0] - self._pending.get(h, 0), -score[h][1]))
        if reserve:
            self._pending[host] = self._pending.get(host, 0) + 1
        return host

    def unreserve(self, host) -> None:
        if self._pending.get(host, 0) > 0:
            self._pending[host] -= 1

    def _strike(self, dev: Device, now: float) -> None:
        dev.infra_fail += 1
        if dev.infra_fail >= self.quarantine_after:
            dev.quarantined_until = now + self.quarantine_s
            dev.quarantines += 1
            dev.infra_fail = 0
            log.warning(f"[{self.name}] {dev.label()} quarantined for {self.quarantine_s:.0f}s after "
                        f"{self.quarantine_after} infrastructure failures")

    def host_failure(self, host) -> None:
        """An infrastructure failure that is the host's, not a device's (e.g. the
        copy to it failed): a strike for each of its devices."""
        now = self.clock()
        for d in self.devices:
            if d.host == host:
                self._strike(d, now)

    async def acquire(self, group=None, avoid: Iterable = (), host=_ANY) -> Device:
        """A free device (waits for one); ``avoid``: device keys to pass over
        while others exist (a retry after an infrastructure failure);
        ``host``: only that host's devices."""
        avoid = set(avoid)
        cond = self._condition()
        async with cond:
            while True:
                dev, wait = self._pick(group, avoid, host)
                if dev is not None:
                    break
                try:
                    await asyncio.wait_for(cond.wait(), timeout=wait or None)
                except asyncio.TimeoutError:
                    pass
            self._busy.add(dev.key)
            if host is not _ANY:
                self.unreserve(host)
            self.peak_busy = max(self.peak_busy, len(self._busy))
            self._started[dev.key] = self.clock()
            if group is not None and group not in self._group_model:
                self._group_model[group] = dev.name
            return dev

    async def release(self, dev: Device, *, infra_error: bool = False) -> None:
        cond = self._condition()
        async with cond:
            now = self.clock()
            self._busy.discard(dev.key)
            dev.runs += 1
            dev.busy_s += now - self._started.pop(dev.key, now)
            dev.last_used = now
            if infra_error:
                self._strike(dev, now)
            else:
                dev.infra_fail = 0
            cond.notify_all()

    @asynccontextmanager
    async def lease(self, group=None, avoid: Iterable = (), host=_ANY):
        """``async with pool.lease(g) as l: ... l.device ...; l.infra_error = True``"""
        dev = await self.acquire(group, avoid, host)
        lease = Lease(dev)
        try:
            yield lease
        finally:
            await self.release(dev, infra_error=lease.infra_error)

    def pinned_model(self, group) -> Optional[str]:
        return self._group_model.get(group)

    def stats(self) -> list[dict]:
        return [{"device": d.label(), "name": d.name, "runs": d.runs, "busy_s": round(d.busy_s, 2),
                 "quarantines": d.quarantines} for d in self.devices]

    def describe(self) -> str:
        return ", ".join(f"{d.label()} ({d.name})" for d in self.devices)


@dataclass
class Lease:
    device: Device
    infra_error: bool = False


# Driver/runtime messages that mean the device or the link to it failed, not
# the candidate (a candidate's own CUDA errors -- illegal address, launch
# failure -- are its fault and stay candidate failures).
_INFRA_PATTERNS = re.compile(
    r"^MB_ERROR \[infra\]|no CUDA-capable device is detected|all CUDA-capable devices are busy or unavailable|"
    r"CUDA driver version is insufficient|cudaErrorNoDevice|cudaErrorDevicesUnavailable|"
    r"cudaErrorInsufficientDriver|cudaErrorSystemNotReady|CUDA_ERROR_NO_DEVICE|"
    r"forward compatibility was attempted|Failed to initialize NVML|"
    r"ssh: connect to host|Connection timed out|Connection refused|Connection reset by peer|"
    r"Could not resolve hostname|Host key verification failed|Permission denied \(publickey",
    re.I | re.M)


def is_infra_failure(text: Optional[str]) -> bool:
    return bool(text) and bool(_INFRA_PATTERNS.search(text))


def local_devices(spec=None, *, inventory: Optional[list[GPUInfo]] = None,
                  logger: Optional[logging.Logger] = None) -> list[Device]:
    """Devices for this host from a GPU spec (see :func:`select_gpus`).

    Without nvidia-smi: explicit ids pass through as ``CUDA_VISIBLE_DEVICES``
    values; auto falls back to one device that leaves the environment alone
    (the pre-pool behaviour)."""
    lg = logger or log
    inv = gpu_inventory() if inventory is None else inventory
    ids = parse_gpu_spec(spec)
    if not inv:
        if ids and ids != ["all"]:
            return [Device(None, i, "gpu") for i in ids]
        return [Device(None, None, "default")]
    chosen, why = select_gpus(spec, inv)
    lg.info(f"local GPUs ({why}): " + ("; ".join(g.short() for g in chosen) or "none"))
    if not chosen:
        if ids and ids != ["all"]:
            return [Device(None, i, "gpu") for i in ids]
        return [Device(None, None, "default")]
    return [Device.from_info(g) for g in chosen]


async def remote_devices(host: str, spec=None, *, ssh_key: Optional[str] = None, ssh: str = "ssh",
                         logger: Optional[logging.Logger] = None) -> list[Device]:
    """Devices on ``host`` (over ssh) from a GPU spec; explicit ids pass through
    when the inventory fails, auto then yields one device with no pinning."""
    lg = logger or log
    inv = await gpu_inventory_async(host, ssh_key=ssh_key, ssh=ssh)
    ids = parse_gpu_spec(spec)
    if not inv:
        lg.warning(f"{host}: GPU inventory failed; " + (f"using {ids} as given" if ids else "one unpinned device"))
        if ids and ids != ["all"]:
            return [Device(host, i, "gpu") for i in ids]
        return [Device(host, None, "default")]
    chosen, why = select_gpus(spec, inv, visible=None)
    lg.info(f"{host} GPUs ({why}): " + ("; ".join(g.short() for g in chosen) or "none"))
    return [Device.from_info(g) for g in chosen]


def gpu_lock_wrap(argv: str, ident: Optional[str], *, wait_s: float = 3600) -> str:
    """Shell text that runs ``argv`` (a shell word list) holding an flock on the
    GPU ``ident``: one timed run per GPU across every KernelBlaster process on
    the host (other runs, other agents), not only within this pool.

    The lock is ``$KERNELBLASTER_GPU_LOCK_DIR`` (default ``/dev/shm``)
    ``/kernelblaster_gpu_<uuid>.lock``. Without flock, or when the lock file
    cannot be opened (another user's file), the command runs unlocked.
    ``KERNELBLASTER_GPU_FLOCK=0`` (read here, on the client) turns it off."""
    if not ident or os.getenv("KERNELBLASTER_GPU_FLOCK", "1").lower() in ("0", "false", "no", "off"):
        return argv
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", ident)
    d = os.getenv("KERNELBLASTER_GPU_LOCK_DIR", "/dev/shm")
    lock = shlex.quote(f"{d}/kernelblaster_gpu_{safe}.lock")
    return (f"if command -v flock >/dev/null 2>&1 && {{ : >> {lock}; }} 2>/dev/null; "
            f"then flock -w {int(wait_s)} {lock} {argv}; else {argv}; fi")


def shell_env(env: dict) -> str:
    return " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
