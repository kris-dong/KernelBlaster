# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-slot problems exported by ModelBlaster (manifest ``mb-kb-problem/1``).

A problem is a graph partition with N *slots* (one editable kernel source per
op kind). ModelBlaster writes the manifest, the seed slot sources, the walker
(head/tail) and a self-contained driver; KernelBlaster edits one slot at a
time, assembles the translation unit and runs the driver, which checks the
partition and every op in isolation and reports per-slot times.

The assembly rule must match ModelBlaster's ``pipeline/kb_export.assemble``:
head, then each slot wrapped in ``namespace <prefix><id> { ... }`` (its
``#include`` lines hoisted, its ``#define`` names ``#undef``'d after), then
tail.
"""
from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict
from typing import Mapping, Optional

FORMAT = "mb-kb-problem/1"


# ---------------------------------------------------------------- problem

@dataclass
class Slot:
    id: str
    op: str
    op_class: str
    entry: str
    signature: str
    semantics: str
    seed_path: Path
    reference_path: Optional[Path]
    instances: list
    seed_source: str = "hand"

    @property
    def signature_sha(self) -> str:
        return hashlib.sha1(" ".join(self.signature.split()).encode()).hexdigest()[:12]

    def shapes(self) -> list[dict]:
        return [i.get("shape", {}) for i in self.instances]


class SlotProblem:
    """A ModelBlaster partition problem, loaded from its directory."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"{root}: not an {FORMAT} problem")
        m = self.manifest
        self.name: str = m["name"]
        self.target: str = m.get("target", "cuda")
        # storage dtype (fp32 / fp16) and the driver's bounds for it
        self.dtype: str = m.get("dtype", "fp32")
        self.tolerance: dict = m.get("tolerance", {})
        self.slots: dict[str, Slot] = {}
        for s in m["slots"]:
            self.slots[s["id"]] = Slot(
                id=s["id"], op=s["op"], op_class=s.get("op_class", "other"), entry=s["entry"],
                signature=s["signature"], semantics=s.get("semantics", ""),
                seed_path=self.root / s["file"],
                reference_path=(self.root / s["reference_file"]) if s.get("reference_file") else None,
                instances=s.get("instances", []), seed_source=s.get("seed_source", "hand"))
        self.order: list[str] = m["assembly"]["order"]
        self.driver_path = self._local_driver(self.root / m["driver"], self.root / m["data"])

    @staticmethod
    def _local_driver(driver: Path, data: Path) -> Path:
        """The driver .incbin's data.bin by absolute path; if the problem
        directory moved, use a copy that points at this directory's data."""
        txt = driver.read_text()
        m = re.search(r'\.incbin \\"([^"\\]+)\\"', txt)
        if m is None or Path(m.group(1)) == data or Path(m.group(1)).exists():
            return driver
        local = driver.with_name(".driver.local.cpp")
        new = txt.replace(m.group(1), str(data))
        if not local.exists() or local.read_text() != new:
            local.write_text(new)
        return local

    def seed_sources(self) -> dict[str, str]:
        return {sid: s.seed_path.read_text() for sid, s in self.slots.items()}

    def reference_text(self, sid: str) -> str:
        p = self.slots[sid].reference_path
        return p.read_text() if p and p.exists() else ""

    def assemble(self, sources: Mapping[str, str]) -> str:
        asm = self.manifest["assembly"]
        prefix = asm.get("namespace_prefix", "mbslot_")
        parts = [(self.root / asm["head"]).read_text()]
        for sid in self.order:
            parts.append(wrap_slot(sid, sources[sid], prefix))
        parts.append((self.root / asm["tail"]).read_text())
        return "\n".join(parts)

    def units(self, sources: Mapping[str, str]) -> list[tuple[str, str]]:
        """The same program as :meth:`assemble`, one TU per slot.

        ``(name, text)`` per TU: ``slot_<id>.cu`` = head + the wrapped slot,
        and ``walker_tail.cu`` = head + a declaration of every slot entry (from
        the manifest signature) + tail. Slot kernels keep their
        ``<prefix><id>::`` names, so NCU attribution is unchanged. A slot
        whose entry no longer matches its manifest signature, that relies on
        another slot's includes, or defines a host global twice fails to
        build this way; evaluators then fall back to the assembled TU.
        """
        asm = self.manifest["assembly"]
        prefix = asm.get("namespace_prefix", "mbslot_")
        head = (self.root / asm["head"]).read_text()
        out = [(f"slot_{sid}.cu", "\n".join([head, wrap_slot(sid, sources[sid], prefix)]))
               for sid in self.order]
        decls = "".join(f"namespace {prefix}{sid} {{ {self.slots[sid].signature}; }}\n" for sid in self.order)
        out.append(("walker_tail.cu", "\n".join([head, decls, (self.root / asm["tail"]).read_text()])))
        return out

    def neighbours(self, sid: str) -> list[str]:
        """Slots that feed or consume this slot's ops (for read-only context)."""
        ops = self.manifest["ops"]
        mine = [o for o in ops if o["slot"] == sid]
        my_in = {b for o in mine for b in o["inputs"]}
        my_out = {b for o in mine for b in o["outputs"]}
        out = []
        for o in ops:
            if o["slot"] != sid and (set(o["outputs"]) & my_in or set(o["inputs"]) & my_out):
                if o["slot"] not in out:
                    out.append(o["slot"])
        return out

    def adjacent_pairs(self) -> list[tuple[str, str, str]]:
        """(producer op, consumer op, tensor) where the consumer is the producer
        output's only reader inside the partition and it isn't a partition
        output -- the fusion candidates."""
        ops = self.manifest["ops"]
        bufs = self.manifest["buffers"]
        readers: dict[int, list[dict]] = {}
        for o in ops:
            for b in o["inputs"]:
                readers.setdefault(b, []).append(o)
        pairs = []
        for o in ops:
            for b in o["outputs"]:
                r = readers.get(b, [])
                if len(r) == 1 and bufs[b]["kind"] == "intermediate":
                    pairs.append((o["name"], r[0]["name"], bufs[b]["name"]))
        return pairs


def wrap_slot(slot_id: str, code: str, prefix: str = "mbslot_") -> str:
    # C linkage would make two slots' entries the same symbol across namespaces
    code = code.replace('extern "C"', "")
    lines = code.splitlines()
    includes = [l for l in lines if l.strip().startswith("#include")]
    body = "\n".join(l for l in lines if not l.strip().startswith("#include"))
    defines = sorted(set(re.findall(r"^\s*#\s*define\s+([A-Za-z_]\w*)", body, re.M)))
    undefs = "".join(f"#undef {d}\n" for d in defines)
    return ("\n".join(includes) + "\n" if includes else "") + (
        f"// ==== slot {slot_id} ====\nnamespace {prefix}{slot_id} {{\n{body}\n}}  // namespace {prefix}{slot_id}\n{undefs}")


# ---------------------------------------------------------------- results

@dataclass
class EvalResult:
    ok: bool                                   # compiled, ran, every check passed
    stage: str                                 # "compile" | "run" | "check" | "ok"
    partition_us: Optional[float] = None
    partition_graph_us: Optional[float] = None         # CUDA-graph replay (driver MB_GRAPH=1), else None
    slot_us: dict = field(default_factory=dict)
    op_us: dict = field(default_factory=dict)          # op name -> us (driver MB_OP_US)
    slot_checks: dict = field(default_factory=dict)    # sid -> {"rel_err", "ok"}
    outputs: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    log: str = ""
    kernel_us: dict = field(default_factory=dict)      # "slot/kernel" -> us (OpenCL driver MB_KERNEL_US)
    device: Optional[str] = None                       # OpenCL driver MB_DEVICE
    # end to end, as deployed (MB_E2E_US): the CUDA-graph replay of the whole
    # partition where it is capturable, else the eager launch sequence
    e2e_us: Optional[float] = None
    e2e_basis: Optional[str] = None                    # "graph" | "eager"
    graph_skip: Optional[str] = None                   # why there is no graph time (MB_GRAPH_SKIP)
    graph_nodes: Optional[int] = None
    # the incumbent timed in the same process, interleaved rep by rep (persistent host, ref=)
    ref_e2e_us: Optional[float] = None
    ref_e2e_basis: Optional[str] = None
    e2e_ratio: Optional[dict] = None                   # candidate / incumbent per rep: med, mean, min, max, n
    # per-op attribution: GPU time of each op's kernels run back to back (CUPTI, MB_ATTR_*)
    attr_slot_us: dict = field(default_factory=dict)
    attr_op_us: dict = field(default_factory=dict)
    attr_slot_launches: dict = field(default_factory=dict)
    attr_sum_us: Optional[float] = None
    attr_launches: Optional[float] = None              # kernel launches (+ memsets/copies) per forward pass
    attr_skip: Optional[str] = None
    launch_overhead_us: Optional[float] = None         # eager partition time - sum of kernel time
    launch_overhead_graph_us: Optional[float] = None
    launch_overhead_frac: Optional[float] = None       # of the eager partition time
    runtime: str = "process"                           # "process" (one driver run) | "host" (persistent host)
    eval_wall_s: Optional[float] = None
    # the GPU the run was timed on (pool evaluators): host, label, index, uuid, name (model), arch
    gpu: Optional[dict] = None

    @property
    def deployed_us(self) -> Optional[float]:
        """The problem's time as deployed: MB_E2E_US, else the graph time, else
        the eager partition time (drivers that predate MB_E2E_US)."""
        return self.e2e_us or self.partition_graph_us or self.partition_us

    def slot_time(self, sid: str) -> Optional[float]:
        """A slot's attributed GPU time (CUPTI), else its event-timed MB_SLOT_US."""
        v = self.attr_slot_us.get(sid)
        return v if v is not None else self.slot_us.get(sid)

    def slot_times(self) -> dict:
        return {s: self.slot_time(s) for s in (self.slot_us or self.attr_slot_us)}

    def launch_gap_us(self, sid: str) -> Optional[float]:
        """Event-timed minus attributed time of a slot: its launch and gap cost."""
        a, e = self.attr_slot_us.get(sid), self.slot_us.get(sid)
        return None if a is None or e is None else max(0.0, e - a)

    def failing_slots(self) -> list[str]:
        return [s for s, c in self.slot_checks.items() if not c.get("ok")]

    def error_text(self, limit: int = 3000) -> str:
        bits = list(self.errors)
        bad = self.failing_slots()
        if bad:
            bits.append("isolated-op check failed for slots: " + ", ".join(
                f"{s} (rel_err={self.slot_checks[s]['rel_err']:.3g})" for s in bad))
        for o, c in self.outputs.items():
            if not c.get("ok"):
                bits.append(f"output {o}: max_abs_err={c['max_abs_err']:.3g} > bound={c['bound']:.3g}")
        txt = "\n".join(bits) or self.log
        return txt[-limit:]


def parse_driver_output(text: str) -> EvalResult:
    passed = bool(re.search(r"^passed$", text, re.M))
    r = EvalResult(ok=passed, stage="ok" if passed else "check", log=text[-4000:])
    r.errors = re.findall(r"^MB_ERROR (.*)$", text, re.M)
    for m in re.finditer(r"^MB_CHECK output (\S+) max_abs_err=(\S+) bound=(\S+) ok=(\d)", text, re.M):
        r.outputs[m.group(1)] = {"max_abs_err": float(m.group(2)), "bound": float(m.group(3)),
                                 "ok": m.group(4) == "1"}
    for m in re.finditer(r"^MB_SLOT_CHECK (\S+) rel_err=(\S+) ok=(\d)", text, re.M):
        r.slot_checks[m.group(1)] = {"rel_err": float(m.group(2)), "ok": m.group(3) == "1"}
    for m in re.finditer(r"^MB_SLOT_US (\S+) (\S+)", text, re.M):
        r.slot_us[m.group(1)] = float(m.group(2))
    for m in re.finditer(r"^MB_OP_US (\S+) (\S+)", text, re.M):
        r.op_us[m.group(1)] = float(m.group(2))
    m = re.search(r"^MB_PARTITION_US (\S+)", text, re.M)
    if m:
        r.partition_us = float(m.group(1))
    m = re.search(r"^MB_PARTITION_GRAPH_US (\S+)", text, re.M)
    if m:
        r.partition_graph_us = float(m.group(1))
    for m in re.finditer(r"^MB_KERNEL_US (\S+) (\S+) (\S+)", text, re.M):
        r.kernel_us[f"{m.group(1)}/{m.group(2)}"] = float(m.group(3))
    m = re.search(r"^MB_DEVICE (.*)$", text, re.M)
    if m:
        r.device = m.group(1).strip()
    _parse_e2e(text, r)
    if r.errors and not passed:
        r.stage = "run"
    return r


def _parse_e2e(text: str, r: EvalResult) -> None:
    m = re.search(r"^MB_E2E_US (\S+) basis=(\w+)", text, re.M)
    if m:
        r.e2e_us, r.e2e_basis = float(m.group(1)), m.group(2)
    m = re.search(r"^MB_GRAPH_SKIP (.*)$", text, re.M)
    if m:
        r.graph_skip = m.group(1).strip()
    m = re.search(r"^MB_PARTITION_GRAPH_US \S+ reps=\d+ nodes=(\d+)", text, re.M)
    if m:
        r.graph_nodes = int(m.group(1))
    m = re.search(r"^MB_REF_E2E_US (\S+) basis=(\w+)", text, re.M)
    if m:
        r.ref_e2e_us, r.ref_e2e_basis = float(m.group(1)), m.group(2)
    m = re.search(r"^MB_E2E_RATIO med=(\S+) mean=(\S+) min=(\S+) max=(\S+) n=(\d+)(?: basis=(\w+))?", text, re.M)
    if m:
        r.e2e_ratio = {"med": float(m.group(1)), "mean": float(m.group(2)), "min": float(m.group(3)),
                       "max": float(m.group(4)), "n": int(m.group(5)), "basis": m.group(6) or "eager"}
    for m in re.finditer(r"^MB_ATTR_SLOT_US (\S+) (\S+)(?: launches=(\S+))?", text, re.M):
        r.attr_slot_us[m.group(1)] = float(m.group(2))
        if m.group(3):
            r.attr_slot_launches[m.group(1)] = float(m.group(3))
    for m in re.finditer(r"^MB_ATTR_OP_US (\S+) (\S+)", text, re.M):
        r.attr_op_us[m.group(1)] = float(m.group(2))
    m = re.search(r"^MB_ATTR_SUM_US (\S+) basis=\w+ reps=\d+ launches=(\S+)", text, re.M)
    if m:
        r.attr_sum_us, r.attr_launches = float(m.group(1)), float(m.group(2))
    m = re.search(r"^MB_ATTR_SKIP (.*)$", text, re.M)
    if m:
        r.attr_skip = m.group(1).strip()
    m = re.search(r"^MB_LAUNCH_OVERHEAD_US eager=(\S+)(?: graph=(\S+))? frac_eager=(\S+)", text, re.M)
    if m:
        r.launch_overhead_us, r.launch_overhead_frac = float(m.group(1)), float(m.group(3))
        r.launch_overhead_graph_us = float(m.group(2)) if m.group(2) else None


# ---------------------------------------------------------------- driver run options

def _env_on(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() not in ("0", "false", "no", "off", "")


def driver_env(only_slots=None, screen: bool = False) -> dict:
    """Environment for one run of a ModelBlaster CUDA driver (MB docs/gpu_harness_perf.md):
    ``only_slots`` -> ``MB_ONLY_SLOTS`` (isolated-op checks for those slots'
    ops only; the partition outputs are still checked, the partition and every
    op still timed), ``screen`` -> ``MB_SCREEN=1`` (short timing budgets).
    Drivers that predate these variables ignore them and run in full."""
    env = {}
    if only_slots:
        env["MB_ONLY_SLOTS"] = ",".join(only_slots)
    if screen:
        env["MB_SCREEN"] = "1"
    return env


def step_eval_options(sid: str) -> dict:
    """``evaluate()`` options for a candidate that changes only slot ``sid``
    (the agent's step; confirm runs and the final composite run in full):
    isolated checks of its ops only (``KERNELBLASTER_MULTISLOT_ONLY_SLOTS``,
    default on) and screening budgets (``KERNELBLASTER_MULTISLOT_SCREEN``,
    default on)."""
    return {"only_slots": [sid] if _env_on("KERNELBLASTER_MULTISLOT_ONLY_SLOTS", "1") else None,
            "screen": _env_on("KERNELBLASTER_MULTISLOT_SCREEN", "1")}


# ---------------------------------------------------------------- evaluators

# the device the current evaluate() call's run was leased (set at each lease, read by evaluate)
_LEASED: contextvars.ContextVar = contextvars.ContextVar("kb_leased_device", default=None)


def _leased(dev) -> None:
    _LEASED.set(dev)


class Evaluator:
    """Compile an assembled candidate with the problem's driver and run it.

    ``only_slots`` / ``screen``: see :func:`driver_env` (ignored where the
    driver has no such modes). ``incumbent``: the sources the candidate is
    compared with; an evaluator with ``pairs_incumbent(problem)`` true times
    both in one process, interleaved (``EvalResult.e2e_ratio``), the others
    ignore it (callers pass it only where it is supported)."""

    def pairs_incumbent(self, problem: SlotProblem) -> bool:
        return False

    async def evaluate(self, problem: SlotProblem, sources: Mapping[str, str], tag: str, *,
                       only_slots=None, screen: bool = False, incumbent=None) -> EvalResult:
        raise NotImplementedError

    async def profile(self, problem: SlotProblem, sources: Mapping[str, str], tag: str, slots=None):
        """NCU per slot (every slot, or only ``slots``): -> ({slot: summary},
        None) or ({}, why not)."""
        return {}, f"{type(self).__name__} does not profile"

    def __init_subclass__(cls, **kw):
        # telemetry spans on evaluate / profile and the build / run stages (telemetry/evaluator.py)
        super().__init_subclass__(**kw)
        from ..telemetry.evaluator import instrument_class
        instrument_class(cls)


def _ncu_result(text: str):
    from .ncu import ncu_error, profile_by_slot

    err = ncu_error(text)
    if err:
        return {}, err
    by = profile_by_slot(text)
    return (by, None) if by else ({}, "ncu output had no mbslot_ kernels")


class LocalEvaluator(Evaluator):
    """nvcc + run on this host (dev/tests; no KernelBlaster servers needed).

    Builds: every nvcc process (slot units, the link, the one-shot assembled
    build) goes through one limiter. ``build_jobs`` / ``max_parallel`` (or
    ``KERNELBLASTER_MULTISLOT_COMPILE_JOBS``) fix its size; by default it
    adapts to the host's free CPUs and memory (``servers/utils/resources.py``).

    Runs: every timed run and NCU pass leases one GPU from a :class:`GPUPool`
    (one run per GPU at a time, ``CUDA_VISIBLE_DEVICES`` set per run). ``gpus``
    is ``"auto"`` (default: idle GPUs), ``"all"``, or a list of nvidia-smi
    indices / UUIDs; ``pool`` passes a ready pool (e.g. shared between
    evaluators). Leases are grouped by problem, so one problem's runs stay on
    one GPU model.

    Runtime: ``runtime="host"`` (the default, ``"auto"``, for drivers that have
    it; ``KERNELBLASTER_MULTISLOT_RUNTIME``) evaluates candidates in a
    persistent host per (problem, GPU) (``knowledge/runtime_host.py``): the
    candidate is a shared object of its slot units, the host keeps the CUDA
    context and the data, the partition is timed as a CUDA graph where it is
    capturable, and ``incumbent=`` sources are timed in the same request
    (paired). ``"process"`` (and the fallback when the host cannot be built or
    started) runs one driver process per evaluation, as before.

    Hardware: ``gpu_model`` (e.g. ``"NVIDIA L40S"``) keeps only the selected
    devices of that model -- none of it is an error (``NoDevice``), never a
    run on another model -- and ``max_devices`` caps the pool (1: every run on
    one GPU). Every result records the device it was timed on
    (``EvalResult.gpu``)."""

    def __init__(self, arch: str = "75", work_root: Optional[Path] = None, nvcc: str = "nvcc",
                 max_parallel: Optional[int] = None, timeout_s: int = 900, keep: bool = False,
                 incremental: Optional[bool] = None, cache_dir: Optional[Path] = None,
                 build_jobs: Optional[int] = None, gpus=None, pool=None, infra_retries: int = 2,
                 graph: Optional[bool] = None, runtime: Optional[str] = None, hosts_per_device: int = 3,
                 gpu_model: Optional[str] = None, max_devices: Optional[int] = None):
        from ..servers.utils.resources import build_limiter
        from .runtime_host import HostPool

        self.runtime = (runtime or os.getenv("KERNELBLASTER_MULTISLOT_RUNTIME", "auto")).lower()
        if self.runtime not in ("auto", "host", "process"):
            raise ValueError(f"runtime must be auto|host|process, not {self.runtime!r}")
        self._hosts = HostPool(hosts_per_device)
        self._host_off: dict = {}                # problem root -> why the host is not used

        self.arch, self.nvcc, self.timeout_s, self.keep = arch, nvcc, timeout_s, keep
        # also time the partition as a CUDA graph (MB_PARTITION_GRAPH_US): every
        # launch on the per-thread default stream so the driver can capture it
        self.graph = _env_on("KERNELBLASTER_MULTISLOT_GRAPH", "0") if graph is None else graph
        self.work_root = Path(work_root) if work_root else None
        # compiler processes (unit objects, links, assembled builds)
        self._jobs = build_limiter(build_jobs or max_parallel, env="KERNELBLASTER_MULTISLOT_COMPILE_JOBS",
                                   name="multislot nvcc")
        self._sem = self._unit_sem = self._jobs
        self.gpus, self._pool, self._pool_lock = gpus, pool, None
        self.gpu_model, self.max_devices = gpu_model, max_devices
        self.infra_retries = infra_retries
        # one TU per slot, objects + binaries cached by content (see _build_units)
        self.incremental = _incremental_default() if incremental is None else incremental
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._cache = None

    # ---- GPUs
    async def _devices(self) -> list:
        from ..servers.utils.resources import local_devices

        return await asyncio.to_thread(local_devices, self.gpus)

    async def pool(self):
        """The GPU pool (built on first use from ``gpus``)."""
        from ..servers.utils.resources import GPUPool, loop_local, restrict_devices

        if self._pool is None:
            self._pool_lock = loop_local(self._pool_lock, asyncio.Lock)
            async with self._pool_lock:
                if self._pool is None:
                    devs = restrict_devices(await self._devices(), gpu_model=self.gpu_model,
                                            max_devices=self.max_devices)
                    self._pool = GPUPool(devs, name=type(self).__name__)
                    _log().info(f"{type(self).__name__}: GPU pool {self._pool.describe()}; "
                                f"compile jobs {self._jobs.describe()}")
        return self._pool

    async def _timed_run(self, problem, exe: Path, env: Optional[dict] = None, prefix: str = "") -> Optional[str]:
        """``_run`` on a leased GPU; an infrastructure failure (device or link
        down, not the candidate) is retried on another device."""
        from ..servers.utils.resources import is_infra_failure

        pool = await self.pool()
        avoid: set = set()
        for attempt in range(self.infra_retries + 1):
            async with pool.lease(str(problem.root), avoid) as lease:
                _leased(lease.device)
                text = await self._run(exe, env, prefix, device=lease.device, problem=problem)
                lease.infra_error = text is not None and is_infra_failure(text) and \
                    not re.search(r"^passed$", text, re.M)
            if not lease.infra_error:
                return text
            dev = lease.device
            # the link to a host failed: retry on another host if there is one
            avoid |= ({d.key for d in pool.devices if d.host == dev.host}
                      if dev.host and "MB_ERROR [infra]" in text else {dev.key})
            _log().warning(f"{problem.name}: infrastructure failure on {lease.device.label()} "
                           f"(attempt {attempt + 1}): {text[-300:]!r}")
        return text

    def _workdir(self, problem, tag) -> Path:
        root = self.work_root or problem.root / "kb_work"
        work = root / f"{re.sub(r'[^A-Za-z0-9_]+', '_', tag)}_{uuid.uuid4().hex[:6]}"
        work.mkdir(parents=True, exist_ok=True)
        return work

    def _flags(self) -> list[str]:
        return ["-O3", f"-arch=sm_{self.arch}", "-std=c++17", "-lineinfo",
                *(["--default-stream", "per-thread"] if getattr(self, "graph", False) else [])]

    def _driver_flags(self) -> list[str]:
        """Extra flags for the driver TU: CUPTI's headers (per-op attribution)."""
        return cupti_flags(self.nvcc)

    def _run_env(self, only_slots=None, screen: bool = False) -> dict:
        return {**driver_env(only_slots, screen), **({"MB_GRAPH": "1"} if getattr(self, "graph", False) else {})}

    def _build_cache(self, problem):
        from ..servers.utils.build_cache import BuildCache

        if self._cache is None:
            root = self._cache_dir or (self.work_root or problem.root / "kb_work") / ".build_cache"
            self._cache = BuildCache(root)
        return self._cache

    async def _nvcc(self, args: list[str]) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(self.nvcc, *args, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), self.timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            return -1, f"nvcc timed out after {self.timeout_s}s"
        return proc.returncode, out.decode(errors="replace")

    async def _build_units(self, problem, sources, work: Path) -> bool:
        """The incremental build: every slot / the tail / the driver as its
        own object, each cached by (flags, text, absolute inputs), then one
        cached link. The same nvcc invocation as the one-shot build, split
        into its compile and link steps. False if any step fails."""
        from ..servers.utils.build_cache import cache_key, external_inputs, link_or_copy

        cache = self._build_cache(problem)
        flags = self._flags()

        class _Failed(Exception):
            pass

        async def obj(src: Path, text: str, extra=()):
            key = cache_key("nvcc-c", self.nvcc, *flags, *extra, text, *external_inputs(text))

            async def build(tmp: Path):
                async with self._unit_sem:
                    rc, _ = await self._nvcc([*flags, *extra, "-c", str(src), "-o", str(tmp)])
                if rc != 0:
                    raise _Failed(src.name)
            return key, (await cache.get_or_build(key, ".o", build))[0]

        jobs = []
        for name, text in problem.units(sources):
            (work / name).write_text(text)
            jobs.append(obj(work / name, text))
        drv = problem.driver_path
        jobs.append(obj(drv, drv.read_text(), self._driver_flags()))
        objs = await asyncio.gather(*jobs, return_exceptions=True)
        if any(isinstance(o, BaseException) for o in objs):
            return False
        link_key = cache_key("nvcc-link", self.nvcc, f"-arch=sm_{self.arch}", *[k for k, _ in objs])

        async def link(tmp: Path):
            rc, _ = await self._nvcc([f"-arch=sm_{self.arch}", *[str(p) for _, p in objs], "-o", str(tmp)])
            if rc != 0:
                raise _Failed("link")
        try:
            exe, _ = await cache.get_or_build(link_key, ".exe", link)
        except _Failed:
            return False
        link_or_copy(exe, work / "problem")
        return True

    async def _build(self, problem, sources, work: Path) -> Optional[EvalResult]:
        """nvcc the assembled TU + driver into work/problem; an EvalResult on failure."""
        if self.incremental and await self._build_units(problem, sources, work):
            return None
        # the one-shot build of the assembled TU is authoritative (errors,
        # and anything the per-slot split cannot express)
        (work / "kernels.cu").write_text(problem.assemble(sources))
        async with self._sem:
            proc = await asyncio.create_subprocess_exec(
                self.nvcc, *self._flags(), *self._driver_flags(), "-o", str(work / "problem"),
                str(work / "kernels.cu"), str(problem.driver_path),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            out, _ = await asyncio.wait_for(proc.communicate(), self.timeout_s)
        if proc.returncode != 0:
            return EvalResult(ok=False, stage="compile", errors=[_compile_errors(out.decode())],
                              log=out.decode()[-4000:])
        return None

    async def _run(self, exe: Path, env: Optional[dict] = None, prefix: str = "", device=None,
                   problem=None) -> Optional[str]:
        """Run exe (optionally under a prefix command, e.g. ncu) on ``device``
        (holding its cross-process lock); stdout+stderr, None on timeout."""
        from ..servers.utils.resources import gpu_lock_wrap

        cmd = f"{prefix} {shlex.quote(str(exe))}" if prefix else shlex.quote(str(exe))
        if device is not None:
            cmd = gpu_lock_wrap(cmd, device.ident, wait_s=self.timeout_s)
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, **(device.env() if device is not None else {}), **(env or {})})
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), self.timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            return None
        return out.decode(errors="replace")

    def gpu_info(self, dev) -> Optional[dict]:
        """What a result records of the device it was timed on."""
        if dev is None:
            return None
        ident = dev.ident
        return {"host": dev.host, "label": dev.label(), "index": dev.index,
                "uuid": ident if ident and str(ident).startswith("GPU-") else None, "name": dev.name,
                "arch": f"sm_{self.arch}"}

    async def evaluate(self, problem, sources, tag, *, only_slots=None, screen=False, incumbent=None):
        tok = _LEASED.set(None)
        try:
            res = await self._evaluate(problem, sources, tag, only_slots, screen, incumbent)
            if res is not None and res.gpu is None:
                res.gpu = self.gpu_info(_LEASED.get())
            return res
        finally:
            _LEASED.reset(tok)

    async def _evaluate(self, problem, sources, tag, only_slots, screen, incumbent):
        if self._use_host(problem):
            res = await self._evaluate_host(problem, sources, tag, only_slots, screen, incumbent)
            if res is not None:
                return res
        work = self._workdir(problem, tag)
        try:
            bad = await self._build(problem, sources, work)
            if bad is not None:
                return bad
            text = await self._timed_run(problem, work / "problem", self._run_env(only_slots, screen))
            if text is None:
                return EvalResult(ok=False, stage="run", errors=[f"timeout after {self.timeout_s}s"])
            return parse_driver_output(text)
        finally:
            if not self.keep:
                shutil.rmtree(work, ignore_errors=True)

    async def profile(self, problem, sources, tag, slots=None):
        from .ncu import ncu_command

        work = self._workdir(problem, tag + "_ncu")
        try:
            bad = await self._build(problem, sources, work)
            if bad is not None:
                return {}, "build failed: " + bad.error_text(300)
            text = await self._timed_run(problem, work / "problem", {"MB_MODE": "profile"}, ncu_command(slots=slots))
            return _ncu_result(text) if text is not None else ({}, "ncu timed out")
        finally:
            if not self.keep:
                shutil.rmtree(work, ignore_errors=True)


    # ---- the persistent host (knowledge/runtime_host.py)
    # telemetry stages of the host path (telemetry/evaluator.py); the host client itself
    # records driver.host_start (start / restart) and run.exec with the driver's phases
    # (load, checks, graph_build = capture, time_partition = graph / eager replay, attribution)
    TELEMETRY_STAGES = {"_build_so": "build", "_host_exe": "build", "_host_request": "run",
                        "_stage_file": "run.stage"}

    def _use_host(self, problem) -> bool:
        from .runtime_host import supports_host

        drv = getattr(problem, "driver_path", None)
        if self.runtime == "process" or getattr(problem, "target", "cuda") != "cuda" or drv is None or \
                str(problem.root) in self._host_off:
            return False
        if not supports_host(drv.read_text()):
            self._host_off[str(problem.root)] = "driver exported without the persistent host"
            if self.runtime == "host":
                _log().warning(f"{problem.name}: {self._host_off[str(problem.root)]}; one process per run")
            return False
        return True

    def pairs_incumbent(self, problem) -> bool:
        return self._use_host(problem)

    def _so_flags(self) -> list[str]:
        """The candidate's shared object: always the per-thread default stream (graph capture)."""
        return ["-O3", f"-arch=sm_{self.arch}", "-std=c++17", "-lineinfo", "--default-stream", "per-thread",
                "-Xcompiler", "-fPIC"]

    async def _build_so(self, problem, sources, work: Path) -> tuple[Optional[Path], Optional[EvalResult]]:
        """The candidate's kernels as a shared object (cached by content):
        slot units + tail linked -shared, else the assembled TU (authoritative errors)."""
        from ..servers.utils.build_cache import cache_key, external_inputs

        cache = self._build_cache(problem)
        flags = self._so_flags()
        work.mkdir(parents=True, exist_ok=True)

        class _Failed(Exception):
            pass

        async def obj(src: Path, text: str):
            key = cache_key("nvcc-c", self.nvcc, *flags, text, *external_inputs(text))

            async def build(tmp: Path):
                async with self._unit_sem:
                    rc, _ = await self._nvcc([*flags, "-c", str(src), "-o", str(tmp)])
                if rc != 0:
                    raise _Failed(src.name)
            return key, (await cache.get_or_build(key, ".o", build))[0]

        if self.incremental:
            jobs = []
            for name, text in problem.units(sources):
                (work / name).write_text(text)
                jobs.append(obj(work / name, text))
            objs = await asyncio.gather(*jobs, return_exceptions=True)
            if not any(isinstance(o, BaseException) for o in objs):
                link_key = cache_key("nvcc-so", self.nvcc, f"-arch=sm_{self.arch}", *[k for k, _ in objs])

                async def link(tmp: Path):
                    async with self._sem:
                        rc, _ = await self._nvcc(["-shared", f"-arch=sm_{self.arch}", *[str(p) for _, p in objs],
                                                  "-o", str(tmp)])
                    if rc != 0:
                        raise _Failed("link")
                try:
                    return (await cache.get_or_build(link_key, ".so", link))[0], None
                except _Failed:
                    pass
        text = problem.assemble(sources)
        (work / "kernels.cu").write_text(text)
        key = cache_key("nvcc-so-tu", self.nvcc, *flags, text, *external_inputs(text))
        log = {}

        async def build_tu(tmp: Path):
            async with self._sem:
                rc, out = await self._nvcc([*flags, "-shared", "-o", str(tmp), str(work / "kernels.cu")])
            if rc != 0:
                log["out"] = out
                raise _Failed("kernels.cu")
        try:
            return (await cache.get_or_build(key, ".so", build_tu))[0], None
        except _Failed:
            out = log.get("out", "")
            return None, EvalResult(ok=False, stage="compile", errors=[_compile_errors(out)], log=out[-4000:],
                                    runtime="host")

    async def _host_exe(self, problem) -> Optional[Path]:
        """driver.cpp built -DMB_HOST (cached by content); None if it does not build."""
        from ..servers.utils.build_cache import cache_key, external_inputs

        drv = problem.driver_path
        text = drv.read_text()
        flags = ["-O3", f"-arch=sm_{self.arch}", "-std=c++17", "-DMB_HOST", *self._driver_flags()]
        key = cache_key("nvcc-host", self.nvcc, *flags, text, *external_inputs(text))
        log = {}

        async def build(tmp: Path):
            async with self._sem:
                rc, out = await self._nvcc([*flags, "-o", str(tmp), str(drv), "-ldl"])
            if rc != 0:
                log["out"] = out
                raise RuntimeError("host build failed")
        try:
            return (await self._build_cache(problem).get_or_build(key, ".host", build))[0]
        except RuntimeError:
            self._host_off[str(problem.root)] = "host build failed: " + _compile_errors(log.get("out", ""), 500)
            _log().warning(f"{problem.name}: {self._host_off[str(problem.root)]}; one process per run")
            return None

    def _host_vars(self, only_slots, screen) -> dict:
        return {**driver_env(only_slots, screen), "MB_GRAPH": "1"}

    async def _evaluate_host(self, problem, sources, tag, only_slots, screen, incumbent) -> Optional[EvalResult]:
        """One request to the problem's host; None to fall back to one process per run."""
        exe = await self._host_exe(problem)
        if exe is None:
            return None
        work = self._workdir(problem, tag)
        try:
            so, bad = await self._build_so(problem, sources, work)
            if bad is not None:
                return bad
            ref = None
            if incumbent is not None and dict(incumbent) != dict(sources):
                ref, rbad = await self._build_so(problem, incumbent, work / "incumbent")
                if rbad is not None:
                    _log().warning(f"{problem.name}: incumbent does not build as a shared object; unpaired run")
            rep = await self._host_request(problem, exe, so, ref, self._host_vars(only_slots, screen))
            if rep is None:
                return None
            return self._host_result(problem, rep)
        finally:
            if not self.keep:
                shutil.rmtree(work, ignore_errors=True)

    def _host_result(self, problem, rep) -> Optional[EvalResult]:
        if rep.status == "start_failed":
            # the host itself failed (it printed): one process per run from now
            # on; a link or device failure before it printed: this run only
            if re.search(r"^MB_", rep.text, re.M):
                self._host_off[str(problem.root)] = "host did not start: " + rep.text[-300:]
            _log().warning(f"{problem.name}: host did not start ({rep.text[-300:]!r}); one process for this run")
            return None
        r = parse_driver_output(rep.text)
        r.runtime, r.eval_wall_s = "host", round(rep.wall_s, 3)
        if rep.status != "done" or rep.poisoned:
            r.ok, r.stage = False, "run"
            r.errors.append({"timeout": f"timeout after {self.timeout_s}s (the runtime was restarted)",
                             "crash": f"the candidate crashed the runtime (exit {rep.rc}); it was restarted"}.get(
                rep.status, "the candidate left the CUDA context unusable (sticky CUDA error); the runtime "
                            "was restarted"))
        return r

    async def _host_request(self, problem, exe: Path, so: Path, ref: Optional[Path], env: dict):
        """Run one request on a leased local GPU's host (started on first use)."""
        from ..servers.utils.resources import gpu_lock_file, is_infra_failure
        from .runtime_host import HostClient, host_env

        pool = await self.pool()
        avoid: set = set()
        rep = None
        for attempt in range(self.infra_retries + 1):
            async with pool.lease(str(problem.root), avoid) as lease:
                dev = lease.device
                _leased(dev)
                key = (str(problem.root), dev.key, str(exe))
                h = await self._hosts.get(key, dev.key, lambda: HostClient(
                    [str(exe)], env=host_env(dev.env()), label=f"{problem.name}@{dev.label()}",
                    start_timeout=self.timeout_s, device=dev.label()))
                rep = await h.request(str(so), ref=str(ref) if ref else None, env=env,
                                      lock=gpu_lock_file(dev.ident), lock_wait=self.timeout_s, timeout=self.timeout_s)
                self._hosts.stats.note(rep, dev.label())
                lease.infra_error = is_infra_failure(rep.text) and not re.search(r"^passed$", rep.text, re.M)
            if not lease.infra_error:
                return rep
            await self._hosts.drop(key)
            avoid.add(dev.key)
            _log().warning(f"{problem.name}: infrastructure failure on {dev.label()} (attempt {attempt + 1}): "
                           f"{rep.text[-300:]!r}")
        return rep

    def host_stats(self) -> dict:
        return {"runtime": self.runtime, **self._hosts.stats.to_json(),
                "fallbacks": {Path(k).name: v for k, v in self._host_off.items()}}

    async def close(self):
        await self._hosts.close()


class SSHEvaluator(LocalEvaluator):
    """Build here, run on remote GPU hosts over ssh (e.g. idle L40S GPUs with
    profiler-counter access). The binary is self-contained (data.bin is
    linked in), so only the executable is copied.

    Problems whose data is too big to link (ModelBlaster ``kb_export`` above
    1 GiB: the driver mmaps data.bin at start-up from ``$MB_DATA_PATH``)
    have data.bin copied once per (host, data sha1) to
    ``<remote_dir>/data_<sha12>/data.bin``, before a GPU is leased, and the
    runs (one process or the persistent host) get ``MB_DATA_PATH`` pointing
    at it. Concurrent runs of the same data on a host share one transfer
    (rsync ``--partial``, resumed by a retry; else scp) into a temporary
    name, verified (size + sha1) and renamed into place; a valid copy left
    by an earlier run is reused. Hosts that already hold a problem's data
    are preferred while they have a GPU to spare; another host gets its own
    copy when it is picked. ``close()`` removes the copies this evaluator
    made unless ``keep_data`` (``KERNELBLASTER_SSH_KEEP_DATA=1``) is set.

    ``host`` is one ``user@host`` or several; ``cuda_visible_devices`` is a GPU
    spec for every host (``"auto"``/None: the host's idle GPUs, ``"all"``, or
    ids) or a ``{host: spec}`` mapping; ``remote_dir`` is one directory used on
    every host, or ``{host: dir}``. Each run leases a free (host, GPU) from the
    pool and is copied to that host's directory. All hosts must run the
    binary built for ``arch``; a problem's runs stay on one GPU model.

    Transport: ssh connections are multiplexed (``ControlMaster``; off with
    ``KERNELBLASTER_SSH_MUX=0``). Where both ends have rsync
    (``KERNELBLASTER_SSH_RSYNC=0`` turns it off), a binary is sent as an rsync
    delta against the problem's previous binary on that host, which stays
    there as ``.kb_basis_<problem>`` (``close()`` removes them): candidates
    differ in one slot while the linked data dominates the size, so a 100 MB
    binary costs a few hundred KB. Otherwise it is scp'd."""

    def __init__(self, host, *, arch: str, remote_dir, ssh_key: Optional[str] = None,
                 cuda_visible_devices=None, remote_path: str = "/usr/local/cuda/bin", ssh: str = "ssh",
                 scp: str = "scp", rsync: Optional[str] = "rsync", keep_data: Optional[bool] = None,
                 remote_manifest: Optional[Path] = None, **kw):
        super().__init__(arch, **kw)
        # resume: what this run staged / started on the hosts (one JSON line each), so a
        # resumed run removes exactly the leftovers of its killed segment (cleanup_previous)
        self.remote_manifest = Path(remote_manifest) if remote_manifest else None
        self._noted: set = set()
        self.hosts = [host] if isinstance(host, str) else list(host)
        if not self.hosts:
            raise ValueError("SSHEvaluator needs at least one host")
        self.host = self.hosts[0]
        self.remote_dirs = {h: str(remote_dir[h] if isinstance(remote_dir, Mapping) else remote_dir).rstrip("/")
                            for h in self.hosts}
        self.remote_dir = self.remote_dirs[self.host]
        self.cvd = {h: (cuda_visible_devices.get(h) if isinstance(cuda_visible_devices, Mapping)
                        else cuda_visible_devices) for h in self.hosts}
        self.key, self.remote_path, self.ssh, self.scp = ssh_key, remote_path, ssh, scp
        use_rsync = os.getenv("KERNELBLASTER_SSH_RSYNC", "1").lower() not in ("0", "false", "no", "off")
        self.rsync = (shutil.which(rsync) if rsync and use_rsync else None)
        self._rsync_hosts: set = set()           # hosts with rsync (probed with the inventory)
        self._mux_dir: Optional[Path] = None
        if os.getenv("KERNELBLASTER_SSH_MUX", "1").lower() not in ("0", "false", "no", "off"):
            d = Path(os.getenv("KERNELBLASTER_SSH_CONTROL_DIR") or Path.home() / ".ssh" / "kb_mux")
            try:
                d.mkdir(parents=True, exist_ok=True, mode=0o700)
                self._mux_dir = d
            except OSError:
                pass
        self.transfer = {"rsync": 0, "scp": 0}
        self._staged: dict = {}                  # host -> OrderedDict of staged host binaries / .so names
        # run-time data.bin (problems too big to link): one copy per (host, data sha1)
        from ..servers.utils.resources import truthy
        self.keep_data = truthy(os.getenv("KERNELBLASTER_SSH_KEEP_DATA")) if keep_data is None else keep_data
        self._data_file: dict = {}               # driver path -> local data.bin or None (linked in)
        self._data_sha: dict = {}                # (path, size, mtime) -> sha1
        self._data_at: dict = {}                 # (host, sha) -> remote data.bin
        self._data_made: dict = {}               # (host, sha) -> remote data.bin this evaluator copied
        self._data_locks: dict = {}              # (host, sha) -> asyncio.Lock (one transfer at a time)
        self._data_err: dict = {}                # (host, sha) -> (monotonic time, error) of the last failure
        self.data_transfer = {"copies": 0, "reused": 0, "failures": 0, "bytes": 0, "seconds": 0.0}

    def _ssh_base(self) -> list[str]:
        mux = (["-o", "ControlMaster=auto", "-o", f"ControlPath={self._mux_dir}/%C", "-o", "ControlPersist=120"]
               if self._mux_dir else [])
        return ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", *mux] + (["-i", self.key] if self.key else [])

    async def _sh(self, *argv, timeout: float = 600) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return -1, f"timed out after {timeout}s"
        return proc.returncode, out.decode(errors="replace")

    async def _devices(self) -> list:
        from ..servers.utils.resources import NoDevice, remote_devices

        devs = []
        for h in self.hosts:
            rc, out = await self._sh(self.ssh, *self._ssh_base(), h,
                                     f"mkdir -p {shlex.quote(self.remote_dirs[h])} && "
                                     "{ command -v rsync >/dev/null && echo KB_HAS_RSYNC; true; }", timeout=60)
            if rc != 0:
                _log().warning(f"{h}: unreachable or {self.remote_dirs[h]} not writable, skipped: {out[-200:]}")
                continue
            if "KB_HAS_RSYNC" in out and self.rsync:
                self._rsync_hosts.add(h)
            devs += await remote_devices(h, self.cvd[h], ssh_key=self.key, ssh=self.ssh, logger=_log())
        if not devs:
            raise NoDevice(f"no usable GPU on {', '.join(self.hosts)}")
        return devs

    @staticmethod
    def _basis(problem) -> Optional[str]:
        if problem is None:
            return None
        return ".kb_basis_" + hashlib.sha1(str(problem.root).encode()).hexdigest()[:12]

    # ---- run-time data.bin (drivers that read it from $MB_DATA_PATH)
    TELEMETRY_STAGES = {**LocalEvaluator.TELEMETRY_STAGES, "_stage_data": "run.stage.data"}
    DATA_TIMEOUT_S = 3600

    def _local_data(self, problem) -> Optional[Path]:
        """The problem's data.bin if its driver reads it at run time, else None (linked in)."""
        drv = getattr(problem, "driver_path", None)
        if drv is None:
            return None
        key = str(drv)
        if key not in self._data_file:
            reads = 'getenv("MB_DATA_PATH")' in drv.read_text()
            data = problem.root / problem.manifest.get("data", "data.bin") if reads else None
            self._data_file[key] = data
        return self._data_file[key]

    async def _problem_data(self, problem) -> Optional[tuple[Path, str, int]]:
        """(local data.bin, its sha1, size) for a run-time-data problem, else None."""
        path = self._local_data(problem)
        if path is None:
            return None
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
        if key not in self._data_sha:
            def sha1():
                h = hashlib.sha1()
                with open(path, "rb") as f:
                    for b in iter(lambda: f.read(1 << 24), b""):
                        h.update(b)
                return h.hexdigest()
            self._data_sha[key] = await asyncio.to_thread(sha1)
        return path, self._data_sha[key], st.st_size

    def _data_hosts(self, problem) -> set:
        """Hosts that already hold the problem's data (pick_host prefers them)."""
        path = self._data_file.get(str(getattr(problem, "driver_path", "")))
        if path is None:
            return set()
        shas = {s for (p, *_), s in self._data_sha.items() if p == str(path)}
        return {h for (h, s) in self._data_at if s in shas}

    async def _ensure_data(self, problem, host: str) -> tuple[Optional[str], Optional[str]]:
        """-> (remote data.bin or None if the data is linked in, error text or None).
        One transfer per (host, sha): concurrent callers wait for it; callers
        that waited on a transfer that failed get its error (the infra retry
        moves them on) instead of starting another one."""
        from ..servers.utils.resources import loop_local

        info = await self._problem_data(problem)
        if info is None:
            return None, None
        path, sha, size = info
        key = (host, sha)
        if key in self._data_at:
            return self._data_at[key], None
        t0 = time.monotonic()
        lock = self._data_locks[key] = loop_local(self._data_locks.get(key), asyncio.Lock)
        async with lock:
            if key in self._data_at:
                return self._data_at[key], None
            failed = self._data_err.get(key)
            if failed is not None and failed[0] >= t0:
                return None, failed[1]
            remote, err = await self._stage_data(host, path, sha, size)
            if err is not None:
                self._data_err[key] = (time.monotonic(), err)
                self.data_transfer["failures"] += 1
                return None, err
            self._data_at[key] = remote
            return remote, None

    def _forget_data(self, host: str, remote: Optional[str]) -> None:
        """The copy on ``host`` went missing (removed under us): copy it again next time."""
        for k, v in list(self._data_at.items()):
            if k[0] == host and v == remote:
                del self._data_at[k]

    @staticmethod
    def _data_tmp_name() -> str:
        """This process's temporary name for a data.bin in flight (stable, so a retry resumes it)."""
        return f".data.bin.part.{re.sub(r'[^A-Za-z0-9_.-]', '_', socket.gethostname())}.{os.getpid()}"

    @staticmethod
    def _data_missing(text: Optional[str], remote: Optional[str]) -> bool:
        return bool(remote and text) and (f"MB_ERROR cannot open {remote}" in text or
                                          f"MB_ERROR cannot map {remote}" in text)

    async def _stage_data(self, host: str, path: Path, sha: str, size: int) -> tuple[Optional[str], Optional[str]]:
        """Make ``<rdir>/data_<sha12>/data.bin`` on ``host`` hold ``path``
        (reuse a valid copy, else copy to a temporary name, verify, rename)
        -> (remote path, error text or None)."""
        from .. import telemetry as tel

        q = shlex.quote
        rdir = self.remote_dirs.get(host, self.remote_dir)
        d = f"{rdir}/data_{sha[:12]}"
        final = f"{d}/data.bin"
        tmp = self._data_tmp_name()
        sz = '{ stat -c %s "$1" 2>/dev/null || wc -c < "$1"; }'
        s1 = '{ sha1sum "$1" 2>/dev/null || shasum -a 1 "$1" 2>/dev/null || echo nosha; }'
        same = f'{{ [ "$h" = {sha} ] || [ "$h" = nosha ]; }}'     # no sha1 tool there: size only
        # KB_DATA_VALID: data.bin already holds this content (an earlier run's copy)
        check = (f"mkdir -p {q(d)} && cd {q(d)} || exit 1; sz() {sz}; s1() {s1}; "
                 f'if [ -f data.bin ] && [ "$(sz data.bin)" = {size} ]; then '
                 f"h=$(s1 data.bin | cut -c1-40); {same} && echo KB_DATA_VALID; fi; true")
        sp = tel.current() if tel.enabled() else None

        def note(**kw):
            if sp is not None and hasattr(sp, "set"):
                sp.set(host=host, **kw)
        rc, out = await self._sh(self.ssh, *self._ssh_base(), host, check, timeout=600)
        if rc != 0:
            return None, f"data check on {host} failed (rc {rc}): {out[-300:]}"
        if "KB_DATA_VALID" in out:
            self.data_transfer["reused"] += 1
            note(reused=True, bytes=0)
            _log().info(f"{host}: reusing {final} ({size / 2**30:.2f} GiB)")
            return final, None
        t0 = time.monotonic()
        method, err = "scp", None
        if host in self._rsync_hosts:
            method = "rsync"
            rc, out = await self._sh(self.rsync, "--partial", "--append-verify", "--timeout=300",
                                     "-e", shlex.join([self.ssh, *self._ssh_base()]),
                                     str(path), f"{host}:{d}/{tmp}", timeout=self.DATA_TIMEOUT_S)
            if rc != 0:
                _log().warning(f"rsync of data.bin to {host} failed ({out[-200:]!r}); using scp")
                method = "scp"
        if method == "scp":
            rc, out = await self._sh(self.scp, "-q", *self._ssh_base(), str(path), f"{host}:{d}/{tmp}",
                                     timeout=self.DATA_TIMEOUT_S)
            if rc != 0:
                err = f"copying data.bin to {host} failed: {out[-300:]}"
        if err is None:
            verify = (f"cd {q(d)} || exit 1; sz() {sz}; s1() {s1}; n=$(sz {q(tmp)}); h=$(s1 {q(tmp)} | cut -c1-40); "
                      f'if [ "$n" = {size} ] && {same}; then mv -f {q(tmp)} data.bin && echo KB_DATA_OK; '
                      f"else rm -f {q(tmp)}; echo KB_DATA_BAD $n $h; fi")
            rc, out = await self._sh(self.ssh, *self._ssh_base(), host, verify, timeout=600)
            if rc != 0 or "KB_DATA_OK" not in out:
                err = f"data.bin on {host} did not verify (rc {rc}): {out[-300:]}"
        secs = time.monotonic() - t0
        if err is not None:
            note(reused=False, method=method, ok=False)
            return None, err
        self._data_made[(host, sha)] = final
        self.data_transfer["copies"] += 1
        self.data_transfer["bytes"] += size
        self.data_transfer["seconds"] = round(self.data_transfer["seconds"] + secs, 3)
        self.transfer[method] += 1
        note(reused=False, method=method, bytes=size, seconds=round(secs, 3),
             mb_per_s=round(size / 2**20 / secs, 1) if secs > 0 else None)
        tel.count("ssh_data_bytes", size)
        _log().info(f"{host}: copied data.bin ({size / 2**30:.2f} GiB) by {method} in {secs:.1f} s")
        return final, None

    async def _send(self, exe: Path, host: str, rdir: str, name: str, basis: Optional[str]) -> Optional[str]:
        """Copy exe to host:rdir/name; None, or the error text."""
        q = shlex.quote
        if basis and host in self._rsync_hosts:
            # seed the destination with the last binary of this problem: rsync then sends a delta
            rc, out = await self._sh(self.ssh, *self._ssh_base(), host,
                                     f"cd {q(rdir)} && {{ cp -f {q(basis)} {q(name)} 2>/dev/null; true; }}", timeout=120)
            if rc == 0:
                rc, out = await self._sh(self.rsync, "-I", "-e", shlex.join([self.ssh, *self._ssh_base()]),
                                         str(exe), f"{host}:{rdir}/{name}", timeout=self.timeout_s)
                if rc == 0:
                    self.transfer["rsync"] += 1
                    return None
            _log().warning(f"rsync to {host} failed ({out[-200:]!r}); using scp")
        rc, out = await self._sh(self.scp, "-q", *self._ssh_base(), str(exe), f"{host}:{rdir}/{name}",
                                 timeout=self.timeout_s)
        if rc == 0:
            self.transfer["scp"] += 1
            return None
        return out

    def _note_remote(self, host: str, kind: str, **kw) -> None:
        """Append to the remote manifest (before the remote side exists)."""
        if self.remote_manifest is None:
            return
        key = (host, kind, tuple(sorted(kw.items())))
        if key in self._noted:
            return
        self._noted.add(key)
        if kind != "segment":
            self._note_remote(host, "segment", tmp=self._data_tmp_name())
        from ..agents.journal import append_jsonl
        try:
            append_jsonl(self.remote_manifest, {"pid": os.getpid(), "host": host, "kind": kind,
                                                "rdir": self.remote_dirs.get(host, self.remote_dir), **kw})
        except OSError as e:
            _log().warning(f"remote manifest {self.remote_manifest}: {e}")

    async def cleanup_previous(self) -> dict:
        """Remove what earlier (killed) processes of this run left on the hosts: the
        one-shot binaries they staged, their partial data.bin transfers, and their
        persistent hosts if still alive (by recorded pid, only a ``kbhost_`` process
        running in the remote dir). Content-named host binaries / shared objects
        (``kbhost_*``, ``kbso_*``) and verified data.bin copies are reused, and removed
        by ``close()`` as usual. Other runs' files are never touched."""
        from ..agents.journal import atomic_write_text, read_jsonl
        if self.remote_manifest is None or getattr(self, "_cleaned", False):
            return {}
        self._cleaned = True
        rows = read_jsonl(self.remote_manifest)
        old = [r for r in rows if r.get("pid") != os.getpid()]
        if not old:
            return {}
        q = shlex.quote
        by: dict = {}
        for r in old:
            by.setdefault((r["host"], r["rdir"]), []).append(r)
        done = {"hosts": 0, "files": 0, "pids": 0}
        for (h, rdir), rs in by.items():
            files = sorted({r["name"] for r in rs if r["kind"] == "file"})
            tmps = sorted({r["tmp"] for r in rs if r["kind"] == "segment" and r.get("tmp")})
            pids = sorted({int(r["host_pid"]) for r in rs if r["kind"] == "host" and r.get("host_pid")})
            cmd = [f"cd {q(rdir)} || exit 0"]
            if files:
                cmd.append("rm -f " + " ".join(q(f) for f in files))
            for t in tmps:
                cmd.append(f"rm -f data_*/{q(t)}")
            for pid in pids:
                cmd.append(f'if grep -q kbhost_ /proc/{pid}/cmdline 2>/dev/null && '
                           f'[ "$(readlink /proc/{pid}/cwd)" = "$(pwd -P)" ]; then kill {pid}; fi')
            rc, out = await self._sh(self.ssh, *self._ssh_base(), h, "; ".join(cmd) + "; true", timeout=60)
            done["hosts"] += 1
            done["files"] += len(files)
            done["pids"] += len(pids)
            if rc != 0:
                _log().warning(f"{h}: cleanup of the earlier segment's files failed: {out[-200:]!r}")
        keep = [r for r in rows if r.get("pid") == os.getpid()]
        atomic_write_text(self.remote_manifest, "".join(json.dumps(r) + "\n" for r in keep))
        _log().info(f"resume: removed the earlier segment's remote leftovers: {done}")
        return done

    async def _stage(self, exe: Path, host: str, problem=None) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Copy exe to ``host`` -> (remote name, rsync basis or None, error text or None)."""
        rdir = self.remote_dirs.get(host, self.remote_dir)
        name = f"{exe.parent.name}_{exe.name}_{uuid.uuid4().hex[:6]}"
        self._note_remote(host, "file", name=name)
        basis = self._basis(problem) if host in self._rsync_hosts else None
        err = await self._send(exe, host, rdir, name, basis)
        return name, basis, err

    async def _exec_staged(self, host: str, name: str, basis: Optional[str], env=None, prefix="",
                           device=None) -> Optional[str]:
        from ..servers.utils.resources import gpu_lock_wrap

        rdir = self.remote_dirs.get(host, self.remote_dir)
        envs = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in
                        {**(device.env() if device is not None else {}), **(env or {})}.items())
        q = shlex.quote(name)
        keep = f"mv -f ./{q} ./{shlex.quote(basis)} 2>/dev/null || rm -f ./{q}" if basis else f"rm -f ./{q}"
        run = gpu_lock_wrap(f"env {envs} {prefix} ./{q}", device.ident if device is not None else None,
                            wait_s=self.timeout_s)
        remote = (f"cd {shlex.quote(rdir)} && export PATH={shlex.quote(self.remote_path)}:$PATH && "
                  f"{run}; rc=$?; {keep}; exit $rc")
        proc = await asyncio.create_subprocess_exec(self.ssh, *self._ssh_base(), host, remote,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), self.timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            return None
        text = out.decode(errors="replace")
        if proc.returncode == 255 and not re.search(r"^passed$", text, re.M):     # ssh itself failed
            text += f"\nMB_ERROR [infra] ssh to {host} failed (rc 255)\n"
        return text

    async def _run(self, exe, env=None, prefix="", device=None, problem=None):
        host = device.host if device is not None and device.host else self.host
        data, err = await self._ensure_data(problem, host) if problem is not None else (None, None)
        if err is not None:
            return f"MB_ERROR [infra] copy of data.bin to {host} failed: {err[-300:]}\nfailed\n"
        name, basis, err = await self._stage(exe, host, problem)
        if err is not None:
            return f"MB_ERROR [infra] copy to {host} failed: {err[-300:]}\nfailed\n"
        return await self._exec_staged(host, name, basis, {**(env or {}), **({"MB_DATA_PATH": data} if data else {})},
                                       prefix, device)

    async def _timed_run(self, problem, exe: Path, env: Optional[dict] = None, prefix: str = "") -> Optional[str]:
        """Copy the binary (and a run-time data.bin, once per host) to the host
        with the most free GPUs *before* leasing one there, so a GPU is never
        held while a file is in flight; an infrastructure failure moves on to
        another host."""
        from ..servers.utils.resources import is_infra_failure

        pool = await self.pool()
        group = str(problem.root)
        bad_hosts: set = set()
        text = None
        await self._problem_data(problem)                 # the sha, so pick_host can prefer the data's hosts
        for attempt in range(self.infra_retries + 1):
            host = pool.pick_host(group, avoid_hosts=bad_hosts, prefer_hosts=self._data_hosts(problem)) or self.host
            data, err = await self._ensure_data(problem, host)
            what = "data.bin"
            if err is None:
                what = "the binary"
                name, basis, err = await self._stage(exe, host, problem)
            if err is not None:
                text = f"MB_ERROR [infra] copy of {what} to {host} failed: {err[-300:]}\nfailed\n"
                pool.unreserve(host)
                pool.host_failure(host)
            else:
                run_env = {**(env or {}), **({"MB_DATA_PATH": data} if data else {})}
                async with pool.lease(group, host=host) as lease:
                    _leased(lease.device)
                    text = await self._exec_staged(host, name, basis, run_env, prefix, device=lease.device)
                    missing = self._data_missing(text, data)
                    lease.infra_error = missing or (text is not None and is_infra_failure(text) and
                                                    not re.search(r"^passed$", text, re.M))
                if missing:
                    self._forget_data(host, data)
                if not lease.infra_error:
                    return text
            _log().warning(f"{problem.name}: infrastructure failure on {host} (attempt {attempt + 1}): "
                           f"{(text or '')[-300:]!r}")
            bad_hosts.add(host)
        return text

    async def close(self):
        """Stop the persistent hosts; remove the rsync basis binaries, the
        staged host binaries / shared objects, and (unless ``keep_data``) the
        data.bin copies this evaluator made from every host."""
        await self._hosts.close()
        q = shlex.quote
        data: dict = {}
        if not self.keep_data:
            for (h, _), f in self._data_made.items():
                data.setdefault(h, []).append(f)
        tmp = self._data_tmp_name()
        for h in sorted(self._rsync_hosts | set(self._staged) | {h for h, _ in self._data_at} |
                        {h for h, _ in self._data_err}):
            rm = [f"cd {q(self.remote_dirs[h])} && rm -f .kb_basis_* kbhost_* kbso_*.so",
                  f"rm -f data_*/{q(tmp)}"]
            for f in data.get(h, []):          # the directory only goes if nothing else is in it
                d = f.rsplit("/", 1)[0]
                rm.append(f"rm -f {q(f)}; rmdir {q(d)} 2>/dev/null")
            await self._sh(self.ssh, *self._ssh_base(), h, "; ".join(rm) + "; true", timeout=60)
        self._staged.clear()
        if not self.keep_data:
            for k in self._data_made:
                self._data_at.pop(k, None)
            self._data_made.clear()

    # ---- the persistent host on a remote GPU: `ssh host ./kbhost_<sha>` over the
    # multiplexed connection; the host binary is copied once per problem and
    # host, each candidate's shared object (~1 MB) per request
    STAGED_SO_LIMIT = 48

    async def _stage_file(self, host: str, path: Path, name: str, keep: set) -> Optional[str]:
        """Copy ``path`` to host:rdir/name once (content-named); evict old shared objects."""
        staged = self._staged.setdefault(host, OrderedDict())
        if name in staged:
            staged.move_to_end(name)
            return None
        rdir = self.remote_dirs.get(host, self.remote_dir)
        err = await self._send(path, host, rdir, name, None)
        if err is not None:
            return err
        staged[name] = True
        old = [n for n in staged if n.startswith("kbso_") and n not in keep]
        drop = old[:max(0, len([n for n in staged if n.startswith("kbso_")]) - self.STAGED_SO_LIMIT)]
        if drop:
            for n in drop:
                staged.pop(n, None)
            await self._sh(self.ssh, *self._ssh_base(), host,
                           f"cd {shlex.quote(rdir)} && rm -f " + " ".join(shlex.quote(n) for n in drop), timeout=60)
        return None

    async def _host_request(self, problem, exe: Path, so: Path, ref: Optional[Path], env: dict):
        from ..servers.utils.resources import gpu_lock_file, is_infra_failure
        from .runtime_host import HostClient, HostReply

        pool = await self.pool()
        group = str(problem.root)
        bad_hosts: set = set()
        hname = f"kbhost_{exe.stem[:16]}"
        names = {so: f"kbso_{so.stem[:16]}.so", **({ref: f"kbso_{ref.stem[:16]}.so"} if ref else {})}
        rep = None
        await self._problem_data(problem)
        for attempt in range(self.infra_retries + 1):
            host = pool.pick_host(group, avoid_hosts=bad_hosts, prefer_hosts=self._data_hosts(problem)) or self.host
            rdir = self.remote_dirs.get(host, self.remote_dir)
            data, err = await self._ensure_data(problem, host)
            what = "data.bin" if err is not None else "the host binary / shared objects"
            if err is None:
                err = await self._stage_file(host, exe, hname, set())
            for path, name in names.items():
                if err is None:
                    err = await self._stage_file(host, path, name, set(names.values()))
            if err is not None:
                rep = HostReply(f"MB_ERROR [infra] copy of {what} to {host} failed: {err[-300:]}\nfailed\n",
                                "done", rc=1)
                pool.unreserve(host)
                pool.host_failure(host)
                bad_hosts.add(host)
                continue
            async with pool.lease(group, host=host) as lease:
                dev = lease.device
                _leased(dev)
                key = (group, dev.key, hname)
                # the host maps data.bin once at start-up: MB_DATA_PATH is the process's
                envs = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in
                                {**dev.env(), **({"MB_DATA_PATH": data} if data else {})}.items())
                # KB_HOST_PID: the remote pid, recorded so a resumed run can stop an orphan
                remote = (f"cd {shlex.quote(rdir)} && export PATH={shlex.quote(self.remote_path)}:$PATH && "
                          f"echo KB_HOST_PID $$ && exec env {envs} ./{hname}")
                self._note_remote(host, "segment", tmp=self._data_tmp_name())
                h = await self._hosts.get(key, dev.key, lambda: HostClient(
                    [self.ssh, *self._ssh_base(), host, remote], label=f"{problem.name}@{dev.label()}",
                    start_timeout=self.timeout_s, device=dev.label()))

                # dlopen searches the library path for a bare name: ./ (the host runs in rdir)
                rep = await h.request(f"./{names[so]}", ref=f"./{names[ref]}" if ref else None, env=env,
                                      lock=gpu_lock_file(dev.ident), lock_wait=self.timeout_s, timeout=self.timeout_s)
                self._hosts.stats.note(rep, dev.label())
                if getattr(h, "remote_pid", None):
                    self._note_remote(host, "host", host_pid=h.remote_pid)
                # ssh itself failing (rc 255 before the host printed anything) is the link, not the candidate
                link_down = rep.status in ("crash", "start_failed") and rep.rc == 255 and \
                    not re.search(r"^MB_", rep.text, re.M)
                missing = self._data_missing(rep.text, data)       # the copy was removed under us
                lease.infra_error = link_down or missing or \
                    (is_infra_failure(rep.text) and not re.search(r"^passed$", rep.text, re.M))
            if missing:
                self._forget_data(host, data)
            if not lease.infra_error:
                return rep
            await self._hosts.drop(key)
            bad_hosts.add(host)
            _log().warning(f"{problem.name}: infrastructure failure on {host} (attempt {attempt + 1}): "
                           f"{rep.text[-300:]!r}")
        return rep

    def host_stats(self) -> dict:
        return {**super().host_stats(), "transfer": dict(self.transfer), "data": dict(self.data_transfer)}


class ServerEvaluator(Evaluator):
    """KernelBlaster's compile + GPU exec servers (``compile_cu`` / ``run_gpu_executable``)."""

    def __init__(self, gpu, work_root: Path, timeout_s: int = 900, incremental: Optional[bool] = None):
        self.gpu, self.work_root, self.timeout_s = gpu, Path(work_root), timeout_s
        # one TU per slot via the compile server's /compile_units (objects
        # cached there by content); off when the server lacks the endpoint
        self.incremental = _incremental_default() if incremental is None else incremental

    async def _compile_units(self, problem, sources, work: Path, cu: Path) -> Optional[Path]:
        from ..agents.utils.commands import UnitsUnsupported, compile_cu_units
        from ..agents.utils import FeedbackError

        paths = []
        for name, text in problem.units(sources):
            # content-addressed name: the server keys objects by text anyway,
            # this keeps lineinfo paths stable across candidates
            d = self.work_root / "units" / hashlib.sha1(text.encode()).hexdigest()[:16]
            d.mkdir(parents=True, exist_ok=True)
            if not (d / name).exists():
                tmp = d / f".{name}.{uuid.uuid4().hex[:6]}"
                tmp.write_text(text)
                os.replace(tmp, d / name)
            paths.append(d / name)
        try:
            return Path(await compile_cu_units(problem.driver_path, paths, self.gpu, timeout=self.timeout_s,
                                               job_name=f"{cu} (units)"))
        except UnitsUnsupported:
            self.incremental = False
        except FeedbackError:
            pass                    # the assembled build reports the error
        return None

    async def _compile(self, problem, sources, tag):
        from ..agents.utils.commands import compile_cu
        from ..agents.utils import FeedbackError

        work = self.work_root / f"{re.sub(r'[^A-Za-z0-9_]+', '_', tag)}_{uuid.uuid4().hex[:6]}"
        work.mkdir(parents=True, exist_ok=True)
        cu = work / "kernels.cu"
        cu.write_text(problem.assemble(sources))
        if self.incremental:
            exe = await self._compile_units(problem, sources, work, cu)
            if exe is not None:
                return cu, exe, None
        try:
            return cu, Path(await compile_cu(problem.driver_path, cu, self.gpu, timeout=self.timeout_s,
                                             job_name=str(cu))), None
        except FeedbackError as e:
            return cu, None, EvalResult(ok=False, stage="compile", errors=[_compile_errors(str(e))],
                                        log=str(e)[-4000:])

    async def evaluate(self, problem, sources, tag, *, only_slots=None, screen=False, incumbent=None):
        from ..agents.utils.commands import run_gpu_executable
        from ..agents.utils import FeedbackError

        cu, exe, bad = await self._compile(problem, sources, tag)
        if bad is not None:
            return bad
        env = driver_env(only_slots, screen)
        prefix = "".join(f"{k}={shlex.quote(v)} " for k, v in env.items()) or None
        try:
            stdout, stderr = await run_gpu_executable(exe, self.gpu, self.timeout_s, job_name=str(cu),
                                                      prefix_command=prefix)
        except FeedbackError as e:
            return EvalResult(ok=False, stage="run", errors=[str(e)[-2000:]], log=str(e)[-4000:])
        return parse_driver_output(stdout + "\n" + stderr)

    async def profile(self, problem, sources, tag, slots=None):
        from ..agents.utils.commands import run_gpu_executable
        from ..agents.utils import FeedbackError
        from .ncu import ncu_command

        cu, exe, bad = await self._compile(problem, sources, tag + "_ncu")
        if bad is not None:
            return {}, "build failed: " + bad.error_text(300)
        try:
            stdout, stderr = await run_gpu_executable(exe, self.gpu, self.timeout_s, job_name=f"{cu} (ncu)",
                                                      prefix_command=f"MB_MODE=profile {ncu_command(slots=slots)} ")
        except FeedbackError as e:
            return {}, str(e)[-300:]
        return _ncu_result(stdout + "\n" + stderr)


def _log():
    try:
        from loguru import logger
        return logger
    except ImportError:                       # pragma: no cover
        import logging
        return logging.getLogger(__name__)


def cupti_flags(nvcc: str = "nvcc") -> list[str]:
    """``-I`` for CUPTI's headers beside ``nvcc`` (the driver's per-op
    attribution, ``MB_ATTR_*``); [] without them (the driver then prints
    MB_ATTR_SKIP). ``KERNELBLASTER_CUPTI=0`` leaves them out."""
    if not _env_on("KERNELBLASTER_CUPTI", "1"):
        return []
    exe = shutil.which(nvcc) or nvcc
    inc = Path(exe).resolve().parent.parent / "extras" / "CUPTI" / "include"
    return [f"-I{inc}"] if (inc / "cupti.h").exists() else []


def _incremental_default() -> bool:
    return os.getenv("KERNELBLASTER_SLOT_UNITS", "1").lower() not in ("0", "false", "no", "off")


def _compile_errors(log: str, limit: int = 3000) -> str:
    errs = [l for l in log.splitlines() if "error" in l.lower()]
    return ("\n".join(errs) or log)[-limit:]


def find_problems(root: Path) -> list[Path]:
    root = Path(root)
    if (root / "manifest.json").exists():
        return [root]
    return sorted(p.parent for p in root.rglob("manifest.json"))


# ---------------------------------------------------------------- OpenCL (Qualcomm Adreno) evaluators
#
# An OpenCL problem (manifest target "opencl", ModelBlaster kb_export_opencl) is
# C++11 on ModelBlaster's OpenCL runtime: the kernels TU (head + slots + tail)
# and driver.cpp build with ``g++ -std=c++11 ... -lOpenCL`` on the machine that
# has the GPU -- the Adreno board itself (KernelBlaster's Adreno flow compiles
# on the RB5) or a host with any OpenCL 2.0 GPU with coarse-grained SVM. The
# driver reads data.bin at start-up (argv[1]). Concatenated, kernels TU and
# driver are one self-contained main.cpp (``opencl_main``).

OPENCL_CXXFLAGS = "-O2 -std=c++11 -w"
OPENCL_LDFLAGS = "-lOpenCL -lm"
# slots launch through mb_launch / mb_copy / mb_zero: the driver times every
# command they enqueue; a direct enqueue would run untimed (and unchecked order)
_CL_FORBIDDEN = re.compile(r"\b(clEnqueue\w+|clFinish|clFlush|clCreateCommandQueue\w*|clWaitForEvents)\s*\(")


def opencl_main(problem: SlotProblem, sources: Mapping[str, str]) -> str:
    """kernels TU + driver as one translation unit."""
    return problem.assemble(sources) + "\n#define MB_SINGLE_TU 1\n" + problem.driver_path.read_text()


def opencl_source_errors(problem: SlotProblem, sources: Mapping[str, str]) -> Optional[EvalResult]:
    bad = [f"slot {sid} calls {m.group(1)}() directly; enqueue only through mb_launch / mb_copy / mb_zero"
           for sid in problem.order for m in [_CL_FORBIDDEN.search(sources[sid])] if m]
    return EvalResult(ok=False, stage="compile", errors=bad) if bad else None


def _instances(problem: SlotProblem) -> dict:
    return {sid: len(s.instances) for sid, s in problem.slots.items()}


def _opencl_profile_result(problem: SlotProblem, text: Optional[str]):
    from .opencl_profile import profile_by_slot

    if text is None:
        return {}, "OpenCL profile run timed out"
    if "MB_PROFILE_DONE" not in text:
        return {}, "OpenCL profile run failed: " + "; ".join(re.findall(r"^MB_ERROR (.*)$", text, re.M))[:300]
    by = profile_by_slot(text, _instances(problem))
    return (by, None) if by else ({}, "profile run reported no kernels")


class OpenCLLocalEvaluator(LocalEvaluator):
    """g++ + this host's OpenCL GPU (functional stand-in for the board: any
    OpenCL 2.0 device with coarse-grained SVM; its times are not Adreno's)."""

    def __init__(self, work_root: Optional[Path] = None, cxx: str = "g++", max_parallel: Optional[int] = None,
                 timeout_s: int = 900, keep: bool = False, env: Optional[dict] = None):
        super().__init__("opencl", work_root=work_root, max_parallel=max_parallel, timeout_s=timeout_s, keep=keep)
        self.cxx, self.env = cxx, dict(env or {})

    async def _devices(self) -> list:
        # the OpenCL platform picks the device; one run at a time on it
        from ..servers.utils.resources import Device

        return [Device(None, None, "opencl")]

    async def _build(self, problem, sources, work):
        bad = opencl_source_errors(problem, sources)
        if bad is not None:
            return bad
        (work / "kernels.cpp").write_text(problem.assemble(sources))
        async with self._sem:
            proc = await asyncio.create_subprocess_exec(
                self.cxx, *OPENCL_CXXFLAGS.split(), "-o", str(work / "problem"), str(work / "kernels.cpp"),
                str(problem.driver_path), *OPENCL_LDFLAGS.split(),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            out, _ = await asyncio.wait_for(proc.communicate(), self.timeout_s)
        if proc.returncode != 0:
            return EvalResult(ok=False, stage="compile", errors=[_compile_errors(out.decode())],
                              log=out.decode()[-4000:])
        return None

    async def _exec(self, problem, sources, tag, env):
        work = self._workdir(problem, tag)
        try:
            bad = await self._build(problem, sources, work)
            if bad is not None:
                return bad, None
            data = problem.root / problem.manifest["data"]
            text = await self._timed_run(problem, work / "problem", {**self.env, **env, "MB_DATA_PATH": str(data)})
            return None, text
        finally:
            if not self.keep:
                shutil.rmtree(work, ignore_errors=True)

    async def evaluate(self, problem, sources, tag, *, only_slots=None, screen=False, incumbent=None):  # full run (OpenCL driver)
        bad, text = await self._exec(problem, sources, tag, {})
        if bad is not None:
            return bad
        if text is None:
            return EvalResult(ok=False, stage="run", errors=[f"timeout after {self.timeout_s}s"])
        return parse_driver_output(text)

    async def profile(self, problem, sources, tag, slots=None):    # one run profiles every slot
        bad, text = await self._exec(problem, sources, tag + "_prof", {"MB_MODE": "profile"})
        if bad is not None:
            return {}, "build failed: " + bad.error_text(300)
        return _opencl_profile_result(problem, text)


class AdrenoEvaluator(Evaluator):
    """Build and run on the Adreno board over ssh, as KernelBlaster's Adreno flow
    does (servers/compile_opencl.py: scp sources, compile on the board, run,
    clean up): one main.cpp per candidate is copied into a per-candidate
    directory, compiled with the board's g++ against its libOpenCL, run, and
    the directory is removed. data.bin is copied once per problem into
    ``remote_dir``; ``close()`` removes ``remote_dir``."""

    def __init__(self, host: Optional[str] = None, *, remote_dir: str = "/tmp/kernelblaster_multislot",
                 ssh_key: Optional[str] = None, cxx: str = "g++", timeout_s: int = 1800,
                 work_root: Optional[Path] = None, keep: bool = False):
        from ..backends.opencl import default_board_host

        self.host = host or default_board_host()
        self.remote_dir, self.key, self.cxx = remote_dir.rstrip("/"), ssh_key, cxx
        self.timeout_s, self.keep = timeout_s, keep
        self.work_root = Path(work_root) if work_root else None
        self._board = asyncio.Lock()            # one compile+run on the board at a time
        self._data: dict[str, str] = {}         # data_sha1 -> remote data.bin

    def _opts(self) -> list[str]:
        return ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"] + (["-i", self.key] if self.key else [])

    async def _cmd(self, *argv, timeout=None, stdin: Optional[bytes] = None) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.PIPE if stdin else None,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(stdin), timeout or self.timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            return -1, "timeout"
        return proc.returncode, out.decode(errors="replace")

    async def _ssh(self, script: str, timeout=None) -> tuple[int, str]:
        return await self._cmd("ssh", *self._opts(), self.host, script, timeout=timeout)

    async def _remote_data(self, problem: SlotProblem) -> str:
        sha = problem.manifest.get("data_sha1") or uuid.uuid4().hex
        if sha not in self._data:
            d = f"{self.remote_dir}/data_{sha[:12]}"
            rc, out = await self._ssh(f"mkdir -p {shlex.quote(d)}", timeout=60)
            if rc == 0:
                rc, out = await self._cmd("scp", "-q", *self._opts(), str(problem.root / problem.manifest["data"]),
                                          f"{self.host}:{d}/data.bin", timeout=1800)
            if rc != 0:
                raise RuntimeError(f"copying data.bin to {self.host} failed: {out[-300:]}")
            self._data[sha] = f"{d}/data.bin"
        return self._data[sha]

    async def _exec(self, problem, sources, tag, env: dict) -> tuple[Optional[EvalResult], Optional[str]]:
        bad = opencl_source_errors(problem, sources)
        if bad is not None:
            return bad, None
        try:
            data = await self._remote_data(problem)
        except RuntimeError as e:
            return EvalResult(ok=False, stage="run", errors=[str(e)]), None
        name = f"{re.sub(r'[^A-Za-z0-9_]+', '_', tag)}_{uuid.uuid4().hex[:6]}"
        d = f"{self.remote_dir}/{name}"
        local = (self.work_root or problem.root / "kb_work") / name
        local.mkdir(parents=True, exist_ok=True)
        (local / "main.cpp").write_text(opencl_main(problem, sources))
        envs = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
        script = (f"cd {shlex.quote(d)} || exit 1; "
                  f"if {self.cxx} {OPENCL_CXXFLAGS} -o main main.cpp {OPENCL_LDFLAGS} > build.log 2>&1; then "
                  f"env {envs} ./main {shlex.quote(data)}; else echo MB_BUILD_FAILED; tail -c 6000 build.log; fi; "
                  f"cd / && rm -rf {shlex.quote(d)}")
        try:
            async with self._board:
                rc, out = await self._ssh(f"mkdir -p {shlex.quote(d)}", timeout=60)
                if rc == 0:
                    rc, out = await self._cmd("scp", "-q", *self._opts(), str(local / "main.cpp"),
                                              f"{self.host}:{d}/main.cpp", timeout=300)
                if rc != 0:
                    return EvalResult(ok=False, stage="run", errors=[f"ssh/scp to {self.host} failed: {out[-300:]}"]), None
                rc, out = await self._ssh(script)
        finally:
            if not self.keep:
                shutil.rmtree(local, ignore_errors=True)
        if rc == -1:
            await self._ssh(f"rm -rf {shlex.quote(d)}", timeout=60)
            return None, None
        if "MB_BUILD_FAILED" in out:
            log = out.split("MB_BUILD_FAILED", 1)[1]
            return EvalResult(ok=False, stage="compile", errors=[_compile_errors(log)], log=log[-4000:]), None
        return None, out

    async def evaluate(self, problem, sources, tag, *, only_slots=None, screen=False, incumbent=None):  # full run (OpenCL driver)
        bad, text = await self._exec(problem, sources, tag, {})
        if bad is not None:
            return bad
        if text is None:
            return EvalResult(ok=False, stage="run", errors=[f"timeout after {self.timeout_s}s"])
        return parse_driver_output(text)

    async def profile(self, problem, sources, tag, slots=None):    # one run profiles every slot
        bad, text = await self._exec(problem, sources, tag + "_prof", {"MB_MODE": "profile"})
        if bad is not None:
            return {}, "build failed: " + bad.error_text(300)
        return _opencl_profile_result(problem, text)

    async def close(self):
        await self._ssh(f"rm -rf {shlex.quote(self.remote_dir)}", timeout=120)
        self._data.clear()


class OpenCLServerEvaluator(Evaluator):
    """KernelBlaster's OpenCL compile server + Adreno exec server
    (``compile_opencl`` / ``run_adreno_executable``): main.cpp is compiled on the
    board (or locally without one), data.bin travels as a kernel file and the
    driver finds it in the run directory."""

    def __init__(self, gpu, work_root: Path, timeout_s: int = 1800, remote: bool = True):
        self.gpu, self.work_root, self.timeout_s, self.remote = gpu, Path(work_root), timeout_s, remote

    async def _compile(self, problem, sources, tag):
        from ..agents.utils import FeedbackError
        from ..agents.utils.commands import compile_opencl

        bad = opencl_source_errors(problem, sources)
        if bad is not None:
            return None, None, bad
        work = self.work_root / f"{re.sub(r'[^A-Za-z0-9_]+', '_', tag)}_{uuid.uuid4().hex[:6]}"
        work.mkdir(parents=True, exist_ok=True)
        main = work / "main.cpp"
        main.write_text(opencl_main(problem, sources))
        kernel = work / "kernel.cl"           # the strategy's slot for a .cl file; programs live in main.cpp
        kernel.write_text("// OpenCL programs are embedded in main.cpp\n")
        try:
            exe = await compile_opencl(main, kernel, self.gpu, timeout=self.timeout_s, job_name=str(main),
                                       remote=self.remote)
            return main, Path(exe), None
        except FeedbackError as e:
            return main, None, EvalResult(ok=False, stage="compile", errors=[_compile_errors(str(e))],
                                          log=str(e)[-4000:])

    async def _exec(self, problem, sources, tag, profile_mode: bool):
        from ..agents.utils import FeedbackError
        from ..agents.utils.commands import run_adreno_executable

        main, exe, bad = await self._compile(problem, sources, tag)
        if bad is not None:
            return bad, None
        try:
            stdout, stderr = await run_adreno_executable(
                exe, self.gpu, self.timeout_s, job_name=str(main),
                kernel_files=[str(problem.root / problem.manifest["data"])],
                extra_args="data.bin" + (" profile" if profile_mode else ""))
        except FeedbackError as e:
            return EvalResult(ok=False, stage="run", errors=[str(e)[-2000:]], log=str(e)[-4000:]), None
        return None, stdout + "\n" + stderr

    async def evaluate(self, problem, sources, tag, *, only_slots=None, screen=False, incumbent=None):  # full run (OpenCL driver)
        bad, text = await self._exec(problem, sources, tag, False)
        return bad if bad is not None else parse_driver_output(text)

    async def profile(self, problem, sources, tag, slots=None):    # one run profiles every slot
        bad, text = await self._exec(problem, sources, tag + "_prof", True)
        if bad is not None:
            return {}, "build failed: " + bad.error_text(300)
        return _opencl_profile_result(problem, text)
