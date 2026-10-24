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
import hashlib
import json
import os
import re
import shlex
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
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
    if r.errors and not passed:
        r.stage = "run"
    return r


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

class Evaluator:
    """Compile an assembled candidate with the problem's driver and run it.

    ``only_slots`` / ``screen``: see :func:`driver_env` (ignored where the
    driver has no such modes)."""

    async def evaluate(self, problem: SlotProblem, sources: Mapping[str, str], tag: str, *,
                       only_slots=None, screen: bool = False) -> EvalResult:
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
    one GPU model."""

    def __init__(self, arch: str = "75", work_root: Optional[Path] = None, nvcc: str = "nvcc",
                 max_parallel: Optional[int] = None, timeout_s: int = 900, keep: bool = False,
                 incremental: Optional[bool] = None, cache_dir: Optional[Path] = None,
                 build_jobs: Optional[int] = None, gpus=None, pool=None, infra_retries: int = 2,
                 graph: Optional[bool] = None):
        from ..servers.utils.resources import build_limiter

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
        from ..servers.utils.resources import GPUPool, loop_local

        if self._pool is None:
            self._pool_lock = loop_local(self._pool_lock, asyncio.Lock)
            async with self._pool_lock:
                if self._pool is None:
                    self._pool = GPUPool(await self._devices(), name=type(self).__name__)
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

        async def obj(src: Path, text: str):
            key = cache_key("nvcc-c", self.nvcc, *flags, text, *external_inputs(text))

            async def build(tmp: Path):
                async with self._unit_sem:
                    rc, _ = await self._nvcc([*flags, "-c", str(src), "-o", str(tmp)])
                if rc != 0:
                    raise _Failed(src.name)
            return key, (await cache.get_or_build(key, ".o", build))[0]

        jobs = []
        for name, text in problem.units(sources):
            (work / name).write_text(text)
            jobs.append(obj(work / name, text))
        drv = problem.driver_path
        jobs.append(obj(drv, drv.read_text()))
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
                self.nvcc, *self._flags(), "-o", str(work / "problem"),
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

    async def evaluate(self, problem, sources, tag, *, only_slots=None, screen=False):
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


class SSHEvaluator(LocalEvaluator):
    """Build here, run on remote GPU hosts over ssh (e.g. idle L40S GPUs with
    profiler-counter access). The binary is self-contained (data.bin is
    linked in), so only the executable is copied. Problems whose data is too
    big to link (read at start-up from a local path) are not supported.

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
                 scp: str = "scp", rsync: Optional[str] = "rsync", **kw):
        super().__init__(arch, **kw)
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

    async def _stage(self, exe: Path, host: str, problem=None) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Copy exe to ``host`` -> (remote name, rsync basis or None, error text or None)."""
        rdir = self.remote_dirs.get(host, self.remote_dir)
        name = f"{exe.parent.name}_{exe.name}_{uuid.uuid4().hex[:6]}"
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
        name, basis, err = await self._stage(exe, host, problem)
        if err is not None:
            return f"MB_ERROR [infra] copy to {host} failed: {err[-300:]}\nfailed\n"
        return await self._exec_staged(host, name, basis, env, prefix, device)

    async def _timed_run(self, problem, exe: Path, env: Optional[dict] = None, prefix: str = "") -> Optional[str]:
        """Copy the binary to the host with the most free GPUs *before* leasing
        one there, so a GPU is never held while a binary is in flight; an
        infrastructure failure moves on to another host."""
        from ..servers.utils.resources import is_infra_failure

        pool = await self.pool()
        group = str(problem.root)
        bad_hosts: set = set()
        text = None
        for attempt in range(self.infra_retries + 1):
            host = pool.pick_host(group, avoid_hosts=bad_hosts) or self.host
            name, basis, err = await self._stage(exe, host, problem)
            if err is not None:
                text = f"MB_ERROR [infra] copy to {host} failed: {err[-300:]}\nfailed\n"
                pool.unreserve(host)
                pool.host_failure(host)
            else:
                async with pool.lease(group, host=host) as lease:
                    text = await self._exec_staged(host, name, basis, env, prefix, device=lease.device)
                    lease.infra_error = text is not None and is_infra_failure(text) and \
                        not re.search(r"^passed$", text, re.M)
                if not lease.infra_error:
                    return text
            _log().warning(f"{problem.name}: infrastructure failure on {host} (attempt {attempt + 1}): "
                           f"{(text or '')[-300:]!r}")
            bad_hosts.add(host)
        return text

    async def close(self):
        """Remove the rsync basis binaries from every host."""
        for h in self._rsync_hosts:
            await self._sh(self.ssh, *self._ssh_base(), h,
                           f"rm -f {shlex.quote(self.remote_dirs[h])}/.kb_basis_*", timeout=60)

    async def _build(self, problem, sources, work):
        if "MB_DATA_PATH" in problem.driver_path.read_text():
            return EvalResult(ok=False, stage="compile", errors=[
                "data.bin is read at run time (too big to link); SSHEvaluator needs a self-contained binary"])
        return await super()._build(problem, sources, work)


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

    async def evaluate(self, problem, sources, tag, *, only_slots=None, screen=False):
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

    async def evaluate(self, problem, sources, tag, *, only_slots=None, screen=False):  # full run (OpenCL driver)
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

    async def evaluate(self, problem, sources, tag, *, only_slots=None, screen=False):  # full run (OpenCL driver)
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

    async def evaluate(self, problem, sources, tag, *, only_slots=None, screen=False):  # full run (OpenCL driver)
        bad, text = await self._exec(problem, sources, tag, False)
        return bad if bad is not None else parse_driver_output(text)

    async def profile(self, problem, sources, tag, slots=None):    # one run profiles every slot
        bad, text = await self._exec(problem, sources, tag + "_prof", True)
        if bad is not None:
            return {}, "build failed: " + bad.error_text(300)
        return _opencl_profile_result(problem, text)
