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
    slot_us: dict = field(default_factory=dict)
    op_us: dict = field(default_factory=dict)          # op name -> us (driver MB_OP_US)
    slot_checks: dict = field(default_factory=dict)    # sid -> {"rel_err", "ok"}
    outputs: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    log: str = ""

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
    if r.errors and not passed:
        r.stage = "run"
    return r


# ---------------------------------------------------------------- evaluators

class Evaluator:
    """Compile an assembled candidate with the problem's driver and run it."""

    async def evaluate(self, problem: SlotProblem, sources: Mapping[str, str], tag: str) -> EvalResult:
        raise NotImplementedError

    async def profile(self, problem: SlotProblem, sources: Mapping[str, str], tag: str, slots=None):
        """NCU per slot (every slot, or only ``slots``): -> ({slot: summary},
        None) or ({}, why not)."""
        return {}, f"{type(self).__name__} does not profile"


def _ncu_result(text: str):
    from .ncu import ncu_error, profile_by_slot

    err = ncu_error(text)
    if err:
        return {}, err
    by = profile_by_slot(text)
    return (by, None) if by else ({}, "ncu output had no mbslot_ kernels")


class LocalEvaluator(Evaluator):
    """nvcc + run on this host (dev/tests; no KernelBlaster servers needed)."""

    def __init__(self, arch: str = "75", work_root: Optional[Path] = None, nvcc: str = "nvcc",
                 max_parallel: int = 2, timeout_s: int = 900, keep: bool = False,
                 incremental: Optional[bool] = None, cache_dir: Optional[Path] = None,
                 build_jobs: int = 4):
        self.arch, self.nvcc, self.timeout_s, self.keep = arch, nvcc, timeout_s, keep
        self.work_root = Path(work_root) if work_root else None
        self._sem = asyncio.Semaphore(max_parallel)
        self._gpu = asyncio.Lock()             # one timed run at a time
        # one TU per slot, objects + binaries cached by content (see _build_units)
        self.incremental = _incremental_default() if incremental is None else incremental
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._cache = None
        self._unit_sem = asyncio.Semaphore(build_jobs)   # nvcc processes for unit objects

    def _workdir(self, problem, tag) -> Path:
        root = self.work_root or problem.root / "kb_work"
        work = root / f"{re.sub(r'[^A-Za-z0-9_]+', '_', tag)}_{uuid.uuid4().hex[:6]}"
        work.mkdir(parents=True, exist_ok=True)
        return work

    def _flags(self) -> list[str]:
        return ["-O3", f"-arch=sm_{self.arch}", "-std=c++17", "-lineinfo"]

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
                self.nvcc, "-O3", f"-arch=sm_{self.arch}", "-std=c++17", "-lineinfo", "-o", str(work / "problem"),
                str(work / "kernels.cu"), str(problem.driver_path),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            out, _ = await asyncio.wait_for(proc.communicate(), self.timeout_s)
        if proc.returncode != 0:
            return EvalResult(ok=False, stage="compile", errors=[_compile_errors(out.decode())],
                              log=out.decode()[-4000:])
        return None

    async def _run(self, exe: Path, env: Optional[dict] = None, prefix: str = "") -> Optional[str]:
        """Run exe (optionally under a prefix command, e.g. ncu); stdout+stderr, None on timeout."""
        cmd = f"{prefix} {shlex.quote(str(exe))}" if prefix else shlex.quote(str(exe))
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, **(env or {})})
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), self.timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            return None
        return out.decode(errors="replace")

    async def evaluate(self, problem, sources, tag):
        work = self._workdir(problem, tag)
        try:
            bad = await self._build(problem, sources, work)
            if bad is not None:
                return bad
            async with self._gpu:
                text = await self._run(work / "problem")
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
            async with self._gpu:
                text = await self._run(work / "problem", {"MB_MODE": "profile"}, ncu_command(slots=slots))
            return _ncu_result(text) if text is not None else ({}, "ncu timed out")
        finally:
            if not self.keep:
                shutil.rmtree(work, ignore_errors=True)


class SSHEvaluator(LocalEvaluator):
    """Build here, run on a remote GPU host over ssh (e.g. an idle L40S with
    profiler-counter access). The binary is self-contained (data.bin is
    linked in), so only the executable is copied. Problems whose data is too
    big to link (read at start-up from a local path) are not supported."""

    def __init__(self, host: str, *, arch: str, remote_dir: str, ssh_key: Optional[str] = None,
                 cuda_visible_devices: Optional[str] = None, remote_path: str = "/usr/local/cuda/bin", **kw):
        super().__init__(arch, **kw)
        self.host, self.remote_dir, self.key = host, remote_dir.rstrip("/"), ssh_key
        self.cvd, self.remote_path = cuda_visible_devices, remote_path

    def _ssh_base(self) -> list[str]:
        return ["-o", "BatchMode=yes"] + (["-i", self.key] if self.key else [])

    async def _run(self, exe, env=None, prefix=""):
        name = f"{exe.parent.name}_{exe.name}"
        dst = f"{self.remote_dir}/{name}"
        cp = await asyncio.create_subprocess_exec("scp", "-q", *self._ssh_base(), str(exe), f"{self.host}:{dst}",
                                                  stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out, _ = await cp.communicate()
        if cp.returncode != 0:
            return f"MB_ERROR scp to {self.host} failed: {out.decode()[-300:]}\nfailed\n"
        envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in {**({"CUDA_VISIBLE_DEVICES": self.cvd} if self.cvd else {}),
                                                            **(env or {})}.items())
        remote = (f"cd {shlex.quote(self.remote_dir)} && export PATH={shlex.quote(self.remote_path)}:$PATH && "
                  f"env {envs} {prefix} ./{shlex.quote(name)}; rc=$?; rm -f ./{shlex.quote(name)}; exit $rc")
        proc = await asyncio.create_subprocess_exec("ssh", *self._ssh_base(), self.host, remote,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), self.timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            return None
        return out.decode(errors="replace")

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

    async def evaluate(self, problem, sources, tag):
        from ..agents.utils.commands import run_gpu_executable
        from ..agents.utils import FeedbackError

        cu, exe, bad = await self._compile(problem, sources, tag)
        if bad is not None:
            return bad
        try:
            stdout, stderr = await run_gpu_executable(exe, self.gpu, self.timeout_s, job_name=str(cu))
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
