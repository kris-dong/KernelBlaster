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
    for m in re.finditer(r"^MB_KERNEL_US (\S+) (\S+) (\S+)", text, re.M):
        r.kernel_us[f"{m.group(1)}/{m.group(2)}"] = float(m.group(3))
    m = re.search(r"^MB_DEVICE (.*)$", text, re.M)
    if m:
        r.device = m.group(1).strip()
    if r.errors and not passed:
        r.stage = "run"
    return r


# ---------------------------------------------------------------- evaluators

class Evaluator:
    """Compile an assembled candidate with the problem's driver and run it."""

    async def evaluate(self, problem: SlotProblem, sources: Mapping[str, str], tag: str) -> EvalResult:
        raise NotImplementedError

    async def profile(self, problem: SlotProblem, sources: Mapping[str, str], tag: str):
        """NCU per slot: -> ({slot: summary}, None) or ({}, why not)."""
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
                 max_parallel: int = 2, timeout_s: int = 900, keep: bool = False):
        self.arch, self.nvcc, self.timeout_s, self.keep = arch, nvcc, timeout_s, keep
        self.work_root = Path(work_root) if work_root else None
        self._sem = asyncio.Semaphore(max_parallel)
        self._gpu = asyncio.Lock()             # one timed run at a time

    def _workdir(self, problem, tag) -> Path:
        root = self.work_root or problem.root / "kb_work"
        work = root / f"{re.sub(r'[^A-Za-z0-9_]+', '_', tag)}_{uuid.uuid4().hex[:6]}"
        work.mkdir(parents=True, exist_ok=True)
        return work

    async def _build(self, problem, sources, work: Path) -> Optional[EvalResult]:
        """nvcc the assembled TU + driver into work/problem; an EvalResult on failure."""
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

    async def profile(self, problem, sources, tag):
        from .ncu import ncu_command

        work = self._workdir(problem, tag + "_ncu")
        try:
            bad = await self._build(problem, sources, work)
            if bad is not None:
                return {}, "build failed: " + bad.error_text(300)
            async with self._gpu:
                text = await self._run(work / "problem", {"MB_MODE": "profile"}, ncu_command())
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

    def __init__(self, gpu, work_root: Path, timeout_s: int = 900):
        self.gpu, self.work_root, self.timeout_s = gpu, Path(work_root), timeout_s

    async def _compile(self, problem, sources, tag):
        from ..agents.utils.commands import compile_cu
        from ..agents.utils import FeedbackError

        work = self.work_root / f"{re.sub(r'[^A-Za-z0-9_]+', '_', tag)}_{uuid.uuid4().hex[:6]}"
        work.mkdir(parents=True, exist_ok=True)
        cu = work / "kernels.cu"
        cu.write_text(problem.assemble(sources))
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

    async def profile(self, problem, sources, tag):
        from ..agents.utils.commands import run_gpu_executable
        from ..agents.utils import FeedbackError
        from .ncu import ncu_command

        cu, exe, bad = await self._compile(problem, sources, tag + "_ncu")
        if bad is not None:
            return {}, "build failed: " + bad.error_text(300)
        try:
            stdout, stderr = await run_gpu_executable(exe, self.gpu, self.timeout_s, job_name=f"{cu} (ncu)",
                                                      prefix_command=f"MB_MODE=profile {ncu_command()} ")
        except FeedbackError as e:
            return {}, str(e)[-300:]
        return _ncu_result(stdout + "\n" + stderr)


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

    def __init__(self, work_root: Optional[Path] = None, cxx: str = "g++", max_parallel: int = 2,
                 timeout_s: int = 900, keep: bool = False, env: Optional[dict] = None):
        super().__init__("opencl", work_root=work_root, max_parallel=max_parallel, timeout_s=timeout_s, keep=keep)
        self.cxx, self.env = cxx, dict(env or {})

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
            async with self._gpu:
                text = await self._run(work / "problem", {**self.env, **env, "MB_DATA_PATH": str(data)})
            return None, text
        finally:
            if not self.keep:
                shutil.rmtree(work, ignore_errors=True)

    async def evaluate(self, problem, sources, tag):
        bad, text = await self._exec(problem, sources, tag, {})
        if bad is not None:
            return bad
        if text is None:
            return EvalResult(ok=False, stage="run", errors=[f"timeout after {self.timeout_s}s"])
        return parse_driver_output(text)

    async def profile(self, problem, sources, tag):
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

    async def evaluate(self, problem, sources, tag):
        bad, text = await self._exec(problem, sources, tag, {})
        if bad is not None:
            return bad
        if text is None:
            return EvalResult(ok=False, stage="run", errors=[f"timeout after {self.timeout_s}s"])
        return parse_driver_output(text)

    async def profile(self, problem, sources, tag):
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

    async def evaluate(self, problem, sources, tag):
        bad, text = await self._exec(problem, sources, tag, False)
        return bad if bad is not None else parse_driver_output(text)

    async def profile(self, problem, sources, tag):
        bad, text = await self._exec(problem, sources, tag + "_prof", True)
        if bad is not None:
            return {}, "build failed: " + bad.error_text(300)
        return _opencl_profile_result(problem, text)
