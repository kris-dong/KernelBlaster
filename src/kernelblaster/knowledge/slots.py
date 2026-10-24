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
    op_us: dict = field(default_factory=dict)          # op name -> us (MB_OP_US; newer drivers)
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


class LocalEvaluator(Evaluator):
    """nvcc + run on this host (dev/tests; no KernelBlaster servers needed)."""

    def __init__(self, arch: str = "75", work_root: Optional[Path] = None, nvcc: str = "nvcc",
                 max_parallel: int = 2, timeout_s: int = 900, keep: bool = False):
        self.arch, self.nvcc, self.timeout_s, self.keep = arch, nvcc, timeout_s, keep
        self.work_root = Path(work_root) if work_root else None
        self._sem = asyncio.Semaphore(max_parallel)
        self._gpu = asyncio.Lock()             # one timed run at a time

    async def evaluate(self, problem, sources, tag):
        root = self.work_root or problem.root / "kb_work"
        work = root / f"{re.sub(r'[^A-Za-z0-9_]+', '_', tag)}_{uuid.uuid4().hex[:6]}"
        work.mkdir(parents=True, exist_ok=True)
        try:
            (work / "kernels.cu").write_text(problem.assemble(sources))
            async with self._sem:
                proc = await asyncio.create_subprocess_exec(
                    self.nvcc, "-O3", f"-arch=sm_{self.arch}", "-std=c++17", "-o", str(work / "problem"),
                    str(work / "kernels.cu"), str(problem.driver_path),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                out, _ = await asyncio.wait_for(proc.communicate(), self.timeout_s)
            if proc.returncode != 0:
                return EvalResult(ok=False, stage="compile", errors=[_compile_errors(out.decode())],
                                  log=out.decode()[-4000:])
            async with self._gpu:
                proc = await asyncio.create_subprocess_exec(
                    str(work / "problem"), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                try:
                    out, _ = await asyncio.wait_for(proc.communicate(), self.timeout_s)
                except asyncio.TimeoutError:
                    proc.kill()
                    return EvalResult(ok=False, stage="run", errors=[f"timeout after {self.timeout_s}s"])
            return parse_driver_output(out.decode(errors="replace"))
        finally:
            if not self.keep:
                shutil.rmtree(work, ignore_errors=True)


class ServerEvaluator(Evaluator):
    """KernelBlaster's compile + GPU exec servers (``compile_cu`` / ``run_gpu_executable``)."""

    def __init__(self, gpu, work_root: Path, timeout_s: int = 900):
        self.gpu, self.work_root, self.timeout_s = gpu, Path(work_root), timeout_s

    async def evaluate(self, problem, sources, tag):
        from ..agents.utils.commands import compile_cu, run_gpu_executable
        from ..agents.utils import FeedbackError

        work = self.work_root / f"{re.sub(r'[^A-Za-z0-9_]+', '_', tag)}_{uuid.uuid4().hex[:6]}"
        work.mkdir(parents=True, exist_ok=True)
        cu = work / "kernels.cu"
        cu.write_text(problem.assemble(sources))
        try:
            exe = await compile_cu(problem.driver_path, cu, self.gpu, timeout=self.timeout_s, job_name=str(cu))
        except FeedbackError as e:
            return EvalResult(ok=False, stage="compile", errors=[_compile_errors(str(e))], log=str(e)[-4000:])
        try:
            stdout, stderr = await run_gpu_executable(Path(exe), self.gpu, self.timeout_s, job_name=str(cu))
        except FeedbackError as e:
            return EvalResult(ok=False, stage="run", errors=[str(e)[-2000:]], log=str(e)[-4000:])
        return parse_driver_output(stdout + "\n" + stderr)


def _compile_errors(log: str, limit: int = 3000) -> str:
    errs = [l for l in log.splitlines() if "error" in l.lower()]
    return ("\n".join(errs) or log)[-limit:]


def find_problems(root: Path) -> list[Path]:
    root = Path(root)
    if (root / "manifest.json").exists():
        return [root]
    return sorted(p.parent for p in root.rglob("manifest.json"))
