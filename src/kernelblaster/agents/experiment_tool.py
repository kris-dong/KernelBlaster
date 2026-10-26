"""Multi-turn codegen with an experiment tool (A2 capabilities item 5; ModelBlaster cuda_vs_flow.md recommendation 5).

The slot optimiser's codegen call is normally one stateless completion: technique name in, slot source out. The
CUDA-only Claude baseline instead ran deliberate probes (timing ablations, in-kernel ``clock64`` phase counters,
spin-wait cost probes) before restructuring: 18/37 MLP evals, 18/33 VGG, 15 MobileNet probes, 2+2 in 41.

With ``KERNELBLASTER_MULTITURN=1`` (ModelBlaster ``--multiturn on``) a codegen step becomes a bounded conversation:

* the prompt gains an "Experiment tool" section (protocol, budget);
* a reply that contains ```` ```experiment ```` blocks (and no final ```` ```cuda ```` block) runs each block as a
  standalone CUDA program: ``nvcc -O3 -std=c++17 -arch=sm_<arch>`` then the binary, on a GPU leased from the run's
  own pool and under that GPU's timing lock (so experiments never overlap a timed candidate on the same GPU); the
  tail of stdout/stderr goes back as the next user message;
* budget per step (``KERNELBLASTER_MULTITURN_BUDGET`` = "turns,runs,seconds", default "3,4,120"): at most ``turns``
  LLM calls, ``runs`` experiment programs, ``seconds`` of compile + run wall time in total (each program also has
  its own 60 s run cap); the last turn tells the model to answer with the slot source;
* restricted: a program is refused (not compiled) when it is larger than 64 KB or uses process / file / network
  APIs (``system``, ``popen``, ``fork``, ``exec*``, sockets, ``fopen`` / ``ofstream``, ``remove``, ``unlink``,
  ``#include`` of ``<sys/...>``, ``<unistd.h>``, ``<fstream>`` ...); it runs in an empty scratch directory with a
  CPU-time and output cap. Not a sandbox: a mitigation for an LLM that writes benchmark code, not for an adversary.

Off (the default) the codegen call is byte-for-byte the single-shot call. Every step's tool use is recorded
(``<out>/multiturn.jsonl``; result.json ``multiturn``).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ENV = "KERNELBLASTER_MULTITURN"
ENV_BUDGET = "KERNELBLASTER_MULTITURN_BUDGET"
MAX_SRC = 64 * 1024
OUT_TAIL = 4000
RUN_CAP_S = 60.0
COMPILE_CAP_S = 120.0
_FORBIDDEN = re.compile(
    r"\b(system|popen|fork|vfork|execl|execlp|execle|execv|execvp|execve|socket|connect|fopen|freopen|remove|"
    r"unlink|rename|rmdir|mkdir|chdir|kill|ptrace|dlopen)\s*\(|"
    r"#\s*include\s*<\s*(sys/|unistd\.h|fstream|filesystem|dlfcn\.h|netinet|arpa/|spawn\.h)|"
    r"\b(std::filesystem|ofstream|ifstream|__asm__\s*\(\s*\"syscall)")
_BLOCK = re.compile(r"```experiment[^\n]*\n(.*?)```", re.S)


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip().lower() in ("1", "on", "true", "yes")


def budget() -> dict:
    s = os.environ.get(ENV_BUDGET, "3,4,120")
    t, r, sec = [x.strip() for x in s.split(",")]
    return {"turns": max(1, int(t)), "runs": max(0, int(r)), "seconds": float(sec)}


def blocks(text: str) -> list[str]:
    return [b for b in _BLOCK.findall(text or "") if b.strip()]


def has_final(text: str) -> bool:
    return bool(re.search(r"```(?:cuda|cpp|c\+\+|cu|c)\s*\n", text or ""))


def refused(code: str) -> Optional[str]:
    if len(code) > MAX_SRC:
        return f"program too large ({len(code)} bytes > {MAX_SRC})"
    m = _FORBIDDEN.search(code)
    if m:
        return f"refused: uses {m.group(0).strip()!r} (no process / file / network APIs in experiments)"
    if not re.search(r"\bint\s+main\s*\(", code):
        return "refused: an experiment is a complete program with int main()"
    return None


def prompt_section(b: dict, arch: str) -> str:
    return (
        "## Experiment tool (optional, before you answer)\n"
        f"You may run up to {b['runs']} experiment program(s) in this step, over at most {b['turns']} replies "
        f"({b['seconds']:.0f} s of compile + run time in total). To run experiments, reply with one or more blocks\n"
        "```experiment\n<a complete standalone CUDA program with int main()>\n```\n"
        "and NO ```cuda block. Each is compiled with `nvcc -O3 -std=c++17 -arch=sm_" + str(arch) + "` and run once "
        f"on the target GPU (cap {RUN_CAP_S:.0f} s); you get its exit code and the last {OUT_TAIL} characters of its "
        "stdout/stderr. The program does not see this slot's harness or data: include the kernels and the inputs "
        "you need (random data of the instance shapes). Useful: cudaEvent timing of two variants back to back "
        "(warm up, median of reps), in-kernel clock64() phase counters copied back and printed, cost of a spin-wait / "
        "L2 round trip between blocks, occupancy (cudaOccupancyMaxActiveBlocksPerMultiprocessor), register/smem use "
        "(compile with your own __launch_bounds__). No file, process or network APIs (refused). Your final reply must "
        "contain the complete slot source in one ```cuda block (no experiment blocks in it).")


@dataclass
class StepLog:
    turns: int = 0
    runs: int = 0
    refused: int = 0
    compile_failed: int = 0
    seconds: float = 0.0
    forced_final: bool = False
    items: list = field(default_factory=list)
    # X-145 (only filled when converse(record=True); never part of to_json / multiturn.jsonl)
    records: list = field(default_factory=list)
    rationale: str = ""

    def to_json(self) -> dict:
        return {"turns": self.turns, "runs": self.runs, "refused": self.refused, "compile_failed": self.compile_failed,
                "seconds": round(self.seconds, 2), "forced_final": self.forced_final, "items": self.items}


class ExperimentTool:
    """Compiles and runs experiment programs for one KernelBlaster run (``work``: scratch root)."""

    def __init__(self, evaluator, problem, arch: str, work: Path, nvcc: str = "nvcc"):
        self.ev, self.p, self.arch, self.work = evaluator, problem, str(arch), Path(work)
        self.nvcc = getattr(evaluator, "nvcc", None) or nvcc
        self.n = 0
        self._nvcc_ver: Optional[dict] = None             # X-145 v1.1: nvcc version, read once (only when recording)

    @staticmethod
    def _recording() -> bool:
        from ..knowledge.v2 import experiments
        return experiments.record_enabled()

    async def _nvcc_info(self, flags: list) -> dict:
        if self._nvcc_ver is None:
            self._nvcc_ver = {}
            try:
                cp = await _proc([self.nvcc, "--version"], cwd=self.work, timeout=20, env=None)
                m = re.search(r"release (\d+\.\d+), V(\d+\.\d+\.\d+)", cp["out"])
                self._nvcc_ver = {"release": m.group(1), "version": m.group(2)} if m else {"version": None}
            except Exception:                              # noqa: BLE001 -- never break an experiment
                self._nvcc_ver = {"version": None}
        return {**self._nvcc_ver, "flags": [f for f in flags if f != "-o" and not f.endswith(("exp", "exp.cu"))]}

    async def run(self, code: str, seconds_left: float) -> dict:
        """-> {ok, stage (refused|compile|run), exit, out (tail), wall_s}"""
        why = refused(code)
        if why:
            return {"ok": False, "stage": "refused", "out": why, "wall_s": 0.0}
        self.n += 1
        d = self.work / f"exp{self.n:03d}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        (d / "exp.cu").write_text(code)
        t0 = time.time()
        arch = self.arch.removeprefix("sm_")
        flag = f"-arch=sm_{arch}" + ("a" if arch in ("90", "100") else "")
        cmd = [self.nvcc, "-O3", "-std=c++17", flag, "-lineinfo", "-o", str(d / "exp"), str(d / "exp.cu")]
        cp = await _proc(cmd, cwd=d, timeout=min(COMPILE_CAP_S, max(5.0, seconds_left)), env=None)
        nv = await self._nvcc_info(cmd[1:]) if self._recording() else None
        if cp["exit"] != 0:
            return {"ok": False, "stage": "compile", "exit": cp["exit"], "out": cp["out"][-OUT_TAIL:],
                    "wall_s": round(time.time() - t0, 2), **({"nvcc": nv} if nv else {})}
        left = max(2.0, seconds_left - (time.time() - t0))
        res = await self._run_leased(d, min(RUN_CAP_S, left))
        res["wall_s"] = round(time.time() - t0, 2)
        if nv:
            res["nvcc"] = nv
        return res

    async def _run_leased(self, d: Path, limit: float) -> dict:
        pool_f = getattr(self.ev, "pool", None)
        if pool_f is None:                                       # remote / server evaluators: not supported
            return {"ok": False, "stage": "run", "out": "experiments need a local evaluator (GPU pool on this host)"}
        from ..servers.utils.resources import acquire_lock_file, gpu_lock_file
        pool = await pool_f()
        async with pool.lease(str(self.p.root)) as lease:
            dev = lease.device
            env = {**os.environ, **dev.env()}
            lk = await acquire_lock_file(gpu_lock_file(dev.ident), getattr(self.ev, "gpu_lock_wait_s", 900.0))
            clock = await self._clock("probe_pre", lease) if self._recording() else None    # v1.1: GPU state at probe time
            try:
                r = await _proc([str(d / "exp")], cwd=d, timeout=limit, env=env, ulimit_cpu=int(limit) + 5)
            finally:
                lk.release()
            if clock is not None:
                post = await self._clock("probe_post", lease)
                clock = {**clock, **({"post": post["pre"]} if post else {})}
        out = {"ok": r["exit"] == 0, "stage": "run", "exit": r["exit"], "out": r["out"][-OUT_TAIL:],
               "timed_out": r.get("timed_out", False), "device": dev.label()}
        if clock is not None:
            out["clock"] = {k: v for k, v in clock.items() if k in ("pre", "post")}
            out["gpu_uuid"] = dev.ident if dev.ident and str(dev.ident).startswith("GPU-") else None
        return out

    async def _clock(self, phase: str, lease) -> Optional[dict]:
        """The leased GPU's clocks (the clock guard's query and classification; knowledge/clock_guard.py), as
        ``{"pre": ClockVerdict dict}``; None when the evaluator has no way to run a command on the GPU's host. A suspect
        reading with the guard on also quarantines the GPU (as before a timed step)."""
        run = getattr(self.ev, "_clock_run", None)
        obj = getattr(self.ev, "_clock_obj", None)
        if run is None or obj is None:
            return None
        try:
            dev = lease.device
            v = await obj().check(lambda cmd: run(dev, cmd), ident=dev.ident, index=dev.index, phase=phase, key=dev.key,
                                  label=dev.label())
            from ..knowledge import clock_guard as cg
            if v.suspect and cg.enabled() and hasattr(self.ev, "_clock_flag"):
                self.ev._clock_flag(lease, v, "around an experiment probe")
            return {"pre": v.to_dict()}
        except Exception:                                  # noqa: BLE001 -- never break an experiment
            return None


async def _proc(cmd, *, cwd, timeout, env, ulimit_cpu: Optional[int] = None) -> dict:
    pre = None
    if ulimit_cpu:
        def pre():                                               # CPU-time cap + no core dumps
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (ulimit_cpu, ulimit_cpu))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    p = await asyncio.create_subprocess_exec(*cmd, cwd=str(cwd), env=env, stdout=asyncio.subprocess.PIPE,
                                             stderr=asyncio.subprocess.STDOUT, preexec_fn=pre)
    try:
        out, _ = await asyncio.wait_for(p.communicate(), timeout)
        return {"exit": p.returncode, "out": out.decode(errors="replace")}
    except asyncio.TimeoutError:
        p.kill()
        out = b""
        try:
            out, _ = await asyncio.wait_for(p.communicate(), 5)
        except Exception:  # noqa: BLE001
            pass
        return {"exit": -9, "out": out.decode(errors="replace") + f"\n[killed after {timeout:.0f} s]", "timed_out": True}


def results_message(results: list[dict], b: dict, log: StepLog, last: bool) -> str:
    parts = []
    for i, r in enumerate(results):
        head = f"### experiment {i + 1}: " + (
            "REFUSED" if r["stage"] == "refused" else "compile FAILED" if r["stage"] == "compile" else
            f"exit {r.get('exit')}" + (" (TIMED OUT)" if r.get("timed_out") else "")) + f", {r.get('wall_s', 0)} s"
        parts.append(f"{head}\n```\n{(r.get('out') or '').strip()}\n```")
    left = f"Budget left: {b['runs'] - log.runs} run(s), {max(0, b['turns'] - log.turns)} reply(ies), " \
           f"{max(0.0, b['seconds'] - log.seconds):.0f} s."
    tail = ("This is your LAST reply: answer now with the complete slot source in one ```cuda block."
            if last else "Run more experiments the same way, or answer with the complete slot source in one ```cuda "
                         "block.")
    return "\n\n".join(parts + [left, tail])


async def converse(llm, messages: list[dict], tool: ExperimentTool, b: dict, *, key: str,
                   extra: Optional[dict] = None, record: bool = False,
                   extra_prompt: str = "") -> tuple[str, StepLog]:
    """The multi-turn codegen of one step: ``llm(messages, key=..., extra=...)`` -> text. Returns (final text,
    step log). Turn 0 uses ``key`` (the single-shot journal key), turn k ``{key}t{k}``. A reply with a ```cuda block
    is final (experiment blocks next to it are ignored); a reply with only experiments runs them (within the
    budget) and continues; when the turns, runs or seconds are used up while the model still experiments, one
    last call asks for the source (``forced_final``).

    ``record=True`` (X-145, ``KERNELBLASTER_EXPERIMENT_RECORD``) additionally keeps, per executed program, its source,
    the model's stated purpose (prose before the block) and conclusion (prose opening the next reply) in
    ``log.records``; ``extra_prompt`` is appended to the experiment-tool prompt section. Both default off: the
    conversation is then exactly what it was."""
    log = StepLog()
    conv = [dict(m) for m in messages]
    conv[-1] = {**conv[-1], "content": conv[-1]["content"] + "\n\n" + prompt_section(b, tool.arch)
                + (("\n\n" + extra_prompt) if extra_prompt else "")}
    text = ""
    pending: list = []
    if record:
        from ..knowledge.v2.experiments import purpose_of, split_purposes
    for turn in range(b["turns"]):
        text = await llm(conv, key=key if turn == 0 else f"{key}t{turn}", extra=extra if turn == 0 else None)
        log.turns += 1
        if record:
            for rec in pending:
                rec["conclusion"] = purpose_of(text)
            pending = []
            log.rationale = purpose_of(text)
        exps = blocks(text)
        if not exps or has_final(text):
            return text, log
        runs_left = b["runs"] - log.runs
        if turn == b["turns"] - 1 or runs_left <= 0 or log.seconds >= b["seconds"]:
            break
        results = []
        purposes = split_purposes(text) if record else []
        for ei, code in enumerate(exps[:runs_left]):
            t0 = time.time()
            r = await tool.run(code, b["seconds"] - log.seconds)
            log.seconds += time.time() - t0
            if r["stage"] == "refused":
                log.refused += 1
            else:
                log.runs += 1
                if r["stage"] == "compile":
                    log.compile_failed += 1
            log.items.append({"turn": turn, "stage": r["stage"], "exit": r.get("exit"), "wall_s": r.get("wall_s"),
                              "out_head": (r.get("out") or "")[:300]})
            results.append(r)
            if record:
                rec = {"turn": turn, "idx": ei, "purpose": purposes[ei] if ei < len(purposes) else "",
                       "program": code, "stage": r["stage"], "ok": r.get("ok"), "exit": r.get("exit"),
                       "out": r.get("out") or "", "wall_s": r.get("wall_s"), "timed_out": r.get("timed_out", False),
                       "device": r.get("device"), "conclusion": "", "clock": r.get("clock"),
                       "gpu_uuid": r.get("gpu_uuid"), "nvcc": r.get("nvcc")}
                log.records.append(rec)
                pending.append(rec)
        conv += [{"role": "assistant", "content": text},
                 {"role": "user", "content": results_message(results, b, log, last=turn + 1 == b["turns"] - 1)}]
    conv += [{"role": "assistant", "content": text},
             {"role": "user", "content": "Experiment budget exhausted. Answer now with the complete slot source in "
                                         "one ```cuda block."}]
    log.forced_final = True
    text = await llm(conv, key=f"{key}tf", extra=None)
    log.turns += 1
    if record:
        for rec in pending:
            rec["conclusion"] = purpose_of(text)
        log.rationale = purpose_of(text)
    return text, log


def model_context_text(manifest: dict) -> str:
    """The manifest's ``model_context`` (ModelBlaster --model-context on) as a prompt section."""
    ctx = manifest.get("model_context")
    if not ctx:
        return ""
    rows = [f"- {o['name']} ({o['op']}): {o['us']:.1f} us, {100 * o['share']:.1f}%"
            + (" [in this problem]" if o.get("in_this_problem") else "") for o in ctx.get("top_ops", [])[:16]]
    kinds = ", ".join(f"{k} x{v['count']} {100 * v['share']:.0f}%" for k, v in list(ctx.get("kinds", {}).items())[:8])
    return (f"## Whole-model context\nThe model ({ctx.get('n_ops')} ops) runs {ctx.get('model_us')} us end to end. "
            f"By kind: {kinds}.\nLargest ops:\n" + "\n".join(rows))
