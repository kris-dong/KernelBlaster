# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-slot RL optimisation of a ModelBlaster partition (one slot per step).

A problem has N kernel slots (``knowledge.slots.SlotProblem``). Each step of a
rollout:

  1. picks a slot by its share of partition time (UCB-style bonus for slots
     tried less often; slots under ``min_share`` are skipped);
  2. picks a technique with a UCB1 bandit keyed by (op class, slot state);
     an arm never tried in this key starts from the technique's mean reward
     in any other key, so a lesson on one slot transfers to similar slots;
  3. asks the LLM to rewrite only that slot, with its neighbours' signatures
     read-only, its C reference, the instance shapes, the measured state,
     implementation-store examples and semantic claims as context;
  4. assembles, builds and runs the driver (partition + isolated-op checks,
     per-slot times), with a bounded fix loop on compile/check failures;
  5. rewards the partition-level gain (slot time saved / partition time),
     appends an OutcomeRecord, and stores verified code in the ImplStore.

Parallel rollouts share a *composite best*: the fastest verified source per
slot. Slots are separate kernels, so wins on different slots compose; the
composite is re-verified as a whole before it is reported.

The slot *state* is a roofline classification from the manifest (bytes moved
and FLOPs per op) and the measured slot time -- no NCU counters needed.
"""
from __future__ import annotations

import asyncio
import difflib
import json
import math
import random
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger as _default_logger

from .. import telemetry as tel
from ..knowledge import ncu as ncu_mod
from ..knowledge.impl_store import ImplStore, shape_bucket
from ..knowledge.outcomes import (COMPILE_ERROR, RUNTIME_ERROR, WRONG_OUTPUT, OutcomeLog,
                                  OutcomeRecord, classify_speedup)
from ..knowledge.slots import EvalResult, Evaluator, SlotProblem, step_eval_options
from .cost_tracker import lookup_pricing
from .model_router import ModelRouter, technique_tier
from .utils import generate_code_retry

# Peak fp32 throughput / DRAM bandwidth per arch, for the roofline state.
ARCH_PEAKS = {  # sm: (GFLOP/s, GB/s)
    "75": (16300.0, 672.0),       # TITAN RTX
    "80": (19500.0, 1555.0),      # A100 40GB
    "86": (35600.0, 936.0),       # RTX 3090
    "89": (91600.0, 864.0),       # L40S
    "90": (67000.0, 3350.0),      # H100 SXM
}

# Built-in technique catalog, by op class ("*" applies to every class).
# Names follow the technique database where one exists.
TECHNIQUES: dict[str, list[tuple[str, str]]] = {
    "*": [
        ("memory_coalescing", "Make consecutive threads touch consecutive addresses; reorder the thread->element mapping."),
        ("vectorized_memory_access", "Use float4/float2 loads and stores where alignment and sizes allow (keep a scalar tail)."),
        ("launch_configuration_tuning", "Choose block size / grid shape for occupancy and to cover small problems; grid-stride loops."),
        ("loop_unrolling_and_ilp", "Unroll inner loops and give each thread several independent elements to hide latency."),
        ("read_only_cache_and_restrict", "Mark inputs const __restrict__ and use __ldg for read-only data."),
    ],
    "gemm": [
        ("shared_memory_tiling", "Tile A and B through shared memory (e.g. 32x32 or 64x64 tiles) so each element is read from DRAM once per tile."),
        ("register_blocking", "Each thread computes a small TMxTN block of outputs in registers (outer-product micro-kernel)."),
        ("double_buffering", "Prefetch the next shared-memory tile while computing the current one."),
    ],
    "conv": [
        ("implicit_gemm", "Treat the convolution as a GEMM (M=N*OH*OW, N=OC, K=IC*KH*KW) without materialising im2col; tile it."),
        ("shared_memory_tiling", "Stage input patches and filter tiles in shared memory and reuse them across output channels/pixels."),
        ("register_blocking", "Each thread computes several output channels and/or pixels, reusing loaded inputs from registers."),
    ],
    "reduction": [
        ("warp_shuffle_reduction", "Reduce within a warp with __shfl_down_sync, then across warps via shared memory."),
        ("one_block_per_row", "Map one block (or warp) to each reduced row; parallelise the reduction dimension."),
        ("single_pass_fusion", "Compute statistics and the normalised output in as few passes over memory as possible."),
    ],
    "elementwise": [
        ("grid_stride_vectorized", "Grid-stride loop with float4 accesses; minimal integer math per element."),
    ],
    "data_movement": [
        ("index_math_hoisting", "Precompute strides/divisions; use fewer integer divides per element."),
    ],
    "attention": [
        ("flash_style_tiling", "Tile Q/K/V through shared memory with an online softmax so the score matrix is never materialised."),
    ],
}


def techniques_for(op_class: str, techniques: Optional[dict] = None) -> list[tuple[str, str]]:
    techniques = techniques or TECHNIQUES
    return techniques.get(op_class, []) + techniques["*"]


# ---------------------------------------------------------------- roofline

# element bytes per manifest buffer dtype (fp16 problems move half the bytes)
BUFFER_BYTES = {"f32": 4, "f16": 2, "i32": 4, "i64": 8}


def _nbytes(buf: dict) -> float:
    return float(BUFFER_BYTES.get(buf.get("dtype", "f32"), 4) * buf["numel"])

def primary_shape(shape: dict) -> dict:
    """A fused op's shape is {"a": stage 0, "b": stage 1, ...}; its work is
    the first stage's (the producer whose epilogue the others join)."""
    return shape["a"] if isinstance(shape.get("a"), dict) else shape


def _op_cost(problem: SlotProblem, op: dict) -> tuple[float, float]:
    """(bytes, flops) estimate for one op instance from the manifest."""
    bufs = problem.manifest["buffers"]
    byt = sum(_nbytes(bufs[b]) for b in op["inputs"] + op["outputs"])
    slot = problem.slots[op["slot"]]
    inst = next((i for i in slot.instances if i["index"] == op["index"]), {})
    sh = primary_shape(inst.get("shape", {}))
    out_n = sum(bufs[b]["numel"] for b in op["outputs"])
    flops = float(out_n)
    if slot.op_class == "gemm":
        k = sh.get("K", 1)
        flops = 2.0 * out_n * k
    elif slot.op_class == "conv":
        g = sh.get("G", 1)
        flops = 2.0 * out_n * (sh.get("IC", 1) / max(g, 1)) * sh.get("KH", 1) * sh.get("KW", 1)
    elif slot.op_class == "attention":
        s, d = sh.get("S", 1), sh.get("D", 1)
        flops = 4.0 * sh.get("BH", 1) * s * s * d
    return byt, flops


def slot_state(problem: SlotProblem, sid: str, slot_us: float, arch: str,
               peaks: Optional[tuple] = None) -> tuple[str, dict]:
    """Roofline regime of a slot and the numbers behind it (``peaks``: (GFLOP/s, GB/s), default by sm)."""
    byt = fl = 0.0
    for op in problem.manifest["ops"]:
        if op["slot"] == sid:
            b, f = _op_cost(problem, op)
            byt += b
            fl += f
    peak_f, peak_b = peaks or ARCH_PEAKS.get(arch, ARCH_PEAKS["75"])
    t = max(slot_us, 1e-3) * 1e-6
    gbs, gfs = byt / t / 1e9, fl / t / 1e9
    pb, pf = gbs / peak_b, gfs / peak_f
    ai = fl / max(byt, 1.0)
    ridge = peak_f / peak_b
    if slot_us < 8.0:
        state = "latency_bound"
    elif max(pb, pf) < 0.15:
        state = "underutilized_compute" if ai > ridge else "underutilized_memory"
    else:
        state = "compute_bound" if pf >= pb else "memory_bound"
    return state, {"bytes": byt, "flops": fl, "arith_intensity": round(ai, 2),
                   "achieved_GBps": round(gbs, 1), "achieved_GFLOPs": round(gfs, 1),
                   "pct_peak_bw": round(100 * pb, 1), "pct_peak_flops": round(100 * pf, 1)}


# ---------------------------------------------------------------- KernelSpec proposals

FUSABLE_CONSUMERS = ("elementwise",)


@tel.traced("proposals")
def propose_fusions(problem: SlotProblem, result: EvalResult, arch: str, min_gain_pct: float = 2.0,
                    arch_label: Optional[str] = None) -> dict:
    """``kb-proposal/1``: fuse producer->consumer pairs where the consumer is an
    elementwise op that is the only reader of the producer's output.

    Estimated gain per pair = the consumer instance's time (its kernel goes
    away into the producer's epilogue) -- the intermediate's DRAM round trip
    is the bulk of an elementwise kernel's time, so this already counts it.
    Pairs are chosen greedily by gain, each op in at most one pair; a pattern
    (producer kind, consumer kind) is proposed if its pairs together save
    ``min_gain_pct`` of the partition. ModelBlaster builds, verifies and
    decides (pipeline/kb_proposals.py).
    """
    ops = {o["name"]: o for o in problem.manifest["ops"]}
    bufs = problem.manifest["buffers"]
    part_us = result.partition_us or sum(result.slot_us.values()) or 1.0
    per_inst = {sid: result.slot_us.get(sid, 0.0) / max(len(s.instances), 1) for sid, s in problem.slots.items()}
    cands = []
    for a, b, t in problem.adjacent_pairs():
        sa, sb = problem.slots[ops[a]["slot"]], problem.slots[ops[b]["slot"]]
        if sb.op_class not in FUSABLE_CONSUMERS:
            continue
        nbytes = _nbytes(next(x for x in bufs if x["name"] == t))
        cands.append((per_inst[sb.id], a, b, t, sa.op, sb.op, nbytes))
    used, by_pattern = set(), {}
    for gain, a, b, t, ka, kb, nb in sorted(cands, reverse=True):
        if a in used or b in used:
            continue
        used.update((a, b))
        g = by_pattern.setdefault((ka, kb), {"pairs": [], "gain_us": 0.0, "bytes": 0})
        g["pairs"].append([a, b, t])
        g["gain_us"] += gain
        g["bytes"] += nb
    props = []
    for (ka, kb), g in sorted(by_pattern.items(), key=lambda kv: -kv[1]["gain_us"]):
        pct = 100.0 * g["gain_us"] / part_us
        if pct < min_gain_pct:
            continue
        props.append({
            "kind": "fuse", "pattern": [ka, kb], "pairs": g["pairs"],
            "evidence": {"est_gain_us": round(g["gain_us"], 2), "est_gain_pct": round(pct, 2),
                         "intermediate_bytes": g["bytes"], "partition_us": part_us,
                         "consumer_slot_us": result.slot_us.get(problem.slots[ops[g["pairs"][0][1]]["slot"]].id)},
            "rationale": (f"{kb} only reads the {ka} output it follows; fusing it into the {ka} epilogue "
                          f"removes {len(g['pairs'])} kernel launch(es) and {g['bytes'] / 1e6:.1f} MB of DRAM "
                          f"round trip, ~{pct:.1f}% of the partition on {arch_label or 'sm_' + arch}."),
        })
    return {"format": "kb-proposal/1", "problem": problem.name, "problem_dir": str(problem.root),
            "arch": arch_label or f"sm_{arch}", "proposals": props}


def _efficiency(problem: SlotProblem, ops: list[dict], us: float, arch: str) -> float:
    """Achieved fraction of the binding roofline peak for a set of op instances."""
    byt = fl = 0.0
    for op in ops:
        b, f = _op_cost(problem, op)
        byt += b
        fl += f
    peak_f, peak_b = ARCH_PEAKS.get(arch, ARCH_PEAKS["75"])
    t = max(us, 1e-3) * 1e-6
    return max(byt / t / 1e9 / peak_b, fl / t / 1e9 / peak_f)


@tel.traced("proposals")
def propose_splits(problem: SlotProblem, result: EvalResult, arch: str, min_gain_pct: float = 2.0) -> list[dict]:
    """``kb-proposal/1`` "split" proposals: give a slot shape-specialised
    variants when its instances fall into several shape buckets and the one
    kernel serves them very unevenly.

    Per bucket group g: measured time t_g (``op_us``) and roofline efficiency
    e_g (achieved fraction of the binding peak). A kernel tuned for one shape
    regime is typically efficient there and not elsewhere, so the estimated
    gain is sum_g t_g * (1 - e_g / max e): every group brought to the best
    group's efficiency by a kernel of its own. Needs per-op times; proposed
    if the gain is ``min_gain_pct`` of the partition. ModelBlaster builds the
    variant (one slot per group, each seeded with the current kernel) and
    decides by measurement."""
    if not result.op_us:
        return []
    part_us = result.partition_us or sum(result.slot_us.values()) or 1.0
    by_slot: dict[str, list[dict]] = {}
    for op in problem.manifest["ops"]:
        by_slot.setdefault(op["slot"], []).append(op)
    props = []
    for sid, ops in by_slot.items():
        slot = problem.slots[sid]
        if len(ops) < 2 or "__v" in sid:
            continue
        groups: dict[str, list[dict]] = {}
        for op in ops:
            inst = next((i for i in slot.instances if i["index"] == op["index"]), {})
            groups.setdefault(shape_bucket(slot.op_class, inst.get("shape", {})), []).append(op)
        if len(groups) < 2:
            continue
        rows = []
        for b, gops in groups.items():
            t = sum(result.op_us.get(o["name"], 0.0) for o in gops)
            rows.append({"bucket": b, "ops": [o["name"] for o in gops], "us": round(t, 3),
                         "eff_pct": round(100 * _efficiency(problem, gops, t, arch), 2)})
        best = max(r["eff_pct"] for r in rows) or 1e-9
        gain = sum(r["us"] * (1 - r["eff_pct"] / best) for r in rows)
        pct = 100.0 * gain / part_us
        if pct < min_gain_pct:
            continue
        lo = min(rows, key=lambda r: r["eff_pct"])
        props.append({
            "kind": "split", "slot": sid, "op": slot.op, "groups": rows,
            "evidence": {"est_gain_us": round(gain, 2), "est_gain_pct": round(pct, 2), "partition_us": part_us,
                         "slot_us": result.slot_us.get(sid)},
            "rationale": (f"`{sid}` serves {len(rows)} shape regimes with one kernel: {best:.1f}% of peak on its best "
                          f"group but {lo['eff_pct']:.1f}% on {lo['bucket']} ({lo['us']:.0f} us); a kernel per regime "
                          f"could save ~{pct:.1f}% of the partition."),
        })
    return sorted(props, key=lambda p: -p["evidence"]["est_gain_us"])


# ---------------------------------------------------------------- bandit

class TechniqueBandit:
    """UCB1 over techniques per key, with cross-key prior transfer."""

    def __init__(self, c: float = 1.0, prior_weight: float = 1.0):
        self.c, self.prior_weight = c, prior_weight
        self.stats: dict[str, dict[str, list[float]]] = {}       # key -> tech -> [n, sum]

    def _global(self, tech: str) -> Optional[float]:
        n = s = 0.0
        for arms in self.stats.values():
            if tech in arms:
                n += arms[tech][0]
                s += arms[tech][1]
        return s / n if n else None

    def select(self, key: str, techs: list[str], rng: random.Random) -> str:
        arms = self.stats.setdefault(key, {})
        total = sum(a[0] for a in arms.values()) + 1
        best, best_v = None, -1e9
        for t in techs:
            n, s = arms.get(t, [0.0, 0.0])
            prior = self._global(t)
            if n == 0 and prior is None:
                v = 1e6 + rng.random()               # untried everywhere: explore first
            else:
                pw = self.prior_weight if prior is not None else 0.0
                mean = (s + pw * (prior or 0.0)) / (n + pw) if (n + pw) else 0.0
                v = mean + self.c * math.sqrt(math.log(total + 1) / (n + 1)) + 1e-6 * rng.random()
            if v > best_v:
                best, best_v = t, v
        return best

    def update(self, key: str, tech: str, reward: float):
        a = self.stats.setdefault(key, {}).setdefault(tech, [0.0, 0.0])
        a[0] += 1
        a[1] += reward

    def to_json(self) -> dict:
        return self.stats

    @classmethod
    def from_json(cls, d: dict, **kw) -> "TechniqueBandit":
        b = cls(**kw)
        b.stats = {k: {t: list(v) for t, v in arms.items()} for k, arms in d.items()}
        return b


# ---------------------------------------------------------------- prompts

SYSTEM_PROMPT_TEMPLATE = """You optimise one CUDA kernel *slot* inside a larger graph partition that ModelBlaster \
compiled. The partition calls this slot's entry function once per op instance listed below; other slots \
are separate kernels you cannot change. Rewrite the slot to be faster while staying correct.

Rules for the slot source you return:
- Define the entry function with EXACTLY the given signature (same name, parameter types and order). All \
pointers are device pointers; scalars are runtime shapes, so handle every listed instance shape (and stay \
correct for other sizes: guard bounds).
- Launch kernels on the default stream. Do not allocate per call unless unavoidable, do not copy data \
to/from the host, do not call cudaDeviceSynchronize, do not define main() or launch_gpu_implementation.
- The source is placed inside a C++ namespace: any helper kernels/functions are fine. #include lines \
are hoisted out. Macros MB_THREADS (256), MB_GRID(n) and MB_TID are predefined; #undef anything you \
#define.
{dtype_rules}
- Return the COMPLETE slot source in a single ```cuda code block, nothing after it."""

# Per storage dtype (the manifest's `dtype`); {slot_rel} is the manifest's isolated-op bound.
DTYPE_RULES = {
    "fp32": "- fp32 in/out. Each op is checked alone against the reference: max error <= {slot_rel} "
            "relative to the largest input/output magnitude. Accumulate dot products in fp32 or better.",
    "fp16": "- fp16 storage, fp32 math: every float pointer is `__half` (IEEE binary16; <cuda_fp16.h> is "
            "included), integer pointers stay int. Widen on load (`__half2float`, or `__half22float2` on "
            "aligned `half2` pairs), compute and accumulate in fp32, round once on store "
            "(`__float2half_rn`); never accumulate or chain arithmetic in `__half`. Tensor cores "
            "(`wmma` / `mma.sync` with fp16 inputs and fp32 accumulators) are allowed.\n"
            "- Each op is checked alone against a reference that rounds the exact fp32 result to fp16 "
            "once: max error <= {slot_rel} relative to the largest input/output magnitude (about two fp16 "
            "ulps).",
}


def system_prompt(dtype: str = "fp32", tolerance: Optional[dict] = None) -> str:
    rel = (tolerance or {}).get("slot_rel", 1e-3)
    rel_s = f"{rel:.0e}".replace("e-0", "e-")                    # 1e-3, 2e-3
    rules = DTYPE_RULES.get(dtype, DTYPE_RULES["fp32"]).format(slot_rel=rel_s)
    return SYSTEM_PROMPT_TEMPLATE.format(dtype_rules=rules)


SYSTEM_PROMPT = system_prompt("fp32")


@dataclass(frozen=True)
class TargetProfile:
    """What differs between slot targets (manifest ``target``): prompt, code fence, technique
    catalogue, roofline peaks, store/outcome arch label, per-slot profiler, slot file suffix."""
    name: str
    system_prompt: str
    fence: str
    techniques: dict
    peaks: dict
    slot_ext: str
    profiler: Any                     # module with render() / delta(): knowledge.ncu or knowledge.opencl_profile

    def prompt_for(self, problem) -> str:
        if self.name == "cuda":                       # dtype rules + the manifest's isolated bound
            return system_prompt(problem.dtype, problem.tolerance)
        return self.system_prompt

    def arch_label(self, arch: str) -> str:
        return f"sm_{arch}" if self.name == "cuda" else arch

    def peaks_for(self, arch: str) -> tuple:
        return self.peaks.get(arch) or next(iter(self.peaks.values()))


def target_profile(target: str) -> TargetProfile:
    if target == "cuda":
        return TargetProfile("cuda", SYSTEM_PROMPT, "cuda", TECHNIQUES, ARCH_PEAKS, ".cu", ncu_mod)
    if target == "opencl":
        from ..knowledge import opencl_profile
        from .multislot_opencl import OPENCL_PEAKS, OPENCL_TECHNIQUES, SYSTEM_PROMPT_OPENCL
        return TargetProfile("opencl", SYSTEM_PROMPT_OPENCL, "cpp", OPENCL_TECHNIQUES, OPENCL_PEAKS, ".cpp",
                             opencl_profile)
    raise ValueError(f"unknown slot target {target!r}")


def _fmt_instances(slot) -> str:
    rows = [json.dumps(i.get("shape", {}), sort_keys=True) for i in slot.instances[:8]]
    more = f"\n  ... and {len(slot.instances) - 8} more" if len(slot.instances) > 8 else ""
    return "\n".join(f"  {r}" for r in rows) + more


def build_codegen_messages(*, problem: SlotProblem, sid: str, source: str, technique: str,
                           technique_desc: str, state: str, state_numbers: dict, slot_us: float,
                           partition_us: float, neighbours: list[tuple[str, str]],
                           examples: list[tuple[str, str]], adapt: Optional[tuple[str, str]],
                           knowledge_text: str, already_tried: str, best_note: str,
                           profile_text: str = "") -> list[dict]:
    slot = problem.slots[sid]
    tp = target_profile(problem.target)
    ref = problem.reference_text(sid)
    parts = [
        f"## Slot `{sid}` (op `{slot.op}`, class `{slot.op_class}`)",
        f"Signature:\n```c\n{slot.signature}\n```",
        f"Semantics: {slot.semantics}" if slot.semantics else "",
        f"Instances (shape scalars per call):\n{_fmt_instances(slot)}",
        f"Measured: {slot_us:.1f} us for all instances = {100 * slot_us / max(partition_us, 1e-9):.1f}% "
        f"of the partition ({partition_us:.1f} us). State: {state} {json.dumps(state_numbers)}",
        profile_text,
        best_note,
        f"C reference implementation (semantics oracle, not fast):\n```c\n{ref[:4000]}\n```" if ref else "",
    ]
    if neighbours:
        parts.append("Neighbouring slots (read-only, for context):\n" + "\n".join(
            f"- `{n}`: {sig}" for n, sig in neighbours))
    if adapt:
        parts.append(f"A verified fast implementation of this same op for overlapping shapes ({adapt[0]}); "
                     f"adapt it to these instances (re-tune what doesn't fit):\n```{tp.fence}\n{adapt[1][:6000]}\n```")
    for label, code in examples:
        parts.append(f"Example: a verified fast kernel of the same class ({label}):\n```{tp.fence}\n{code[:5000]}\n```")
    if knowledge_text:
        parts.append("Lessons from earlier optimisation runs:\n" + knowledge_text)
    if already_tried:
        parts.append("Already tried on this slot (avoid repeating dead ends):\n" + already_tried)
    parts.append(f"## Technique to apply: {technique}\n{technique_desc}")
    parts.append(f"Current slot source:\n```{tp.fence}\n{source}\n```")
    return [{"role": "system", "content": tp.prompt_for(problem)},
            {"role": "user", "content": "\n\n".join(p for p in parts if p)}]


def build_fix_messages(*, problem: SlotProblem, sid: str, source: str, error: str,
                       pitfalls: str = "") -> list[dict]:
    slot = problem.slots[sid]
    tp = target_profile(problem.target)
    known = f"Known pitfalls from earlier runs:\n{pitfalls}\n\n" if pitfalls else ""
    return [{"role": "system", "content": tp.prompt_for(problem)},
            {"role": "user", "content":
             f"The slot `{sid}` (op `{slot.op}`, signature `{slot.signature}`) failed:\n```\n{error}\n```\n\n"
             f"Instances:\n{_fmt_instances(slot)}\n\n{known}Fix it (keep the optimisation if you can). "
             f"Current source:\n```{tp.fence}\n{source}\n```"}]


@tel.traced("extract")
def extract_code(text: str) -> Optional[str]:
    blocks = re.findall(r"```(?:cuda|cpp|c\+\+|cu|c)?\s*\n(.*?)```", text, re.S)
    if not blocks:
        return None
    return max(blocks, key=len).strip() + "\n"


def diff_excerpt(a: str, b: str, max_lines: int = 80) -> str:
    """The first lines of a unified diff (code mining / state discovery context)."""
    d = list(difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=1))
    return "\n".join(d[2:2 + max_lines])


def diff_summary(a: str, b: str) -> str:
    d = list(difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0))
    add = sum(1 for l in d if l.startswith("+") and not l.startswith("+++"))
    rm = sum(1 for l in d if l.startswith("-") and not l.startswith("---"))
    return f"+{add}/-{rm} lines"


# ---------------------------------------------------------------- agent

@dataclass
class SlotBest:
    source: str
    slot_us: float
    impl_id: Optional[str] = None
    techniques: list = field(default_factory=list)


@dataclass
class StepLog:
    rollout: int
    step: int
    slot: str
    technique: str
    status: str
    slot_us_before: Optional[float]
    slot_us_after: Optional[float]
    reward: float
    fix_attempts: int
    seconds: float
    context: dict = field(default_factory=dict)      # what the prompt was given (store tiers, claims)


class MultiSlotAgent:
    def __init__(self, problem: SlotProblem, evaluator: Evaluator, *, model: str, out_dir: Path,
                 arch: str = "75", impl_store: Optional[ImplStore] = None,
                 outcome_log: Optional[OutcomeLog] = None, claims: Any = None, semantic_updater: Any = None,
                 rollouts: int = 4, steps: int = 6, max_fix_attempts: int = 2, min_share: float = 0.01,
                 confirm_pct: float = 3.0, stall_decay: float = 0.6, bandit: Optional[TechniqueBandit] = None,
                 model_fix: Optional[str] = None, seed: int = 0, logger=None, run_id: Optional[str] = None,
                 problem_id: Optional[str] = None, propose: bool = True, ncu: str = "off",
                 knowledge_v2: Any = None, router: Optional[ModelRouter] = None):
        self.p, self.ev, self.model = problem, evaluator, model
        # which model runs which call (model_router.py); default: `model` everywhere, `model_fix` for fixes
        self.router = router or ModelRouter({"codegen_simple": model, "codegen_hard": model,
                                             "fix": model_fix or model}, policy=[])
        # knowledge v2 (knowledge/v2): predicate states, ontology, hierarchical
        # stats, hypotheses; replaces the bandit, roofline state and claims lookup
        self.kv = knowledge_v2
        self.propose = propose
        if ncu not in ("off", "best", "all"):
            raise ValueError(f"ncu must be off|best|all, not {ncu!r}")
        # NCU per slot: "best" profiles the baseline and every new slot best,
        # "all" every verified candidate; disabled on the first failure
        self.ncu = ncu
        self.ncu_disabled: Optional[str] = None
        self.ncu_cache: dict[tuple, dict] = {}         # (slot, code sha1) -> ncu.summarize()
        self.ncu_runs = 0
        self.model_fix = self.router.models["fix"]
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.arch, self.store, self.olog = arch, impl_store, outcome_log
        # target (manifest): prompt, techniques, peaks; records keyed (target, arch label) so
        # CUDA and OpenCL implementations / lessons never mix
        self.tp = target_profile(problem.target)
        self.arch_label = self.tp.arch_label(arch)
        if self.kv is not None:                # targets.json entries win; the profile's peaks fill gaps
            self.kv.registry.ensure(self._tkey, peaks=self.tp.peaks_for(arch))
        self.claims, self.semantic = claims, semantic_updater
        self.rollouts, self.steps, self.max_fix = rollouts, steps, max_fix_attempts
        self.min_share, self.confirm_pct = min_share, confirm_pct
        self.bandit = bandit or TechniqueBandit()
        self.rng = random.Random(seed)
        self.log = logger or _default_logger
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.problem_id = problem_id or problem.name
        self.best: dict[str, SlotBest] = {}
        self.partition_us: Optional[float] = None
        self.baseline: Optional[EvalResult] = None
        self.slot_tries: dict[str, int] = {s: 0 for s in problem.slots}
        self.slot_stall: dict[str, int] = {s: 0 for s in problem.slots}   # attempts since last slot best
        self.stall_decay = stall_decay
        self.steps_log: list[StepLog] = []
        self.llm_calls = 0
        self.llm_usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
        self.llm_usage_by: dict[str, dict] = {}         # "<role> <model>" -> calls / tokens / cost
        self._lock = asyncio.Lock()

    # ---- helpers
    def _sources(self) -> dict[str, str]:
        return {s: b.source for s, b in self.best.items()}

    def _share(self, sid: str) -> float:
        tot = sum(b.slot_us for b in self.best.values()) or 1.0
        return self.best[sid].slot_us / tot

    def _pick_slot(self) -> Optional[str]:
        """Sample a slot ~ time share x stall decay x (1 + exploration bonus).

        Sampling (not argmax) spreads parallel rollouts over slots; the stall
        decay stops a big slot that keeps failing from starving the others."""
        total = sum(self.slot_tries.values()) + 1
        cands = [s for s in self.best if self._share(s) >= self.min_share] or list(self.best)
        w = [self._share(s) * (self.stall_decay ** self.slot_stall[s])
             * (1.0 + 0.5 * math.sqrt(math.log(total + 1) / (1 + self.slot_tries[s]))) for s in cands]
        if sum(w) <= 0:
            return self.rng.choice(cands)
        return self.rng.choices(cands, weights=w, k=1)[0]

    def _knowledge(self, sid: str, state: str, technique: Optional[str]) -> tuple[str, str]:
        if self.claims is None:
            return "", ""
        slot = self.p.slots[sid]
        try:
            cl = self.claims.relevant(target=self.p.target, arch=self.arch_label, op_class=slot.op_class,
                                      state=state, technique=technique, problem_id=self.problem_id, slot=sid)
            return self.claims.render(cl), self.claims.already_tried(self.problem_id, sid)
        except Exception as e:                                    # knowledge must never block a step
            self.log.warning(f"claims lookup failed: {e}")
            return "", ""

    @staticmethod
    def _sha(code: str) -> str:
        import hashlib
        return hashlib.sha1(code.encode()).hexdigest()

    async def _profile(self, sources: dict, tag: str) -> dict:
        """NCU the slots of ``sources`` whose code has no cached profile; cache
        per (slot, code). {} if off/unavailable.

        Slots already profiled with this exact code are left out of the ncu
        filter (their kernels are not replayed); their cached summaries are
        returned alongside the new ones."""
        if self.ncu == "off" or self.ncu_disabled:
            return {}
        todo = [sid for sid in sources if self._ncu_of(sid, sources[sid]) is None]
        cached = {sid: self._ncu_of(sid, sources[sid]) for sid in sources if sid not in todo}
        if not todo:
            return cached
        self.ncu_runs += 1
        only = {} if len(todo) == len(sources) else {"slots": todo}
        by, err = await self.ev.profile(self.p, sources, tag, **only)
        if err:
            self.ncu_disabled = err
            self.log.warning(f"[{self.p.name}] slot profiles disabled for this run: {err} (roofline states instead)")
            return {}
        for sid, summ in by.items():
            if sid in sources:
                self.ncu_cache[(sid, self._sha(sources[sid]))] = summ
        return {**cached, **by}

    def _ncu_of(self, sid: str, code: str) -> Optional[dict]:
        return self.ncu_cache.get((sid, self._sha(code)))

    def _pitfalls(self, sid: str, technique: str) -> str:
        if self.claims is None:
            return ""
        try:
            cl = self.claims.relevant(target=self.p.target, arch=self.arch_label,
                                      op_class=self.p.slots[sid].op_class, technique=technique,
                                      problem_id=self.problem_id, slot=sid, kinds=["pitfall"], k=6)
            return self.claims.render(cl)
        except Exception:
            return ""

    def _retrieve(self, sid: str) -> dict:
        if self.store is None:
            return {"seed": None, "adapt": None, "examples": []}
        s = self.p.slots[sid]
        return self.store.retrieve(target=self.p.target, arch=self.arch_label, op=s.op, op_class=s.op_class,
                                   signature=s.signature, shapes=s.shapes(), exclude_problem=self.problem_id,
                                   dtype=self.p.dtype)

    async def _llm(self, messages, model, role: str = "codegen") -> str:
        self.llm_calls += 1
        with tel.attrs(role=role):            # telemetry: the llm.<role> span + token record
            resp = await generate_code_retry(messages, model, self.log, n_tasks=1)
        u = getattr(resp, "usage", None) or {}
        if not isinstance(u, dict):
            u = {k: getattr(u, k, 0) for k in ("prompt_tokens", "completion_tokens")}
        # OpenAI-style (prompt/completion) or Bedrock Converse-style (input/output) keys
        pt = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
        ct = int(u.get("completion_tokens") or u.get("output_tokens") or 0)
        cost = float(u.get("cost_usd") or 0.0)                          # providers that report it
        pin, pout = lookup_pricing(model)                               # else the price table (0 if unknown)
        est = cost or (pt * pin + ct * pout) / 1e6
        self.llm_usage["prompt_tokens"] += pt
        self.llm_usage["completion_tokens"] += ct
        self.llm_usage["cost_usd"] += cost
        by = self.llm_usage_by.setdefault(f"{role} {model}", {"role": role, "model": model, "calls": 0,
                                                               "prompt_tokens": 0, "completion_tokens": 0,
                                                               "cost_usd": 0.0, "est_usd": 0.0})
        by["calls"] += 1
        by["prompt_tokens"] += pt
        by["completion_tokens"] += ct
        by["cost_usd"] += cost
        by["est_usd"] += est
        return resp.generations[0] if resp.generations else ""

    def _outcome(self, **kw) -> OutcomeRecord:
        sid = kw.pop("sid")
        s = self.p.slots[sid]
        return OutcomeRecord(target=self.p.target, arch=self.arch_label, problem_id=self.problem_id,
                             run_id=self.run_id, slot=sid, op_kind=s.op, op_class=s.op_class,
                             shape_bucket=shape_bucket(s.op_class, s.shapes()[0] if s.instances else {}),
                             dtype=self.p.dtype, **kw)

    @property
    def _tkey(self) -> str:
        return f"{self.p.target}:{self.arch_label}"

    @property
    def _profile_kind(self) -> str:
        """knowledge-v2 metric adapter for this target's profiler module (ncu | opencl_profile)."""
        return self.tp.profiler.__name__.rsplit(".", 1)[-1]

    def _roofline_input(self, sid: str, slot_us: float):
        """Manifest bytes/flops of the slot's instances + measured time (knowledge v2 estimate)."""
        from ..knowledge.v2.metrics import RooflineInput
        byt = fl = 0.0
        work = 0.0
        bufs = self.p.manifest["buffers"]
        n = 0
        for op in self.p.manifest["ops"]:
            if op["slot"] == sid:
                b, f = _op_cost(self.p, op)
                byt += b
                fl += f
                n += 1
                work = max(work, float(sum(bufs[x]["numel"] for x in op["outputs"])))
        return RooflineInput(bytes=byt, flops=fl, time_us=slot_us, work_items=work or None, launches=max(n, 1))

    def _record_outcome(self, **kw) -> Optional[OutcomeRecord]:
        if self.olog is None:
            return None
        rec = self._outcome(**kw)
        self.olog.append(rec)
        if self.claims is not None and hasattr(self.claims, "observe"):
            self.claims.observe(rec)          # "already tried" visible to parallel rollouts now
        return rec

    def _store(self, sid: str, code: str, slot_us: float, baseline_us: Optional[float],
               techniques: list, parent: Optional[str], rel_err: Optional[float]) -> Optional[str]:
        if self.store is None:
            return None
        s = self.p.slots[sid]
        seed_us = self.baseline.slot_us.get(sid) if self.baseline else None
        rec = self.store.add(code, target=self.p.target, arch=self.arch_label, op=s.op, op_class=s.op_class,
                             signature_sha=s.signature_sha, shapes=s.shapes(),
                             buckets=sorted({shape_bucket(s.op_class, x) for x in s.shapes()}),
                             slot_us=slot_us, baseline_us=baseline_us,
                             speedup=(baseline_us / slot_us) if baseline_us and slot_us else None,
                             seed_us=seed_us, speedup_vs_seed=(seed_us / slot_us) if seed_us and slot_us else None,
                             techniques=techniques, parent_id=parent, problem_id=self.problem_id,
                             run_id=self.run_id, slot=sid, max_rel_err=rel_err, dtype=self.p.dtype)
        return rec.id

    # ---- initialise: baseline + store seeds
    @tel.traced("init")
    async def initialize(self) -> EvalResult:
        seeds = self.p.seed_sources()
        base = await self.ev.evaluate(self.p, seeds, "baseline")
        if not base.ok:
            raise RuntimeError(f"baseline failed ({base.stage}): {base.error_text()}")
        self.baseline = base
        self.partition_us = base.partition_us
        for sid, code in seeds.items():
            self.best[sid] = SlotBest(code, base.slot_us.get(sid, 0.0), techniques=[])
        self.log.info(f"[{self.p.name}] baseline {base.partition_us:.1f} us; slots "
                      + ", ".join(f"{s}={u:.1f}" for s, u in sorted(base.slot_us.items(), key=lambda kv: -kv[1])))
        # seed from the implementation store where it has this op+shapes on this arch
        trial = dict(seeds)
        from_store = {}
        for sid in self.p.slots:
            r = self._retrieve(sid)
            if r["seed"] is not None and self.store is not None:
                trial[sid] = self.store.code(r["seed"])
                from_store[sid] = r["seed"]
        if from_store:
            self.log.info(f"[{self.p.name}] store seeds for {sorted(from_store)}")
            res = await self.ev.evaluate(self.p, trial, "store_seeds")
            for sid, rec in from_store.items():
                ok = res.ok or res.slot_checks.get(sid, {}).get("ok", False)
                us = res.slot_us.get(sid)
                if ok and us is not None and us < self.best[sid].slot_us:
                    self.log.info(f"[{self.p.name}] {sid}: store seed {rec.id} {us:.1f} us "
                                  f"(hand {self.best[sid].slot_us:.1f})")
                    self.best[sid] = SlotBest(trial[sid], us, rec.id, list(rec.techniques))
            if res.ok and res.partition_us:
                self.partition_us = min(self.partition_us, res.partition_us)
        await self._profile(self._sources(), "baseline")
        for sid, b in self.best.items():
            if b.impl_id is None:
                b.impl_id = self._store(sid, b.source, b.slot_us, None, [], None,
                                        base.slot_checks.get(sid, {}).get("rel_err"))
        return base

    # ---- one step
    async def _step(self, ridx: int, step: int, cur: dict[str, str], cur_us: dict[str, float]) -> None:
        """One step inside a telemetry ``step`` span; its breakdown goes into the step's context."""
        n = len(self.steps_log)
        with tel.span("step", collect=True, laps=True, rollout=ridx, step=step) as sp:
            tel.count("step")
            try:
                await self._step_body(ridx, step, cur, cur_us)
            finally:
                tel.end_laps(sp)
        summ = tel.ctx_summary(sp)
        if summ is not None:
            for s in reversed(self.steps_log[n:]):
                if s.rollout == ridx and s.step == step:
                    s.context["telemetry"] = summ
                    break

    async def _step_body(self, ridx: int, step: int, cur: dict[str, str], cur_us: dict[str, float]) -> None:
        t0 = time.time()
        tel.lap("knowledge.select")
        async with self._lock:
            sid = self._pick_slot()
            self.slot_tries[sid] += 1
        slot = self.p.slots[sid]
        state, nums = slot_state(self.p, sid, cur_us[sid], self.arch, self.tp.peaks_for(self.arch))
        prof_before = self._ncu_of(sid, cur[sid])
        if prof_before:
            state = prof_before.get("state") or state    # measured bound beats the roofline estimate
        choice = cls = vec_before = None
        if self.kv is not None:
            vec_before = self.kv.vector(self._tkey, estimate=self._roofline_input(sid, cur_us[sid]),
                                        profile=prof_before, profile_kind=self._profile_kind)
            cls = self.kv.classify(vec_before, target_key=self._tkey, op_class=slot.op_class, dtype=self.p.dtype)
            choice = self.kv.select(target_key=self._tkey, op_class=slot.op_class, dtype=self.p.dtype, cls=cls,
                                    vec=vec_before, rng=self.rng, op_kind=slot.op)
            tech, desc, state = choice.item_id, choice.desc, cls.label()
            nums = {**nums, **cls.decided_by}
        else:
            techs = techniques_for(slot.op_class, self.tp.techniques)
            # fp16 arms are their own keys (half2 / tensor-core techniques pay off
            # differently); fp32 keys keep their names, and the cross-key prior
            # still seeds an untried fp16 arm from its fp32 record
            key = f"{slot.op_class}|{state}" + ("" if self.p.dtype == "fp32" else f"|{self.p.dtype}")
            tech = self.bandit.select(key, [t for t, _ in techs], self.rng)
            desc = dict(techs)[tech]
        tel.lap("knowledge.lookup", slot=sid)
        r = self._retrieve(sid)
        examples = []
        if self.store is not None:
            examples = [(f"{e.op}, {e.speedup_vs_seed or 1:.1f}x over its seed, tuned for shape buckets "
                         f"{', '.join(e.buckets[:4])}" + ("; same op, different shapes: for ideas only, its "
                         "blocking may not suit these instances" if e.op == slot.op else ""),
                         self.store.code(e)) for e in r["examples"]]
        adapt = None
        if r["adapt"] is not None and self.store is not None:
            adapt = (f"{json.dumps(r['adapt'].shapes[:2])}", self.store.code(r["adapt"]))
        if self.kv is not None:
            ktext = self.kv.retrieve(target_key=self._tkey, op_class=slot.op_class, dtype=self.p.dtype, cls=cls,
                                     vec=vec_before, problem_id=self.problem_id, slot=sid, technique=tech,
                                     op_kind=slot.op)
            tried = self.kv.already_tried(self.problem_id, sid)
        else:
            ktext, tried = self._knowledge(sid, state, tech)
        tel.lap("prompt.build")
        part_us = sum(cur_us.values())
        neigh = [(n, self.p.slots[n].signature) for n in self.p.neighbours(sid)]
        best_note = (f"Best verified so far for this slot: {self.best[sid].slot_us:.1f} us."
                     if self.best[sid].slot_us < cur_us[sid] else "")
        msgs = build_codegen_messages(problem=self.p, sid=sid, source=cur[sid], technique=tech,
                                      technique_desc=desc, state=state, state_numbers=nums,
                                      slot_us=cur_us[sid], partition_us=part_us, neighbours=neigh,
                                      examples=examples, adapt=adapt, knowledge_text=ktext,
                                      already_tried=tried, best_note=best_note,
                                      profile_text=self.tp.profiler.render(prof_before) if prof_before else "")
        tdef = self.kv.ontology.techniques.get(tech) if self.kv is not None else None
        model, role, why = self.router.codegen(technique=tech, tier=technique_tier(tech, tdef),
                                               share=self._share(sid), stall=self.slot_stall[sid],
                                               kind=choice.kind if choice is not None else "technique")
        text = await self._llm(msgs, model, role)
        tel.lap("candidate", technique=tech)
        code = extract_code(text)
        fix_models = []
        fixes, res, status = 0, None, None
        before = cur_us[sid]
        while True:
            if code is None:
                res = EvalResult(ok=False, stage="compile", errors=[f"no ```{self.tp.fence} block in the response"])
            else:
                # only `sid` changed: isolated checks of its ops (the partition outputs are
                # always checked), screening budgets; a win is re-measured below in full
                res = await self.ev.evaluate(self.p, {**cur, sid: code}, f"r{ridx}_s{step}_{sid}_{tech}_f{fixes}",
                                             **step_eval_options(sid))
            if res.ok or fixes >= self.max_fix or code is None:
                break
            other_bad = [s for s in res.failing_slots() if s != sid]
            if other_bad:                                         # shouldn't happen: only `sid` changed
                break
            fixes += 1
            fmodel, frole = self.router.fix(attempt=fixes, max_attempts=self.max_fix)
            fix_models.append(fmodel)
            text = await self._llm(build_fix_messages(problem=self.p, sid=sid, source=code,
                                                      error=res.error_text(),
                                                      pitfalls=self._pitfalls(sid, tech)), fmodel, frole)
            new = extract_code(text)
            if not new or new == code:
                break
            code = new
        after = res.slot_us.get(sid) if res.ok else None
        if res.ok and after is not None and after < before * (1 - self.confirm_pct / 100):
            # confirm a win with a second, full measurement (every check; timing noise on shared GPUs)
            res2 = await self.ev.evaluate(self.p, {**cur, sid: code}, f"r{ridx}_s{step}_{sid}_confirm")
            if res2.ok and sid in res2.slot_us:
                after = max(after, res2.slot_us[sid])
        if res.ok:
            speedup = before / after if after else None
            status = classify_speedup(speedup)
            reward = max(-1.0, min(1.0, (before - after) / max(part_us, 1e-9)))
        else:
            speedup = None
            status = {"compile": COMPILE_ERROR, "run": RUNTIME_ERROR}.get(res.stage, WRONG_OUTPUT)
            reward = -0.05 if status == COMPILE_ERROR else -0.1
        tel.lap("knowledge.record")
        if self.kv is None:
            self.bandit.update(key, tech, reward)
        prof_after = None
        if res.ok and (self.ncu == "all" or (self.ncu == "best" and after < self.best[sid].slot_us *
                                                (1 - self.confirm_pct / 100))):
            by = await self._profile({**cur, sid: code}, f"r{ridx}_s{step}_{sid}")
            prof_after = by.get(sid)
        impl_id, parent = None, self.best[sid].impl_id
        dsum = diff_summary(cur[sid], code) if code else None
        if res.ok:
            impl_id = self._store(sid, code, after, before, self.best[sid].techniques + [tech], parent,
                                  res.slot_checks.get(sid, {}).get("rel_err"))
            if after <= before * 1.05:              # keep improving/neutral moves; revert regressions
                cur[sid], cur_us[sid] = code, after
        async with self._lock:
            if res.ok and after < self.best[sid].slot_us * (1 - self.confirm_pct / 100):
                self.best[sid] = SlotBest(code, after, impl_id, self.best[sid].techniques + [tech])
                self.slot_stall[sid] = 0
                tel.count("win")
                self.log.info(f"[{self.p.name}] r{ridx} s{step}: {sid} {before:.1f} -> {after:.1f} us "
                              f"via {tech} (new slot best)")
            else:
                self.slot_stall[sid] += 1
        okw = dict(sid=sid, technique=tech, status=status, state_before=state,
                   state_after=prof_after.get("state") if prof_after else None,
                   metric_before=before, metric_after=after, speedup=speedup,
                   metrics_delta=self.tp.profiler.delta(prof_before, prof_after),
                   error=None if res.ok else res.error_text(1500),
                   diff_summary=dsum, impl_id=impl_id, parent_impl_id=parent)
        v2 = None
        if self.kv is not None:
            vec_after = (self.kv.vector(self._tkey, estimate=self._roofline_input(sid, after), profile=prof_after,
                                        profile_kind=self._profile_kind) if res.ok and after else None)
            try:
                v2 = self.kv.record(self._outcome(**okw), choice=choice, reward=reward, cls=cls, before=vec_before,
                                    after=vec_after, op_class=slot.op_class, dtype=self.p.dtype,
                                    extra={"lineage": list(self.best[sid].techniques),
                                           "ncu_rules": (prof_before or {}).get("rules", []),
                                           "diff_excerpt": diff_excerpt(cur[sid], code) if code else ""})
            except Exception as e:                                # knowledge must never break a step
                self.log.warning(f"knowledge-v2 record failed: {e!r}")
        else:
            self._record_outcome(**okw)
        ctx = {"adapt": r["adapt"].id if r["adapt"] is not None else None,
               "examples": [e.id for e in r["examples"]],
               "claims": ktext.count("\n") + 1 if ktext else 0, "already_tried": bool(tried),
               "ncu": bool(prof_before), "state": state, "model": model, "model_role": role,
               "model_why": why}
        if fix_models:
            ctx["fix_models"] = fix_models
        if v2 is not None:
            ctx.update(kind=choice.kind, est=round(choice.est, 4), evidence=choice.evidence,
                       explore=choice.explore, mechanism=v2["mechanism"]["label"], chain=cls.chain)
        self.steps_log.append(StepLog(ridx, step, sid, tech, status, before, after, reward, fixes,
                                      round(time.time() - t0, 1), ctx))
        self.log.info(f"[{self.p.name}] r{ridx} s{step}: {sid} {tech} -> {status}"
                      + (f" {before:.1f}->{after:.1f} us" if after else "") + f" (fixes={fixes})")

    async def _rollout(self, ridx: int):
        async with self._lock:
            cur = {s: b.source for s, b in self.best.items()}
            cur_us = {s: b.slot_us for s, b in self.best.items()}
        for step in range(self.steps):
            try:
                await self._step(ridx, step, cur, cur_us)
            except Exception as e:
                self.log.error(f"[{self.p.name}] rollout {ridx} step {step} failed: {e!r}")
            if self.semantic is not None:
                try:
                    await self.semantic.maybe_reflect()
                except Exception as e:
                    self.log.warning(f"semantic update failed: {e!r}")
            elif self.kv is not None:
                try:
                    await self.kv.maybe_upkeep()
                except Exception as e:
                    self.log.warning(f"knowledge-v2 upkeep failed: {e!r}")

    # ---- run
    async def run(self) -> dict:
        """``_run`` inside a telemetry ``problem`` span; result.json gets its breakdown."""
        with tel.span("problem", collect=True, problem=self.problem_id) as sp:
            result = await self._run()
        summ = tel.full_summary(sp)
        if summ is not None:
            result["telemetry"] = summ
            (self.out / "result.json").write_text(json.dumps(result, indent=1, default=str))
        return result

    async def _run(self) -> dict:
        if self.baseline is None:
            await self.initialize()
        await asyncio.gather(*[self._rollout(i) for i in range(self.rollouts)])
        final = await self.ev.evaluate(self.p, self._sources(), "final_composite")
        if not final.ok:
            # a composition broke something: fall back slot by slot to the baseline seed
            self.log.warning(f"[{self.p.name}] composite failed ({final.error_text(300)}); repairing")
            seeds = self.p.seed_sources()
            for sid in final.failing_slots() or list(self.best):
                self.best[sid] = SlotBest(seeds[sid], self.baseline.slot_us.get(sid, 0.0))
            final = await self.ev.evaluate(self.p, self._sources(), "final_repaired")
        result = self._write_result(final)
        if final.ok and self.propose:
            props = propose_fusions(self.p, final, self.arch, arch_label=self.arch_label)
            props["proposals"] += propose_splits(self.p, final, self.arch)
            if self.kv is not None:                  # graph-level hypotheses (ids = MB graph_hints ids)
                self.kv.hypotheses.from_proposals(props, self._tkey, self.kv.review_path)
            (self.out / "proposals.json").write_text(json.dumps(props, indent=1))
            result["proposals"] = [{"kind": x["kind"], "pattern": x.get("pattern") or [x["slot"]],
                                    "instances": len(x.get("pairs") or x.get("groups")),
                                    "est_gain_pct": x["evidence"]["est_gain_pct"]} for x in props["proposals"]]
            (self.out / "result.json").write_text(json.dumps(result, indent=1))
        if self.semantic is not None:
            try:
                await self.semantic.maybe_reflect(force=True)
            except Exception as e:
                self.log.warning(f"final semantic update failed: {e!r}")
        elif self.kv is not None:
            try:
                await self.kv.maybe_upkeep(force=True)
            except Exception as e:
                self.log.warning(f"final knowledge-v2 upkeep failed: {e!r}")
        if self.kv is not None:
            result["knowledge"] = self.kv.summary()
            (self.out / "result.json").write_text(json.dumps(result, indent=1, default=str))
        if self.claims is not None and hasattr(self.claims, "save"):
            self.claims.save()
        return result

    @tel.traced("io.result")
    def _write_result(self, final: EvalResult) -> dict:
        best_dir = self.out / "best" / "slots"
        best_dir.mkdir(parents=True, exist_ok=True)
        for sid, b in self.best.items():
            (best_dir / f"{sid}{self.tp.slot_ext}").write_text(b.source)
        base = self.baseline
        res = {
            "problem": self.p.name, "problem_dir": str(self.p.root), "run_id": self.run_id,
            "arch": self.arch_label, "target": self.p.target, "model": self.model, "models": self.router.describe(),
            "llm_calls": self.llm_calls,
            "llm_usage": {**self.llm_usage, "cost_usd": round(self.llm_usage["cost_usd"], 4)},
            "llm_usage_by_model": sorted(self.llm_usage_by.values(), key=lambda d: (d["role"], d["model"])),
            "ncu": {"mode": self.ncu, "runs": self.ncu_runs, "disabled": self.ncu_disabled,
                    "final": {sid: self._ncu_of(sid, b.source) for sid, b in self.best.items()
                              if self._ncu_of(sid, b.source)}},
            "baseline_partition_us": base.partition_us, "final_ok": final.ok,
            "final_partition_us": final.partition_us, "op_us": final.op_us,
            "speedup": (base.partition_us / final.partition_us) if final.ok and final.partition_us else None,
            "slots": {sid: {"baseline_us": base.slot_us.get(sid), "final_us": final.slot_us.get(sid),
                            "best_us": b.slot_us, "impl_id": b.impl_id, "techniques": b.techniques,
                            "rel_err": final.slot_checks.get(sid, {}).get("rel_err")}
                      for sid, b in self.best.items()},
            "steps": [asdict(s) for s in self.steps_log],
            "bandit": self.bandit.to_json(),
        }
        (self.out / "result.json").write_text(json.dumps(res, indent=1))
        return res
