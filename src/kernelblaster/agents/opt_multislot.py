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
import copy
import difflib
import json
import math
import os
import random
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger as _default_logger

from .. import telemetry as tel
from .. import trace
from ..knowledge import ncu as ncu_mod
from ..knowledge.impl_store import ImplStore, shape_bucket
from ..knowledge.outcomes import (COMPILE_ERROR, IMPROVED, NEUTRAL, REGRESSED, RUNTIME_ERROR, WRONG_OUTPUT,
                                  OutcomeLog, OutcomeRecord, classify_speedup)
from ..knowledge.slots import EvalResult, Evaluator, SlotProblem, step_eval_options
from ..knowledge.clock_guard import clock_suspect
from . import hw_profile
from . import experiment_tool
from ..knowledge.v2 import experiments
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
    "100": (80000.0, 8000.0),     # B200
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
    # sequence recurrences (gru_seq / lstm_seq) with ModelBlaster --recurrent-class on: latency per timestep, not
    # tiling or DRAM, bounds them (techniques/45_recurrent.json has the v2 entries)
    "recurrent": [
        ("persistent_recurrence_state_on_chip", "One persistent launch for all timesteps; each block keeps its W_hh "
                                                "slice resident (registers as mma fragments, or smem) and its h slice "
                                                "on chip; only the h exchange goes through L2."),
        ("recurrence_input_projection_hoisting", "Compute X W_ih^T + b for all timesteps as one tensor-core GEMM "
                                                 "(before or overlapped with the recurrence); each serial step only "
                                                 "does h W_hh^T + gates."),
        ("step_sync_minimization", "Replace grid-wide barriers by point-to-point readiness: publish h_t slices as "
                                   "tagged {value, step} words (release store), readers poll only what they need "
                                   "(acquire load); replicate the exchange buffer for coalesced reads."),
        ("layer_wavefront_pipeline", "Use few blocks per recurrence (size the grid with mb_sm_budget() when present) "
                                     "so concurrent siblings (other direction / layer) fit beside it; overlap "
                                     "independent recurrences."),
        ("recurrent_gate_fast_math", "Gate math is on the serial critical path: __expf / tanh.approx.f32, one "
                                     "(batch, unit) per thread, fp32 accumulation."),
    ],
    "attention": [
        ("flash_style_tiling", "Tile Q/K/V through shared memory with an online softmax so the score matrix is never materialised."),
        ("eliminate_composite_intermediates", "If the seed is a COMPOSITE (copies + bmm + softmax + bmm called back to "
                                              "back with a workspace), replace it by one kernel: read q/k/v in place "
                                              "through their strides, keep scores in registers, write O through its "
                                              "strides; drop the workspace."),
    ],
}


# Offered (v1 bandit) when the slot or its partition is launch-bound: the end-to-end
# reward credits fewer launches, which the slot's own kernel time does not show.
LAUNCH_TECHNIQUES: list[tuple[str, str]] = [
    ("kernel_fusion_in_slot", "Fuse the kernels this slot launches per call into one (fold the separate init / bias / "
                              "activation / second pass into the main kernel): fewer launches and no intermediate "
                              "DRAM round trip."),
    ("persistent_kernel", "Replace a host loop of many small launches (timesteps, layers, chunks) with one persistent "
                          "kernel or a few large launches (grid-stride work loop, in-kernel synchronisation)."),
]


def techniques_for(op_class: str, techniques: Optional[dict] = None,
                   launch_bound: bool = False, arch: Optional[str] = None,
                   dtype: str = "fp32") -> list[tuple[str, str]]:
    techniques = techniques or TECHNIQUES
    extra = LAUNCH_TECHNIQUES if launch_bound and techniques is TECHNIQUES else []
    # tensor-core / arch-specific GEMM techniques (hw_profile) go first: they are what the fast kernels use
    tc = hw_profile.tensor_core_techniques(arch, dtype, op_class) if arch and techniques is TECHNIQUES else []
    return tc + techniques.get(op_class, []) + techniques["*"] + extra


# ---------------------------------------------------------------- roofline

# element bytes per manifest buffer dtype (fp16 problems move half the bytes)
BUFFER_BYTES = {"f32": 4, "f16": 2, "i32": 4, "i64": 8}


def _is_infra(text: Optional[str]) -> bool:
    from ..servers.utils.resources import is_infra_failure
    return bool(text) and (is_infra_failure(text) or "MB_ERROR [infra]" in text)


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
    elif slot.op_class == "recurrent":                  # G gates x (input + hidden) GEMV per step and batch row
        g = 4 if "lstm" in slot.op else 3
        flops = 2.0 * sh.get("T", 1) * sh.get("B", 1) * g * sh.get("H", 1) * (sh.get("I", 0) + sh.get("H", 1))
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
    rec = problem.slots[sid].op_class == "recurrent"
    if slot_us < 8.0:
        state = "latency_bound"
    elif rec and max(pb, pf) < 0.5:                     # T dependent steps: per-step latency bounds it
        state = "latency_bound"
    elif max(pb, pf) < 0.15:
        state = "underutilized_compute" if ai > ridge else "underutilized_memory"
    else:
        state = "compute_bound" if pf >= pb else "memory_bound"
    nums = {"bytes": byt, "flops": fl, "arith_intensity": round(ai, 2),
            "achieved_GBps": round(gbs, 1), "achieved_GFLOPs": round(gfs, 1),
            "pct_peak_bw": round(100 * pb, 1), "pct_peak_flops": round(100 * pf, 1)}
    if rec:
        steps = sum(int(primary_shape(i.get("shape", {})).get("T", 1)) for i in problem.slots[sid].instances)
        nums.update(serial_steps=steps, us_per_step=round(slot_us / max(steps, 1), 3))
    return state, nums


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
    part_us = result.deployed_us or sum(result.slot_us.values()) or 1.0
    # a consumer instance's share of the deployed time: its kernels' GPU time
    # (attributed where the driver measured it) plus its launches' overhead
    w = slot_weights(result)
    per_inst = {sid: w.get(sid, 0.0) / max(len(s.instances), 1) for sid, s in problem.slots.items()}
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
                         "consumer_slot_us": result.slot_us.get(problem.slots[ops[g["pairs"][0][1]]["slot"]].id),
                         **({"e2e_basis": result.e2e_basis, "overhead_us_per_launch": round(overhead_per_launch(result), 3),
                             "launch_bound": partition_runtime(result)["launch_bound"]} if result.attr_slot_us else {})},
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
            "(`wmma`, `mma.sync`, and where the target hardware below has them `wgmma` / `tcgen05`, with fp16 "
            "inputs and fp32 accumulators) are allowed.\n"
            "- Each op is checked alone against a reference that rounds the exact fp32 result to fp16 "
            "once: max error <= {slot_rel} relative to the largest input/output magnitude (about two fp16 "
            "ulps).",
}


def system_prompt(dtype: str = "fp32", tolerance: Optional[dict] = None, arch: Optional[str] = None) -> str:
    rel = (tolerance or {}).get("slot_rel", 1e-3)
    rel_s = f"{rel:.0e}".replace("e-0", "e-")                    # 1e-3, 2e-3
    rules = DTYPE_RULES.get(dtype, DTYPE_RULES["fp32"]).format(slot_rel=rel_s)
    hw = hw_profile.hardware_text(arch, dtype) if arch else ""
    return SYSTEM_PROMPT_TEMPLATE.format(dtype_rules=rules) + (f"\n\n## Target hardware\n{hw}" if hw else "")


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

    def prompt_for(self, problem, arch: Optional[str] = None) -> str:
        if self.name == "cuda":                       # dtype rules + the manifest's isolated bound + the target GPU
            return system_prompt(problem.dtype, problem.tolerance, arch)
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


def build_codegen_messages(*, problem: SlotProblem, sid: str, source: str, technique: str, arch: Optional[str] = None,
                           technique_desc: str, state: str, state_numbers: dict, slot_us: float,
                           partition_us: float, neighbours: list[tuple[str, str]],
                           examples: list[tuple[str, str]], adapt: Optional[tuple[str, str]],
                           knowledge_text: str, already_tried: str, best_note: str,
                           profile_text: str = "", runtime_text: str = "") -> list[dict]:
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
        runtime_text,
        profile_text,
        best_note,
        f"C reference implementation (semantics oracle, not fast):\n```c\n{ref[:4000]}\n```" if ref else "",
        hw_profile.recurrent_text(arch) if slot.op_class == "recurrent" and tp.name == "cuda" else "",
        experiment_tool.model_context_text(problem.manifest),     # ModelBlaster --model-context (absent: "")
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
    return [{"role": "system", "content": tp.prompt_for(problem, arch)},
            {"role": "user", "content": "\n\n".join(p for p in parts if p)}]


def build_fix_messages(*, problem: SlotProblem, sid: str, source: str, error: str,
                       pitfalls: str = "", arch: Optional[str] = None) -> list[dict]:
    slot = problem.slots[sid]
    tp = target_profile(problem.target)
    known = f"Known pitfalls from earlier runs:\n{pitfalls}\n\n" if pitfalls else ""
    return [{"role": "system", "content": tp.prompt_for(problem, arch)},
            {"role": "user", "content":
             f"The slot `{sid}` (op `{slot.op}`, signature `{slot.signature}`) failed:\n```\n{error}\n```\n\n"
             f"Instances:\n{_fmt_instances(slot)}\n\n{known}Fix it (keep the optimisation if you can). "
             f"Current source:\n```{tp.fence}\n{source}\n```"}]


@tel.traced("extract")
def extract_code(text: str) -> Optional[str]:
    blocks = re.findall(r"```(?:cuda|cpp|c\+\+|cu|c)?\s*\n(.*?)```", text, re.S)
    code = max(blocks, key=len).strip() + "\n" if blocks else None
    trace.on_extract(text, code)              # the step's candidate (or that the response had none)
    return code


def diff_excerpt(a: str, b: str, max_lines: int = 80) -> str:
    """The first lines of a unified diff (code mining / state discovery context)."""
    d = list(difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=1))
    return "\n".join(d[2:2 + max_lines])


def diff_summary(a: str, b: str) -> str:
    d = list(difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0))
    add = sum(1 for l in d if l.startswith("+") and not l.startswith("+++"))
    rm = sum(1 for l in d if l.startswith("-") and not l.startswith("---"))
    return f"+{add}/-{rm} lines"


# ---------------------------------------------------------------- end-to-end reward

REWARDS = ("e2e", "slot")
LAUNCH_BOUND_FRAC = 0.15        # eager partition time not spent in kernels
TINY_KERNEL_US = 5.0


def e2e_gain(res: EvalResult, before_us: Optional[float]) -> tuple[Optional[float], str]:
    """Fraction of the deployed partition time a candidate saves (> 0: faster).

    Paired: ``1 - median(candidate / incumbent)`` over reps timed alternately in
    one process (the persistent host's ``ref=``). Else against ``before_us``,
    the incumbent's time from an earlier run."""
    r = res.e2e_ratio
    if r and r.get("n") and r.get("med"):
        return 1.0 - r["med"], "paired"
    after = res.deployed_us
    if after and before_us:
        return (before_us - after) / before_us, "unpaired"
    return None, "none"


def overhead_per_launch(res: EvalResult) -> float:
    """Deployed time not spent in kernels, per launch (0 without attribution)."""
    if not res.attr_sum_us or not res.attr_launches or not res.deployed_us:
        return 0.0
    return max(0.0, res.deployed_us - res.attr_sum_us) / max(res.attr_launches, 1.0)


def slot_weights(res: EvalResult) -> dict:
    """Each slot's share of the deployed time: its kernels' GPU time plus its
    launches' share of the launch/idle overhead (the slot's attributed time
    alone where there is no attribution)."""
    per = overhead_per_launch(res)
    return {s: (res.slot_time(s) or 0.0) + per * res.attr_slot_launches.get(s, 0.0) for s in res.slot_times()}


def partition_runtime(res: EvalResult) -> dict:
    """The deployed-runtime numbers of a run (result.json, prompts, outcome records)."""
    eager = res.partition_us
    out = {"runtime": res.runtime, "e2e_us": res.deployed_us, "e2e_basis": res.e2e_basis or "eager",
           "eager_us": eager, "graph_us": res.partition_graph_us, "graph_skip": res.graph_skip,
           "kernel_us": res.attr_sum_us, "launches": res.attr_launches, "graph_nodes": res.graph_nodes,
           "launch_overhead_us": res.launch_overhead_us, "launch_overhead_frac": res.launch_overhead_frac,
           "attribution": "cupti" if res.attr_slot_us else ("events" if res.slot_us else None),
           "attr_skip": res.attr_skip}
    if eager and res.partition_graph_us:
        out["graph_saving_frac"] = round(1.0 - res.partition_graph_us / eager, 4)
    out["launch_bound"] = bool((res.launch_overhead_frac or 0.0) >= LAUNCH_BOUND_FRAC
                               or (out.get("graph_saving_frac") or 0.0) >= 0.05)
    return out


def slot_launch_bound(res: Optional[EvalResult], sid: str, problem: SlotProblem) -> bool:
    """The slot launches more than once per call, or its kernels are tiny."""
    if res is None or sid not in res.attr_slot_launches:
        return False
    n = res.attr_slot_launches[sid]
    calls = max(len(problem.slots[sid].instances), 1)
    t = res.attr_slot_us.get(sid, 0.0)
    return n > 1.5 * calls or (n > 0 and t / n < TINY_KERNEL_US)


def runtime_text(res: Optional[EvalResult], sid: str, problem: SlotProblem) -> str:
    """Prompt lines: how the partition runs end to end and what this slot's launches cost."""
    if res is None or not res.deployed_us:
        return ""
    rt = partition_runtime(res)
    how = "CUDA-graph replay of the whole partition" if rt["e2e_basis"] == "graph" else "eager launches"
    lines = [f"Reward: the END-TO-END partition time as deployed ({how}): {rt['e2e_us']:.1f} us per forward pass."]
    if rt["kernel_us"] is not None and rt["launches"]:
        lines.append(f"Of that, kernels take {rt['kernel_us']:.1f} us in {rt['launches']:.0f} launches; launch and idle "
                     f"overhead is {rt['e2e_us'] - rt['kernel_us']:.1f} us as deployed and "
                     f"{rt['launch_overhead_us'] or 0.0:.1f} us ({100 * (rt['launch_overhead_frac'] or 0.0):.0f}%) "
                     f"when launched eagerly.")
    n = res.attr_slot_launches.get(sid)
    if n is not None:
        calls = max(len(problem.slots[sid].instances), 1)
        lines.append(f"This slot: {res.attr_slot_us.get(sid, 0.0):.1f} us of GPU time in {n:.0f} launches "
                     f"({n / calls:.1f} per call).")
    if rt["launch_bound"] or slot_launch_bound(res, sid, problem):
        lines.append("Launch-bound: fewer launches count as much as faster kernels here. Fuse what this slot "
                     "launches per call into one kernel, replace host loops of small launches with a persistent "
                     "kernel, and keep the launch configuration small-problem friendly.")
    if rt["e2e_basis"] == "graph":
        lines.append("Keep the slot capturable in a CUDA graph (no host sync, allocation or host copy per call): a "
                     "slot that breaks capture makes the whole partition fall back to eager launches.")
    return "\n".join(lines)


# ---------------------------------------------------------------- agent

@dataclass
class SlotBest:
    source: str
    slot_us: float                      # the slot's time: attributed GPU time (e2e reward) or MB_SLOT_US (slot reward)
    impl_id: Optional[str] = None
    techniques: list = field(default_factory=list)
    event_us: Optional[float] = None    # MB_SLOT_US (events around each op), for the per-slot comparison


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
    reward_basis: str = "slot"                       # "e2e": end-to-end partition time; "slot": slot time
    e2e_before: Optional[float] = None               # deployed partition time of the state the step started from
    e2e_after: Optional[float] = None                # ... with the candidate (paired: before x ratio)


def _sf(s, k):
    return s.get(k) if isinstance(s, dict) else getattr(s, k, None)


def patience_trigger(steps: list, patience: int) -> Optional[dict]:
    """E7 patience rule (l3_farm/analysis/efficiency_rules.md) over committed steps in commit order
    (``steps_log`` order: across the parallel rollouts, as each step committed).

    A step sets a *new best* iff its status is ``improved`` and its ``slot_us_after`` is below the running best
    of its slot; the running best starts at the slot's time the rollouts started from (``slot_us_before`` of the
    slot's first committed step-0 step, else of its first committed step) -- exactly the replay's definition
    (e7_build.py / e7_analyze.py). Every other committed step is a *miss*: neutral, regressed, an improved step
    that does not beat the best, and compile / run / wrong-output failures after their fix attempts (a step with
    its fix attempts is one step). Steps that raised (not in the log) do not count. -> {"trigger_commit": n
    (1-based commit number of the ``patience``-th consecutive miss), "misses": patience} or None."""
    if not patience or patience <= 0:
        return None
    base: dict = {}
    for s in steps:                                   # slot baselines: first step-0 commit, else first commit
        sid = _sf(s, "slot")
        if _sf(s, "step") in (0, "0") and sid not in base and _sf(s, "slot_us_before") is not None:
            base[sid] = float(_sf(s, "slot_us_before"))
    for s in steps:
        sid = _sf(s, "slot")
        if sid not in base and _sf(s, "slot_us_before") is not None:
            base[sid] = float(_sf(s, "slot_us_before"))
    best, miss = dict(base), 0
    for i, s in enumerate(steps):
        sid, after = _sf(s, "slot"), _sf(s, "slot_us_after")
        after = float(after) if after not in (None, "") else None
        if _sf(s, "status") == IMPROVED and after is not None and after < best.get(sid, float("inf")):
            best[sid], miss = after, 0
        else:
            miss += 1
            if miss >= patience:
                return {"trigger_commit": i + 1, "misses": miss}
    return None


# ---- rollout acceptance and lineage (ModelBlaster search fixes, l3_farm/analysis/search_fixes.md item 4)
# KERNELBLASTER_MULTISLOT_ROLLOUT_ACCEPT:
#   composite (default): a step's decision uses g = min(gain over the rollout's own state, confirmed gain over the
#             composite best): a rollout that lags the composite cannot keep an improvement that does not beat it.
#   local:    status / reward / keep use the gain over the rollout's own state, so the rollout continues from its own
#             improvement (its lineage); merging into the composite still needs the confirmed composite gain (promotion
#             = the existing confirm + merge when the lineage overtakes the composite).
#   lineage:  local, plus resync (default factor 1.5): see should_resync.
# KERNELBLASTER_MULTISLOT_ROLLOUT_RESYNC=X (0 / unset = off, except lineage): before a step, a rollout whose own
#   state is more than X times slower end to end than the composite best AND whose previous step did not improve its
#   own state (stalled) restarts from the composite best. Patience (E7) is unchanged: it counts committed steps that
#   set no new SLOT BEST, so a local lineage improvement that does not beat the best is a miss.
ROLLOUT_MODES = ("composite", "local", "lineage")
LINEAGE_RESYNC = 1.5


def rollout_mode() -> str:
    m = (os.getenv("KERNELBLASTER_MULTISLOT_ROLLOUT_ACCEPT") or "composite").strip().lower()
    return m if m in ROLLOUT_MODES else "composite"


def rollout_resync_factor(mode: Optional[str] = None) -> float:
    v = os.getenv("KERNELBLASTER_MULTISLOT_ROLLOUT_RESYNC")
    if v not in (None, ""):
        try:
            return max(0.0, float(v))
        except ValueError:
            return 0.0
    return LINEAGE_RESYNC if (mode or rollout_mode()) == "lineage" else 0.0


def rollout_decision(gain: Optional[float], g: float, acc: float, mode: str = "composite") -> tuple:
    """(gr, status, keep) of a verified candidate: ``gain`` = gain over the rollout's own state (None: unpaired),
    ``g`` = min(gain, confirmed composite gain) as measured (= gain when no confirm ran), ``acc`` = the accept
    fraction. composite decides on g; local / lineage on the rollout's own gain."""
    gr = gain if (mode in ("local", "lineage") and gain is not None) else g
    status = IMPROVED if gr >= acc else REGRESSED if gr <= -acc else NEUTRAL
    return gr, status, gr > -acc / 2


def should_resync(state_e2e: Optional[float], comp_e2e: Optional[float], factor: float, stalled: bool) -> bool:
    """Restart a rollout from the composite best: its state lags the composite by more than ``factor`` x end to end
    and its last step did not improve its own state."""
    return bool(factor and factor > 0 and stalled and state_e2e and comp_e2e and state_e2e > comp_e2e * factor)


class MultiSlotAgent:
    def __init__(self, problem: SlotProblem, evaluator: Evaluator, *, model: str, out_dir: Path,
                 arch: str = "75", impl_store: Optional[ImplStore] = None,
                 outcome_log: Optional[OutcomeLog] = None, claims: Any = None, semantic_updater: Any = None,
                 rollouts: int = 4, steps: int = 6, max_fix_attempts: int = 2, min_share: float = 0.01,
                 confirm_pct: float = 3.0, stall_decay: float = 0.6, bandit: Optional[TechniqueBandit] = None,
                 model_fix: Optional[str] = None, seed: int = 0, logger=None, run_id: Optional[str] = None,
                 problem_id: Optional[str] = None, propose: bool = True, ncu: str = "off",
                 knowledge_v2: Any = None, router: Optional[ModelRouter] = None, reward: Optional[str] = None,
                 e2e_accept_pct: Optional[float] = None, journal: Any = None,
                 stop_event: Optional[asyncio.Event] = None, on_checkpoint: Any = None,
                 stop_patience: int = 0):
        self.p, self.ev, self.model = problem, evaluator, model
        # E7 patience (patience_trigger): stop starting steps after this many consecutive committed steps
        # without a new slot best; 0 = off (the class default; run_multislot.py defaults to 8, --efficiency-rules)
        self.stop_patience = max(0, int(stop_patience or 0))
        self.patience_stop: Optional[dict] = None
        # what a step is rewarded and accepted on: "e2e" (default; KERNELBLASTER_MULTISLOT_REWARD):
        # the deployed partition time (CUDA graph where capturable), paired with the incumbent
        # where the evaluator can; "slot": the changed slot's own time, as before
        self.reward = (reward or os.getenv("KERNELBLASTER_MULTISLOT_REWARD", "e2e")).lower()
        if self.reward not in REWARDS:
            raise ValueError(f"reward must be one of {REWARDS}, not {self.reward!r}")
        # an end-to-end gain counts from this percentage of the partition (and is re-measured
        # in full against the composite before it joins it); smaller moves are neutral
        self.e2e_accept_pct = float(os.getenv("KERNELBLASTER_MULTISLOT_E2E_ACCEPT_PCT", "1.0")) \
            if e2e_accept_pct is None else e2e_accept_pct
        self.e2e_us: Optional[float] = None             # deployed time of the composite best
        self.runtime_info: dict = {}                    # partition_runtime() of the latest composite run
        self.last_full: Optional[EvalResult] = None     # latest full run of the composite (attribution)
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
        self.seed = seed
        self.log = logger or _default_logger
        # resume (agents/journal.py): the snapshot + LLM responses of this problem, the
        # run's stop request (SIGTERM / SIGINT) and a hook called after every checkpoint
        self.journal, self.stop_event, self.on_checkpoint = journal, stop_event, on_checkpoint
        prev = (journal.state or {}) if journal is not None else {}
        self.run_id = prev.get("run_id") or run_id or uuid.uuid4().hex[:8]
        self.segment = int(prev.get("segment", -1)) + 1 if prev else 0
        self.resumed = bool(prev)
        self._rstate: dict[int, dict] = {}              # rollout -> state after its last completed step
        self._inflight: dict[str, dict] = {}            # "r,s" -> the slot / technique the step picked
        self._replay: dict[str, dict] = {}              # in-flight plans restored from the journal
        self._counted: set = set()                      # LLM journal keys whose usage is in llm_usage
        self._existing: dict[str, set] = {"impl": set(), "outcome": set()}   # ids recorded before a kill
        self.llm_replayed = 0                           # responses taken from the journal (not re-paid)
        self.resume_info: dict = {}
        self.phase: Optional[str] = None                # journal phase: init | rollouts | final | done
        self.segments: list = []
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
        """The slot's share of the partition: of the deployed time (kernels plus its
        launches' overhead) with the end-to-end reward, of the slot times otherwise."""
        if self.reward == "e2e" and self.last_full is not None and self.last_full.attr_slot_us:
            w = slot_weights(self.last_full)
            tot = sum(w.values()) or 1.0
            return w.get(sid, 0.0) / tot
        tot = sum(b.slot_us for b in self.best.values()) or 1.0
        return self.best[sid].slot_us / tot

    def _slot_us(self, res: EvalResult, sid: str) -> Optional[float]:
        """The slot time the agent tracks: attributed (e2e reward) or event-timed (slot reward)."""
        return res.slot_time(sid) if self.reward == "e2e" else res.slot_us.get(sid)

    def _eval_kw(self, incumbent: dict) -> dict:
        """Pair the run with the incumbent where the evaluator can (end-to-end reward)."""
        pairs = getattr(self.ev, "pairs_incumbent", None)
        if self.reward != "e2e" or pairs is None or not pairs(self.p):
            return {}
        if not hasattr(self, "_takes_incumbent"):       # evaluators written before the parameter existed
            import inspect
            try:
                self._takes_incumbent = "incumbent" in inspect.signature(self.ev.evaluate).parameters
            except (TypeError, ValueError):
                self._takes_incumbent = False
        return {"incumbent": incumbent} if self._takes_incumbent else {}

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
        # A5 (opt-in): KERNELBLASTER_IMPL_SIMILAR=K adds pool-based similarity retrieval (ImplStore.retrieve similar_k)
        k_sim = int(os.environ.get("KERNELBLASTER_IMPL_SIMILAR", "0") or 0)
        return self.store.retrieve(target=self.p.target, arch=self.arch_label, op=s.op, op_class=s.op_class,
                                   signature=s.signature, shapes=s.shapes(), exclude_problem=self.problem_id,
                                   dtype=self.p.dtype, **({"similar_k": k_sim} if k_sim > 0 else {}))

    def _impl_links_text(self, sid: str, r: dict) -> str:
        """A5 (opt-in, KERNELBLASTER_IMPL_LINKS=1): claims / hypotheses / negative results linked to the retrieved
        and current implementations (knowledge/impl_links.py), capped at KERNELBLASTER_IMPL_LINKS_CHARS (600)."""
        if os.environ.get("KERNELBLASTER_IMPL_LINKS", "0") in ("", "0", "off") or self.kv is None:
            return ""
        try:
            from ..knowledge.impl_links import ImplLinks
            lk = getattr(self, "_impl_links", None)
            if lk is None or time.time() - lk.built > 60:
                lk = self._impl_links = ImplLinks.from_db(self.kv.root / "knowledge.db")
            ids, labels = [], {}
            for tag, rec in (("current best", self.best[sid] if sid in self.best else None), ("adapt", r.get("adapt"))):
                iid = getattr(rec, "impl_id", None) or getattr(rec, "id", None)
                if iid:
                    ids.append(iid)
                    labels[iid] = tag
            for e in r.get("examples") or []:
                ids.append(e.id)
                labels.setdefault(e.id, "example")
            return lk.render(ids, labels=labels,
                             max_chars=int(os.environ.get("KERNELBLASTER_IMPL_LINKS_CHARS", "600") or 600))
        except Exception:                                   # noqa: BLE001 -- knowledge must never block a step
            return ""

    async def _llm(self, messages, model, role: str = "codegen", key: Optional[str] = None,
                   digest: Optional[str] = None, extra: Optional[dict] = None) -> str:
        """One LLM call. With a journal and ``key``, a response journaled under ``key``
        (and ``digest``, where given) is returned instead of calling the model again; a
        new response is journaled before it is returned (so before its candidate is
        evaluated). Usage is counted once per key across segments."""
        jr = self.journal
        if jr is not None and key is not None:
            hit = jr.llm_get(key)
            if hit is not None and (digest is None or hit.get("digest") == digest):
                self.llm_replayed += 1
                tel.count("llm.replayed")
                trace.llm_replayed(key, hit, role=role, model=model)   # not stored as a second call
                if key not in self._counted:          # paid in a segment that died before its checkpoint
                    self._count_usage(hit.get("usage") or {}, hit.get("model", model), hit.get("role", role), key)
                return hit.get("text", "")
        with tel.attrs(role=role):            # telemetry: the llm.<role> span + token record
            resp = await generate_code_retry(messages, model, self.log, n_tasks=1)
        u = getattr(resp, "usage", None) or {}
        if not isinstance(u, dict):
            u = {k: getattr(u, k, 0) for k in ("prompt_tokens", "completion_tokens")}
        text = resp.generations[0] if resp.generations else ""
        if jr is not None and key is not None:
            jr.llm_put(key, {"model": model, "role": role, "digest": digest, "text": text, **(extra or {}),
                             "usage": {k: v for k, v in u.items() if isinstance(v, (int, float))}})
            tel.flush()                        # its telemetry record is on disk with it
        self._count_usage(u, model, role, key)
        return text

    async def _multiturn(self, msgs, model, role, jkey, sid, tech, extra=None) -> str:
        """One codegen step as a bounded conversation with the experiment tool (agents/experiment_tool.py)."""
        if getattr(self, "_xtool", None) is None:
            self._xtool = experiment_tool.ExperimentTool(self.ev, self.p, self.arch, self.out / "experiments")
            self.mt_stats = {"steps": 0, "turns": 0, "runs": 0, "refused": 0, "compile_failed": 0, "seconds": 0.0,
                             "forced_final": 0, "budget": experiment_tool.budget()}
        b = experiment_tool.budget()

        async def llm(conv, key, extra=None):
            return await self._llm(conv, model, role, key=key, extra=extra)
        rec_on = experiments.record_enabled()                  # X-145 (default off)
        xtext, fact_ids = self._experiment_prompt(sid, tech) if rec_on or experiments.prompt_n() else ("", [])
        kw = {"record": True, "extra_prompt": xtext} if (rec_on or xtext) else {}
        text, log = await experiment_tool.converse(llm, msgs, self._xtool, b, key=f"{jkey}c0", extra=extra, **kw)
        st = self.mt_stats
        st["steps"] += 1
        for k in ("turns", "runs", "refused", "compile_failed"):
            st[k] += getattr(log, k)
        st["seconds"] = round(st["seconds"] + log.seconds, 2)
        st["forced_final"] += int(log.forced_final)
        with open(self.out / "multiturn.jsonl", "a") as f:
            f.write(json.dumps({"key": jkey, "slot": sid, "technique": tech, **log.to_json()}) + "\n")
        if rec_on:
            try:
                sl = self.p.slots[sid]
                base_key = f"{jkey}c0"                           # turn 0 uses it, turn k {key}t{k}, the forced last {key}tf
                ukey = getattr(self, "_usage_key", {})
                usage = {t: ukey[k] for t, k in [(t, base_key if t == 0 else f"{base_key}t{t}") for t in range(log.turns)]
                         if k in ukey}
                if f"{base_key}tf" in ukey:
                    usage["final"] = ukey[f"{base_key}tf"]
                self._exp_recorder().record_step(
                    jkey, log, slot=sid, technique=tech, op=sl.op, op_class=sl.op_class,
                    shape_bucket=shape_bucket(sl.op_class, sl.shapes()[0] if sl.instances else {}),
                    model=model, role=role, rationale=log.rationale, extra_prompt=xtext, fact_ids=fact_ids,
                    usage=usage)
            except Exception as e:                              # noqa: BLE001 -- recording must never break a step
                self.log.warning(f"experiment record failed: {e!r}")
        return text

    # ---- X-145 experiment recording (knowledge/v2/experiments.py); every method is a no-op with the flags off
    def _exp_recorder(self):
        rec = getattr(self, "_exp", None)
        if rec is None:
            rec = self._exp = experiments.ExperimentRecorder(self.out, run_id=self.run_id, problem=self.p.name,
                                                             arch=self.arch_label, target=self.p.target,
                                                             dtype=self.p.dtype,
                                                             lineage=experiments.lineage_from_env(
                                                                 getattr(self, "seed", None)),
                                                             flags=self._exp_flags)
        return rec

    def _exp_flags(self) -> dict:
        """Run-state flags the experiment records' validity depends on (v1.1): ncu disabled (a timeout can leave the
        clocks locked), steps already timed on a clock-suspect GPU."""
        cs = self.ev.clock_stats() if hasattr(self.ev, "clock_stats") else None    # None: the clock guard is off
        return {"ncu_disabled": self.ncu_disabled, "clock_suspect": bool(getattr(self, "clock_suspect_steps", 0)),
                "clock_clean_after_ncu": bool(cs) and not cs.get("suspect")}

    def _experiment_prompt(self, sid: str, tech: str) -> tuple:
        n = experiments.prompt_n()
        if not n or self.kv is None:
            return "", []
        try:
            return experiments.prompt_text(self.kv.claims, self.kv.root, arch=self.arch_label,
                                           op_class=self.p.slots[sid].op_class, technique=tech, n=n)
        except Exception:                                       # noqa: BLE001
            return "", []

    def _experiment_outcome(self, ridx, step, sid, seed_src, code, status, speedup, reward, d, before, after,
                            impl_id, oid, res=None) -> None:
        rec = getattr(self, "_exp", None)
        if rec is None:
            return
        try:
            key = f"r{ridx}s{step}"
            ov = None
            if code and rec.n_records:
                progs = [experiments.load_program(r, rec.blobs) for r in experiments.read_jsonl(rec.path)
                         if r.get("type") == "experiment" and r.get("step_key") == key and r.get("program_sha")]
                progs = [p for p in progs if p]
                if progs:
                    ov = experiments.reference_overlap(seed_src, code, "\n".join(progs))
            # v1.1: an unchanged source is its own class (its measured delta is noise, not a regression); a step timed
            # on a clock-suspect GPU is flagged; absolute candidate time next to the paired-chain one
            unchanged = bool(code and experiments.same_source(seed_src, code))
            susp = bool(res is not None and getattr(res, "_clock_suspect", False))
            noise = None
            if unchanged and before and after:
                noise = round(100.0 * (after - before) / before, 3)
            tm = None
            if res is not None and getattr(res, "ok", False):
                c = d["ctx"] if d is not None else {}
                tm = {"basis": getattr(res, "e2e_basis", None), "abs_candidate_us": res.deployed_us,
                      "abs_incumbent_us": res.ref_e2e_us, "paired_before_us": c.get("before_us"),
                      "paired_after_us": (d or {}).get("e2e_after"), "pairing": c.get("pairing"),
                      "ratio": (res.e2e_ratio or {}).get("med") if getattr(res, "e2e_ratio", None) else None}
            rec.record_outcome(key, status=("clock_suspect" if susp else "unchanged" if unchanged else status),
                               speedup=None if (unchanged or susp) else speedup, reward=reward,
                               e2e_gain=(d["ctx"].get("gain") if d is not None else None),
                               slot_us_before=before, slot_us_after=after, impl_id=impl_id, outcome_id=oid, overlap=ov,
                               unchanged=unchanged, noise_pct=noise, time=tm, step_clock_suspect=susp)
            if experiments.ingest_enabled() and self.kv is not None:
                gpu = None
                try:
                    gpu = {"name": getattr(self.ev, "gpu_name", None), "arch": self.arch_label}
                except Exception:                               # noqa: BLE001
                    pass
                experiments.ingest(self.out, self.kv.root, self.kv.claims, gpu=gpu)
        except Exception as e:                                  # noqa: BLE001 -- never break a committed step
            self.log.warning(f"experiment outcome record failed: {e!r}")

    def _count_usage(self, u: dict, model: str, role: str, key: Optional[str]) -> None:
        if key is not None:
            self._counted.add(key)
        self.llm_calls += 1
        # OpenAI-style (prompt/completion) or Bedrock Converse-style (input/output) keys
        pt = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
        ct = int(u.get("completion_tokens") or u.get("output_tokens") or 0)
        cost = float(u.get("cost_usd") or 0.0)                          # providers that report it
        pin, pout = lookup_pricing(model)                               # else the price table (0 if unknown)
        est = cost or (pt * pin + ct * pout) / 1e6
        if key is not None:                                   # X-145 v1.1: tokens / $ per LLM turn (the experiment record)
            if not hasattr(self, "_usage_key"):
                self._usage_key = {}
            self._usage_key[key] = {"prompt_tokens": pt, "completion_tokens": ct, "usd": round(est, 6), "key": key}
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

    def _outcome(self, **kw) -> OutcomeRecord:
        sid = kw.pop("sid")
        s = self.p.slots[sid]
        rec = OutcomeRecord(target=self.p.target, arch=self.arch_label, problem_id=self.problem_id,
                            run_id=self.run_id, slot=sid, op_kind=s.op, op_class=s.op_class,
                            shape_bucket=shape_bucket(s.op_class, s.shapes()[0] if s.instances else {}),
                            dtype=self.p.dtype, **kw)
        trace.link(outcome_id=rec.id)
        return rec

    def _rid(self, *parts) -> str:
        """Deterministic record id of this run (impl store / outcome log idempotency key)."""
        from .journal import sha
        return sha(self.problem_id, self.run_id, *parts)

    @property
    def _tkey(self) -> str:
        return f"{self.p.target}:{self.arch_label}"

    @property
    def _profile_kind(self) -> str:
        """knowledge-v2 metric adapter for this target's profiler module (ncu | opencl_profile)."""
        return self.tp.profiler.__name__.rsplit(".", 1)[-1]

    def _roofline_input(self, sid: str, slot_us: float, res: Optional[EvalResult] = None):
        """Manifest bytes/flops of the slot's instances + measured time (knowledge v2
        estimate); with a run's attribution (``res``), its measured launches and
        launch-overhead share."""
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
        launches, lfrac = max(n, 1), None
        if res is not None and sid in res.attr_slot_launches:
            launches = max(int(round(res.attr_slot_launches[sid])), 1)
            ovh = overhead_per_launch(res) * res.attr_slot_launches[sid]
            lfrac = ovh / max(slot_us + ovh, 1e-9)
        pk = hw_profile.peaks(self.arch, self.p.dtype, self.p.slots[sid].op_class) if self.tp.name == "cuda" else None
        return RooflineInput(bytes=byt, flops=fl, time_us=slot_us, work_items=work or None, launches=launches,
                             launch_frac=lfrac, peak_gflops=pk[0] if pk else None)

    def _record_outcome(self, **kw) -> Optional[OutcomeRecord]:
        if self.olog is None:
            return None
        rec = self._outcome(**kw)
        if rec.id in self._existing["outcome"]:      # recorded by a segment killed before its checkpoint
            return rec
        self.olog.append(rec)
        if self.claims is not None and hasattr(self.claims, "observe"):
            self.claims.observe(rec)          # "already tried" visible to parallel rollouts now
        return rec

    def _store(self, sid: str, code: str, slot_us: float, baseline_us: Optional[float],
               techniques: list, parent: Optional[str], rel_err: Optional[float],
               rid: Optional[str] = None) -> Optional[str]:
        """Add verified code to the store; ``rid``: a deterministic id (idempotent on resume)."""
        if self.store is None:
            return None
        if rid is not None and rid in self._existing["impl"]:
            return rid
        s = self.p.slots[sid]
        seed_us = self.baseline.slot_us.get(sid) if self.baseline else None
        rec = self.store.add(code, target=self.p.target, arch=self.arch_label, op=s.op, op_class=s.op_class,
                             signature_sha=s.signature_sha, shapes=s.shapes(),
                             buckets=sorted({shape_bucket(s.op_class, x) for x in s.shapes()}),
                             slot_us=slot_us, baseline_us=baseline_us,
                             speedup=(baseline_us / slot_us) if baseline_us and slot_us else None,
                             seed_us=seed_us, speedup_vs_seed=(seed_us / slot_us) if seed_us and slot_us else None,
                             techniques=techniques, parent_id=parent, problem_id=self.problem_id,
                             run_id=self.run_id, slot=sid, max_rel_err=rel_err, dtype=self.p.dtype,
                             **({"id": rid} if rid else {}))
        return rec.id

    # ---- initialise: baseline + store seeds
    @tel.traced("init")
    async def initialize(self) -> EvalResult:
        seeds = self.p.seed_sources()
        base = await self.ev.evaluate(self.p, seeds, "baseline")
        if not base.ok and "MB_CONC_DEADLOCK" in (base.log or "") + base.error_text():
            # ModelBlaster --graph-concurrency: the seeds' concurrent CUDA-graph branches hung (driver watchdog);
            # this problem's runs launch sequentially from here on (recorded in result.json)
            from ..knowledge import slots as _slots
            _slots.PROBLEM_DRIVER_ENV[str(self.p.root)] = {"MB_CONCURRENCY": "0"}
            self.concurrency_fallback = "deadlock at baseline: sequential launches"
            self.log.warning(f"[{self.p.name}] {self.concurrency_fallback}")
            base = await self.ev.evaluate(self.p, seeds, "baseline_sequential")
        if not base.ok:
            raise RuntimeError(f"baseline failed ({base.stage}): {base.error_text()}")
        self.baseline = base
        self.partition_us = base.partition_us
        self._set_composite(base)
        for sid, code in seeds.items():
            self.best[sid] = SlotBest(code, self._slot_us(base, sid) or 0.0, techniques=[],
                                      event_us=base.slot_us.get(sid))
        self.log.info(f"[{self.p.name}] baseline {base.partition_us:.1f} us"
                      + (f", end to end {base.deployed_us:.1f} us ({base.e2e_basis or 'eager'}, {base.runtime})"
                         if self.reward == "e2e" else "") + "; slots "
                      + ", ".join(f"{s}={b.slot_us:.1f}" for s, b in sorted(self.best.items(), key=lambda kv: -kv[1].slot_us)))
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
            res = await self.ev.evaluate(self.p, trial, "store_seeds", **self._eval_kw(seeds))
            taken = []
            for sid, rec in from_store.items():
                ok = res.ok or res.slot_checks.get(sid, {}).get("ok", False)
                us = self._slot_us(res, sid)
                if ok and us is not None and us < self.best[sid].slot_us:
                    self.log.info(f"[{self.p.name}] {sid}: store seed {rec.id} {us:.1f} us "
                                  f"(hand {self.best[sid].slot_us:.1f})")
                    self.best[sid] = SlotBest(trial[sid], us, rec.id, list(rec.techniques),
                                              event_us=res.slot_us.get(sid))
                    taken.append(sid)
            if res.ok and res.partition_us:
                self.partition_us = min(self.partition_us, res.partition_us)
            if self.reward == "e2e" and taken:
                # the seeded composite must not be slower end to end than the hand seeds
                comp = res if res.ok and len(taken) == len(from_store) else \
                    await self.ev.evaluate(self.p, self._sources(), "store_composite", **self._eval_kw(seeds))
                gain, _ = e2e_gain(comp, base.deployed_us)
                if comp.ok and gain is not None and gain >= 0:
                    self._set_composite(comp)
                    for sid in taken:
                        self.best[sid].slot_us = self._slot_us(comp, sid) or self.best[sid].slot_us
                else:
                    self.log.info(f"[{self.p.name}] store seeds not faster end to end ({gain}); hand seeds kept")
                    for sid in taken:
                        self.best[sid] = SlotBest(seeds[sid], self._slot_us(base, sid) or 0.0, techniques=[],
                                                  event_us=base.slot_us.get(sid))
        await self._seed_candidates(seeds, base)
        await self._profile(self._sources(), "baseline")
        for sid, b in self.best.items():
            if b.impl_id is None:
                b.impl_id = self._store(sid, b.source, b.slot_us, None, [], None,
                                        base.slot_checks.get(sid, {}).get("rel_err"), rid=self._rid("seed", sid))
        return base

    async def _seed_candidates(self, seeds: dict, base: EvalResult) -> None:
        """ModelBlaster ``--reuse-winners``: the problem's ``seed_candidates`` (verified kernels of the same op and
        signature from sibling problems / the global store, for other shapes) are tried like store seeds: one run
        per candidate rank, a slot takes one only if it passes its isolated check on this problem's shapes and is
        faster; the seeded composite must not be slower end to end."""
        cands = self.p.manifest.get("seed_candidates") or {}
        cands = {sid: lst for sid, lst in cands.items() if sid in self.p.slots and lst}
        if not cands:
            return
        taken = []
        for r in range(max(len(v) for v in cands.values())):
            trial, tried = self._sources(), {}
            for sid, lst in cands.items():
                if r < len(lst):
                    try:
                        trial[sid] = (self.p.root / lst[r]["file"]).read_text()
                    except OSError:
                        continue
                    tried[sid] = lst[r]
            if not tried:
                continue
            res = await self.ev.evaluate(self.p, trial, f"seed_candidates_{r}", **self._eval_kw(seeds))
            for sid, c in tried.items():
                ok = res.ok or res.slot_checks.get(sid, {}).get("ok", False)
                us = self._slot_us(res, sid)
                if ok and us is not None and us < self.best[sid].slot_us * 0.98:
                    self.log.info(f"[{self.p.name}] {sid}: seed candidate {c.get('from')} {us:.1f} us "
                                  f"(was {self.best[sid].slot_us:.1f})")
                    self.best[sid] = SlotBest(trial[sid], us, None, ["reuse_seed"], event_us=res.slot_us.get(sid))
                    taken.append(sid)
        self.seed_candidates_taken = sorted(set(taken))
        if taken and self.reward == "e2e":
            comp = await self.ev.evaluate(self.p, self._sources(), "seed_candidates_composite", **self._eval_kw(seeds))
            gain, _ = e2e_gain(comp, self.e2e_us or base.deployed_us)
            if comp.ok and gain is not None and gain >= 0:
                self._set_composite(comp)
                if comp.partition_us:
                    self.partition_us = min(self.partition_us, comp.partition_us)
            else:
                self.log.info(f"[{self.p.name}] seed candidates not faster end to end ({gain}); kept the seeds")
                for sid in set(taken):
                    self.best[sid] = SlotBest(seeds[sid], self._slot_us(base, sid) or 0.0, techniques=[],
                                              event_us=base.slot_us.get(sid))
                self.seed_candidates_taken = []

    def _set_composite(self, res: EvalResult) -> None:
        """A full run of the composite best: its deployed time and attribution."""
        self.e2e_us = res.deployed_us
        self.last_full = res
        note = getattr(self.ev, "note_incumbent_wall", None)       # adaptive run timeout (slots.py)
        if note is not None and res.eval_wall_s:
            note(self.p, res.eval_wall_s)
        self.runtime_info = partition_runtime(res)
        self.composite_version = getattr(self, "composite_version", 0) + 1

    # ---- one step
    async def _step(self, ridx: int, step: int, cur: dict[str, str], cur_us: dict[str, float],
                    meta: Optional[dict] = None) -> None:
        """One step inside a telemetry ``step`` span; its breakdown goes into the step's context."""
        n = len(self.steps_log)
        with tel.span("step", collect=True, laps=True, rollout=ridx, step=step,
                      **({"replay": True} if f"{ridx},{step}" in self._replay else {})) as sp, \
                trace.scope(rollout=ridx, step=step) as tsc:
            try:
                await self._step_body(ridx, step, cur, cur_us, meta)
            finally:
                tel.end_laps(sp)
        summ = tel.ctx_summary(sp)
        if summ is not None:
            for s in reversed(self.steps_log[n:]):
                if s.rollout == ridx and s.step == step:
                    s.context["telemetry"] = summ
                    break
        trace.step_done(self, ridx, step, n, sp, tsc)     # the decision record (docs/trace.md)

    async def _step_body(self, ridx: int, step: int, cur: dict[str, str], cur_us: dict[str, float],
                         meta: Optional[dict] = None) -> None:
        """One step on rollout state ``cur`` (sources), ``cur_us`` (slot times) and
        ``meta`` (``e2e``: its deployed time; ``ev``: its event-timed slot times)."""
        t0 = time.time()
        if meta is None:
            meta = {"e2e": self.e2e_us, "ev": {s: b.event_us for s, b in self.best.items()}}
        e2e = self.reward == "e2e"
        info = self.last_full if e2e else None
        skey = f"{ridx},{step}"
        tel.lap("knowledge.select")
        plan = self._replay.pop(skey, None)       # picked before a kill: its tries / RNG draws are in the snapshot
        choice = cls = vec_before = key = None
        if plan is not None:
            sid = plan["sid"]
            adopted = plan.get("count_try")
            plan = {k: v for k, v in plan.items() if k != "count_try"}
            self._inflight[skey] = plan               # in flight again (a later checkpoint keeps it)
            if adopted:                               # from the LLM journal: its pick was never checkpointed
                self.slot_tries[sid] += 1
                if plan.get("rng"):                   # the draws the pick made
                    from .journal import set_rng_state
                    set_rng_state(self.rng, plan["rng"])
            slot = self.p.slots[sid]
            tech, desc, state, nums, key = plan["tech"], plan["desc"], plan["state"], plan["nums"], plan.get("key")
            prof_before = self._ncu_of(sid, cur[sid])
            if self.kv is not None:
                from ..knowledge.v2.knowledge import Choice
                from ..knowledge.v2.states import Classification
                choice, cls, vec_before = Choice(**plan["choice"]), Classification(**plan["cls"]), plan["vec"]
        else:
            async with self._lock:
                sid = self._pick_slot()
                self.slot_tries[sid] += 1
            slot = self.p.slots[sid]
            state, nums = slot_state(self.p, sid, cur_us[sid], self.arch,
                                     (hw_profile.peaks(self.arch, self.p.dtype, slot.op_class)
                                      if self.tp.name == "cuda" else None) or self.tp.peaks_for(self.arch))
            prof_before = self._ncu_of(sid, cur[sid])
            if prof_before:
                state = prof_before.get("state") or state    # measured bound beats the roofline estimate
            if self.kv is not None:
                vec_before = self.kv.vector(self._tkey, estimate=self._roofline_input(sid, cur_us[sid], info),
                                            profile=prof_before, profile_kind=self._profile_kind)
                cls = self.kv.classify(vec_before, target_key=self._tkey, op_class=slot.op_class,
                                       dtype=self.p.dtype)
                choice = self.kv.select(target_key=self._tkey, op_class=slot.op_class, dtype=self.p.dtype, cls=cls,
                                        vec=vec_before, rng=self.rng, op_kind=slot.op)
                tech, desc, state = choice.item_id, choice.desc, cls.label()
                nums = {**nums, **cls.decided_by}
            else:
                lb = e2e and (self.runtime_info.get("launch_bound") or slot_launch_bound(info, sid, self.p))
                techs = techniques_for(slot.op_class, self.tp.techniques, launch_bound=bool(lb),
                                       arch=self.arch if self.tp.name == "cuda" else None, dtype=self.p.dtype)
                # fp16 arms are their own keys (half2 / tensor-core techniques pay off
                # differently); fp32 keys keep their names, and the cross-key prior
                # still seeds an untried fp16 arm from its fp32 record
                key = f"{slot.op_class}|{state}" + ("" if self.p.dtype == "fp32" else f"|{self.p.dtype}")
                tech = self.bandit.select(key, [t for t, _ in techs], self.rng)
                desc = dict(techs)[tech]
            # no await since the pick: a checkpoint never sees the tries / draws without the plan
            from .journal import rng_state
            self._inflight[skey] = {"sid": sid, "tech": tech, "desc": desc, "state": state, "nums": nums,
                                    "key": key, "rng": rng_state(self.rng),
                                    **({"choice": asdict(choice), "cls": asdict(cls), "vec": vec_before}
                                       if self.kv is not None else {})}
        jkey = f"r{ridx}s{step}"                  # LLM journal keys: <jkey>c0 codegen, c1.. fixes
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
        ltext = self._impl_links_text(sid, r)
        if ltext:
            ktext = (ktext + "\n\n" + ltext) if ktext else ltext
        tel.lap("prompt.build")
        trace.step_input(slot=sid, technique=tech, source=cur[sid], state=state, choice_kind=getattr(choice, "kind", None))
        part_us = (meta.get("e2e") if e2e else None) or sum(cur_us.values())
        neigh = [(n, self.p.slots[n].signature) for n in self.p.neighbours(sid)]
        best_note = (f"Best verified so far for this slot: {self.best[sid].slot_us:.1f} us."
                     if self.best[sid].slot_us < cur_us[sid] else "")
        msgs = build_codegen_messages(problem=self.p, sid=sid, source=cur[sid], technique=tech, arch=self.arch,
                                      technique_desc=desc, state=state, state_numbers=nums,
                                      slot_us=cur_us[sid], partition_us=part_us, neighbours=neigh,
                                      examples=examples, adapt=adapt, knowledge_text=ktext,
                                      already_tried=tried, best_note=best_note,
                                      profile_text=self.tp.profiler.render(prof_before) if prof_before else "",
                                      runtime_text=runtime_text(info, sid, self.p) if e2e else "")
        tdef = self.kv.ontology.techniques.get(tech) if self.kv is not None else None
        model, role, why = self.router.codegen(technique=tech, tier=technique_tier(tech, tdef),
                                               share=self._share(sid), stall=self.slot_stall[sid],
                                               kind=choice.kind if choice is not None else "technique")
        if plan is not None and plan.get("model"):
            model, role, why = plan["model"], plan["role"], plan["why"]
        if skey in self._inflight:
            self._inflight[skey].update(model=model, role=role, why=why)
        # the plan goes with the response: a step killed before any checkpoint saw it is
        # resumed with the same slot / technique, so its journaled response still applies
        if experiment_tool.enabled():             # multi-turn codegen with the experiment tool (A2; default off)
            text = await self._multiturn(msgs, model, role, jkey, sid, tech,
                                         extra={"plan": self._inflight.get(skey) or plan})
        else:
            text = await self._llm(msgs, model, role, key=f"{jkey}c0",
                                   extra={"plan": self._inflight.get(skey) or plan})
        tel.lap("candidate", technique=tech)
        code = extract_code(text)
        fix_models, capture_defects = [], []
        fixes, res, status = 0, None, None
        before = cur_us[sid]
        while True:
            if code is None:
                res = EvalResult(ok=False, stage="compile", errors=[f"no ```{self.tp.fence} block in the response"])
            else:
                # only `sid` changed: isolated checks of its ops (the partition outputs are
                # always checked), screening budgets; a win is re-measured below in full
                res = await self.ev.evaluate(self.p, {**cur, sid: code}, f"r{ridx}_s{step}_{sid}_{tech}_f{fixes}",
                                             **step_eval_options(sid), **self._eval_kw(cur))
            # a verified candidate that breaks CUDA-graph capture of a partition that was
            # capturable is a fixable defect (it costs the graph replay end to end)
            capture = self._capture_defect(res)
            if (res.ok and not capture) or fixes >= self.max_fix or code is None:
                break
            other_bad = [s for s in res.failing_slots() if s != sid]
            if other_bad:                                         # shouldn't happen: only `sid` changed
                break
            fixes += 1
            trace.attempt(fixes)
            fmodel, frole = self.router.fix(attempt=fixes, max_attempts=self.max_fix, writer=model)
            fix_models.append(fmodel)
            if capture:
                capture_defects.append(res.graph_skip)
            err = capture or res.error_text()
            from .journal import sha as _jsha
            text = await self._llm(build_fix_messages(problem=self.p, sid=sid, source=code, error=err,
                                                      pitfalls=self._pitfalls(sid, tech), arch=self.arch), fmodel, frole,
                                   key=f"{jkey}c{fixes}", digest=_jsha(code, err))
            new = extract_code(text)
            if not new or new == code:
                break
            code = new
        if e2e:
            d = await self._decide_e2e(ridx, step, sid, tech, code, res, before, cur, meta)
            after, speedup, status, reward = d["after"], d["speedup"], d["status"], d["reward"]
        else:
            d = None
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
        if not res.ok:
            speedup = None
            status = {"compile": COMPILE_ERROR, "run": RUNTIME_ERROR}.get(res.stage, WRONG_OUTPUT)
            reward = -0.05 if status == COMPILE_ERROR else -0.1
        elif clock_suspect(res) or (d is not None and clock_suspect(d.get("res2"))):
            # X-149: timed on a GPU whose clocks are locked / throttled and no clean GPU was left to re-time on: the
            # numbers are not evidence. Neutral, no reward, never adopted, never recorded as a regression.
            res._clock_suspect = True
            self.clock_suspect_steps = getattr(self, "clock_suspect_steps", 0) + 1
            self.log.warning(f"[{self.p.name}] r{ridx} s{step}: {sid} {tech} timed on a clock-suspect GPU: step marked "
                             f"inconclusive (not adopted, not counted as a regression)")
            tel.count("clock.suspect_step")
            speedup, status, reward = None, NEUTRAL, 0.0
            if d is not None:
                d["keep"], d["merged"] = False, False
        tel.lap("knowledge.record")
        prof_after = None
        if res.ok and (self.ncu == "all" or (self.ncu == "best" and (
                d["merged"] if d is not None else after < self.best[sid].slot_us * (1 - self.confirm_pct / 100)))):
            by = await self._profile({**cur, sid: code}, f"r{ridx}_s{step}_{sid}")
            prof_after = by.get(sid)
        # ---- commit: everything below up to the checkpoint runs without an await, so a
        # checkpoint written by another rollout never sees half a step; the store /
        # outcome records carry deterministic ids, so a replayed step does not add them twice
        async with self._lock:
            self._commit_step(ridx, step, t0, sid, slot, tech, key, state, choice, cls, vec_before, prof_before,
                              prof_after, code, res, d, before, after, speedup, status, reward, fixes, fix_models,
                              capture_defects, cur, cur_us, meta, r, ktext, tried, model, role, why, e2e)

    def _commit_step(self, ridx, step, t0, sid, slot, tech, key, state, choice, cls, vec_before, prof_before,
                     prof_after, code, res, d, before, after, speedup, status, reward, fixes, fix_models,
                     capture_defects, cur, cur_us, meta, r, ktext, tried, model, role, why, e2e) -> None:
        if self.kv is None:
            self.bandit.update(key, tech, reward)
        impl_id, parent = None, self.best[sid].impl_id
        cur_prev = cur[sid]                                     # the step's seed source (X-145 reference overlap)
        dsum = diff_summary(cur[sid], code) if code else None
        csus = bool(getattr(res, "_clock_suspect", False))       # X-149: timed on a clock-suspect GPU
        if res.ok:
            impl_id = self._store(sid, code, after, before, self.best[sid].techniques + [tech], parent,
                                  res.slot_checks.get(sid, {}).get("rel_err"), rid=self._rid("impl", ridx, step))
            keep = d["keep"] if d is not None else after <= before * 1.05   # keep improving/neutral moves
            keep = keep and not csus
            if keep:
                cur[sid], cur_us[sid] = code, after
                meta["ev"][sid] = res.slot_us.get(sid)
                if d is not None and d["e2e_after"]:
                    meta["e2e"] = d["e2e_after"]
        if d is not None:
            merged = d["merged"] and self._merge(sid, code, d, impl_id, tech)
        else:
            merged = res.ok and after < self.best[sid].slot_us * (1 - self.confirm_pct / 100) and not csus
            if merged:
                self.best[sid] = SlotBest(code, after, impl_id, self.best[sid].techniques + [tech],
                                          event_us=res.slot_us.get(sid))
        if merged:
            self.slot_stall[sid] = 0
            tel.count("win")
            self.log.info(f"[{self.p.name}] r{ridx} s{step}: {sid} {before:.1f} -> {after:.1f} us "
                          f"via {tech} (new slot best" + (f", end to end -{100 * d['reward']:.1f}%)" if d else ")"))
        else:
            self.slot_stall[sid] += 1
        oid = self._rid("outcome", ridx, step)
        if getattr(self, "_exp", None) is not None:             # X-145: the outcome of the step the experiments fed
            self._experiment_outcome(ridx, step, sid, cur_prev, code, status, speedup, reward, d, before, after,
                                     impl_id, oid, res=res)
        okw = dict(sid=sid, technique=tech, status=status, state_before=state, id=oid,
                   state_after=prof_after.get("state") if prof_after else None,
                   metric_before=before, metric_after=after, speedup=speedup,
                   metrics_delta=self.tp.profiler.delta(prof_before, prof_after),
                   error=None if res.ok else res.error_text(1500),
                   diff_summary=dsum, impl_id=impl_id, parent_impl_id=parent,
                   **(d["outcome"] if d is not None else {"timing_basis": "slot_events"}))
        v2 = None
        if csus:                                    # X-149: inconclusive timing: nothing is learned from it
            pass
        elif self.kv is not None:
            vec_after = (self.kv.vector(self._tkey, estimate=self._roofline_input(sid, after, res if e2e else None),
                                        profile=prof_after, profile_kind=self._profile_kind)
                         if res.ok and after else None)
            try:
                if oid in self._existing["outcome"]:     # recorded before a kill: its stats are in the DB
                    rows = self.kv.outcomes.rows("id=?", (oid,))
                    v2 = rows[0]["v2"] if rows else None
                else:
                    v2 = self.kv.record(self._outcome(**okw), choice=choice, reward=reward, cls=cls,
                                        before=vec_before, after=vec_after, op_class=slot.op_class,
                                        dtype=self.p.dtype,
                                        extra={"lineage": list(self.best[sid].techniques),
                                               "ncu_rules": (prof_before or {}).get("rules", []),
                                               "diff_excerpt": diff_excerpt(cur[sid], code) if code else "",
                                               **({"e2e": d["ctx"]} if d is not None else {})})
            except Exception as e:                                # knowledge must never break a step
                self.log.warning(f"knowledge-v2 record failed: {e!r}")
        else:
            self._record_outcome(**okw)
        ctx = {"adapt": r["adapt"].id if r["adapt"] is not None else None,
               "examples": [e.id for e in r["examples"]],
               "claims": ktext.count("\n") + 1 if ktext else 0, "already_tried": bool(tried),
               "ncu": bool(prof_before), "state": state, "model": model, "model_role": role,
               "model_why": why, **({"clock_suspect": True, "gpu_clock": (res.gpu or {}).get("clock")} if csus else {})}
        if fix_models:
            ctx["fix_models"] = fix_models
        if capture_defects or (res is not None and self._capture_defect(res)):
            ctx["graph_capture"] = {"defects": capture_defects, "final_skip": res.graph_skip if res else None}
        if d is not None:
            ctx["e2e"] = d["ctx"]
        if v2 is not None:
            ctx.update(kind=choice.kind, est=round(choice.est, 4), evidence=choice.evidence,
                       explore=choice.explore, mechanism=v2["mechanism"]["label"], chain=cls.chain)
        self.steps_log.append(StepLog(ridx, step, sid, tech, status, before, after, reward, fixes,
                                      round(time.time() - t0, 1), ctx, reward_basis=self.reward,
                                      e2e_before=d["ctx"]["before_us"] if d else None,
                                      e2e_after=d["e2e_after"] if d else None))
        self.log.info(f"[{self.p.name}] r{ridx} s{step}: {sid} {tech} -> {status}"
                      + (f" {before:.1f}->{after:.1f} us" if after else "") + f" (fixes={fixes})")
        tel.count("step")                      # completed steps (a step replayed after a kill counts once)
        self._check_patience()
        self._step_done(ridx, step, cur, cur_us, meta)

    def _check_patience(self, announce: bool = True) -> None:
        """E7 patience: once ``stop_patience`` consecutive committed steps set no new slot best, no rollout
        starts another step (in-flight ones finish and commit). Sticky; recomputed from the steps log on resume."""
        if not self.stop_patience or self.patience_stop is not None:
            return
        trig = patience_trigger(self.steps_log, self.stop_patience)
        if trig is None:
            return
        planned = self.rollouts * self.steps
        inflight = max(len(self._inflight) - 1, 0)       # the committing step is still in _inflight here
        self.patience_stop = {**trig, "patience": self.stop_patience, "steps_planned": planned,
                              "inflight_at_trigger": inflight}
        if not announce:                                  # restored from the journal: logged when it happened
            return
        self.log.info(f"[{self.p.name}] efficiency: patience {self.stop_patience} reached at commit "
                      f"{trig['trigger_commit']} of {planned} planned steps ({trig['misses']} committed steps "
                      f"without a new slot best); no new steps, {inflight} in flight finish")
        tel.emit({"t": "efficiency", "ts": time.time(), "problem": self.problem_id, "rule": "patience",
                  **self.patience_stop})

    def _external_stop(self) -> bool:
        """``<out>/mb_stop.json`` (written by ModelBlaster's graph search when a mid-run graph decision removes this
        problem's ops from the graph): no rollout starts another step; in-flight steps finish. Sticky; absent = no
        effect."""
        if getattr(self, "external_stop", None) is not None:
            return True
        f = self.out / "mb_stop.json"
        if not f.exists():
            return False
        try:
            why = json.loads(f.read_text())
        except (OSError, ValueError):
            why = {}
        self.external_stop = {"commit": len(self.steps_log), **(why if isinstance(why, dict) else {})}
        self.log.info(f"[{self.p.name}] external stop at commit {len(self.steps_log)}: {str(why)[:200]}; no new steps")
        tel.emit({"t": "efficiency", "ts": time.time(), "problem": self.problem_id, "rule": "external",
                  **self.external_stop})
        return True

    def efficiency_info(self) -> dict:
        """result.json ``efficiency``: the rule and, if it stopped the problem, where (E7-style attribution)."""
        planned = self.rollouts * self.steps
        n = len(self.steps_log)
        out = {"stop_patience": self.stop_patience, "stopped": None, "steps_planned": planned,
               "steps_committed": n, "steps_not_run": max(planned - n, 0)}
        if self.patience_stop is not None:
            tc = int(self.patience_stop["trigger_commit"])
            out.update(stopped="patience", trigger_commit=tc, steps_after_trigger=max(n - tc, 0),
                       inflight_at_trigger=self.patience_stop.get("inflight_at_trigger"))
        if getattr(self, "external_stop", None) is not None:
            out["external_stop"] = self.external_stop
            out["stopped"] = out["stopped"] or "external"
        return out

    def _capture_defect(self, res: Optional[EvalResult]) -> Optional[str]:
        """The fix-loop error text when a verified candidate made the partition fall back
        from a CUDA graph to eager launches (the composite best is capturable), else None."""
        if self.reward != "e2e" or res is None or not res.ok or not res.graph_skip:
            return None
        if self.runtime_info.get("e2e_basis") != "graph":
            return None
        return ("The rewritten slot is correct, but the partition can no longer be captured into a CUDA graph: "
                f"{res.graph_skip}. The deployed runtime replays the whole partition as a graph, so this costs the "
                "graph replay end to end. Make every call capturable: no cudaDeviceSynchronize/cudaStreamSynchronize, "
                "no cudaMalloc/cudaFree or host copies per call (allocate scratch once, lazily, outside the timed "
                "path), launch on the default stream only.")

    async def _decide_e2e(self, ridx: int, step: int, sid: str, tech: str, code: Optional[str], res: EvalResult,
                          before: float, cur: dict, meta: dict) -> dict:
        """End-to-end decision for a verified candidate: reward = fraction of the
        deployed partition time saved (paired with the rollout's state where the
        evaluator can). A gain of ``e2e_accept_pct`` or more is re-measured with
        every check against the composite best, and joins it if it holds there
        too. Also records what the per-slot rule (slot time vs ``confirm_pct``)
        would have decided on the same runs (``ctx["slot_rule"]``)."""
        acc = self.e2e_accept_pct / 100.0
        out = {"after": None, "speedup": None, "status": None, "reward": 0.0, "keep": False, "merged": False,
               "e2e_after": None, "outcome": {}, "ctx": {}}
        ev_before = meta["ev"].get(sid)
        ctx = out["ctx"] = {"before_us": meta.get("e2e"), "basis": res.e2e_basis, "runtime": res.runtime,
                            "slot_attr_before": before}
        if not res.ok:
            return out
        after = self._slot_us(res, sid)
        gain, how = e2e_gain(res, meta.get("e2e"))
        ctx.update(gain=None if gain is None else round(gain, 5), pairing=how, after_us=res.deployed_us,
                   ratio=res.e2e_ratio, slot_attr_after=after, graph_skip=res.graph_skip,
                   launches=res.attr_slot_launches.get(sid))
        g = gain if gain is not None else 0.0
        res2, gain_c = None, None
        if gain is not None and gain >= acc:
            comp = self._sources()
            out["base_src"] = comp[sid]
            res2 = await self.ev.evaluate(self.p, {**comp, sid: code}, f"r{ridx}_s{step}_{sid}_confirm",
                                          **self._eval_kw(comp))
            if res2.ok:
                gain_c, _ = e2e_gain(res2, self.e2e_us)
                if gain_c is not None:
                    g = min(g, gain_c)
                    after = max(after or 0.0, self._slot_us(res2, sid) or 0.0) or after
            ctx.update(confirm_ok=res2.ok, confirm_gain=None if gain_c is None else round(gain_c, 5),
                       confirm_us=res2.deployed_us if res2.ok else None)
        # KERNELBLASTER_MULTISLOT_ROLLOUT_ACCEPT=local|lineage (ModelBlaster --reuse-winners / --rollout-accept): a
        # rollout keeps a verified gain over its own state even when it does not beat the composite best (that still
        # decides the merge below); default "composite": g = min(own gain, confirmed gain vs the composite), so a
        # lagging rollout cannot climb (rollout_decision)
        mode = rollout_mode()
        gr, status, keep = rollout_decision(gain, g, acc, mode)
        if mode != "composite" and gain is not None:
            ctx["rollout_accept"] = mode
        reward = max(-1.0, min(1.0, gr))
        e2e_after = res.deployed_us
        if how == "paired" and meta.get("e2e"):
            e2e_after = meta["e2e"] * (1.0 - (gain or 0.0))     # on the scale of the state's own time
        out.update(after=after, speedup=(before / after) if after else None, status=status, reward=reward,
                   keep=keep, e2e_after=e2e_after,
                   merged=bool(res2 is not None and res2.ok and gain_c is not None and gain_c >= acc and g >= acc),
                   res2=res2)
        # the per-slot rule on the same runs (what the previous reward would have done)
        ev_after = res.slot_us.get(sid)
        if res2 is not None and res2.ok and sid in res2.slot_us and ev_after is not None:
            ev_after = max(ev_after, res2.slot_us[sid])
        part = sum(v for v in meta["ev"].values() if v) or 1.0
        ctx["slot_rule"] = None if ev_before is None or ev_after is None else {
            "before_us": ev_before, "after_us": ev_after,
            "reward": round(max(-1.0, min(1.0, (ev_before - ev_after) / part)), 5),
            "accept": ev_after < ev_before * (1 - self.confirm_pct / 100)}
        ctx["accept"] = out["merged"]
        out["outcome"] = {"timing_basis": f"e2e_{res.e2e_basis or 'eager'}", "e2e_before": meta.get("e2e"),
                          "e2e_after": e2e_after, "partition_effect": round(g, 6)}
        return out

    def _merge(self, sid: str, code: str, d: dict, impl_id: Optional[str], tech: str) -> bool:
        """Join a confirmed end-to-end win to the composite (under the lock). A
        change of the same slot by another rollout since the confirm run started
        wins (it was measured against a composite this one never saw)."""
        res2 = d.get("res2")
        if res2 is None:
            return False
        if self.best[sid].source != d.get("base_src"):
            return False
        self.best[sid] = SlotBest(code, self._slot_us(res2, sid) or d["after"], impl_id,
                                  self.best[sid].techniques + [tech], event_us=res2.slot_us.get(sid))
        self._set_composite(res2)
        return True

    async def _rollout(self, ridx: int):
        async with self._lock:
            st = self._rstate.get(ridx)
            if st is None:
                cur = {s: b.source for s, b in self.best.items()}
                cur_us = {s: b.slot_us for s, b in self.best.items()}
                meta = {"e2e": self.e2e_us, "ev": {s: b.event_us for s, b in self.best.items()}}
                first = 0
                self._rstate[ridx] = {"next_step": 0, "cur": dict(cur), "cur_us": dict(cur_us),
                                      "meta": copy.deepcopy(meta)}
            else:                                   # resumed: from the rollout's last completed step
                cur, cur_us, meta = dict(st["cur"]), dict(st["cur_us"]), copy.deepcopy(st["meta"])
                first = st["next_step"]
        for step in range(first, self.steps):
            if self.stopping:                       # SIGTERM / SIGINT: no new steps
                break
            if self.patience_stop is not None:      # E7 patience: no new steps (in-flight ones finish)
                break
            if self._external_stop():               # ModelBlaster graph search: the slot's ops were fused away
                break
            if step > 0:
                self._maybe_resync(ridx, step, cur, cur_us, meta)
            try:
                await self._step(ridx, step, cur, cur_us, meta)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.log.error(f"[{self.p.name}] rollout {ridx} step {step} failed: {e!r}")
                if self._rstate[ridx]["next_step"] <= step:      # a failed step is done too
                    self._step_done(ridx, step, cur, cur_us, meta)
            if self.stopping:
                break
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

    def _maybe_resync(self, ridx: int, step: int, cur: dict, cur_us: dict, meta: dict) -> None:
        """Lineage resync (should_resync): a stalled rollout lagging the composite restarts from it (in place: the
        step code holds these dicts). Deterministic from the steps log, so a resumed rollout decides alike."""
        factor = rollout_resync_factor()
        if not factor or self.reward != "e2e":
            return
        mine = [x for x in self.steps_log if x.rollout == ridx and x.step < step]
        stalled = bool(mine) and mine[-1].status != IMPROVED
        if not should_resync(meta.get("e2e"), self.e2e_us, factor, stalled):
            return
        was = meta.get("e2e")
        cur.clear()
        cur.update({s: b.source for s, b in self.best.items()})
        cur_us.clear()
        cur_us.update({s: b.slot_us for s, b in self.best.items()})
        meta.clear()
        meta.update({"e2e": self.e2e_us, "ev": {s: b.event_us for s, b in self.best.items()}})
        ev = {"rollout": ridx, "before_step": step, "state_e2e": was, "composite_e2e": self.e2e_us,
              "factor": factor}
        if not hasattr(self, "resyncs"):
            self.resyncs = []
        self.resyncs.append(ev)
        self.log.info(f"[{self.p.name}] r{ridx} s{step}: lineage resync ({was:.1f} us > {factor} x composite "
                      f"{self.e2e_us:.1f} us after a non-improving step)")
        tel.emit({"t": "rollout_resync", "ts": time.time(), "problem": self.problem_id, **ev})

    # ---- resume (agents/journal.py)
    @property
    def stopping(self) -> bool:
        return self.stop_event is not None and self.stop_event.is_set()

    def _step_done(self, ridx: int, step: int, cur: dict, cur_us: dict, meta: dict) -> None:
        """A step finished (or failed): the rollout continues from here after a kill."""
        self._inflight.pop(f"{ridx},{step}", None)
        self._replay.pop(f"{ridx},{step}", None)
        self._rstate[ridx] = {"next_step": step + 1, "cur": dict(cur), "cur_us": dict(cur_us),
                              "meta": copy.deepcopy(meta)}
        self._checkpoint()

    def _checkpoint(self, phase: Optional[str] = None) -> None:
        if phase is not None:
            self.phase = phase
        if self.journal is None:
            return
        with tel.span("io.checkpoint"):
            # v1 "already tried" tallies live in claims.json; saved first (observe is idempotent
            # by outcome id), so a kill between the two writes never loses a completed step's tally
            if self.claims is not None and self.kv is None and hasattr(self.claims, "save"):
                try:
                    self.claims.save()
                except Exception as e:
                    self.log.warning(f"claims save failed: {e!r}")
            self.journal.save(self._snapshot())
        tel.flush()                            # the telemetry of completed steps is on disk with them
        if self.on_checkpoint is not None:
            try:
                self.on_checkpoint(self)
            except Exception as e:
                self.log.warning(f"checkpoint hook failed: {e!r}")

    @staticmethod
    def _er(res: Optional[EvalResult]) -> Optional[dict]:
        if res is None:
            return None
        d = asdict(res)
        d["log"] = (d.get("log") or "")[-4000:]
        return d

    @staticmethod
    def _er_load(d: Optional[dict]) -> Optional[EvalResult]:
        if d is None:
            return None
        known = EvalResult.__dataclass_fields__
        return EvalResult(**{k: v for k, v in d.items() if k in known})

    def _snapshot(self) -> dict:
        from .journal import problem_hash, rng_state
        code: dict[str, str] = {}

        def ref(src: str) -> str:
            h = self._sha(src)
            code[h] = src
            return h

        if not hasattr(self, "_phash"):
            self._phash = problem_hash(self.p)
        return {
            "phase": self.phase, "problem": self.p.name, "problem_id": self.problem_id, "problem_hash": self._phash,
            "run_id": self.run_id, "segment": self.segment, "segments": self.segments,
            "steps": self.steps, "rollouts": self.rollouts,
            "stop_patience": self.stop_patience, "patience_stop": self.patience_stop,
            "baseline": self._er(self.baseline), "partition_us": self.partition_us, "e2e_us": self.e2e_us,
            "last_full": self._er(self.last_full), "runtime_info": self.runtime_info,
            "composite_version": getattr(self, "composite_version", 0),
            "best": {sid: {**asdict(b), "source": ref(b.source)} for sid, b in self.best.items()},
            "slot_tries": self.slot_tries, "slot_stall": self.slot_stall,
            "steps_log": [asdict(x) for x in self.steps_log],
            "llm_calls": self.llm_calls, "llm_usage": self.llm_usage, "llm_usage_by": self.llm_usage_by,
            "llm_counted": sorted(self._counted), "llm_replayed": self.llm_replayed,
            "bandit": self.bandit.to_json(), "rng": rng_state(self.rng),
            "ncu": {"cache": [[k[0], k[1], v] for k, v in self.ncu_cache.items()], "runs": self.ncu_runs,
                    "disabled": self.ncu_disabled},
            "rollout_state": {str(r): {"next_step": st["next_step"], "cur": {s: ref(c) for s, c in st["cur"].items()},
                                       "cur_us": st["cur_us"], "meta": st["meta"]}
                              for r, st in self._rstate.items()},
            "inflight": {**self._replay, **self._inflight},
            "gpu": getattr(self.baseline, "gpu", None) if self.baseline is not None else None,
            # knowledge v2's exploration budget counters (in memory, shared by the run's problems)
            "kv_counters": ({"n_selected": getattr(self.kv, "n_selected", 0), "n_explore": getattr(self.kv, "n_explore", 0)}
                            if self.kv is not None else None),
            "code": code,
        }

    def _restore(self, st: dict) -> None:
        from .journal import set_rng_state
        code = st.get("code", {})
        self.phase = st.get("phase")
        self.segments = list(st.get("segments", []))
        self.baseline = self._er_load(st.get("baseline"))
        self.partition_us, self.e2e_us = st.get("partition_us"), st.get("e2e_us")
        self.last_full = self._er_load(st.get("last_full"))
        self.runtime_info = st.get("runtime_info") or {}
        self.composite_version = st.get("composite_version", 0)
        self.best = {sid: SlotBest(**{**b, "source": code[b["source"]]}) for sid, b in st.get("best", {}).items()}
        self.slot_tries.update(st.get("slot_tries", {}))
        self.slot_stall.update(st.get("slot_stall", {}))
        self.steps_log = [StepLog(**x) for x in st.get("steps_log", [])]
        self.llm_calls = st.get("llm_calls", 0)
        self.llm_usage = st.get("llm_usage") or self.llm_usage
        self.llm_usage_by = st.get("llm_usage_by") or {}
        self._counted = set(st.get("llm_counted", []))
        self.llm_replayed = st.get("llm_replayed", 0)
        self.bandit = TechniqueBandit.from_json(st.get("bandit", {}), c=self.bandit.c,
                                                prior_weight=self.bandit.prior_weight)
        if st.get("rng"):
            set_rng_state(self.rng, st["rng"])
        n = st.get("ncu", {})
        self.ncu_cache = {(a, b): v for a, b, v in n.get("cache", [])}
        self.ncu_runs, self.ncu_disabled = n.get("runs", 0), n.get("disabled")
        self._rstate = {int(r): {"next_step": x["next_step"], "cur": {s: code[h] for s, h in x["cur"].items()},
                                 "cur_us": x["cur_us"], "meta": x["meta"]}
                        for r, x in st.get("rollout_state", {}).items()}
        self._replay = dict(st.get("inflight", {}))
        # the patience the problem was started with wins (a resume never changes it silently); a journal from
        # before the rule (no key) keeps whatever the run resolved (run_multislot.py: off for a legacy out dir)
        if "stop_patience" in st and int(st.get("stop_patience") or 0) != self.stop_patience:
            self.log.warning(f"[{self.p.name}] resume: keeping the journal's stop_patience "
                             f"{st.get('stop_patience')} (this run asked for {self.stop_patience})")
            self.stop_patience = int(st.get("stop_patience") or 0)
        self.patience_stop = None
        self._check_patience(announce=False)
        kc = st.get("kv_counters")
        if kc and self.kv is not None:
            for k, v in kc.items():
                if hasattr(self.kv, k):
                    setattr(self.kv, k, max(getattr(self.kv, k), v))
        if self.journal is not None:                 # steps that got a response but no checkpoint
            for k in self.journal.llm_keys():
                m = re.fullmatch(r"r(\d+)s(\d+)c0", k)
                if not m:
                    continue
                ridx, step = int(m.group(1)), int(m.group(2))
                skey = f"{ridx},{step}"
                plan = self.journal.llm_get(k).get("plan")
                done = self._rstate.get(ridx, {}).get("next_step", 0) > step
                if plan and not done and skey not in self._replay:
                    self._replay[skey] = {**plan, "count_try": True}
        # records a killed segment wrote after its last checkpoint: not written again
        try:
            if self.store is not None:
                self._existing["impl"] = {r.id for r in self.store.records()
                                          if r.run_id == self.run_id and r.problem_id == self.problem_id}
            if self.olog is not None:
                self._existing["outcome"] = {r.id for r in self.olog
                                             if r.run_id == self.run_id and r.problem_id == self.problem_id}
        except Exception as e:
            self.log.warning(f"[{self.p.name}] scanning earlier records failed: {e!r}")

    async def _verify_restored(self) -> None:
        """Re-measure the restored composite best and rollout states once (the GPU or
        host may differ from the killed segment's): a state that no longer verifies
        falls back (slots to their seeds; a rollout to the composite)."""
        if not any(st["next_step"] < self.steps for st in self._rstate.values()) and \
                len(self._rstate) >= self.rollouts:
            return
        seeds = self.p.seed_sources()
        comp = self._sources()
        res = await self.ev.evaluate(self.p, comp, "resume_verify", **self._eval_kw(seeds))
        info = {"composite_ok": res.ok, "rollouts_reset": []}
        if not res.ok:
            self.log.warning(f"[{self.p.name}] restored composite failed ({res.error_text(300)}); repairing")
            for sid in res.failing_slots() or list(self.best):
                self.best[sid] = SlotBest(seeds[sid], self._slot_us(self.baseline, sid) or 0.0,
                                          event_us=self.baseline.slot_us.get(sid))
            comp = self._sources()
            res = await self.ev.evaluate(self.p, comp, "resume_repaired", **self._eval_kw(seeds))
            if not res.ok:
                raise RuntimeError(f"restored composite does not verify ({res.stage}): {res.error_text(500)}")
        self._set_composite(res)
        for sid, b in self.best.items():
            b.slot_us = self._slot_us(res, sid) or b.slot_us
            b.event_us = res.slot_us.get(sid, b.event_us)
        for ridx, st in sorted(self._rstate.items()):
            if st["next_step"] >= self.steps:
                continue
            r2 = res if st["cur"] == comp else \
                await self.ev.evaluate(self.p, st["cur"], f"resume_verify_r{ridx}", **self._eval_kw(comp))
            if not r2.ok:
                info["rollouts_reset"].append(ridx)
                st["cur"], r2 = dict(comp), res
            st["cur_us"] = {s: (self._slot_us(r2, s) or st["cur_us"].get(s, 0.0)) for s in st["cur"]}
            st["meta"] = {"e2e": r2.deployed_us if r2 is res else
                          ((self.e2e_us * (1.0 - (e2e_gain(r2, self.e2e_us)[0] or 0.0))) if self.e2e_us else
                           r2.deployed_us),
                          "ev": {s: r2.slot_us.get(s) for s in st["cur"]}}
        self.resume_info.update(verify=info)
        self._checkpoint()

    # ---- run
    async def run(self) -> dict:
        """``_run`` inside a telemetry ``problem`` span; result.json gets its breakdown."""
        with tel.span("problem", collect=True, problem=self.problem_id, segment=self.segment) as sp, \
                trace.scope(run=self.run_id, problem=self.problem_id):
            result = await self._run()
        summ = tel.full_summary(sp)
        if summ is not None and not result.get("resume", {}).get("skipped"):
            result["telemetry"] = summ
            (self.out / "result.json").write_text(json.dumps(result, indent=1, default=str))
        return result

    async def _run(self) -> dict:
        from .journal import RunInterrupted
        st = self.journal.state if self.journal is not None else None
        self.phase, self.segments = None, []
        if st:
            self._restore(st)
            res_path = self.out / "result.json"
            if self.phase == "done" and res_path.exists():
                self.log.info(f"[{self.p.name}] already done (journal); skipped")
                result = json.loads(res_path.read_text())
                result.setdefault("resume", {})["skipped"] = True
                return result
            self.log.info(f"[{self.p.name}] resuming segment {self.segment} from phase {self.phase}: "
                          f"{len(self.steps_log)} step(s) done, {len(self._replay)} in flight, "
                          f"{sum(1 for _ in self.journal.llm_keys())} journaled LLM response(s)")
            tel.emit({"t": "resume", "ts": time.time(), "problem": self.problem_id, "segment": self.segment,
                      "phase": self.phase, "steps_done": len(self.steps_log), "inflight": len(self._replay)})
        self.segments.append({"n": self.segment, "start": time.time(), "phase": self.phase,
                              "steps_done": len(self.steps_log), "pid": os.getpid()})
        if self.phase in (None, "init"):
            self._checkpoint("init")                 # the run id is on disk before anything is recorded
            if self.baseline is None:
                await self.initialize()
            self._checkpoint("rollouts")
        elif self.phase == "rollouts":
            await self._verify_restored()
        if self.phase == "rollouts":
            await asyncio.gather(*[self._rollout(i) for i in range(self.rollouts)])
            if self.stopping:
                self._checkpoint()
                raise RunInterrupted(f"{self.p.name}: stopped after {len(self.steps_log)} step(s)")
            self._checkpoint("final")
        final = await self.ev.evaluate(self.p, self._sources(), "final_composite",
                                       **self._eval_kw(self.p.seed_sources()))
        if not final.ok:
            # a composition broke something: fall back slot by slot to the baseline seed
            self.log.warning(f"[{self.p.name}] composite failed ({final.error_text(300)}); repairing")
            seeds = self.p.seed_sources()
            for sid in final.failing_slots() or list(self.best):
                self.best[sid] = SlotBest(seeds[sid], self._slot_us(self.baseline, sid) or 0.0,
                                          event_us=self.baseline.slot_us.get(sid))
            final = await self.ev.evaluate(self.p, self._sources(), "final_repaired", **self._eval_kw(seeds))
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
        if getattr(self, "_exp", None) is not None and self.kv is not None and experiments.ingest_enabled():
            try:                                              # X-145: final ingest, with the GPU of the result
                experiments.ingest(self.out, self.kv.root, self.kv.claims, gpu=result.get("gpu"))
            except Exception as e:                            # noqa: BLE001
                self.log.warning(f"experiment ingest failed: {e!r}")
        if self.claims is not None and hasattr(self.claims, "save"):
            self.claims.save()
        self._checkpoint("done")
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
            **({"multiturn": self.mt_stats} if getattr(self, "mt_stats", None) else {}),
            **({"experiments": self._exp.pointer()} if getattr(self, "_exp", None) is not None else {}),
            **({"clock_suspect": {"steps": self.clock_suspect_steps}} if getattr(self, "clock_suspect_steps", 0) else {}),
            "ncu": {"mode": self.ncu, "runs": self.ncu_runs, "disabled": self.ncu_disabled,
                    "final": {sid: self._ncu_of(sid, b.source) for sid, b in self.best.items()
                              if self._ncu_of(sid, b.source)}},
            "baseline_partition_us": base.partition_us, "final_ok": final.ok,
            # why the final run failed, and whether the GPU / link / staging failed rather than
            # the composite (ModelBlaster leaves a hint undecided then instead of rejecting it)
            "final_error": None if final.ok else " ".join(final.error_text(800).split()),
            "final_infra": None if final.ok else _is_infra(final.log or final.error_text(3000)),
            "final_partition_us": final.partition_us, "op_us": final.op_us,
            "speedup": (base.partition_us / final.partition_us) if final.ok and final.partition_us else None,
            # end to end as deployed (CUDA graph where capturable): the reward's basis
            "reward": self.reward,
            "baseline_e2e_us": base.deployed_us, "final_e2e_us": final.deployed_us if final.ok else None,
            "e2e_speedup": (base.deployed_us / final.deployed_us) if final.ok and final.deployed_us and
            base.deployed_us else None,
            "final_e2e_ratio": final.e2e_ratio,           # final vs the seeds, paired (persistent host)
            # the GPU the final (else the baseline) numbers were timed on: model, uuid, host, arch
            "gpu": getattr(final, "gpu", None) or getattr(base, "gpu", None),
            "runtime": {"baseline": partition_runtime(base), "final": partition_runtime(final),
                        **({"evaluator": self.ev.host_stats()} if hasattr(self.ev, "host_stats") else {})},
            "attr_op_us": final.attr_op_us,
            "slots": {sid: {"baseline_us": base.slot_us.get(sid), "final_us": final.slot_us.get(sid),
                            "best_us": b.slot_us, "impl_id": b.impl_id, "techniques": b.techniques,
                            "rel_err": final.slot_checks.get(sid, {}).get("rel_err"),
                            "baseline_attr_us": base.attr_slot_us.get(sid), "final_attr_us": final.attr_slot_us.get(sid),
                            "launches": final.attr_slot_launches.get(sid)}
                      for sid, b in self.best.items()},
            "steps": [asdict(s) for s in self.steps_log],
            # E7 rules (efficiency_rules.md): patience and, if it fired, at which commit
            "efficiency": self.efficiency_info(),
            # rollout acceptance / lineage (search fixes item 4): mode, resync factor, resync events
            "rollout_accept": {"mode": rollout_mode(), "resync": rollout_resync_factor(),
                               "resyncs": getattr(self, "resyncs", [])},
            "bandit": self.bandit.to_json(),
            # segments of a resumed run (journal); llm_replayed: responses reused, not paid again
            "resume": {"segments": len(getattr(self, "segments", []) or [None]), "llm_replayed": self.llm_replayed,
                       **({"run_segments": self.segments} if getattr(self, "segments", None) else {}),
                       **self.resume_info},
        }
        (self.out / "result.json").write_text(json.dumps(res, indent=1))
        return res
