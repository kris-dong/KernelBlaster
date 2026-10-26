# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical metrics and the per-target adapters that produce them (design section 3).

A *metric vector* is ``{name: {"v": float, "src": "measured"|"estimated", "from": str}}``.
Missing metrics are absent, never zero: predicates over them evaluate to
unknown. Adapters share one interface::

    adapter.vector(profile=None, estimate=None) -> MetricVector

``estimate`` is a :class:`RooflineInput` (bytes, flops, time, work items) that
every target can provide; ``profile`` is whatever the target's profiler gave
(NCU per-slot summary, NCU text log, ``opencl_profile`` summary, mcycle counts).
The adapter is chosen by the agent's ``TargetProfile.profiler`` module
(``knowledge.ncu`` -> :class:`NCUAdapter`, ``knowledge.opencl_profile`` ->
:class:`OpenCLProfileAdapter`). Measured
values override estimated ones metric by metric.

Units: utilisations, hit rates, occupancy and register pressure in percent;
``parallel_fill`` and ``launch_frac`` are fractions in [0, 1];
``arith_intensity`` is FLOP per DRAM byte.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .targets import Target

MEASURED, ESTIMATED = "measured", "estimated"

#: canonical metric -> (meaning, unit); the vocabulary states and predicts are written in
CANONICAL = {
    "bw_util": ("% of peak DRAM bandwidth", "%"),
    "compute_util": ("% of peak compute", "%"),
    "mem_util": ("% of the busiest memory unit (L1/L2/DRAM)", "%"),
    "arith_intensity": ("FLOP per DRAM byte", "flop/B"),
    "parallel_fill": ("work items / machine width (capped at 1)", "frac"),
    "occupancy": ("achieved occupancy, % of the hardware's max resident warps", "%"),
    "launch_frac": ("share of slot time that is launch/sync overhead", "frac"),
    "l1_hit": ("L1 hit rate", "%"),
    "l2_hit": ("L2 hit rate", "%"),
    "reg_pressure": ("registers per thread, % of the 255 limit", "%"),
    "divergence": ("100 - branch efficiency", "%"),
    # measured only when the profiler provides them (NCU SourceCounters / MemoryWorkloadAnalysis
    # tables, or their rule hints); absent otherwise, so states over them stay at the parent
    "bank_conflicts": ("excess shared-memory wavefronts from bank conflicts, % of ideal", "%"),
    "uncoalesced": ("excess global-memory sectors from uncoalesced access, % of ideal", "%"),
}
#: |delta| below which a metric counts as unchanged ("flat"), per unit
FLAT = {"%": 2.0, "frac": 0.03, "flop/B": 0.05}

MetricVector = dict


def metric(v: float, src: str, frm: str) -> dict:
    return {"v": round(float(v), 4), "src": src, "from": frm}


def values(vec: Optional[MetricVector]) -> dict[str, float]:
    return {k: m["v"] for k, m in (vec or {}).items() if m and m.get("v") is not None}


def measured_names(vec: Optional[MetricVector]) -> set[str]:
    return {k for k, m in (vec or {}).items() if m and m.get("src") == MEASURED}


def merge(*vecs: Optional[MetricVector]) -> MetricVector:
    """Later vectors win, but a measured value is never replaced by an estimate."""
    out: MetricVector = {}
    for vec in vecs:
        for k, m in (vec or {}).items():
            if m is None or m.get("v") is None:
                continue
            if k in out and out[k]["src"] == MEASURED and m["src"] != MEASURED:
                continue
            out[k] = m
    return out


def delta(before: Optional[MetricVector], after: Optional[MetricVector], *,
          measured_only: bool = True) -> dict[str, float]:
    """after - before on metrics both vectors have (measured on both sides by default)."""
    b, a = before or {}, after or {}
    out = {}
    for k in set(a) & set(b):
        if measured_only and not (a[k]["src"] == MEASURED and b[k]["src"] == MEASURED):
            continue
        out[k] = round(a[k]["v"] - b[k]["v"], 4)
    return out


def flat_band(name: str) -> float:
    return FLAT.get(CANONICAL.get(name, ("", "%"))[1], 2.0)


# ---------------------------------------------------------------- roofline estimate (any target)

@dataclass
class RooflineInput:
    bytes: float                      # DRAM bytes the op instances must move (manifest)
    flops: float
    time_us: float                    # measured slot time (all instances)
    work_items: Optional[float] = None  # largest instance's output elements
    launches: int = 1                 # kernel calls in the slot time (>= instances)
    launch_frac: Optional[float] = None  # measured share of launch/idle overhead (runtime attribution)
    peak_gflops: Optional[float] = None  # overrides the target peak (fp16 tensor-core ops on tensor-core GPUs)


def roofline_vector(t: Target, est: RooflineInput) -> MetricVector:
    out: MetricVector = {}
    if est.bytes > 0:
        out["arith_intensity"] = metric(est.flops / est.bytes, ESTIMATED, "manifest flops/bytes")
    sec = max(est.time_us, 1e-3) * 1e-6
    if t.peak_gbps:
        bw = min(100.0, 100.0 * est.bytes / sec / 1e9 / t.peak_gbps)
        out["bw_util"] = metric(bw, ESTIMATED, "bytes/time/peak")
        out["mem_util"] = metric(bw, ESTIMATED, "= bw_util (estimate)")
    peak_f = est.peak_gflops or t.peak_gflops
    if peak_f:
        out["compute_util"] = metric(min(100.0, 100.0 * est.flops / sec / 1e9 / peak_f),
                                     ESTIMATED, "flops/time/peak")
    if est.work_items:
        out["parallel_fill"] = metric(min(1.0, est.work_items / t.width), ESTIMATED,
                                      f"work items/{t.width} lanes")
    if est.launch_frac is not None:
        out["launch_frac"] = metric(max(0.0, min(1.0, est.launch_frac)), MEASURED,
                                    f"{est.launches} launches: deployed overhead share (runtime attribution)")
    elif est.time_us > 0:
        out["launch_frac"] = metric(min(1.0, est.launches * t.launch_us / est.time_us), ESTIMATED,
                                    f"{est.launches} launches x {t.launch_us} us / time")
    return out


# ---------------------------------------------------------------- adapters

class MetricAdapter:
    """Profiler output -> canonical metrics. Subclasses override :meth:`from_profile`."""

    profiler = "roofline"

    def __init__(self, target: Target):
        self.target = target

    def from_profile(self, profile: Optional[dict]) -> MetricVector:
        return {}

    def vector(self, profile: Optional[dict] = None, estimate: Optional[RooflineInput] = None,
               time_us: Optional[float] = None) -> MetricVector:
        est = roofline_vector(self.target, estimate) if estimate is not None else {}
        meas = self.from_profile(profile) if profile else {}
        return merge(est, meas)


class NCUAdapter(MetricAdapter):
    """CUDA: :func:`knowledge.ncu.summarize` per-slot summaries (duration-weighted)."""

    profiler = "ncu"
    _MAP = {"dram_throughput": "bw_util", "compute_throughput": "compute_util",
            "memory_throughput": "mem_util", "achieved_occupancy": "occupancy",
            "l1_hit_rate": "l1_hit", "l2_hit_rate": "l2_hit"}

    def from_profile(self, s: Optional[dict], slot_us: Optional[float] = None) -> MetricVector:
        out: MetricVector = {}
        if not s:
            return out
        for src, dst in self._MAP.items():
            if s.get(src) is not None:
                out[dst] = metric(s[src], MEASURED, f"ncu:{src}")
        if s.get("registers_per_thread_max") is not None:
            out["reg_pressure"] = metric(100.0 * s["registers_per_thread_max"] / 255.0, MEASURED,
                                         "ncu:registers_per_thread_max/255")
        if s.get("waves_per_sm_heaviest") is not None:
            out["parallel_fill"] = metric(min(1.0, float(s["waves_per_sm_heaviest"])), MEASURED,
                                          "ncu:waves_per_sm (heaviest kernel)")
        for key, dst in (("smem_bank_conflict_pct", "bank_conflicts"), ("global_excess_sector_pct", "uncoalesced")):
            if s.get(key) is not None:
                out[dst] = metric(s[key], MEASURED, f"ncu:{key}")
        for r in s.get("rules") or []:            # NCU rule hints carry an estimated speedup
            name, sp = r.get("rule") or "", r.get("est_speedup_pct")
            dst = "bank_conflicts" if "BankConflict" in name else "uncoalesced" if "Uncoalesced" in name else None
            if dst and sp is not None and dst not in out:
                out[dst] = metric(float(sp), MEASURED, f"ncu rule {name} (est. speedup %)")
        if slot_us and s.get("duration_us"):
            # kernel time under the profiler vs the driver's wall time for the slot
            out["launch_frac"] = metric(max(0.0, min(1.0, 1.0 - s["duration_us"] / slot_us)), MEASURED,
                                        "1 - ncu kernel duration / slot time")
        return out

    def vector(self, profile=None, estimate=None, time_us=None) -> MetricVector:
        est = roofline_vector(self.target, estimate) if estimate is not None else {}
        meas = self.from_profile(profile, slot_us=time_us or (estimate.time_us if estimate else None))
        return merge(est, meas)


def canonical_ncu_delta(d: dict) -> dict:
    """v1 ``metrics_delta`` (``ncu.delta`` summary keys) -> measured canonical deltas."""
    out = {dst: d[src] for src, dst in NCUAdapter._MAP.items() if isinstance(d.get(src), (int, float))}
    if isinstance(d.get("registers_per_thread_max"), (int, float)):
        out["reg_pressure"] = 100.0 * d["registers_per_thread_max"] / 255.0
    for src, dst in (("smem_bank_conflict_pct", "bank_conflicts"), ("global_excess_sector_pct", "uncoalesced")):
        if isinstance(d.get(src), (int, float)):
            out[dst] = d[src]
    for k in CANONICAL:                          # already canonical (v2 records)
        if k not in out and isinstance(d.get(k), (int, float)):
            out[k] = d[k]
    return out


#: legacy single-file flow: NCU *text* report lines (``agents/utils/parsing.parse_ncu_metrics``
#: returns 0.0 for missing lines, so the adapter parses the log itself and drops what is absent)
_NCU_TEXT = {
    "bw_util": r"DRAM\s+Throughput",
    "compute_util": r"Compute\s*\(SM\)\s*Throughput",
    "mem_util": r"(?<!DRAM )Memory\s+Throughput",
    "occupancy": r"Achieved\s+Occupancy",
    "l1_hit": r"L1/TEX\s+Hit\s+Rate",
    "l2_hit": r"L2\s+(?:Cache\s+)?Hit\s+Rate",
    "_regs": r"Registers\s+Per\s+Thread",
    "_waves": r"Waves\s+Per\s+SM",
}


class NCUTextAdapter(MetricAdapter):
    """CUDA, single-file flows: the raw ``ncu`` text report (``profile={"log": text}``)."""

    profiler = "ncu"

    def from_profile(self, profile: Optional[dict]) -> MetricVector:
        text = (profile or {}).get("log") or ""
        out: MetricVector = {}
        for name, pat in _NCU_TEXT.items():
            m = re.search(rf"{pat}[^\n0-9]*?([0-9][0-9,]*(?:\.[0-9]+)?)", text, re.I)
            if not m:
                continue
            v = float(m.group(1).replace(",", ""))
            if name == "_regs":
                out["reg_pressure"] = metric(100.0 * v / 255.0, MEASURED, "ncu text: registers/255")
            elif name == "_waves":
                out["parallel_fill"] = metric(min(1.0, v), MEASURED, "ncu text: waves per SM")
            else:
                out[name] = metric(v, MEASURED, f"ncu text: {pat}")
        return out


class OpenCLProfileAdapter(MetricAdapter):
    """OpenCL / Adreno: :func:`knowledge.opencl_profile.summarize` per-slot summaries.

    Adreno exposes no counters to OpenCL, so only what ``cl_event`` and the
    kernel info give is measured: launch overhead (kernel time vs slot time)
    and register pressure (a kernel's max work-group size below the device's
    means registers limit residency). Utilisations stay roofline estimates.
    Counter names a future profiler may add (``alu_util``, ``l1_hit``, ...) map
    directly."""

    profiler = "opencl_profile"
    _MAP = {"alu_util": "compute_util", "bw_util": "bw_util", "mem_util": "mem_util", "l1_hit": "l1_hit",
            "l2_hit": "l2_hit", "occupancy": "occupancy", "texture_l1_hit": "l1_hit"}

    def from_profile(self, s: Optional[dict], slot_us: Optional[float] = None) -> MetricVector:
        s = s or {}
        out = {dst: metric(s[src], MEASURED, f"opencl:{src}") for src, dst in self._MAP.items() if s.get(src) is not None}
        if slot_us and s.get("duration_us"):
            out["launch_frac"] = metric(max(0.0, min(1.0, 1.0 - s["duration_us"] / slot_us)), MEASURED,
                                        "1 - cl_event kernel time / slot time")
        if s.get("kernel_wg_max_min") and s.get("device_max_wg"):
            out["reg_pressure"] = metric(100.0 * (1.0 - s["kernel_wg_max_min"] / s["device_max_wg"]), MEASURED,
                                         "1 - kernel wg_max / device max_wg")
        return out

    def vector(self, profile=None, estimate=None, time_us=None) -> MetricVector:
        est = roofline_vector(self.target, estimate) if estimate is not None else {}
        meas = self.from_profile(profile, slot_us=time_us or (estimate.time_us if estimate else None))
        return merge(est, meas)


AdrenoAdapter = OpenCLProfileAdapter


class MCycleAdapter(MetricAdapter):
    """RISC-V V (stub): ``mcycle`` / ``minstret`` and optional vector-unit counters."""

    profiler = "mcycle"

    def from_profile(self, profile: Optional[dict]) -> MetricVector:
        p = profile or {}
        out: MetricVector = {}
        if p.get("vector_util") is not None:
            out["compute_util"] = metric(p["vector_util"], MEASURED, "rvv:vector_util")
        if p.get("l1d_hit") is not None:
            out["l1_hit"] = metric(p["l1d_hit"], MEASURED, "rvv:l1d_hit")
        if p.get("mem_stall_frac") is not None:
            out["mem_util"] = metric(100.0 * p["mem_stall_frac"], MEASURED, "rvv:mem_stall_frac")
        return out


_ADAPTERS = {"ncu": NCUAdapter, "ncu_text": NCUTextAdapter, "opencl_profile": OpenCLProfileAdapter,
             "adreno": OpenCLProfileAdapter,
             "mcycle": MCycleAdapter, "roofline": MetricAdapter}


def adapter_for(target: Target, kind: Optional[str] = None) -> MetricAdapter:
    """The adapter for ``target``'s profiler (``kind`` overrides, e.g. ``ncu_text``)."""
    return _ADAPTERS.get(kind or target.profiler, MetricAdapter)(target)


def render(vec: Optional[MetricVector], *, keys: Optional[list] = None) -> str:
    """``bw_util=62.1% (measured), arith_intensity=0.25 (est.)``"""
    bits = []
    for k in keys or list(CANONICAL):
        m = (vec or {}).get(k)
        if not m:
            continue
        unit = CANONICAL[k][1]
        v = f"{m['v']:.1f}%" if unit == "%" else f"{m['v']:.3g}"
        bits.append(f"{k}={v}" + ("" if m["src"] == MEASURED else " (est.)"))
    return ", ".join(bits)
