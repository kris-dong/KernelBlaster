# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NCU profiles for multi-slot problems, per slot.

The driver's ``MB_MODE=profile`` runs every op exactly once, and every kernel
lives in its slot's namespace (``mbslot_<id>::``), so ONE ncu run over
``regex:mbslot_`` profiles every slot and each kernel row is attributed by
its demangled name. Per slot, metrics are duration-weighted over its
kernels; NCU's rule hints (with estimated speedups) are kept for prompts.

Needs GPU performance-counter access (``ERR_NVGPUCTRPERM`` otherwise);
callers fall back to the roofline state when :func:`ncu_error` says so.
"""
from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass, field
from typing import Optional

SECTIONS = ("SpeedOfLight", "Occupancy", "LaunchStats", "MemoryWorkloadAnalysis")

#: raw counters added with ``--metrics`` (no extra section): shared-memory bank
#: conflicts vs wavefronts, and global sectors moved vs bytes the instructions
#: asked for (ideal sectors = bytes / 32), loads and stores. They feed the
#: canonical ``bank_conflicts`` / ``uncoalesced`` metrics. Off with
#: ``KB_NCU_ACCESS_METRICS=0``.
ACCESS_METRICS = (
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "smsp__sass_data_bytes_mem_global_op_ld.sum",
    "smsp__sass_data_bytes_mem_global_op_st.sum",
)

# summary key -> (section, metric) in ncu's --csv --page details output
_PCT = {
    "compute_throughput": ("GPU Speed Of Light Throughput", "Compute (SM) Throughput"),
    "memory_throughput": ("GPU Speed Of Light Throughput", "Memory Throughput"),
    "dram_throughput": ("GPU Speed Of Light Throughput", "DRAM Throughput"),
    "l1_throughput": ("GPU Speed Of Light Throughput", "L1/TEX Cache Throughput"),
    "l2_throughput": ("GPU Speed Of Light Throughput", "L2 Cache Throughput"),
    "achieved_occupancy": ("Occupancy", "Achieved Occupancy"),
    "theoretical_occupancy": ("Occupancy", "Theoretical Occupancy"),
    "l1_hit_rate": ("Memory Workload Analysis", "L1/TEX Hit Rate"),
    "l2_hit_rate": ("Memory Workload Analysis", "L2 Hit Rate"),
}
_DURATION = ("GPU Speed Of Light Throughput", "Duration")
_REGS = ("Launch Statistics", "Registers Per Thread")
_SMEM = ("Launch Statistics", "Static Shared Memory Per Block")
_WAVES = ("Launch Statistics", "Waves Per SM")


def ncu_command(*, prefix: str = "mbslot_", slot: Optional[str] = None, slots=None,
                launch_count: int = 256) -> str:
    """ncu prefix for ``MB_MODE=profile <exe>``: every slot, just ``slot``, or
    the kernels of ``slots`` (fewer kernels -> fewer replays)."""
    if slot:
        slots = [slot]
    if slots:
        rx = (f"{prefix}{slots[0]}::" if len(slots) == 1 else
              f"{prefix}(" + "|".join(sorted(slots)) + ")::")
    else:
        rx = prefix
    secs = " ".join(f"--section={s}" for s in SECTIONS)
    if os.getenv("KB_NCU_ACCESS_METRICS", "1").lower() not in ("0", "false", "no", "off"):
        secs += " --metrics " + ",".join(ACCESS_METRICS)
    return (f"ncu --kernel-name-base demangled --print-kernel-base mangled -k regex:'{rx}' "
            f"--launch-count {launch_count} --page details {secs} --csv")


def ncu_error(text: str) -> Optional[str]:
    """Why ncu produced no profile (None if it did)."""
    if "ERR_NVGPUCTRPERM" in text:
        return "no GPU performance-counter permission (ERR_NVGPUCTRPERM)"
    if "No kernels were profiled" in text:
        return "no kernels matched the slot filter"
    m = re.search(r"==ERROR== (.*)", text)
    if m and '"ID"' not in text:
        return m.group(1).strip()
    return None


def _num(v: str) -> Optional[float]:
    try:
        return float(v.replace(",", ""))
    except (ValueError, AttributeError):
        return None


@dataclass
class KernelProfile:
    id: str
    name: str
    metrics: dict = field(default_factory=dict)     # (section, metric) -> (unit, value)
    rules: list = field(default_factory=list)       # (rule, description, est. speedup %)

    def value(self, key) -> Optional[float]:
        mv = self.metrics.get(key)
        return mv[1] if mv else None

    def raw(self, name: str) -> Optional[float]:
        """A ``--metrics`` counter by name (whatever section ncu files it under)."""
        for (_, m), (_, v) in self.metrics.items():
            if m == name:
                return v
        return None

    @property
    def duration_us(self) -> float:
        mv = self.metrics.get(_DURATION)
        if not mv or mv[1] is None:
            return 0.0
        unit = (mv[0] or "").lower()
        scale = {"nsecond": 1e-3, "ns": 1e-3, "usecond": 1.0, "us": 1.0, "msecond": 1e3, "ms": 1e3}.get(unit, 1.0)
        return mv[1] * scale


def parse_ncu_csv(text: str) -> list[KernelProfile]:
    lines = [l for l in text.splitlines() if l.startswith('"')]
    if not lines:
        return []
    out: dict[str, KernelProfile] = {}
    for r in csv.DictReader(io.StringIO("\n".join(lines))):
        kid = r.get("ID", "")
        k = out.setdefault(kid, KernelProfile(kid, r.get("Kernel Name", "")))
        if r.get("Metric Name"):
            k.metrics[(r.get("Section Name", ""), r["Metric Name"])] = (r.get("Metric Unit", ""), _num(r.get("Metric Value", "")))
        if r.get("Rule Name") and r.get("Rule Description"):
            k.rules.append((r["Rule Name"], r["Rule Description"].strip(), _num(r.get("Estimated Speedup", ""))))
    return list(out.values())


def slot_of(kernel_name: str, prefix: str = "mbslot_") -> Optional[str]:
    """Slot id of a kernel row. Mangled names (``_ZN<len>mbslot_<id>...``: ncu >= 2024 drops the namespace
    from demangled names, so the command prints mangled ones) or demangled ``mbslot_<id>::`` names."""
    m = re.match(r"_ZN(\d+)" + re.escape(prefix), kernel_name)
    if m:
        ident = kernel_name[m.start(1) + len(m.group(1)):][:int(m.group(1))]
        return ident[len(prefix):] or None
    m = re.search(rf"{prefix}(\w+?)::", kernel_name)
    return m.group(1) if m else None


def summarize(kernels: list[KernelProfile]) -> dict:
    """Duration-weighted metrics of a slot's kernels, its bound, and top rule hints."""
    if not kernels:
        return {}
    tot = sum(k.duration_us for k in kernels) or float(len(kernels))
    w = {k.id: (k.duration_us or tot / len(kernels)) / tot for k in kernels}
    s: dict = {"n_kernels": len(kernels), "duration_us": round(sum(k.duration_us for k in kernels), 3)}
    for key, mk in _PCT.items():
        vals = [(w[k.id], k.value(mk)) for k in kernels if k.value(mk) is not None]
        if vals:
            s[key] = round(sum(a * v for a, v in vals) / sum(a for a, _ in vals), 2)
    regs = [k.value(_REGS) for k in kernels if k.value(_REGS) is not None]
    if regs:
        s["registers_per_thread_max"] = int(max(regs))
    sm = [k.value(_SMEM) for k in kernels if k.value(_SMEM) is not None]
    if sm:
        s["static_smem_per_block_max"] = int(max(sm))
    s.update(access_pct(kernels))
    heavy = max(kernels, key=lambda k: k.duration_us)
    waves = heavy.value(_WAVES)
    if waves is not None:
        s["waves_per_sm_heaviest"] = waves
    best: dict = {}
    for k in kernels:
        for name, desc, sp in k.rules:
            if name not in best or (sp or 0) > (best[name][1] or 0):
                best[name] = (desc, sp)
    ranked = sorted(best.items(), key=lambda kv: -(kv[1][1] or 0))
    s["rules"] = [{"rule": n, "est_speedup_pct": sp, "hint": d[:300]} for n, (d, sp) in ranked[:3]]
    s["state"] = ncu_state(s)
    return s


def access_pct(kernels: list[KernelProfile]) -> dict:
    """Bank-conflict and uncoalesced-access excess over the slot's kernels (counters summed)."""
    def tot(name):
        vals = [k.raw(name) for k in kernels if k.raw(name) is not None]
        return sum(vals) if vals else None
    out = {}
    conf, wav = tot(ACCESS_METRICS[0]), tot(ACCESS_METRICS[1])
    if conf is not None and wav:
        out["smem_bank_conflict_pct"] = round(100.0 * conf / max(wav - conf, 1.0), 2)
    sect = [tot(ACCESS_METRICS[2]), tot(ACCESS_METRICS[3])]
    byts = [tot(ACCESS_METRICS[4]), tot(ACCESS_METRICS[5])]
    if all(x is not None for x in sect + byts) and sum(byts) > 0:
        out["global_excess_sector_pct"] = round(max(0.0, 100.0 * (32.0 * sum(sect) / sum(byts) - 1.0)), 2)
    return out


def ncu_state(s: dict) -> str:
    """Bound from measured SOL throughputs (same names as the roofline state)."""
    c, m = s.get("compute_throughput", 0.0), s.get("memory_throughput", 0.0)
    if max(c, m) < 40.0:
        # low on both: too little parallel work or latency (occupancy says which)
        return "latency_bound" if s.get("achieved_occupancy", 100.0) < 30.0 else \
            ("underutilized_compute" if c >= m else "underutilized_memory")
    return "compute_bound" if c >= m else "memory_bound"


def profile_by_slot(text: str, prefix: str = "mbslot_") -> dict[str, dict]:
    by: dict[str, list[KernelProfile]] = {}
    for k in parse_ncu_csv(text):
        sid = slot_of(k.name, prefix)
        if sid:
            by.setdefault(sid, []).append(k)
    return {sid: summarize(ks) for sid, ks in by.items()}


def delta(before: Optional[dict], after: Optional[dict]) -> dict:
    """after - before for the numeric summary metrics both have."""
    if not before or not after:
        return {}
    keys = [k for k in list(_PCT) + ["registers_per_thread_max", "duration_us", "smem_bank_conflict_pct",
                                     "global_excess_sector_pct"] if k in before and k in after]
    return {k: round(after[k] - before[k], 3) for k in keys}


def render(s: dict) -> str:
    if not s:
        return ""
    keys = ["duration_us", "compute_throughput", "memory_throughput", "dram_throughput", "achieved_occupancy",
            "theoretical_occupancy", "l1_hit_rate", "l2_hit_rate", "registers_per_thread_max",
            "static_smem_per_block_max", "waves_per_sm_heaviest", "smem_bank_conflict_pct",
            "global_excess_sector_pct"]
    nums = ", ".join(f"{k}={s[k]}" for k in keys if k in s)
    lines = [f"NCU (measured, {s.get('n_kernels', 0)} kernel launches; throughputs and occupancy in % of peak): {nums}"]
    for r in s.get("rules", []):
        sp = f" (est. {r['est_speedup_pct']:.0f}% speedup)" if r.get("est_speedup_pct") else ""
        lines.append(f"- NCU {r['rule']}{sp}: {r['hint']}")
    return "\n".join(lines)
