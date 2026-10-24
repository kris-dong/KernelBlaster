# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCL (Adreno) per-slot profiles for multi-slot problems -- the NCU analogue.

Adreno exposes no hardware counters to an OpenCL program (Snapdragon
Profiler is an interactive tool), so the Adreno flow's profile is what
``cl_event`` profiling and the runtime can tell
(``docs/adreno_opencl_profiling_methodology.md``). The ModelBlaster OpenCL
driver's ``MB_MODE=profile`` run prints, per slot:

  MB_KERNEL_US <slot> <kernel> <us> n=<launches>      cl_event START..END, summed
  MB_KINFO <slot> <kernel> wg_max=.. pref_mult=.. local_mem=.. private_mem=..
  MB_DEVICE_INFO max_wg=.. compute_units=.. local_mem=.. global_mem=.. fp16=..

From these a slot summary has per-kernel times and launch counts, the
kernel's maximum work-group size (on Adreno it drops below the device
maximum when a kernel needs many registers), private memory (register
spills / private arrays in slow memory), local memory, and hints. It has no
measured bound: ``state`` is set only for launch-bound slots (many short
launches); otherwise the agent keeps the roofline state.

Same interface as ``knowledge.ncu``: ``profile_by_slot``, ``render``, ``delta``.
"""
from __future__ import annotations

import re
from typing import Optional

_KUS = re.compile(r"^MB_KERNEL_US (\S+) (\S+) (\S+)(?: n=(\d+))?", re.M)
_KINFO = re.compile(r"^MB_KINFO (\S+) (\S+) wg_max=(\d+) pref_mult=(\d+) local_mem=(\d+) private_mem=(\d+)", re.M)
_DEV = re.compile(r"^MB_DEVICE_INFO (.*)$", re.M)

# per-launch time below which a slot with several launches per op is launch-bound
LAUNCH_US = 10.0


def device_info(text: str) -> dict:
    m = _DEV.search(text)
    if not m:
        return {}
    return {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", m.group(1))}


def profile_by_slot(text: str, instances: Optional[dict] = None) -> dict[str, dict]:
    """``instances``: slot -> number of op instances (for launches per op)."""
    dev = device_info(text)
    kern: dict[str, dict[str, dict]] = {}
    for m in _KUS.finditer(text):
        kern.setdefault(m.group(1), {})[m.group(2)] = {"us": float(m.group(3)), "launches": int(m.group(4) or 1)}
    for m in _KINFO.finditer(text):
        k = kern.setdefault(m.group(1), {}).setdefault(m.group(2), {"us": 0.0, "launches": 0})
        k.update(wg_max=int(m.group(3)), pref_mult=int(m.group(4)), local_mem=int(m.group(5)),
                 private_mem=int(m.group(6)))
    return {sid: summarize(ks, dev, (instances or {}).get(sid)) for sid, ks in kern.items()}


def summarize(kernels: dict[str, dict], dev: Optional[dict] = None, n_instances: Optional[int] = None) -> dict:
    if not kernels:
        return {}
    dev = dev or {}
    tot = sum(k["us"] for k in kernels.values())
    launches = sum(k["launches"] for k in kernels.values())
    s: dict = {"n_kernels": len(kernels), "launches": launches, "duration_us": round(tot, 3),
               "kernels": [{"name": n, **k} for n, k in sorted(kernels.items(), key=lambda kv: -kv[1]["us"])]}
    wgs = [k["wg_max"] for k in kernels.values() if "wg_max" in k]
    if wgs:
        s["kernel_wg_max_min"] = min(wgs)
    s["private_mem_max"] = max((k.get("private_mem", 0) for k in kernels.values()), default=0)
    s["local_mem_max"] = max((k.get("local_mem", 0) for k in kernels.values()), default=0)
    if dev.get("max_wg"):
        s["device_max_wg"] = dev["max_wg"]
    hints = []
    per_inst = launches / n_instances if n_instances else None
    if launches and tot / launches < LAUNCH_US and (per_inst or launches) >= 2:
        s["state"] = "latency_bound"
        hints.append(f"{launches} launches averaging {tot / launches:.1f} us"
                     + (f" ({per_inst:.1f} per op instance)" if per_inst else "")
                     + ": launch overhead dominates; do more work per launch (fuse kernels/timesteps, coarsen)")
    for n, k in kernels.items():
        if dev.get("max_wg") and "wg_max" in k and k["wg_max"] < dev["max_wg"]:
            hints.append(f"{n}: max work-group size {k['wg_max']} < device {dev['max_wg']} -- register pressure; "
                         "fewer live values / smaller unroll, or a smaller work-group")
        if k.get("private_mem", 0) > 0:
            hints.append(f"{n}: {k['private_mem']} bytes of private memory per work-item (private arrays or "
                         "spills live in slow memory on Adreno); keep them in registers or __local")
    heavy = s["kernels"][0]
    if len(kernels) > 1 and tot > 0:
        hints.append(f"{heavy['name']} is {100 * heavy['us'] / tot:.0f}% of the slot's kernel time")
    s["hints"] = hints[:4]
    return s


def delta(before: Optional[dict], after: Optional[dict]) -> dict:
    if not before or not after:
        return {}
    keys = [k for k in ("duration_us", "launches", "private_mem_max", "kernel_wg_max_min") if k in before and k in after]
    return {k: round(after[k] - before[k], 3) for k in keys}


def render(s: dict) -> str:
    if not s:
        return ""
    ks = "; ".join(f"{k['name']} {k['us']:.1f} us x{k['launches']}"
                   + (f" (wg_max {k['wg_max']}, private {k.get('private_mem', 0)} B, local {k.get('local_mem', 0)} B)"
                      if "wg_max" in k else "") for k in s.get("kernels", [])[:6])
    lines = [f"OpenCL profile (cl_event times, {s.get('launches', 0)} launches): {ks}"]
    lines += [f"- {h}" for h in s.get("hints", [])]
    return "\n".join(lines)
