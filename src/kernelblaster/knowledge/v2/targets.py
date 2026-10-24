# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fixed abstraction levels and the target registry (families + targets as data).

    universal  >  family (simt_gpu, vector_cpu, systolic)  >  target (cuda:sm_75, ...)

A *target key* is ``<backend>:<arch>`` (``cuda:sm_89``, ``opencl:adreno740``).
Unknown targets are accepted: their family comes from the backend name and
they have no peaks (the roofline adapter then leaves utilisations absent).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

UNIVERSAL, FAMILY, TARGET = "universal", "family", "target"
LEVELS = (TARGET, FAMILY, UNIVERSAL)            # most specific first

DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "kernelblaster" / "knowledge"
BACKEND_FAMILY = {"cuda": "simt_gpu", "opencl": "simt_gpu", "riscv": "vector_cpu", "rvv": "vector_cpu",
                  "gemmini": "systolic"}


def target_key(target: str, arch: Optional[str] = None) -> str:
    """``("cuda", "sm_75")`` / ``("cuda", "75")`` -> ``cuda:sm_75``; keys pass through."""
    if ":" in (target or ""):
        return target
    a = arch or "unknown"
    if target == "cuda" and a.isdigit():
        a = f"sm_{a}"
    return f"{target}:{a}"


@dataclass
class Target:
    key: str
    family: str
    name: str = ""
    peak_gflops: Optional[float] = None
    peak_gbps: Optional[float] = None
    sms: int = 1
    threads_per_sm: int = 1024
    warp: int = 32
    smem_kb: Optional[int] = None
    l2_kb: Optional[int] = None
    launch_us: float = 3.0
    profiler: str = "roofline"
    aliases: list = field(default_factory=list)

    @property
    def backend(self) -> str:
        return self.key.split(":", 1)[0]

    @property
    def width(self) -> int:
        """Parallel lanes needed to fill the machine once."""
        return max(1, self.sms * self.threads_per_sm)

    @property
    def ridge(self) -> Optional[float]:
        if self.peak_gflops and self.peak_gbps:
            return self.peak_gflops / self.peak_gbps
        return None


class TargetRegistry:
    def __init__(self, path: Optional[Path] = None):
        path = Path(path) if path else DATA_DIR / "targets.json"
        d = json.loads(path.read_text()) if path.exists() else {"families": {}, "targets": {}}
        self.families: dict[str, dict] = d.get("families", {})
        self.targets: dict[str, Target] = {}
        self._alias: dict[str, str] = {}
        for k, t in d.get("targets", {}).items():
            self.targets[k] = Target(key=k, **t)
            for a in t.get("aliases", []):
                self._alias[a] = k

    def get(self, key: str, arch: Optional[str] = None) -> Target:
        key = target_key(key, arch)
        key = self._alias.get(key, key)
        t = self.targets.get(key)
        if t is None:
            fam = BACKEND_FAMILY.get(key.split(":", 1)[0], "unknown")
            t = self.targets[key] = Target(key=key, family=fam)
        return t

    def ensure(self, key: str, *, peaks: Optional[tuple] = None) -> Target:
        """The target for ``key``; fill missing (GFLOP/s, GB/s) peaks, e.g. from the agent's TargetProfile."""
        t = self.get(key)
        if peaks and not (t.peak_gflops and t.peak_gbps):
            t.peak_gflops, t.peak_gbps = float(peaks[0]), float(peaks[1])
        return t

    def family_of(self, key: str) -> str:
        return self.get(key).family

    def scopes(self, key: str) -> list[tuple[str, Optional[str]]]:
        """(level, scope) from the target up: target key, family, universal."""
        return [(TARGET, self.get(key).key), (FAMILY, self.family_of(key)), (UNIVERSAL, None)]

    def covers(self, level: str, scope: Optional[str], key: str) -> bool:
        """Does knowledge at (level, scope) apply to target ``key``?"""
        if level == UNIVERSAL:
            return True
        if level == FAMILY:
            return scope == self.family_of(key)
        return scope is not None and (self.get(scope).key == self.get(key).key)
