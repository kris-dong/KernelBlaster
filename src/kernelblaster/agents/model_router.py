# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model heterogeneity for the multi-slot flow: which model runs which LLM call.

Roles (each defaults to ``--model``):

    codegen_simple  mechanical edits (coalescing, vectorising, launch config)
    codegen_hard    structural rewrites (tiling, tensor cores, fusion, flash)
    fix             compile / runtime / mismatch repairs (default codegen_simple)
    reflect         the semantic update pass (v1 and v2 claims)
    upkeep          v2 hypothesis generation + state discovery

The legacy optimized-RL flow's env vars set the codegen roles: ``MODEL_CODEGEN_SIMPLE``,
``MODEL_CODEGEN_HARD``, ``MODEL_FIX``. ``reflect`` and ``upkeep`` stay opt-in (unset = that
pass makes no LLM calls), so ``MODEL_PLAN`` does not turn them on. A JSON config
(``--models-config``) may give ``{"models": {role: model}, "policy": "category,stall:3"}``.

The policy says when a codegen call uses ``codegen_hard`` (any rule firing
escalates; comma-separated, applied in order, the first that fires is logged):

    category      the technique is structurally hard (the legacy flow's split;
                  a v2 technique may say so with ``params.tier``)
    stall:N       the slot has had >= N attempts since its last best
    share:X       the slot is >= X (0..1) of the partition time
    explore       a v2 hypothesis (an untested idea) rather than a known technique
    fix-escalate  the last fix attempt goes to codegen_hard (a fix rule)
    fix-follow    a fix goes to the model that wrote the kernel (a fix rule; before fix-escalate)
    always / never

Default policy: ``category,fix-escalate``. When every role names the same
model the router is a no-op.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROLES = ("codegen_simple", "codegen_hard", "fix", "reflect", "upkeep")
DEFAULT_POLICY = "category,fix-escalate"

# legacy flow's patterns (backends/cuda.py) plus the multi-slot / OpenCL / v2 technique names
HARD_PATTERNS = ("shared_memory_tiling", "tensor_core", "wmma", "mma", "kernel_fusion", "fused_operations",
                 "register_tiling", "register_blocking", "algorithmic", "composite", "hybrid", "implicit_gemm",
                 "double_buffering", "flash", "online_softmax", "split_k", "persistent", "cp_async",
                 "single_pass_fusion", "local_memory_tiling", "winograd", "im2col", "epilogue_fusion")
ENV = {"codegen_simple": "MODEL_CODEGEN_SIMPLE", "codegen_hard": "MODEL_CODEGEN_HARD", "fix": "MODEL_FIX"}


def technique_tier(name: str, technique=None) -> str:
    """'hard' | 'simple': a v2 technique's ``params.tier`` wins, else the name patterns."""
    tier = ((getattr(technique, "params", None) or {}).get("tier") if technique is not None else None)
    if tier in ("simple", "hard"):
        return tier
    low = (name or "").lower()
    return "hard" if any(p in low for p in HARD_PATTERNS) else "simple"


@dataclass(frozen=True)
class Rule:
    kind: str
    arg: Optional[float] = None

    @classmethod
    def parse(cls, s: str) -> "Rule":
        kind, _, arg = s.strip().partition(":")
        if kind not in ("category", "stall", "share", "explore", "fix-escalate", "fix-follow", "always", "never"):
            raise ValueError(f"unknown heterogeneity rule {s!r}")
        if kind in ("stall", "share"):
            if not arg:
                raise ValueError(f"rule {kind} needs an argument ({kind}:N)")
            return cls(kind, float(arg))
        return cls(kind)


@dataclass
class ModelRouter:
    models: dict
    policy: list = field(default_factory=lambda: parse_policy(DEFAULT_POLICY))

    def __post_init__(self):
        missing = [r for r in ("codegen_simple", "codegen_hard") if not self.models.get(r)]
        if missing:
            raise ValueError(f"model router needs {missing}")
        self.models.setdefault("fix", self.models["codegen_simple"])
        for r in ("reflect", "upkeep"):
            self.models.setdefault(r, None)

    @classmethod
    def build(cls, model: str, *, simple: Optional[str] = None, hard: Optional[str] = None,
              fix: Optional[str] = None, reflect: Optional[str] = None, upkeep: Optional[str] = None,
              policy: Optional[str] = None, config: Optional[str] = None, env=os.environ) -> "ModelRouter":
        """CLI flags > config file > env vars > ``model``."""
        cfg = json.loads(Path(config).read_text()) if config else {}
        cm = cfg.get("models", {})
        unknown = set(cm) - set(ROLES)
        if unknown:
            raise ValueError(f"models config: unknown roles {sorted(unknown)} (roles: {', '.join(ROLES)})")
        flags = {"codegen_simple": simple, "codegen_hard": hard, "fix": fix, "reflect": reflect, "upkeep": upkeep}
        m = {}
        for r in ROLES:
            m[r] = flags[r] or cm.get(r) or (env.get(ENV[r]) if r in ENV else None)
        m["codegen_simple"] = m["codegen_simple"] or model
        m["codegen_hard"] = m["codegen_hard"] or model
        m["fix"] = m["fix"] or m["codegen_simple"]
        pol = policy or cfg.get("policy") or env.get("KB_HETERO_POLICY") or DEFAULT_POLICY
        return cls(m, parse_policy(pol))

    @property
    def heterogeneous(self) -> bool:
        return len({self.models["codegen_simple"], self.models["codegen_hard"], self.models["fix"]}) > 1

    def codegen(self, *, technique: str, tier: str, share: float = 0.0, stall: int = 0,
                kind: str = "technique") -> tuple[str, str, str]:
        """-> (model, role, why)."""
        for r in self.policy:
            why = None
            if r.kind == "always":
                why = "always"
            elif r.kind == "category" and tier == "hard":
                why = f"hard technique {technique}"
            elif r.kind == "stall" and stall >= r.arg:
                why = f"slot stalled {stall} attempts"
            elif r.kind == "share" and share >= r.arg:
                why = f"slot is {share:.0%} of the partition"
            elif r.kind == "explore" and kind == "hypothesis":
                why = "untested hypothesis"
            elif r.kind == "never":
                break
            if why:
                return self.models["codegen_hard"], "codegen_hard", why
        return self.models["codegen_simple"], "codegen_simple", "default"

    def fix(self, *, attempt: int, max_attempts: int, writer: Optional[str] = None) -> tuple[str, str]:
        """-> (model, role) for fix attempt ``attempt`` (1-based). ``writer`` is the model that wrote the kernel
        being fixed (used by the ``fix-follow`` rule; None = the fix role's model)."""
        if attempt >= max_attempts and any(r.kind == "fix-escalate" for r in self.policy):
            return self.models["codegen_hard"], "fix_escalated"
        if writer and any(r.kind == "fix-follow" for r in self.policy):
            return writer, "fix_escalated" if writer == self.models["codegen_hard"] != self.models["fix"] else "fix"
        return self.models["fix"], "fix"

    def describe(self) -> dict:
        return {"models": dict(self.models), "policy": ",".join(r.kind + (f":{r.arg:g}" if r.arg is not None
                                                                         else "") for r in self.policy),
                "heterogeneous": self.heterogeneous}


def parse_policy(s: str) -> list[Rule]:
    return [Rule.parse(x) for x in s.split(",") if x.strip()]
