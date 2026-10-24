# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional knowledge object for the single-file RL flows.

Bundles an :class:`OutcomeLog`, a :class:`ClaimStore` and (when a reflection
model is configured) a :class:`SemanticUpdater`, fixed to one run context
(target, arch, op_class, problem). Agents hold ``knowledge: Optional[...]``;
with ``None`` they behave exactly as before.

Env (``from_env``): ``KB_KNOWLEDGE_DIR`` enables it (outcomes.jsonl and
claims.json live there; share the directory across runs to accumulate),
``KB_REFLECT_MODEL`` names the reflection model (unset: record outcomes and
use existing claims, no reflection), ``KB_REFLECT_EVERY`` the cadence.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from .claims import ClaimStore
from .outcomes import (
    COMPILE_ERROR,
    RUNTIME_ERROR,
    WRONG_OUTPUT,
    OutcomeLog,
    OutcomeRecord,
)
from .reflect import SemanticUpdater

_WRONG = re.compile(r"mismatch|incorrect|allclose|max(imum)? (abs|diff)|verification|wrong (output|result)"
                    r"|does not match|not close|nan", re.I)
_COMPILE = re.compile(r"\berror:|nvcc|compil|undefined reference|ptxas|syntax error|was not declared", re.I)


def classify_error(text: Optional[str]) -> str:
    """Map a FeedbackError / tool message to an outcome status."""
    t = text or ""
    if _COMPILE.search(t):
        return COMPILE_ERROR
    if _WRONG.search(t):
        return WRONG_OUTPUT
    return RUNTIME_ERROR


class SemanticKnowledge:
    def __init__(self, root: Path, *, target: str, arch: Optional[str] = None,
                 op_class: str = "other", problem_id: Optional[str] = None,
                 slot: Optional[str] = None, reflect_model: Optional[str] = None,
                 every_n: int = 8, k: int = 12, logger=None,
                 known_techniques=None, known_states=None, run_id: Optional[str] = None):
        self.root = Path(root)
        self.target, self.arch, self.op_class = target, arch, op_class
        self.problem_id, self.slot, self.run_id, self.k = problem_id, slot, run_id, k
        self.logger = logger
        self.outcomes = OutcomeLog(self.root / "outcomes.jsonl")
        self.claims = ClaimStore(self.root / "claims.json")
        self.updater = (SemanticUpdater(self.outcomes, self.claims, model=reflect_model,
                                        every_n=every_n, logger=logger,
                                        known_techniques=known_techniques,
                                        known_states=known_states,
                                        trace_dir=self.root / "reflect_traces")
                        if reflect_model else None)

    @classmethod
    def from_env(cls, **ctx) -> Optional["SemanticKnowledge"]:
        root = os.getenv("KB_KNOWLEDGE_DIR")
        if not root:
            return None
        ctx.setdefault("reflect_model", os.getenv("KB_REFLECT_MODEL") or None)
        ctx.setdefault("every_n", int(os.getenv("KB_REFLECT_EVERY", "8")))
        return cls(Path(root), **ctx)

    # -- prompts ----------------------------------------------------------
    @property
    def stats_context(self) -> dict:
        return {"op_class": self.op_class, "arch": self.arch}

    def notes(self, *, state: Optional[str] = None, technique: Optional[str] = None) -> str:
        """Claims in scope + "already tried" for this problem; "" when there is nothing."""
        claims = self.claims.relevant(target=self.target, arch=self.arch, op_class=self.op_class,
                                      state=state, technique=technique,
                                      problem_id=self.problem_id, slot=self.slot, k=self.k)
        blocks = []
        if claims:
            blocks.append("Lessons from past attempts (+support/-contradict counts):\n"
                          + self.claims.render(claims))
        tried = self.claims.already_tried(self.problem_id, self.slot) if self.problem_id else ""
        if tried:
            blocks.append("Already tried on this kernel (avoid repeating dead ends):\n" + tried)
        return "\n\n".join(blocks)

    def plan_notes(self, state: Optional[str] = None) -> str:
        return self.notes(state=state)

    def codegen_notes(self, technique: str, state: Optional[str] = None) -> str:
        return self.notes(state=state, technique=technique)

    # -- recording --------------------------------------------------------
    def record(self, **fields) -> OutcomeRecord:
        """Append an outcome (context fields filled in) and update the tallies."""
        for k, v in (("target", self.target), ("arch", self.arch), ("op_class", self.op_class),
                     ("problem_id", self.problem_id), ("slot", self.slot), ("run_id", self.run_id)):
            fields.setdefault(k, v)
        rec = self.outcomes.append(OutcomeRecord(**fields))
        self.claims.observe(rec)
        return rec

    async def maybe_reflect(self, *, force: bool = False) -> list[dict]:
        if self.updater is None:
            self.claims.save()
            return []
        try:
            return await self.updater.maybe_reflect(force=force)
        except Exception as exc:   # knowledge upkeep must never break optimisation
            if self.logger is not None:
                self.logger.warning(f"[semantic-kb] reflection failed: {exc}")
            return []

    def save(self) -> None:
        self.claims.save()
