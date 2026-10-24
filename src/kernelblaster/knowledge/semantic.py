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
use existing claims, no reflection), ``KB_REFLECT_EVERY`` the cadence. ``KB_KNOWLEDGE=v2`` switches to knowledge v2
(``knowledge.db``; NCU text reports are mapped onto the predicate states
through ``v2.metrics.NCUTextAdapter``), ``KB_UPKEEP_MODEL`` names its
hypothesis / state-discovery model.
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
                 known_techniques=None, known_states=None, run_id: Optional[str] = None,
                 version: str = "v1", upkeep_model: Optional[str] = None):
        self.root = Path(root)
        self.target, self.arch, self.op_class = target, arch, op_class
        self.problem_id, self.slot, self.run_id, self.k = problem_id, slot, run_id, k
        self.logger = logger
        self.kv = None
        if version == "v2":
            # knowledge v2: SQLite stores, NCU text -> canonical metrics -> predicate states
            from .v2 import KnowledgeV2
            from .v2.reflect_v2 import SemanticUpdaterV2
            self.kv = KnowledgeV2(self.root, upkeep_model=upkeep_model, logger=logger)
            self.outcomes, self.claims = self.kv.outcomes, self.kv.claims
            self.updater = (SemanticUpdaterV2(self.kv, model=reflect_model, every_n=every_n, logger=logger,
                                              known_techniques=known_techniques,
                                              trace_dir=self.root / "reflect_traces")
                            if reflect_model else None)
            return
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
        ctx.setdefault("version", os.getenv("KB_KNOWLEDGE", "v1"))
        ctx.setdefault("upkeep_model", os.getenv("KB_UPKEEP_MODEL") or None)
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
    @property
    def is_v2(self) -> bool:
        return self.kv is not None

    def record(self, *, ncu_log_before: Optional[str] = None, ncu_log_after: Optional[str] = None,
               **fields) -> OutcomeRecord:
        """Append an outcome (context fields filled in) and update the tallies.

        v2: the NCU text reports become canonical metric vectors, the state
        chain replaces the technique-DB state (kept as ``legacy_state``), and
        the outcome feeds the hierarchical statistics."""
        for k, v in (("target", self.target), ("arch", self.arch), ("op_class", self.op_class),
                     ("problem_id", self.problem_id), ("slot", self.slot), ("run_id", self.run_id)):
            fields.setdefault(k, v)
        if self.kv is not None:
            return self._record_v2(OutcomeRecord(**fields), ncu_log_before, ncu_log_after)
        rec = self.outcomes.append(OutcomeRecord(**fields))
        self.claims.observe(rec)
        return rec

    def _record_v2(self, rec: OutcomeRecord, log_before: Optional[str], log_after: Optional[str]) -> OutcomeRecord:
        from .v2 import Choice
        kv = self.kv
        tk = kv.registry.get(rec.target, rec.arch).key
        before = kv.vector(tk, profile={"log": log_before}, profile_kind="ncu_text") if log_before else {}
        after = kv.vector(tk, profile={"log": log_after}, profile_kind="ncu_text") if log_after else None
        cls = kv.classify(before, target_key=tk, op_class=rec.op_class, dtype=rec.dtype)
        legacy = rec.state_before
        item = kv.ontology.resolve(rec.technique) or rec.technique
        sp = rec.speedup
        reward = (1.0 - 1.0 / sp) if sp else (-0.05 if rec.status == "compile_error" else -0.1)
        kv.record(rec, choice=Choice(item, "technique", ""), reward=max(-1.0, min(1.0, reward)), cls=cls,
                  before=before, after=after, extra={"legacy_state": legacy,
                                                     "legacy_state_mapped": kv.hierarchy.map_legacy(legacy)})
        return rec

    def state_chain(self, ncu_log: str) -> Optional[list]:
        """v2: the predicate-state chain an NCU text report classifies to (None in v1)."""
        if self.kv is None or not ncu_log:
            return None
        tk = self.kv.registry.get(self.target, self.arch).key
        vec = self.kv.vector(tk, profile={"log": ncu_log}, profile_kind="ncu_text")
        return self.kv.classify(vec, target_key=tk, op_class=self.op_class).chain

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
