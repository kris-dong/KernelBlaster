# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Shared base for backend-agnostic optimized RL agents (Phase 5.7).

Owns everything the CUDA-optimized flow's tricks share across backends:
tiered-model dispatch (cheap plan/fix + hard codegen), UCB1 bandit,
content-hash profile cache, best-of-N seed buffer, adaptive token
budget, plus the shared LLM-call helpers ``_llm_codegen`` /
``_llm_fix`` that consume :meth:`Backend.categorise_technique` and
:meth:`Backend.deterministic_fix` (P5.5).

Concrete backend-specific optimized agents subclass this and only own
what's genuinely target-specific:

  - CUDA: NCU + nsys profile capture, NCU-cycles parsing, per-solution-
    kernel gpu_time_ns integration.
  - OpenCL: ``[PROFILE] name: <ms>`` marker parsing, on-board
    reference-gen for verification.
  - RISC-V: modelblaster ``MODELBLASTER_WALL_CYCLES`` + per-op mcycle
    CSV parsing (via :class:`SpikeExecStrategy` output).

The base does NOT touch profile capture — subclasses implement
``_gather_perf_metrics`` and the caller flow around it. The base does
own the full LLM-call surface (codegen + fix dispatch, cost tracking,
token-budget bump) so subclasses inherit the model-heterogeneity
flow verbatim.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..backends import Backend
from .cost_tracker import CostTracker
from .feedback import FeedbackAgent, FeedbackConfig
from .progress_writer import ProgressWriter
from .rl import (
    ProfileCache,
    TopKSeedBuffer,
    UCB1Bandit,
    maybe_bump_token_budget,
)
from .rl_agents import ReplayBuffer
from .utils import generate_code_retry
from .utils.perf_log import perf_span


class RLOptimizedAgentBase(FeedbackAgent):
    """Backend-agnostic scaffolding for optimized RL agents.

    Subclasses pass their :class:`Backend` instance via ``backend=`` to
    :meth:`__init__` and are expected to:

      - Populate ``self.database`` in their own ``__init__`` (the class
        varies — CUDA uses :class:`OptimizedOptimizationDatabase`;
        other backends may use a plain :class:`OptimizationDatabase`
        with ``cheap_llm=`` passed in).
      - Implement backend-specific profile capture and the outer
        ``initialize`` / ``run`` entry points (these vary because the
        artifact naming, baseline-marker paths, and fix-loop details
        depend on the backend).

    What subclasses inherit for free:

      - Tiered model config (``model_plan`` / ``model_codegen_simple``
        / ``model_codegen_hard`` / ``model_fix``) driven by env vars.
      - UCB1 bandit, profile cache, top-K seed buffer, replay buffer.
      - Pruning knobs.
      - The shared LLM-call helpers ``_llm_codegen`` (tier-dispatched
        via ``self.backend.categorise_technique``) and ``_llm_fix``.
      - Adaptive token-budget bump on every LLM response.
    """

    #: Short label used for perf spans / logs. Subclasses override so
    #: dashboard filters see distinct agent names.
    agent_perf_label: str = "opt_rl_optimized"

    def __init__(
        self,
        fb_config: FeedbackConfig,
        code_to_optimize_fp: Path,
        *,
        backend: Backend,
        max_rollout_steps: int = 5,
        replay_buffer_size: int = 1000,
        num_rl_iterations: int = 50,
        seed_from_init_count: int = 10,
        bandit_exploration: float = 1.4,
        prune_patience: int = 2,
        prune_regression_pct: float = -5.0,
        max_fix_attempts: int = 2,
        cost_tracker: Optional[CostTracker] = None,
        problem_id: Optional[str] = None,
        progress_writer: Optional[ProgressWriter] = None,
    ):
        super().__init__(fb_config)
        self.backend = backend
        self.cost_tracker = cost_tracker
        self.problem_id = problem_id
        self.progress_writer = progress_writer

        # Source / test code refs. Kept under the canonical names the
        # existing CUDA agent used — subclasses can add per-backend
        # aliases (kernel_source_fp / kernel_to_optimize_fp / …) in
        # their own __init__.
        self.test_code_fp: Path = fb_config.test_code_fp
        self.test_code: str = fb_config.test_code_fp.read_text()
        self.code_to_optimize_fp: Path = code_to_optimize_fp
        self.code_to_optimize: str = code_to_optimize_fp.read_text()

        # Tiered model dispatch (env-driven). All default to
        # ``self.model`` so a clean run with no extra config still
        # works — the heterogeneity is opt-in per env var.
        self.model_plan: str = os.getenv("MODEL_PLAN") or self.model
        self.model_codegen_simple: str = os.getenv("MODEL_CODEGEN_SIMPLE") or self.model
        self.model_codegen_hard: str = os.getenv("MODEL_CODEGEN_HARD") or self.model
        self.model_fix: str = os.getenv("MODEL_FIX") or self.model_codegen_simple

        # Extracted RL machinery (P5.1-P5.4).
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
        self.bandit = UCB1Bandit(exploration_c=bandit_exploration)
        self.profile_cache = ProfileCache()
        self.seed_buffer = TopKSeedBuffer(
            top_k=5,
            init_count=seed_from_init_count,
            init_code=self.code_to_optimize,
        )

        # Run config.
        self.max_rollout_steps = max_rollout_steps
        self.num_rl_iterations = num_rl_iterations
        self.seed_from_init_count = seed_from_init_count
        self.prune_patience = prune_patience
        self.prune_regression_pct = prune_regression_pct
        self.max_fix_attempts = max_fix_attempts

        # Tracking — shared shape. Subclasses that use different
        # metric units (ms for OpenCL, cycles for CUDA/RISC-V) can
        # still read these; the naming is legacy from the CUDA
        # original but the semantics are backend-agnostic (lower is
        # better).
        self.total_trajectories = 0
        self.best_cycles: float = float("inf")
        self.initial_cycles: Optional[int] = None
        self.last_ncu_log: str = ""

        self._trajectory_lock: asyncio.Lock = asyncio.Lock()

        # Subclasses populate ``self.database`` before use — its type
        # varies (OptimizedOptimizationDatabase for CUDA, ... for
        # others). Declared here so the type checker sees the
        # attribute exists.
        self.database = None

    # ------------------------------------------------------------------
    # LLM dispatchers (backend-agnostic; drive the heterogeneous flow)
    # ------------------------------------------------------------------

    async def _llm_codegen(
        self, messages: List[Dict[str, str]], *, technique_name: str
    ) -> str:
        """Tier-routed codegen call.

        ``self.backend.categorise_technique(name) -> "simple" | "hard"``
        drives which model runs. Cost tracking + adaptive token budget
        are applied uniformly regardless of tier.
        """
        category = self.backend.categorise_technique(technique_name)
        model = (
            self.model_codegen_hard
            if category == "hard"
            else self.model_codegen_simple
        )
        self.agent_logger.info(
            f"Codegen dispatch: technique={technique_name} "
            f"category={category} model={model}"
        )
        with perf_span(
            phase="llm_codegen",
            problem_id=self.problem_id,
            agent=self.agent_perf_label,
            model=model,
        ) as span:
            span.set_extra(technique=technique_name, category=category)
            response = await generate_code_retry(
                messages=messages,
                model=model,
                logger=self.agent_logger,
                max_retries=2,
            )
            text = response.generations[0] if response.generations else ""
            usage = getattr(response, "usage", None)
            if usage:
                span.set_extra(
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                )
        if self.cost_tracker is not None:
            self.cost_tracker.record(
                model=model,
                usage=usage,
                role=f"codegen_{category}",
                problem_id=self.problem_id,
                logger=self.agent_logger,
            )
        maybe_bump_token_budget(text, usage, logger=self.agent_logger)
        return text

    async def _llm_fix(self, messages: List[Dict[str, str]]) -> str:
        """Cheap-model fix call. Uses ``self.model_fix`` (defaults to
        ``model_codegen_simple``)."""
        with perf_span(
            phase="llm_fix",
            problem_id=self.problem_id,
            agent=self.agent_perf_label,
            model=self.model_fix,
        ) as span:
            response = await generate_code_retry(
                messages=messages,
                model=self.model_fix,
                logger=self.agent_logger,
                max_retries=2,
            )
            text = response.generations[0] if response.generations else ""
            usage = getattr(response, "usage", None)
            if usage:
                span.set_extra(
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                )
        if self.cost_tracker is not None:
            self.cost_tracker.record(
                model=self.model_fix,
                usage=usage,
                role="fix",
                problem_id=self.problem_id,
                logger=self.agent_logger,
            )
        maybe_bump_token_budget(text, usage, logger=self.agent_logger)
        return text
