# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Shared base class for the per-backend RL optimization agents (Phase 4f).

After Phases 4a–4e established the Backend abstraction (technique_map,
parse_profile, extract_code_from_response, format_result_artifact,
get_default_optimizations, metric_name / format_metric / extract_primary_metric,
RLNodeConfig), the two RL agents share enough method bodies to extract
a base class. This module owns the genuinely-shared methods; per-backend
subclasses (``RLNCUAgent``, ``RLOpenCLAgent``) override what's truly
hardware-specific:

  - NCU-flavoured profile gathering / Speed-Of-Light section extraction (CUDA)
  - OpenCL event-time profile gathering / on-board reference generation (OpenCL)
  - Verification-pool global-best tracking (OpenCL-only)
  - Database policy-update lifecycle hook (CUDA-only)

Methods that live here:

  - ``calculate_reward`` — byte-identical between agents.
  - ``_lookup_optim_entry_by_name`` — byte-identical lookup.
  - ``_try_add_default_optimizations`` — uses ``backend.get_default_optimizations``.
  - ``get_performance_summary`` — backend-named JSON keys produced by the
    ``_perf_summary_extras`` hook so existing JSON consumers see the same
    schema as before this refactor.

Phase 4f-future work that can lift more methods to this base once the
per-backend kernel-source attribute names (``code_to_optimize_fp`` /
``kernel_to_optimize_fp``) are unified onto a single canonical attribute:
``__init__`` skeleton, ``initialize``, ``run``, ``run_rollout``,
``apply_optimization``, ``get_feedback`` skeleton.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from .feedback import FeedbackAgent, FeedbackConfig
from .database import (
    OptimizationDatabase,
    OptimizationEntry,
    CompositeOptimization,
    LLMInterface,
)
from .rl_agents import (
    ReplayBuffer,
    PolicyEvaluationAgent,
    PerfGapAnalysisAgent,
    ParameterUpdateAgent,
)


class RLAgentBase(FeedbackAgent):
    """Shared scaffolding for ``RLNCUAgent`` and ``RLOpenCLAgent``.

    The shared ``__init__`` sets up everything common to both backends:
    the Backend object (``self.backend`` via ``self.gpu.backend()``),
    test-code references, kernel-source references under canonical names
    (``self.kernel_source_fp`` / ``self.kernel_source``), the
    ``OptimizationDatabase``, ``ReplayBuffer``, RL sub-agents, state
    counters, and concurrency locks. Subclasses override
    ``_init_backend_extras`` to add backend-specific attributes (e.g.
    OpenCL's global-best verification-pool fields and SSH-exec timeout).

    Subclasses MUST pass their per-backend kernel-source path under the
    canonical name ``kernel_source_fp`` to ``super().__init__``; they may
    keep their original public parameter name (``code_to_optimize_fp`` /
    ``kernel_to_optimize_fp``) in their own ``__init__`` signature for
    back-compat with existing callers.
    """

    def __init__(
        self,
        fb_config: FeedbackConfig,
        kernel_source_fp: Path,
        database_path: Path,
        max_rollout_steps: int = 5,
        replay_buffer_size: int = 1000,
        update_frequency: int = 10,
        database: Optional[OptimizationDatabase] = None,
    ):
        super().__init__(fb_config)

        # Phase 2 Backend abstraction: single source of truth for technique
        # map, file-naming, profile parsing, board_host (OpenCL), etc.
        # Routed via gpu.backend() so this picks the right backend for the
        # GPU type carried in fb_config.
        self.backend = self.gpu.backend()

        # Test driver + kernel source (canonical names — Phase 4f rename).
        self.test_code_fp = fb_config.test_code_fp
        self.test_code = fb_config.test_code_fp.read_text()
        self.kernel_source_fp = kernel_source_fp
        self.kernel_source = kernel_source_fp.read_text()

        # Database: use the shared instance if one was passed in (graph
        # nodes do this when running multiple problems in one process);
        # otherwise construct a fresh one rooted at ``database_path``.
        gpu_report_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "algo-sol-modeling/algo-space/gpu_optimization_report.md"
        )
        llm_interface = LLMInterface(self.model, self.agent_logger)
        if database is not None:
            self.database = database
        else:
            self.database = OptimizationDatabase(
                database_path, gpu_report_path, llm_interface, backend=self.backend
            )

        # RL components.
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
        self.max_rollout_steps = max_rollout_steps
        self.update_frequency = update_frequency

        self.policy_evaluation_agent = PolicyEvaluationAgent()
        self.perf_gap_analysis_agent = PerfGapAnalysisAgent()
        self.parameter_update_agent = ParameterUpdateAgent()

        # State tracking — unified naming since Phase 4d.
        self.iteration_count = 0
        self.total_trajectories = 0
        self.best_metric = float("inf")
        self.initial_metric = None

        # Concurrency.
        self._trajectory_lock: asyncio.Lock = asyncio.Lock()
        self.current_trajectory = None

        # Default RL iteration count (can be overridden by the workflow).
        self.num_rl_iterations = 50

        # Backend-specific extras (OpenCL global-best fields, etc.) go in
        # the subclass hook so they always run AFTER all shared state is set.
        self._init_backend_extras()

    def _init_backend_extras(self) -> None:
        """Subclass hook for backend-specific instance attributes.

        Default: no-op (CUDA agent has no extras beyond what the base sets).
        OpenCL agent overrides to set up SSH timeout + global-best fields.
        """
        return None

    # ------------------------------------------------------------------
    # Shared state-derivation glue (Phase 4f.3a)
    # ------------------------------------------------------------------
    async def _derive_state(self, profile_result, code: str) -> str:
        """Compute the optimisation-state string for the current kernel.

        Backend-agnostic — the per-backend bits (metrics extraction, cycles
        arg conversion) come from ``backend.derive_metrics_for_state`` and
        ``backend.state_cycles_arg``. On any database exception, returns a
        ``"<backend>_unknown"`` sentinel so callers can proceed.
        """
        metrics = self.backend.derive_metrics_for_state(profile_result)
        cycles_arg = self.backend.state_cycles_arg(profile_result)
        try:
            return await self.database.get_state_from_ncu_report(
                profile_result.raw_log, metrics, code, elapsed_cycles=cycles_arg
            )
        except Exception:
            return f"{self.backend.name}_unknown"

    # ------------------------------------------------------------------
    # Reward calculation — backend-independent.
    # Byte-identical in both agents pre-Phase-4f.
    # ------------------------------------------------------------------
    def calculate_reward(
        self,
        predicted_improvement: Optional[float],
        actual_improvement: float,
        is_faster: bool,
    ) -> float:
        """Calculate reward based on prediction accuracy and actual performance.

        Safely handles None/zero predicted_improvement by skipping accuracy bonus.
        """
        base_reward = actual_improvement / 100.0
        try:
            safe_predicted = float(predicted_improvement) if predicted_improvement is not None else 0.0
        except (TypeError, ValueError):
            safe_predicted = 0.0

        if safe_predicted > 0.0:
            accuracy = min(actual_improvement / safe_predicted, 2.0)
            accuracy_bonus = 0.2 if 0.8 <= accuracy <= 1.2 else -0.1 * abs(accuracy - 1.0)
        else:
            accuracy_bonus = 0.0

        penalty = -0.5 if not is_faster else 0.0
        return base_reward + accuracy_bonus + penalty

    # ------------------------------------------------------------------
    # Database lookup — backend-independent.
    # Byte-identical in both agents pre-Phase-4f.
    # ------------------------------------------------------------------
    def _lookup_optim_entry_by_name(
        self, technique_name: str
    ) -> Optional[OptimizationEntry | CompositeOptimization]:
        """Search the database for an OptimizationEntry or CompositeOptimization by name."""
        # Search individual techniques.
        for state_data in self.database.optimization_strategies.values():
            for opt in state_data.get("optimizations", []):
                if opt.technique == technique_name:
                    return opt
        # Search composite optimisations.
        for comps in self.database.composite_optimizations.values():
            for comp in comps:
                if comp.get_composite_id() == technique_name:
                    return comp
        return None

    # ------------------------------------------------------------------
    # Default optimisations fallback — backend-aware via Phase 4c.
    # ------------------------------------------------------------------
    def _try_add_default_optimizations(self, current_state: str) -> bool:
        """Fallback when no optimizations are recorded for a discovered state.

        The catalog (bottleneck -> [(technique, predicted_pct), ...]) lives on
        the backend — see ``Backend.get_default_optimizations``.
        """
        try:
            defaults = self.backend.get_default_optimizations()
            primary_bottleneck = next(
                (b for b in defaults if b in current_state), None
            )
            if primary_bottleneck is not None:
                for technique, improvement in defaults[primary_bottleneck]:
                    self.database.add_new_optimization(current_state, technique, improvement)
                self.agent_logger.info(
                    f"Added {len(defaults[primary_bottleneck])} default optimizations for state: {current_state}"
                )
                return True
        except Exception as e:
            self.agent_logger.error(f"Error adding default optimizations: {e}")
        return False

    # ------------------------------------------------------------------
    # Performance summary — shared skeleton with backend-named JSON keys.
    # ------------------------------------------------------------------
    def get_performance_summary(self) -> Dict[str, Any]:
        """Comprehensive performance summary as a JSON-serializable dict.

        Backend-named metric keys (``initial_cycles`` / ``best_cycles`` for CUDA,
        ``initial_time_ms`` / ``best_time_ms`` for OpenCL) come from the
        ``_perf_summary_extras`` hook so the persisted JSON schema is
        preserved across the Phase 4f refactor (downstream analytics tools
        keep working).
        """
        overall_improvement = (
            ((self.initial_metric - self.best_metric) / self.initial_metric * 100)
            if self.initial_metric
            else 0
        )
        return {
            "total_trajectories": self.total_trajectories,
            "iteration_count": self.iteration_count,
            "overall_improvement": overall_improvement,
            "buffer_stats": self.replay_buffer.get_statistics(),
            "database_stats": self.database.get_database_stats(),
            **self._perf_summary_extras(),
        }

    def _perf_summary_extras(self) -> Dict[str, Any]:
        """Subclass hook for backend-named metric keys in the summary dict.

        Default: empty (no extra keys). Subclasses return e.g.
        ``{"initial_cycles": self.initial_metric, "best_cycles": self.best_metric}``.
        """
        return {}
