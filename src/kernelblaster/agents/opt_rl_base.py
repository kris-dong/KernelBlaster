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

from typing import Any, Dict, Optional

from .feedback import FeedbackAgent
from .database import OptimizationDatabase, OptimizationEntry, CompositeOptimization


class RLAgentBase(FeedbackAgent):
    """Shared scaffolding for ``RLNCUAgent`` and ``RLOpenCLAgent``.

    Subclasses must set ``self.backend`` in ``__init__`` (typically via
    ``self.gpu.backend()``) before any base method that reads
    ``self.backend`` is called. They must also populate the canonical
    state fields ``self.iteration_count``, ``self.total_trajectories``,
    ``self.best_metric``, ``self.initial_metric``, ``self.database``,
    and ``self.replay_buffer`` during their ``__init__``.
    """

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
