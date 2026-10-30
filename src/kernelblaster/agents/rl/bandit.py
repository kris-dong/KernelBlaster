# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""UCB1 multi-arm bandit over (state, technique) pairs.

Extracted verbatim from ``opt_ncu_rl_optimized`` so every backend's
optimized RL agent (CUDA, OpenCL, RISC-V, …) can share the same
exploration policy. The reward semantics — actual improvement
fraction, clamped to ``[-1, 2]`` — are backend-agnostic.

Cold-start behaviour is worth flagging: when there's more than one
unseen arm AND the caller passes ``traj_idx``, the bandit does a
deterministic round-robin over the relevance-sorted candidates
instead of cubed-relevance sampling. That guarantees the first
``len(unseen)`` parallel trajectories try DISTINCT actions; without
it, sampling by relevance would collapse most cold-start trajectories
onto the top-ranked candidate and starve the bandit of exploration
data.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class BanditArm:
    pulls: int = 0
    reward_sum: float = 0.0

    def mean(self) -> float:
        return self.reward_sum / self.pulls if self.pulls > 0 else 0.0

    def update(self, reward: float) -> None:
        self.pulls += 1
        self.reward_sum += reward


class UCB1Bandit:
    """Per-state UCB1 over technique names.

    Reward is the actual improvement fraction (0.12 for a 12% speedup),
    clamped to ``[-1.0, 2.0]`` so a single catastrophic step doesn't
    poison an arm forever.
    """

    def __init__(self, *, exploration_c: float = 1.4):
        self.exploration_c = exploration_c
        # arms[(state, technique)] -> BanditArm
        self._arms: Dict[Tuple[str, str], BanditArm] = {}
        self._total_pulls_per_state: Dict[str, int] = {}

    def select(
        self,
        state: str,
        candidates: List[str],
        weights: Optional[List[float]] = None,
        traj_idx: Optional[int] = None,
    ) -> str:
        """Select one candidate.

        Cold-start (any arm with 0 pulls):
        - If ``traj_idx`` is provided AND there are multiple unseen arms,
          spread parallel/early trajectories across distinct unseen arms via
          deterministic round-robin over the relevance-sorted list. This
          guarantees that the first ``len(unseen)`` trajectories each try a
          different action, instead of cubed-relevance sampling collapsing
          them all to the dominant arm before the bandit has any data.
        - Otherwise: cubed-relevance weighted sampling (legacy behavior).
          Falls back to uniform random if no weights or all-zero.

        Warm: standard UCB1 exploitation.
        """
        if not candidates:
            raise ValueError("UCB1Bandit.select called with empty candidates")

        # Find unseen arms.
        unseen: List[Tuple[str, float]] = []
        for i, c in enumerate(candidates):
            arm = self._arms.get((state, c))
            if arm is None or arm.pulls == 0:
                w = float(weights[i]) if weights is not None and i < len(weights) else 1.0
                unseen.append((c, max(0.0, w)))

        if unseen:
            if traj_idx is not None and len(unseen) > 1:
                # Deterministic spread: traj_idx 0 → top relevance, traj 1 →
                # 2nd, etc. With T trajectories and K unseen arms, each arm is
                # tried by ⌈T/K⌉ trajectories. Stable sort with arm name as
                # tiebreaker so identical-relevance arms still get distinct
                # round-robin slots.
                unseen_sorted = sorted(unseen, key=lambda t: (-t[1], t[0]))
                return unseen_sorted[traj_idx % len(unseen_sorted)][0]
            # Cubed-relevance weighted sampling among unseen arms.
            cubed = [(c, w * w * w) for c, w in unseen]
            total_w = sum(w for _, w in cubed)
            if total_w <= 0.0:
                return random.choice([c for c, _ in unseen])
            r = random.random() * total_w
            acc = 0.0
            for c, w in cubed:
                acc += w
                if r <= acc:
                    return c
            return cubed[-1][0]  # numerical safety

        # All arms pulled at least once → UCB1.
        total = max(1, self._total_pulls_per_state.get(state, 0))
        ln_total = math.log(total)
        best = None
        best_score = -float("inf")
        for c in candidates:
            arm = self._arms[(state, c)]
            score = arm.mean() + self.exploration_c * math.sqrt(ln_total / arm.pulls)
            if score > best_score:
                best_score = score
                best = c
        return best  # type: ignore[return-value]

    def update(self, state: str, technique: str, reward: float) -> None:
        reward = max(-1.0, min(2.0, reward))
        arm = self._arms.setdefault((state, technique), BanditArm())
        arm.update(reward)
        self._total_pulls_per_state[state] = self._total_pulls_per_state.get(state, 0) + 1


__all__ = ["BanditArm", "UCB1Bandit"]
