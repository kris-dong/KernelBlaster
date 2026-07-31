# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Best-of-N seed buffer for RL rollouts.

The RL loop's first N rollouts always start from the original ``init``
kernel; subsequent rollouts pick from a running top-K buffer of the
best variants seen so far. Extracted from
``opt_ncu_rl_optimized._top_k_seeds`` / ``_pick_seed_code`` /
``_update_top_k`` so every optimized backend agent inherits the same
seeding behaviour without reimplementing.

Key convention: ``metric`` is lower-is-better everywhere in this
codebase (cycles for CUDA/RISC-V, ms for OpenCL). The buffer keeps
the ``top_k`` smallest metrics and their corresponding source code.
"""
from __future__ import annotations

from typing import List, Optional, Tuple


class TopKSeedBuffer:
    """Running top-K (metric, code) buffer.

    Not thread-safe — the RL agent's ``_trajectory_lock`` (or its
    equivalent) serialises writes from parallel rollouts. Reads are
    lock-free because ``_seeds`` is only mutated inside ``update``.
    """

    def __init__(
        self,
        *,
        top_k: int = 5,
        init_count: int = 10,
        init_code: str = "",
    ):
        """
        Args:
            top_k: Buffer size — keep this many best variants.
            init_count: Rollouts 0..init_count-1 seed from ``init_code``.
                Later rollouts round-robin over the top-K buffer.
            init_code: The original ``init`` kernel source. Also acts
                as the fallback when the buffer is empty (rollouts
                past ``init_count`` before anything's been recorded).
        """
        self._top_k = top_k
        self._init_count = init_count
        self._init_code = init_code
        self._seeds: List[Tuple[float, str]] = []

    def set_init_code(self, code: str) -> None:
        """Late-bind the init code (e.g. after ``initialize`` repaired it)."""
        self._init_code = code

    def update(self, metric: float, code: str) -> None:
        """Record a new (metric, code) — keep the top_k smallest."""
        if not code:
            return
        self._seeds.append((metric, code))
        self._seeds = sorted(self._seeds, key=lambda t: t[0])[: self._top_k]

    def pick(self, rollout_idx: int) -> Tuple[str, Optional[float]]:
        """Pick the starting code for ``rollout_idx``.

        Returns ``(code, seed_metric)``. ``seed_metric`` is ``None``
        when we're seeding from init (no prior baseline to
        propagate). Callers log ``seed_metric`` for observability.
        """
        if rollout_idx < self._init_count or not self._seeds:
            return self._init_code, None
        metric, code = self._seeds[rollout_idx % len(self._seeds)]
        return code, metric

    def __len__(self) -> int:
        return len(self._seeds)

    def snapshot(self) -> List[Tuple[float, str]]:
        """Read-only view for logging / debugging."""
        return list(self._seeds)


__all__ = ["TopKSeedBuffer"]
