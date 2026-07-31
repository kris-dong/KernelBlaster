# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Backend-agnostic RL machinery.

Every module in this package is deliberately free of CUDA / OpenCL /
RISC-V vocabulary — the pieces are pure algorithms (bandit,
content-hash cache, adaptive token budget, top-K seed buffer) that
each backend's RL agent can compose without inheriting a specific
backend's shape.

The CUDA-optimized agent (``opt_ncu_rl_optimized``) was the first
consumer; OpenCL / RISC-V variants pick up the same tricks by
importing from here rather than reimplementing.
"""
from .bandit import BanditArm, UCB1Bandit
from .profile_cache import ProfileCache, ProfileCacheEntry
from .seeds import TopKSeedBuffer
from .token_budget import (
    DEFAULT_TOKEN_BUDGET_TIERS,
    current_max_tokens,
    maybe_bump_token_budget,
)

__all__ = [
    # Bandit selection
    "BanditArm",
    "UCB1Bandit",
    # Profile cache
    "ProfileCache",
    "ProfileCacheEntry",
    # Top-K seeds
    "TopKSeedBuffer",
    # Token budget
    "DEFAULT_TOKEN_BUDGET_TIERS",
    "current_max_tokens",
    "maybe_bump_token_budget",
]
