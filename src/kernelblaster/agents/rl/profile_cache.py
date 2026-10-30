# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Content-hash cache for kernel profile results.

Two trajectories that converge on the same kernel source share one
expensive profile run. Extracted from ``opt_ncu_rl_optimized``'s
``NCUProfileCache`` and generalised so the entry is a
:class:`~backends.base.ProfileResult` — no CUDA-specific fields in
the cache shape, so OpenCL / RISC-V agents can plug into the same
mechanism.

Backend-agnostic cache entry: ``(primary_metric, ProfileResult, stderr)``.
The primary metric is duplicated at the top level for cheap read-back
(the RL loop wants the scalar to compare against ``initial_metric``);
the full ``ProfileResult`` carries per-op detail + raw_log for the
LLM state-analysis path.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional

from ...backends import ProfileResult


@dataclass
class ProfileCacheEntry:
    """One cached profile.

    ``primary_metric`` is what the RL loop compares (cycles for
    CUDA/RISC-V, ms for OpenCL). ``profile`` is the full ProfileResult
    including per-op detail + raw_log. ``stderr`` is the profiler's
    stderr, kept for diagnostics only.
    """
    primary_metric: float
    profile: ProfileResult
    stderr: str = ""


class ProfileCache:
    """Process-local content-hash cache.

    Not size-bounded — the cache is per-run, and a run tops out at a
    few hundred distinct kernels. If a much larger workload lands, wrap
    with an ``OrderedDict`` + LRU eviction here.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, ProfileCacheEntry] = {}

    @staticmethod
    def _hash(code: str) -> str:
        return hashlib.sha1(code.encode("utf-8")).hexdigest()

    def get(self, code: str) -> Optional[ProfileCacheEntry]:
        return self._cache.get(self._hash(code))

    def put(self, code: str, entry: ProfileCacheEntry) -> None:
        self._cache[self._hash(code)] = entry

    def __len__(self) -> int:
        return len(self._cache)


__all__ = ["ProfileCache", "ProfileCacheEntry"]
