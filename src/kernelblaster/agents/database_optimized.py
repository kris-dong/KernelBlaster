# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Back-compat shim for the legacy ``database_optimized`` module.

Phase 3a merged ``OptimizedOptimizationDatabase``, ``TieredLLMInterface``,
and ``extract_metrics_json`` into ``database.py``. The unified
``GPUOptimizationDatabase`` provides the optimized path opt-in via the
``cheap_llm=`` / ``cost_tracker=`` constructor kwargs; when those are not
passed, behavior is identical to the legacy base class.

This shim re-exports the old names so existing callers continue to work:

  - ``OptimizedOptimizationDatabase`` is now an alias of ``GPUOptimizationDatabase``.
  - ``TieredLLMInterface`` and ``extract_metrics_json`` are re-exported
    from ``database``.

New code should import directly from ``database``; this shim will be
removed in a future cleanup.
"""
from __future__ import annotations

from .database import (
    GPUOptimizationDatabase as OptimizedOptimizationDatabase,
    LLMInterface,
    TieredLLMInterface,
    extract_metrics_json,
)

__all__ = [
    "OptimizedOptimizationDatabase",
    "LLMInterface",
    "TieredLLMInterface",
    "extract_metrics_json",
]
