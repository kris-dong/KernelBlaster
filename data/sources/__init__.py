# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Problem-source abstraction — Item 2, Phase 1.

Public surface:

  - ``Problem`` — dataclass describing a single benchmark problem.
  - ``ProblemSource`` — ABC for sources.
  - ``get_source(name, **kwargs)`` — factory. Names mirror the legacy
    ``get_dataset()`` names for a smooth migration.
  - Three concrete sources: ``KernelBenchSource``,
    ``KernelBenchCUDASource``, ``KernelBenchOpenCLSource``. Each is a
    Phase 1 facade over the corresponding ``Dataset`` in ``data/``.

Phase 1 lands the surface only — no consumers migrated. See
``notes/dataset_abstraction_audit.md`` for the full audit and the
phased migration plan.
"""
from __future__ import annotations

from typing import Any

from .base import Problem, ProblemSource
from .kernelbench_source import KernelBenchSource
from .kernelbench_cuda_source import KernelBenchCUDASource
from .kernelbench_opencl_source import KernelBenchOpenCLSource

__all__ = [
    "Problem",
    "ProblemSource",
    "KernelBenchSource",
    "KernelBenchCUDASource",
    "KernelBenchOpenCLSource",
    "get_source",
]


_SOURCE_REGISTRY: dict[str, type[ProblemSource]] = {
    "kernelbench": KernelBenchSource,
    "kernelbench-cuda": KernelBenchCUDASource,
    "kernelbench-opencl": KernelBenchOpenCLSource,
}


def get_source(name: str, **kwargs: Any) -> ProblemSource:
    """Look up a ``ProblemSource`` by name and instantiate it.

    ``**kwargs`` are forwarded to the source constructor — only
    ``KernelBenchSource`` currently accepts them (for ``precision=``).
    Names match the existing ``get_dataset()`` names so a caller can
    migrate one call site at a time.

    Raises ``ValueError`` for unknown names — helps catch typos before
    they turn into confusing "no problems returned" runs.
    """
    try:
        cls = _SOURCE_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_SOURCE_REGISTRY))
        raise ValueError(f"Unknown ProblemSource: {name!r}. Known: {known}") from None
    return cls(**kwargs)
