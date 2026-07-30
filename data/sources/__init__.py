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

from typing import Any, Optional

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
    "parse_problem_numbers",
]


def parse_problem_numbers(spec: Optional[str]) -> Optional[list[int]]:
    """Parse a ``--problem-numbers``-style CLI spec into a sorted-unique list.

    Accepts comma-separated numbers and inclusive ranges. Example:
    ``"1,3,5-9"`` -> ``[1, 3, 5, 6, 7, 8, 9]``. Whitespace tolerated.
    Duplicates are removed (identical entries -> one item). Returns
    ``None`` when ``spec`` is falsy — matches the pre-Phase-6 convention
    where "no filter" is signalled by ``None``.

    Item 2, Phase 6: replaces four near-identical inline parsers
    (``data/__init__.py``, ``scripts/run_kgen_opencl.py``,
    ``scripts/run_opt_ncu_rl_optimized.py``, and one in
    ``scripts/run_reprofile.py`` that returned strings — that one is
    NOT migrated onto this helper because its consumer expects strings).
    """
    if not spec:
        return None
    parsed: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part and not part.startswith("-"):
            lo, hi = part.split("-", 1)
            parsed.update(range(int(lo), int(hi) + 1))
        else:
            parsed.add(int(part))
    return sorted(parsed)


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
