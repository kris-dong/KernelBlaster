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

from pathlib import Path
from typing import Any, Optional, Tuple

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
    "TIER_MARKERS",
    "path_tier_and_problem",
]


# Every source-native tier name that can appear as a path segment in
# run-output trees. Kept in one place so ``reprofile.py`` and
# ``reprofile_nsys.py`` (and any future tier-aware consumer) don't
# reinvent it. Ordering is unimportant — this is a membership set.
TIER_MARKERS: frozenset[str] = frozenset({
    "level1", "level2", "level3",
    "L1", "L2", "L3",
    "sol-level1", "sol-level2", "sol-level3",
})


def path_tier_and_problem(
    path: Path,
) -> Tuple[Optional[str], Optional[str]]:
    """Extract ``(tier, problem_name)`` from a run-output-tree path.

    Convention: ``.../<tier>/<problem_name>/…`` where ``<tier>`` is one
    of :data:`TIER_MARKERS`. Both reprofile scripts (Item 2, Phase 7)
    parse paths this way. Uses the LAST tier marker in the path — paths
    can legitimately contain the marker more than once (e.g. an output
    dir named ``level2_results``); the tier associated with the leaf
    kernel wins.

    Returns ``(None, None)`` if no tier marker is found — callers fall
    back to their own heuristic (usually ``path.parent.name``).
    """
    parts = path.parts
    last_idx: Optional[int] = None
    for i, part in enumerate(parts):
        if part in TIER_MARKERS:
            last_idx = i
    if last_idx is None:
        return None, None
    tier = parts[last_idx]
    problem_name = parts[last_idx + 1] if last_idx + 1 < len(parts) else None
    return tier, problem_name


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
