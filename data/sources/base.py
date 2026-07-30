# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Problem-source abstraction — Item 2, Phase 1 (adapter layer).

The ``ProblemSource`` ABC decouples the *which benchmark* dimension from
the *which backend* dimension. Phase 1 lands the surface + facade
implementations over the existing ``Dataset`` classes; no consumers are
migrated yet. Phase 2+ progressively moves consumers off
``get_dataset(name)`` onto ``get_source(name)`` + ``Problem`` objects.

The audit that motivated this abstraction is ``notes/dataset_abstraction_audit.md``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional


@dataclass(frozen=True)
class Problem:
    """A single benchmark problem, source-agnostic.

    ``curated_artifacts`` is role-keyed (``"driver"``, ``"kernel"``,
    ``"reference"``, …) rather than filename-keyed — the audit's core
    complaint about the existing ``Dataset`` classes is that each
    invents its own key names (``driver_cpp_fp`` vs ``driver_c_fp``)
    and downstream code has to switch on them. Role names normalise
    that: whatever the concrete filename is, ``problem.curated_artifacts["driver"]``
    always points at the driver.

    ``reference_code`` carries an inlined string (typically a PyTorch
    ``.py`` module) for sources like KernelBench where the reference
    isn't a file the graph consumes directly but LLM prompts do.

    ``backends_supported`` names the set of backend keys (``"cuda"``,
    ``"opencl"``, …) this problem can be run against. Sources with a
    1:1 backend mapping declare a single-element set; sources whose
    reference is backend-agnostic (KernelBench torch `.py`) declare
    both.
    """

    id: str                                # e.g. "kernelbench-cuda:level1/003_Foo"
    source: str                            # ProblemSource.name, e.g. "kernelbench-cuda"
    tier: str                              # source-native tier ("level1" / "sol-level1" / "L1")
    problem_num: int
    problem_name: str
    curated_artifacts: Mapping[str, Path] = field(default_factory=dict)
    reference_code: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    backends_supported: frozenset[str] = frozenset()

    def artifact(self, role: str) -> Path:
        """Convenience accessor: ``problem.artifact("driver")`` etc.

        Raises ``KeyError`` if the role isn't present. Callers that
        want a soft lookup should use ``problem.curated_artifacts.get``.
        """
        return self.curated_artifacts[role]


class ProblemSource(ABC):
    """Contract every problem source satisfies.

    Concrete sources own their disk layout, their tier taxonomy, and
    their backend compatibility. The graph and RL agents consume
    ``Problem`` objects and never touch the source's internals
    (this is the Phase 5 endpoint).
    """

    name: str = ""                         # e.g. "kernelbench" / "kernelbench-cuda" / "kernelbench-opencl"

    @abstractmethod
    def iter_problems(
        self,
        *,
        tier: Optional[str] = None,
        problem_numbers: Optional[list[int]] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Iterator[Problem]:
        """Yield problems matching the (optional) tier + numeric filters.

        Sources with no tier concept ignore ``tier``. Sources with
        their own filter semantics (KernelBench's precision-injected
        variants) may honour additional kwargs through subclasses.
        """

    @abstractmethod
    def supports_backend(self, backend_name: str) -> bool:
        """Whether this source's problems can be run against a given backend."""

    # ---- optional metadata surface (default no-ops; sources override) ----

    def curated_root_for(self, backend_name: str) -> Optional[Path]:
        """Filesystem root under which this source's artifacts live for
        the given backend. Returns ``None`` if the source produces no
        curated artifacts for that backend (e.g. KernelBench torch
        source: no on-disk driver/kernel, reference is inlined).
        """
        return None

    def tier_dir_for(self, backend_name: str, tier: str) -> str:
        """Map a source-native tier name to the backend's on-disk tier
        directory. Default: identity. OpenCL's ``sol-level2 -> L2``
        mapping (currently in ``data.kernelbench_opencl``) will move
        here in Phase 2.
        """
        return tier

    def artifact_filenames(self, backend_name: str) -> Mapping[str, str]:
        """Role -> filename mapping for a given backend. Default: empty.
        Concrete sources return e.g. ``{"driver": "driver.cpp",
        "kernel": "init.cu"}`` for CUDA.
        """
        return {}
