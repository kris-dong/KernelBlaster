# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""KernelBench (PyTorch reference) source — Phase 1 facade over ``KernelBenchDataset``.

Wraps ``data.kernelbench.KernelBenchDataset`` and yields ``Problem``
objects with the torch reference inlined as ``reference_code``. This
source declares no curated artifacts — the graph must run kgen to
produce them for either backend.

Precision-injection lives inside ``KernelBenchDataset._load_dataset``
today and is preserved verbatim; Phase 4 will migrate it here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Mapping, Optional

from .base import Problem, ProblemSource


class KernelBenchSource(ProblemSource):
    """PyTorch KernelBench reference source (backend-agnostic).

    Supports both CUDA and OpenCL — the graph invokes kgen to translate
    the torch reference into a backend-specific driver+kernel pair.
    """

    name = "kernelbench"

    def __init__(
        self,
        precision: Optional[str] = None,
        **_ignored,
    ):
        # Precision is source-level metadata (injected into the .py at
        # load time inside the wrapped dataset); we thread it through.
        self._precision = precision

    def supports_backend(self, backend_name: str) -> bool:
        return backend_name in {"cuda", "opencl"}

    def curated_root_for(self, backend_name: str) -> Optional[Path]:
        # No curated artifacts — kgen produces them per problem.
        return None

    def iter_problems(
        self,
        *,
        tier: Optional[str] = None,
        problem_numbers: Optional[list[int]] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Iterator[Problem]:
        # Lazy import so ``data.sources`` doesn't force torch onto every
        # caller — some downstream tools import Problem for its typing
        # only.
        from ..kernelbench import KernelBenchDataset

        ds = KernelBenchDataset(
            level_str=tier,
            problem_numbers=problem_numbers,
            precision=self._precision,
            start=start,
            end=end,
        )
        for entry in ds:
            yield Problem(
                id=f"{self.name}:{entry['id']}",
                source=self.name,
                tier=str(entry.get("level", tier or "unknown")),
                problem_num=int(entry["problem_num"]),
                problem_name=str(entry["problem_name"]),
                curated_artifacts={
                    # ``filepath`` is the torch .py — technically a
                    # reference file, but the current schema also
                    # exposes the on-disk path in case downstream
                    # tools want to re-read it.
                    "reference_py": Path(entry["filepath"]),
                },
                reference_code=entry.get("reference_code"),
                metadata={
                    "precision": entry.get("precision", self._precision),
                },
                backends_supported=frozenset({"cuda", "opencl"}),
            )
