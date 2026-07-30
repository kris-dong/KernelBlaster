# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Curated CUDA-artifacts source — Phase 1 facade over ``KernelBenchCUDADataset``.

Layout: ``data/kernelbench-cuda/<tier>/<problem>/{driver.cpp,init.cu}``.
This source ships curated CUDA driver + kernel per problem; consumers
skip kgen and go straight to compile + run + optimize.

Currently accepts tiers ``level1``/``level2``/``level3`` only — the
``sol-level{1,2}`` cases are still routed through
``KernelBenchDataset`` (torch) for legacy reasons. Phase 3 splits
those out into a dedicated ``SOLExecBenchCUDASource``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Mapping, Optional

from .base import Problem, ProblemSource


class KernelBenchCUDASource(ProblemSource):
    """Curated CUDA artifacts source."""

    name = "kernelbench-cuda"

    def supports_backend(self, backend_name: str) -> bool:
        return backend_name == "cuda"

    def curated_root_for(self, backend_name: str) -> Optional[Path]:
        if backend_name != "cuda":
            return None
        # Mirrors the default in ``KernelBenchCUDADataset.__init__``.
        return Path(__file__).resolve().parent.parent / "kernelbench-cuda"

    def artifact_filenames(self, backend_name: str) -> Mapping[str, str]:
        if backend_name != "cuda":
            return {}
        return {"driver": "driver.cpp", "kernel": "init.cu"}

    def iter_problems(
        self,
        *,
        tier: Optional[str] = None,
        problem_numbers: Optional[list[int]] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Iterator[Problem]:
        from ..kernelbench_cuda import KernelBenchCUDADataset

        ds = KernelBenchCUDADataset(
            level_str=tier,
            problem_numbers=problem_numbers,
            start=start,
            end=end,
        )
        for entry in ds:
            yield Problem(
                id=f"{self.name}:{entry['id']}",
                source=self.name,
                tier=str(entry["level"]),
                problem_num=int(entry["problem_num"]),
                problem_name=str(entry["problem_name"]),
                curated_artifacts={
                    "driver": Path(entry["driver_cpp_fp"]),
                    "kernel": Path(entry["init_cuda_fp"]),
                },
                backends_supported=frozenset({"cuda"}),
            )
