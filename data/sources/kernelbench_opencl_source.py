# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Curated OpenCL/Adreno-artifacts source — Phase 1 facade over ``KernelBenchOpenCLDataset``.

Layout: ``data/benchmark-opencl/<L*>/<problem>/{driver.c,kernel.cl}``
(the primary root) with a fallback to ``data/kernelbench-opencl/…`` for
the legacy port tree. The two-map tier canonicalization
(``SUBSET_TO_BENCHMARK_DIR`` / ``RUN_FOLDER_PARENT_TO_BENCHMARK_DIR``) is
still owned by ``data.kernelbench_opencl`` in Phase 1; the migration
into ``tier_dir_for`` happens in Phase 2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Mapping, Optional

from .base import Problem, ProblemSource


class KernelBenchOpenCLSource(ProblemSource):
    """Curated OpenCL/Adreno artifacts source."""

    name = "kernelbench-opencl"

    def supports_backend(self, backend_name: str) -> bool:
        return backend_name == "opencl"

    def curated_root_for(self, backend_name: str) -> Optional[Path]:
        if backend_name != "opencl":
            return None
        # Lazy import — ``kernelbench_opencl`` pulls its own tier maps
        # + fallback resolution helpers.
        from ..kernelbench_opencl import default_benchmark_opencl_root
        return default_benchmark_opencl_root()

    def artifact_filenames(self, backend_name: str) -> Mapping[str, str]:
        if backend_name != "opencl":
            return {}
        return {"driver": "driver.c", "kernel": "kernel.cl"}

    def tier_dir_for(self, backend_name: str, tier: str) -> str:
        # Phase 1 facade — defer to the existing map. Phase 2 folds
        # this in and eliminates the module-level dict.
        from ..kernelbench_opencl import SUBSET_TO_BENCHMARK_DIR
        return SUBSET_TO_BENCHMARK_DIR.get(tier, tier)

    def iter_problems(
        self,
        *,
        tier: Optional[str] = None,
        problem_numbers: Optional[list[int]] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Iterator[Problem]:
        from ..kernelbench_opencl import KernelBenchOpenCLDataset

        ds = KernelBenchOpenCLDataset(
            level_str=tier,
            problem_numbers=problem_numbers,
            start=start,
            end=end,
        )
        for entry in ds:
            artifacts: dict[str, Path] = {}
            if entry.get("driver_c_fp"):
                artifacts["driver"] = Path(entry["driver_c_fp"])
            artifacts["kernel"] = Path(entry["kernel_cl_fp"])
            if entry.get("reference_py_fp"):
                artifacts["reference_py"] = Path(entry["reference_py_fp"])

            yield Problem(
                id=f"{self.name}:{entry['id']}",
                source=self.name,
                tier=str(entry["level"]),
                problem_num=int(entry["problem_num"]),
                problem_name=str(entry["problem_name"]),
                curated_artifacts=artifacts,
                backends_supported=frozenset({"opencl"}),
            )
