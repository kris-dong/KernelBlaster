# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Back-compat shim over :mod:`data.sources.kernelbench_cuda_source`.

Item 2, Phase 3 (2026-07): the CUDA dataset loading + tier constants
now live on :class:`KernelBenchCUDASource`. This module keeps
``KernelBenchCUDADataset`` importable (only ``data/__init__.py`` and its
factory currently consume it) so callers using the legacy factory path
keep working. Phase 6 migrates the factory itself onto sources.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .dataset import Dataset
from .sources.kernelbench_cuda_source import KernelBenchCUDASource


class KernelBenchCUDADataset(Dataset):
    """Back-compat facade — yields the pre-refactor dict-shaped entries.

    New code should use :class:`data.sources.KernelBenchCUDASource` directly.
    """

    def __init__(
        self,
        level_str: str | None = None,
        problem_numbers: list[int] | None = None,
        start: int | None = None,
        end: int | None = None,
        root_dir: str | Path | None = None,
    ):
        assert level_str is None or level_str in ["level1", "level2", "level3"], (
            f"Invalid level: {level_str}"
        )
        source = KernelBenchCUDASource(root_dir=root_dir)
        super().__init__(source.curated_root_for("cuda"))
        self.level_str = level_str
        for problem in source.iter_problems(
            tier=level_str,
            problem_numbers=problem_numbers,
            start=start,
            end=end,
        ):
            self.data.append(_problem_to_legacy_entry(problem))


def _problem_to_legacy_entry(problem) -> dict[str, Any]:
    """Reverse of :meth:`KernelBenchCUDASource._entry_to_problem`.

    Preserves the pre-Phase-3 dict keys: ``id`` without the source
    prefix, ``driver_cpp_fp``/``init_cuda_fp`` as strings, plus the
    ``final_cuda_fp`` alias legacy callers depend on.
    """
    legacy_id = problem.id.split(":", 1)[1] if ":" in problem.id else problem.id
    init_cu = problem.curated_artifacts["kernel"]
    return {
        "id": legacy_id,
        "problem_name": problem.problem_name,
        "problem_num": problem.problem_num,
        "level": problem.tier,
        "driver_cpp_fp": str(problem.curated_artifacts["driver"]),
        "init_cuda_fp": str(init_cu),
        # Backwards compat: older callers expect ``final_cuda_fp``.
        "final_cuda_fp": str(init_cu),
    }
