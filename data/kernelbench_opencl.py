# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Back-compat shim over :mod:`data.sources.kernelbench_opencl_source`.

Item 2, Phase 2 (2026-07): the OpenCL dataset loading + tier maps + root
helpers all live on :class:`KernelBenchOpenCLSource` now. This module
keeps the pre-refactor public surface (``KernelBenchOpenCLDataset``,
``default_benchmark_opencl_root``, ``run_output_parent_to_benchmark_dir``,
``subset_flag_to_benchmark_dir``, ``kernelbench_opencl_port_root``,
``SUBSET_TO_BENCHMARK_DIR``, ``RUN_FOLDER_PARENT_TO_BENCHMARK_DIR``) so
callers in ``graph/nodes/kgen_opencl.py`` and ``backends/opencl.py``
continue to import from ``data.kernelbench_opencl`` unchanged.

Phase 5 will migrate those two call sites onto the source directly and
this shim can be deleted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .dataset import Dataset
from .sources.kernelbench_opencl_source import (
    KernelBenchOpenCLSource,
    _SUBSET_TO_BENCHMARK_DIR,
)


# ---- Re-exports (module-level, matching pre-Phase-2 surface) -------------
# Note: the two maps used to be distinct dicts with identical contents. Phase 2
# collapsed them since the audit confirmed no divergence in either usage. Both
# names remain as aliases so callers that referenced them by name still work.
SUBSET_TO_BENCHMARK_DIR = dict(_SUBSET_TO_BENCHMARK_DIR)
RUN_FOLDER_PARENT_TO_BENCHMARK_DIR = dict(_SUBSET_TO_BENCHMARK_DIR)


def subset_flag_to_benchmark_dir(subset: str | None) -> str | None:
    if subset is None:
        return None
    if subset not in _SUBSET_TO_BENCHMARK_DIR:
        raise ValueError(f"Invalid kernelbench-opencl subset: {subset}")
    return _SUBSET_TO_BENCHMARK_DIR[subset]


def run_output_parent_to_benchmark_dir(parent: str) -> str:
    return _SUBSET_TO_BENCHMARK_DIR.get(parent, parent)


def default_benchmark_opencl_root() -> Path:
    return KernelBenchOpenCLSource._default_benchmark_root()


def kernelbench_opencl_port_root() -> Path:
    return KernelBenchOpenCLSource._port_root()


class KernelBenchOpenCLDataset(Dataset):
    """Back-compat facade — yields the pre-refactor dict-shaped entries.

    New code should use :class:`data.sources.KernelBenchOpenCLSource` directly.
    Existing callers that iterate the dataset via ``for entry in dataset:``
    or index it (``dataset[i]``) keep working because we materialize the
    entries into ``self.data`` at construction time — same as pre-Phase-2.
    """

    def __init__(
        self,
        level_str: str | None = None,
        problem_numbers: list[int] | None = None,
        start: int | None = None,
        end: int | None = None,
        root_dir: str | Path | None = None,
    ):
        source = KernelBenchOpenCLSource(root_dir=root_dir)
        super().__init__(source.curated_root_for("opencl"))
        self.level_str = level_str
        # Materialize the source's iter into the legacy ``data`` list of
        # dicts — one Problem -> one dict — preserving key names other
        # code depends on.
        for problem in source.iter_problems(
            tier=level_str,
            problem_numbers=problem_numbers,
            start=start,
            end=end,
        ):
            self.data.append(_problem_to_legacy_entry(problem))


def _problem_to_legacy_entry(problem) -> dict[str, Any]:
    """Reverse of :meth:`KernelBenchOpenCLSource._entry_to_problem` — keeps
    pre-Phase-2 dict keys stable for legacy consumers."""
    # Strip the source prefix so ``entry["id"]`` matches the old shape
    # (``"L1/001_vector_add"``, not ``"kernelbench-opencl:L1/001_vector_add"``).
    legacy_id = problem.id.split(":", 1)[1] if ":" in problem.id else problem.id
    entry: dict[str, Any] = {
        "id": legacy_id,
        "problem_name": problem.problem_name,
        "problem_num": problem.problem_num,
        "level": problem.tier,
        "driver_c_fp": (
            str(problem.curated_artifacts["driver"])
            if "driver" in problem.curated_artifacts
            else None
        ),
        "kernel_cl_fp": str(problem.curated_artifacts["kernel"]),
    }
    if "reference_py" in problem.curated_artifacts:
        entry["reference_py_fp"] = str(problem.curated_artifacts["reference_py"])
    return entry
