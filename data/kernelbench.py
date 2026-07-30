# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Back-compat shim over :mod:`data.sources.kernelbench_source`.

Item 2, Phase 4 (2026-07): the KernelBench (PyTorch reference) load
logic + precision injection + SOL-level filtering now live on
:class:`KernelBenchSource`. This module re-exports
``KernelBenchDataset`` (and the two convenience lookup methods) so
callers using the legacy factory ``get_dataset`` keep working. Phase 6
migrates the factory itself onto sources.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .dataset import Dataset
from .sources.kernelbench_source import KernelBenchSource


class KernelBenchDataset(Dataset):
    """Back-compat facade — yields the pre-refactor dict-shaped entries.

    New code should use :class:`data.sources.KernelBenchSource` directly.
    """

    def __init__(
        self,
        level_str: str | None = None,
        problem_numbers: list[int] | None = None,
        precision: str = "fp32",
        start: int | None = None,
        end: int | None = None,
    ):
        assert level_str is None or level_str in [
            "level1", "level2", "level3", "sol-level1", "sol-level2",
        ], "Invalid level"
        source = KernelBenchSource(precision=precision)
        super().__init__(Path(__file__).parent / "kernelbench")
        self.precision = precision
        self.sol_level1 = level_str == "sol-level1"
        self.sol_level2 = level_str == "sol-level2"
        self.level_num = (
            None
            if self.sol_level1 or self.sol_level2
            else (int(level_str.split("level")[1]) if level_str else None)
        )
        for problem in source.iter_problems(
            tier=level_str,
            problem_numbers=problem_numbers,
            start=start,
            end=end,
        ):
            self.data.append(_problem_to_legacy_entry(problem))

    def get_sample(self, level: int, problem_num: int) -> dict[str, Any]:
        for entry in self.data:
            if entry["level"] == level and entry["problem_num"] == problem_num:
                return entry
        raise ValueError(
            f"No sample found for level {level} and problem number {problem_num}"
        )

    def get_sample_by_id(self, id_substring: str) -> dict[str, Any]:
        for entry in self.data:
            if id_substring in entry["id"]:
                return entry
        raise ValueError(f"No sample found for id {id_substring}")


def _problem_to_legacy_entry(problem) -> dict[str, Any]:
    """Reverse of :meth:`KernelBenchSource._entry_to_problem` — keeps
    pre-Phase-4 dict keys stable for legacy consumers."""
    legacy_id = problem.id.split(":", 1)[1] if ":" in problem.id else problem.id
    return {
        "id": legacy_id,
        "problem_name": problem.problem_name,
        "problem_num": problem.problem_num,
        "level": problem.tier,
        "reference_code": problem.reference_code or "",
        "filepath": str(problem.curated_artifacts["reference_py"]),
        "precision": problem.metadata.get("precision", "fp32"),
    }


if __name__ == "__main__":
    dataset = KernelBenchDataset()
    print(f"Dataset size: {len(dataset)}")
    for idx, sample in enumerate(dataset):
        if idx < 3:
            print(f"Sample {idx}:", sample)
        else:
            break
