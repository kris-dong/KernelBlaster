# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from .kernelbench import KernelBenchDataset
from .kernelbench_cuda import KernelBenchCUDADataset

__all__ = [
    "KernelBenchDataset",
    "KernelBenchCUDADataset",
    "get_dataset",
]


def get_dataset(
    name: str,
    subset: str | None = None,
    split: str | None = None,
    precision: str | None = None,
    problem_numbers: str | None = None,
    start: int | None = None,
    end: int | None = None,
    single_file_path: str | None = None,
):
    """Get a dataset by name.
    
    Args:
        name: Dataset name (currently only "kernelbench" is supported)
        subset: Subset name (e.g., "level1", "level2", "level3")
        split: Dataset split (not supported for kernelbench)
        precision: Precision ("fp32", "fp16", "bf16")
        problem_numbers: Comma-separated problem numbers or ranges (e.g., "1,2,3" or "8-60")
        start: Start problem number
        end: End problem number
        single_file_path: Path to single file (not supported for kernelbench)
    
    Returns:
        tuple: (dataset, dataset_iterator)
    """
    # Item 2, Phase 6: shared parse_problem_numbers helper.
    from .sources import parse_problem_numbers
    problem_numbers = parse_problem_numbers(problem_numbers)

    match name:
        case "kernelbench":
            assert subset in [None, "level1", "level2", "level3"], f"Invalid subset: {subset}"
            assert split is None, "dataset-split is not supported for kernelbench"
            dataset = KernelBenchDataset(
                level_str=subset,
                problem_numbers=problem_numbers,
                precision=precision,
                start=start,
                end=end,
            )
        case "kernelbench-cuda":
            assert subset in [None, "level1", "level2", "level3"], f"Invalid subset: {subset}"
            assert split is None, "dataset-split is not supported for kernelbench-cuda"
            dataset = KernelBenchCUDADataset(
                level_str=subset,
                problem_numbers=problem_numbers,
                start=start,
                end=end,
            )
        case "sol-execbench":
            # SOL split (2026-07): torch reference for the sol-level*
            # tiers. Route through KernelBenchDataset — which dispatches
            # to SOLExecBenchTorchSource internally when it sees a SOL
            # tier — so the returned dataset is dict-shaped like every
            # other legacy caller expects.
            assert subset in [None, "sol-level1", "sol-level2"], f"Invalid subset: {subset}"
            assert split is None, "dataset-split is not supported for sol-execbench"
            dataset = KernelBenchDataset(
                level_str=subset or "sol-level1",
                problem_numbers=problem_numbers,
                precision=precision,
                start=start,
                end=end,
            )
        case "sol-execbench-cuda":
            # Curated CUDA artifacts for the SOL suite. Uses the new
            # source directly — no legacy dataset shim exists for this
            # tier (the pre-Phase-3 ``KernelBenchCUDADataset`` never
            # accepted sol-level tiers).
            from .sources import SOLExecBenchCUDASource
            assert subset in [None, "sol-level1", "sol-level2"], f"Invalid subset: {subset}"
            assert split is None, "dataset-split is not supported for sol-execbench-cuda"
            dataset = _SourceBackedDataset(
                SOLExecBenchCUDASource(),
                tier=subset,
                problem_numbers=problem_numbers,
                start=start,
                end=end,
            )
        case _:
            raise ValueError(
                f"Unknown dataset: {name}. Supported: 'kernelbench', 'kernelbench-cuda', "
                f"'sol-execbench', 'sol-execbench-cuda'."
            )

    return dataset, dataset.get_iter(split)


class _SourceBackedDataset:
    """Minimal Dataset-shaped adapter over a ``ProblemSource``.

    Used only for sources that don't have a legacy back-compat dataset
    shim (currently ``sol-execbench-cuda``). Exposes ``__len__``,
    ``__iter__``, and ``get_iter(split)`` so it plugs into
    :func:`get_dataset`'s ``(dataset, iterator)`` return contract.
    """

    def __init__(self, source, *, tier, problem_numbers, start, end):
        # Materialise entries into a list of Problem objects so
        # ``__len__`` / ``__iter__`` / ``get_iter`` don't re-scan.
        self._problems = list(source.iter_problems(
            tier=tier, problem_numbers=problem_numbers, start=start, end=end,
        ))

    def __len__(self):
        return len(self._problems)

    def __iter__(self):
        return iter(self._problems)

    def get_iter(self, split=None):
        assert split is None, "dataset-split is not supported for source-backed datasets"
        return iter(self._problems)
