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
    if problem_numbers:
        # Parse problem_numbers to support both comma-separated and range syntax
        parsed_numbers = []
        for part in problem_numbers.split(","):
            part = part.strip()
            if "-" in part and not part.startswith("-"):
                # Handle range syntax like "8-60"
                start_num, end_num = part.split("-", 1)
                parsed_numbers.extend(range(int(start_num), int(end_num) + 1))
            else:
                # Handle individual numbers
                parsed_numbers.append(int(part))
        problem_numbers = parsed_numbers

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
        case _:
            raise ValueError(
                f"Unknown dataset: {name}. Supported: 'kernelbench', 'kernelbench-cuda'."
            )

    return dataset, dataset.get_iter(split)
