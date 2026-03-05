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
from glob import glob
from typing import Iterator, Dict, Any
from pathlib import Path

from .dataset import Dataset


class KernelBenchDataset(Dataset):
    """Dataset class for KernelBench dataset."""

    def __init__(
        self,
        level_str: str = None,
        problem_numbers: list[int] | None = None,
        precision: str = "fp32",
        start: int = None,
        end: int = None,
    ):
        """Initialize the KernelBench dataset.

        Args:
            level (str): Level of the dataset to load
            problem_name (str): Problem id to load
        """
        # Use a lowercase dataset directory name to avoid hard-coded capitalized paths
        super().__init__(Path(__file__).parent / "kernelbench")
        assert level_str is None or level_str in [
            "level1",
            "level2",
            "level3",
        ], "Invalid level"
        assert precision in ["fp32", "fp16", "bf16"], "Invalid precision"
        self.precision = precision
        self.level_num = int(level_str.split("level")[1]) if level_str else None
        self._load_dataset(problem_numbers, start, end)

    def _load_dataset(self, problem_numbers: list[int] | None, start: int | None, end: int | None) -> None:
        """Load all JSON files from the dataset directory."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory {self.data_dir} not found")

        # Rename files so they have the standard form level/###_problem.py
        # Look for dataset files under a dataset subdirectory (lowercase)
        for path in self.data_dir.glob("kernelbench/**/*.py"):
            parts = list(filter(lambda f: f, path.stem.split("_")))
            num = int(parts[0])
            level = int(path.parent.stem.split("level")[1])
            new_name = f'{num:03d}_{"_".join(parts[1:])}'
            id = f"level{level}/{new_name}"
            if self.level_num is not None and self.level_num != level:
                continue
            if self.level_num is None and (level == 3 or level == 4):
                # skip level3 and level4 problems unless explicitly specified
                # because they are outside the scope of the current agent.
                continue
            if problem_numbers is not None and int(num) not in problem_numbers:
                continue
            if start is not None and int(num) < start:
                continue
            if end is not None and int(num) > end:
                continue

            # modify code for precision
            reference_code = path.read_text()
            match self.precision:
                case "fp32":
                    snippet = f"# Use fp32 datatype for all tensors\ntorch.set_default_dtype(torch.float32)"
                case "fp16":
                    snippet = f"# Use fp16 datatype for all tensors\ntorch.set_default_dtype(torch.float16)"
                case "bf16":
                    snippet = f"# Use bf16 datatype for all tensors\ntorch.set_default_dtype(torch.bfloat16)"
                case _:
                    raise ValueError(f"Invalid precision: {self.precision}")
            insertion_point = reference_code.find("class Model")
            reference_code = (
                reference_code[:insertion_point]
                + snippet
                + "\n\n"
                + reference_code[insertion_point:]
            )

            entry = {
                "id": id,
                "problem_name": new_name,
                "problem_num": num,
                "level": f"level{level}",
                "reference_code": reference_code,
                "filepath": str(path),
                "precision": self.precision,
            }
            self.data.append(entry)
        self.data.sort(key=lambda x: x["id"])

    def get_sample(self, level: int, problem_num: int) -> dict[str, Any]:
        """Get a sample from the dataset by level and problem number."""
        for entry in self.data:
            if entry["level"] == level and entry["problem_num"] == problem_num:
                return entry
        raise ValueError(
            f"No sample found for level {level} and problem number {problem_num}"
        )

    def get_sample_by_id(self, id_substring: str) -> dict[str, Any]:
        """Get a sample from the dataset by id."""
        for entry in self.data:
            if id_substring in entry["id"]:
                return entry
        raise ValueError(f"No sample found for id {id_substring}")


if __name__ == "__main__":
    # Example usage
    dataset = KernelBenchDataset()
    print(f"Dataset size: {len(dataset)}")

    # Example of iterating through the dataset
    for idx, sample in enumerate(dataset):
        if idx < 3:  # Print first 3 samples
            print(f"Sample {idx}:", sample)
        else:
            break
