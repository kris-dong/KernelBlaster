# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Data package — problem sources for the KernelBlaster benchmarks.

The canonical entry point is :func:`data.sources.get_source`.
See ``data/sources/`` for the ``ProblemSource`` ABC + concrete sources.

Item 2 cleanup Phase 3 removed the ``get_dataset`` factory + three
back-compat dataset shims (``KernelBenchDataset`` /
``KernelBenchCUDADataset`` / ``KernelBenchOpenCLDataset``); every
in-tree consumer had already migrated to
``get_source(name).iter_problems(tier=..., problem_numbers=...)`` in
earlier phases. External callers migrating from the old API:

  # Before:
  #   from data import get_dataset
  #   ds, ds_iter = get_dataset("kernelbench-cuda", subset="level1",
  #                             problem_numbers="1,5")
  #   for entry in ds_iter:
  #       driver = entry["driver_cpp_fp"]
  #       kernel = entry["init_cuda_fp"]

  # After:
  #   from data.sources import get_source, parse_problem_numbers
  #   src = get_source("kernelbench-cuda")
  #   for problem in src.iter_problems(
  #       tier="level1",
  #       problem_numbers=parse_problem_numbers("1,5"),
  #   ):
  #       driver = problem.artifact("driver")
  #       kernel = problem.artifact("kernel")
"""
from .sources import (
    Problem,
    ProblemSource,
    get_source,
    parse_problem_numbers,
)

__all__ = [
    "Problem",
    "ProblemSource",
    "get_source",
    "parse_problem_numbers",
]
