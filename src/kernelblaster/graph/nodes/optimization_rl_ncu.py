# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""RL-based CUDA optimization graph node.

Phase 4e shrank this file to a thin wrapper over the shared
``_run_rl_optimization_node`` helper. The per-backend specifics
(curated paths, state keys, agent class, etc.) come from
``CUDABackend.rl_node_config()``.
"""
from ...backends import get_backend
from ..state import GraphState
from ._rl_node import _run_rl_optimization_node


async def optimization_rl_ncu(state: GraphState):
    """RL-based NCU optimization node.

    Takes the CUDA kernel from the curated KernelBench CUDA dataset and
    applies RL-based optimization using ``RLNCUAgent``. Requires curated
    ``driver.cpp`` + ``init.cu`` under ``data/kernelbench-cuda/<level>/<problem>/``;
    falls back to per-run files if curated artifacts are missing.
    """
    return await _run_rl_optimization_node(state, get_backend("cuda"))
