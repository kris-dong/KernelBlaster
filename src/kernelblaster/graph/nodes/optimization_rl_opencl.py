# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""RL-based OpenCL optimization graph node for Qualcomm Adreno GPUs.

Phase 4e shrank this file to a thin wrapper over the shared
``_run_rl_optimization_node`` helper. The per-backend specifics
(curated paths, state keys, agent class, global-best preference, etc.)
come from ``OpenCLBackend.rl_node_config()``.
"""
from ...backends import get_backend
from ..state import GraphState
from ._rl_node import _run_rl_optimization_node


async def optimization_rl_opencl(state: GraphState):
    """RL-based OpenCL optimization node for Qualcomm Adreno GPUs.

    Takes the OpenCL kernel from the curated benchmark and applies
    RL-based optimization using ``RLOpenCLAgent``. Requires curated
    ``driver.c`` + ``kernel.cl`` under
    ``data/<benchmark>/<bench_tier>/<problem>/``; falls back to per-run
    files if curated artifacts are missing.
    """
    return await _run_rl_optimization_node(state, get_backend("opencl"))
