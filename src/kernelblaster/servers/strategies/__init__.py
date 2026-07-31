# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Backend-specific compile / exec strategies.

Public surface consumed by :mod:`servers.compile_server` and
:mod:`servers.exec_server`. All strategies self-register on import:

* Compile strategies register **instances** in
  :mod:`servers.utils.compile_strategy` — they're parameter-free.
* Exec strategies register **classes** in
  :mod:`servers.utils.exec_strategy` — they take per-server config
  (GPU IDs, board host, bitstream path) so the exec server instantiates
  one at startup after resolving CLI flags.

The FPGA strategy pair (``ZephyrCompileStrategy`` + ``FPGAExecStrategy``)
lives here alongside CUDA / OpenCL, giving a RISC-V + Zephyr target a
first-class home without touching the framework core.
"""
from ..utils.compile_strategy import (
    CompileStrategy,
    get_compile_strategy,
    register_compile_strategy,
)
from ..utils.exec_strategy import (
    BatchTooLargeError,
    ExecJob,
    ExecJobResult,
    ExecStrategy,
    get_exec_strategy_cls,
    register_exec_strategy,
)
from .cuda_compile import CUDACompileStrategy
from .opencl_compile import OpenCLCompileStrategy
from .zephyr_compile import ZephyrCompileStrategy
from .local_exec import LocalExecStrategy
from .remote_exec import RemoteExecStrategy
from .fpga_exec import FPGAExecStrategy
from .spike_exec import SpikeExecStrategy

# Self-registration on import — the unified compile server just
# imports this package and can dispatch per-request.
register_compile_strategy(CUDACompileStrategy())
register_compile_strategy(OpenCLCompileStrategy())
register_compile_strategy(ZephyrCompileStrategy())

# Self-registration for exec strategies. Classes (not instances) —
# the exec server instantiates the chosen one at startup with per-
# server config.
register_exec_strategy(LocalExecStrategy)
register_exec_strategy(RemoteExecStrategy)
register_exec_strategy(FPGAExecStrategy)
register_exec_strategy(SpikeExecStrategy)

__all__ = [
    # Compile strategies
    "CompileStrategy",
    "CUDACompileStrategy",
    "OpenCLCompileStrategy",
    "ZephyrCompileStrategy",
    "get_compile_strategy",
    "register_compile_strategy",
    # Exec strategies
    "ExecStrategy",
    "ExecJob",
    "ExecJobResult",
    "BatchTooLargeError",
    "LocalExecStrategy",
    "RemoteExecStrategy",
    "FPGAExecStrategy",
    "SpikeExecStrategy",
    "get_exec_strategy_cls",
    "register_exec_strategy",
]
