# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Backend-specific compile / exec strategies (Phase C).

Public surface consumed by :mod:`servers.compile_server` and (later)
``servers.exec_server``. Both compile strategies self-register with
the :mod:`servers.utils.compile_strategy` registry on import so the
unified server can look them up by backend name.
"""
from ..utils.compile_strategy import (
    CompileStrategy,
    get_compile_strategy,
    register_compile_strategy,
)
from .cuda_compile import CUDACompileStrategy
from .opencl_compile import OpenCLCompileStrategy

# Self-registration on import — the unified server just imports
# this package and can dispatch immediately.
register_compile_strategy(CUDACompileStrategy())
register_compile_strategy(OpenCLCompileStrategy())

__all__ = [
    "CompileStrategy",
    "CUDACompileStrategy",
    "OpenCLCompileStrategy",
    "get_compile_strategy",
    "register_compile_strategy",
]
