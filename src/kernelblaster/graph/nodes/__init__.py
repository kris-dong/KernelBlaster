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
from .optimization_rl_ncu import optimization_rl_ncu

# OpenCL/Adreno nodes are imported defensively: their dependencies (board
# servers, OpenCL agents) may not be available in NVIDIA-only environments.
try:
    from .kgen_opencl import kgen_opencl
except Exception:
    kgen_opencl = None

try:
    from .optimization_rl_opencl import optimization_rl_opencl
except Exception:
    optimization_rl_opencl = None

__all__ = [
    "optimization_rl_ncu",
    "kgen_opencl",
    "optimization_rl_opencl",
]
