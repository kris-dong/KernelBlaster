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
"""Hardware backend registry.

``get_backend(name)`` returns a ``Backend`` instance for the given identifier.
See ``base.py`` for the contract and ``cuda.py`` / ``opencl.py`` for the
Phase 2 facade implementations.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Backend, ProfileResult
from .cuda import CUDABackend
from .opencl import OpenCLBackend

if TYPE_CHECKING:
    from ..config import GPUType


_REGISTRY: dict[str, type[Backend]] = {
    "cuda": CUDABackend,
    "opencl": OpenCLBackend,
}


def get_backend(name: str, **kwargs: Any) -> Backend:
    """Return a Backend instance for ``name`` (``"cuda"`` or ``"opencl"``).

    Extra kwargs are forwarded to the backend constructor (e.g. ``gpu=``,
    or ``board_host=`` for OpenCL).
    """
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


def backend_for_gpu(gpu: "GPUType") -> Backend:
    """Return the canonical backend for a given ``GPUType``."""
    if gpu.is_nvidia:
        return CUDABackend(gpu=gpu)
    if gpu.is_adreno:
        return OpenCLBackend(gpu=gpu)
    raise ValueError(f"No backend registered for GPU {gpu!r}")


__all__ = [
    "Backend",
    "CUDABackend",
    "OpenCLBackend",
    "ProfileResult",
    "backend_for_gpu",
    "get_backend",
]
