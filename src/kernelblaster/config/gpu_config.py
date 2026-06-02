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
from enum import Enum
import subprocess


# StrEnum class is not included in python <3.11, so we define it here
class StrEnum(str, Enum):
    pass


_current_gpu = None

_SM_MAP = {
    # Please modify test_gpu_config.py if you modify this map
    "a100": "sm_80",
    "a6000": "sm_86",
    "l40": "sm_89",
    "l40s": "sm_89",
    "l40g": "sm_89",
    "rtx4090": "sm_89",
    "rtx5000": "sm_89",
    "rtx6000": "sm_89",
    "h100": "sm_90",
    "h200": "sm_90",
    "b200": "sm_100",
}

# Qualcomm Adreno GPU targets, mapped to their OpenCL version.
_ADRENO_MAP = {
    "adreno650": "opencl_2.0",
    "adreno660": "opencl_2.0",
    "adreno730": "opencl_3.0",
    "adreno740": "opencl_3.0",
    "adreno750": "opencl_3.0",
}


class GPUType(StrEnum):
    A100 = "a100"
    A6000 = "a6000"
    L40 = "l40"
    L40S = "l40s"
    L40G = "l40g"
    RTX4090 = "rtx4090"
    H100 = "h100"
    H200 = "h200"
    B200 = "b200"
    RTX5000 = "rtx5000"
    RTX6000 = "rtx6000"
    # Qualcomm Adreno targets
    ADRENO_650 = "adreno650"
    ADRENO_660 = "adreno660"
    ADRENO_730 = "adreno730"
    ADRENO_740 = "adreno740"
    ADRENO_750 = "adreno750"

    @property
    def is_nvidia(self) -> bool:
        return self.value in _SM_MAP

    @property
    def is_adreno(self) -> bool:
        return self.value in _ADRENO_MAP

    @property
    def sm(self):
        assert self.value in _SM_MAP, f"Not an NVIDIA GPU: {self.value}"
        return _SM_MAP[self.value]

    @property
    def opencl_version(self) -> str:
        assert self.value in _ADRENO_MAP, f"Not an Adreno GPU: {self.value}"
        return _ADRENO_MAP[self.value]

    @property
    def target_lang(self) -> str:
        """Return 'cuda' for NVIDIA, 'opencl' for Adreno.

        Note: kept for back-compat with existing consumers. New code should
        prefer ``self.backend()`` (returns a full ``Backend`` object) and read
        ``backend.name`` when only the language string is needed. The
        ``Backend`` abstraction subsumes this — see ``kernelblaster.backends``.
        """
        return self.backend().name

    def backend(self):
        """Return the canonical ``Backend`` instance for this GPU type.

        Lazy import avoids a circular dependency (backends -> agents.utils ->
        config). The returned object carries the GPU type and is the single
        source of truth for compile/profile/database conventions per the
        Phase 2 abstraction.
        """
        from ..backends import backend_for_gpu
        return backend_for_gpu(self)

    @staticmethod
    def current():
        """
        Return the current GPU type.
        This is cached to avoid repeated calls to nvidia-smi.
        """
        global _current_gpu
        if _current_gpu is None:
            try:
                name = (
                    subprocess.check_output(
                        "nvidia-smi --query-gpu=gpu_name --format=csv,noheader", shell=True
                    )
                    .decode("utf-8")
                    .strip()
                )
                name = name.replace(" ", "").lower()
                _current_gpu = _parse_gpu_name(name)
            except (subprocess.CalledProcessError, FileNotFoundError):
                _current_gpu = _detect_adreno_gpu()
        return _current_gpu


def _detect_adreno_gpu() -> "GPUType":
    """Detect Adreno GPU by querying OpenCL device name on the board."""
    import os
    if os.path.exists("/dev/kgsl-3d0"):
        try:
            output = subprocess.check_output(
                "clinfo 2>/dev/null | grep 'Device Name' | head -1",
                shell=True,
            ).decode("utf-8").strip().lower()
            for adreno_key in sorted(_ADRENO_MAP.keys(), key=len, reverse=True):
                if adreno_key.replace("adreno", "adreno ") in output or adreno_key.replace("adreno", "adreno(tm) ") in output:
                    return GPUType(adreno_key)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        return GPUType.ADRENO_650
    raise ValueError("No NVIDIA or Adreno GPU detected")


def _parse_gpu_name(nvidia_smi_name: str) -> GPUType:
    """
    Parse the GPU type from the nvidia-smi output.
    """
    nvidia_smi_name = nvidia_smi_name.replace(" ", "").lower()

    # Sort the gpu types in descending order of lengths to match the longest possible name.
    # This covers the case where the gpu name is a substring of a different gpu name like l40 and l40s.
    # Restrict to NVIDIA-mapped values so Adreno enum entries can't accidentally match nvidia-smi output.
    avail_types = sorted(
        [gpu.value for gpu in GPUType if gpu.value in _SM_MAP], key=lambda x: len(x), reverse=True
    )
    for gpu in avail_types:
        if gpu.lower() in nvidia_smi_name:
            return GPUType(gpu)
    raise ValueError(f"Unknown GPU type: {nvidia_smi_name}")
