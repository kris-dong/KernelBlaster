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

    @property
    def sm(self):
        assert self.value in _SM_MAP, f"Unknown GPU type: {self.value}"
        return _SM_MAP[self.value]

    @staticmethod
    def current():
        """
        Return the current GPU type.
        This is cached to avoid repeated calls to nvidia-smi.
        """
        global _current_gpu
        if _current_gpu is None:
            name = (
                subprocess.check_output(
                    "nvidia-smi --query-gpu=gpu_name --format=csv,noheader", shell=True
                )
                .decode("utf-8")
                .strip()
            )
            name = name.replace(" ", "").lower()
            _current_gpu = _parse_gpu_name(name)
        return _current_gpu


def _parse_gpu_name(nvidia_smi_name: str) -> GPUType:
    """
    Parse the GPU type from the nvidia-smi output.
    """
    nvidia_smi_name = nvidia_smi_name.replace(" ", "").lower()

    # Sort the gpu types in descending order of lengths to match the longest possible name.
    # This covers the case where the gpu name is a substring of a different gpu name like l40 and l40s.
    avail_types = sorted(
        [gpu.value for gpu in GPUType], key=lambda x: len(x), reverse=True
    )
    for gpu in avail_types:
        if gpu.lower() in nvidia_smi_name:
            return GPUType(gpu)
    raise ValueError(f"Unknown GPU type: {nvidia_smi_name}")
