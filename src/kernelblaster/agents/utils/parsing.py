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
from pathlib import Path
import re

from ...config import GPUType

# allows for monkeypatching commands during testing
from . import commands as commands

from ...config import GPUType

__all__ = [
    "find_kernel_names_ncu",
    "find_kernel_names",
    "find_kernel_name",
    "get_elapsed_cycles_ncu_log",
    "find_kernel_launch_header",
]


async def find_kernel_names_ncu(
    executable: Path, source_path: Path, gpu: GPUType, timeout: int
) -> list[str]:
    """Find the kernel names by running NCU on a given executable and comparing it to the source code."""

    INVALID_KERNEL_NAME = "invalid_magic_kernel_name_here"

    kernels_in_source = find_kernel_names(source_path)

    if not kernels_in_source:
        raise RuntimeError(f"No kernels found in the source code:\n{source_path}")

    # Run ncu on the executable
    # Should print a list of available kernels in the log like:
    # ==PROF== Connected to process 2138337 (/tmp/kernelagent/compile_env/build/main)
    # ==PROF== Disconnected from process 2138337
    # ==WARNING== No kernels were profiled.
    # Available Kernels:
    # 1. distribution_elementwise_grid_stride_kernel
    # 2. kernel
    # 3. matmul_fp16_kernel_8x8
    # 4. reduce_kernel
    # 5. vectorized_elementwise_kernel
    stdout, stderr = await commands.run_gpu_executable(
        executable,
        gpu,
        timeout,
        job_name=str(executable),
        prefix_command=f"NVIDIA_TF32_OVERRIDE=0 ncu -k {INVALID_KERNEL_NAME}",
    )

    # Parse the stdout to get the kernel names
    kernel_section = stdout.split("Available Kernels:")[1]
    ncu_kernel_names = re.findall(r"\s*\d+\.\s*(\w+)", kernel_section)
    if not ncu_kernel_names:
        raise RuntimeError(
            f"Failed to find NCU kernel names in:\n stdout: {stdout}\n stderr: {stderr}"
        )

    # find the intersection of the kernel names
    kernel_names = list(set(ncu_kernel_names) & set(kernels_in_source))

    # Check if the kernel names are in the source code
    if not kernel_names:
        raise RuntimeError(
            f"Failed to find kernels running in both the executable and the source code:\n Source code kernels: {kernels_in_source}\n NCU kernels: {ncu_kernel_names}"
        )

    return kernel_names


def find_kernel_names(filename: Path) -> str:
    """Find the kernel names from a given cuda file.

    Args:
        filename: The filename of the kernel code.

    Returns:
        The kernel names.
    """
    kernel_code = filename.read_text()
    # kernel may also be launched from a cuda function, so
    # try parsing the kernel name from the kernel declaration
    kernel_names_launches = re.findall(r"__global__ void (\S+)\(", kernel_code)
    kernel_names_decls = re.findall(r"(\w+)(?:<[^>]*>)?\s*<<<", kernel_code)

    kernel_names = list(set(kernel_names_launches) | set(kernel_names_decls))

    # filter out names with __launch_bounds__ at the prefix because
    # these are launch definitions not names
    kernel_names = list(
        filter(lambda x: not x.startswith("__launch_bounds__"), kernel_names)
    )

    if len(kernel_names) == 0:
        raise RuntimeError(
            f"Failed to find kernel name in:\n{kernel_code}\n Please define the kernel with __global__ void kernel_name()"
        )
    return kernel_names


def find_kernel_name(filename: Path) -> str:
    """Find the only kernel name from a given cuda file.

    Args:
        filename: The filename of the kernel code.

    Returns:
        The only kernel name.
    """
    kernel_names = find_kernel_names(filename)
    if len(kernel_names) > 1:
        raise RuntimeError(
            f"Found multiple kernel names in:\n{filename.read_text()}\n Please generate only one kernel in the output."
        )
    return kernel_names[0]


def get_elapsed_cycles_ncu_log(ncu_log: str) -> int:
    """Get the elapsed cycles from a given ncu log.
    
    Parses the "Elapsed Cycles" metric from the "GPU Speed Of Light Throughput" section.
    Supports both table format and CSV format.

    Args:
        ncu_log: The ncu log.

    Returns:
        The elapsed cycles.
    """
    # Try multiple patterns to match different NCU output formats
    patterns = [
        # Table format: "Elapsed Cycles                cycle        12675"
        r"Elapsed Cycles\s+\w+\s+(\d[\d,]*)",
        # CSV format or other formats: "Elapsed Cycles,cycle,12675" or "Elapsed Cycles: 12675"
        r"Elapsed Cycles[,\s:]+(?:cycle[,\s]+)?(\d[\d,]*)",
        # Fallback: any format with "Elapsed Cycles" followed by digits
        r"Elapsed Cycles.*?(\d[\d,]*)",
    ]
    
    for pattern in patterns:
        elapsed_cycles = re.search(pattern, ncu_log, re.IGNORECASE | re.MULTILINE)
        if elapsed_cycles:
            try:
                return int(elapsed_cycles.group(1).replace(",", ""))
            except ValueError:
                continue
    
    raise RuntimeError(f"Failed to find elapsed cycles in NCU log. Patterns tried: {len(patterns)}")


def find_kernel_launch_header(code: str) -> str:
    """Find the kernel launch header in a given code.

    Args:
        code: The code.

    Returns:
        The kernel launch header.
    """
    launch_headers = re.findall(
        r"(void launch_gpu_implementation\(.*?\);)", code, flags=re.DOTALL
    )
    if len(launch_headers) == 0:
        raise RuntimeError(f"Failed to find kernel launch header in:\n{code}")
    if len(launch_headers) > 1:
        raise RuntimeError(
            f"Found multiple kernel launch headers in:\n{code}\n Please generate only one kernel in the output."
        )
    return launch_headers[0]
