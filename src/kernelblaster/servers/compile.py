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
import argparse
import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
from pydantic import BaseModel
import logging
import shutil
import tempfile
import uuid
import re
import sysconfig
import time
from torch.utils import cmake_prefix_path

from .utils.queue_server import queue_worker_loop
from .utils.subprocess import run_subprocess_shell
from ..agents.utils import find_kernel_launch_header

logger = logging.getLogger("uvicorn")

# Fix the cmake_prefix_path format to properly use quotes for paths that may contain spaces
CMAKE_PREFIX_PATH = f'"{cmake_prefix_path};{sysconfig.get_path("include")}"'
QUEUE = asyncio.Queue()
CUDA_ENV_PATH = Path(__file__).parent / "cuda_env"
ENV_VARS = os.environ.copy()
# Module-level variables (will be set during server startup)
_ARTIFACTS_DIR = None


def get_cmake_prefix_path() -> str:
    """Get the CMAKE_PREFIX_PATH for compilation"""
    return f'"{cmake_prefix_path};{sysconfig.get_path("include")}"'


def get_cuda_env_template_path() -> Path:
    """Get the path to the cuda_env template directory"""
    return Path(__file__).parent / "cuda_env"


def extract_arch_version(sm_version: str) -> str:
    """Extract architecture version from SM version string"""
    assert "sm" in sm_version, f"Invalid sm version format: {sm_version}"
    arch_version = sm_version.split("sm_")[1]
    assert int(arch_version) >= 50, f"Invalid sm version: {sm_version}"
    return arch_version


def write_compilation_files(
    work_dir: Path, main_file_path: str, cuda_file_path: str | None
) -> tuple[Path, Path, Path]:
    """
    Split and write compilation files to the work directory.

    Returns:
        Tuple of (main_fp_out, header_fp_out, cuda_fp_out) paths
    """
    main_fp_out = work_dir / "main.cpp"
    header_fp_out = work_dir / "cuda_model.cuh"
    cuda_fp_out = work_dir / "cuda_model.cu"

    main_file_text, header_file_text, cuda_file_text = split_files_for_compilation(
        main_file_path, cuda_file_path
    )

    main_fp_out.write_text(main_file_text)
    header_fp_out.write_text(header_file_text)
    cuda_fp_out.write_text(cuda_file_text)

    logger.info(
        "Prepared compilation units: main.cpp lines=%d chars=%d | cuda_model.cuh lines=%d chars=%d | cuda_model.cu lines=%d chars=%d",
        len(main_file_text.splitlines()),
        len(main_file_text),
        len(header_file_text.splitlines()),
        len(header_file_text),
        len(cuda_file_text.splitlines()),
        len(cuda_file_text),
    )

    return main_fp_out, header_fp_out, cuda_fp_out


def build_cmake_command(
    sm_build_dir: Path,
    arch_version: str,
    build_type: str = "Release",
) -> str:
    """Build the cmake configuration command"""
    return (
        f"mkdir -p {sm_build_dir} && cd {sm_build_dir} && "
        f"cmake -DCMAKE_PREFIX_PATH={get_cmake_prefix_path()} "
        f"-DCMAKE_BUILD_TYPE={build_type} "
        f'-DGPU_ARCH_VERSION="{arch_version}" '
        ".."
    )


# Lifespan defined at module import time; reads `args` lazily on startup
# (after __main__ has parsed them).


def get_cuda_env_root(thread_id: int) -> Path:
    path = ENV_DIR / f"cuda_eval_{thread_id}"
    if not path.exists():
        setup_cuda_envs(path)
    assert path.exists()
    return path


def get_persistent_root(unique_name: str) -> Path:
    # Create a unique directory based on the output filename. This directory's artifacts will not be overwritten by subsequent compilations.
    persistent_artifacts_dir = ENV_DIR / "persistent" / unique_name
    assert (
        not persistent_artifacts_dir.exists()
    ), f"Persistent artifacts directory {persistent_artifacts_dir} already exists"
    setup_cuda_envs(persistent_artifacts_dir)
    return persistent_artifacts_dir


def setup_cuda_envs(directory: Path):
    shutil.copytree(CUDA_ENV_PATH, directory)
    logger.info(f"Set up CUDA environment at {directory}")


def free_cuda_envs():
    if ENV_DIR.exists():
        shutil.rmtree(ENV_DIR)
    logger.info("Cleaned up CUDA environment")


def get_all_includes(main_file_text: str) -> list[str]:
    # get includes that are surrounded by angle brackets
    system_includes = re.findall(r"(#include\s+<[^>]+>)", main_file_text)
    # get includes that are surrounded by quotes
    user_includes = re.findall(r'(#include\s+"[^"]+")', main_file_text)
    return system_includes + user_includes


def split_files_for_compilation(
    main_file_path: str, cuda_file_path: str | None
) -> tuple[str, str, str]:
    """
    This method parses the driver file and cuda file and separates it into two compilable units.
    """
    main_file_text = Path(main_file_path).read_text()
    header_file_text = ""
    cuda_file_text = ""

    if cuda_file_path:
        cuda_file_text = Path(cuda_file_path).read_text()
        # If we are compiling a cuda kernel, we must construct a separate compilable unit
        # parse the header from the test file and move to a header file
        try:
            header_decl = find_kernel_launch_header(main_file_text)
        except Exception as e:
            logger.error(
                f"Failed to find kernel launch header in {main_file_path}: {e}"
            )
            raise CompilationError(
                f"Failed to find kernel launch header in {main_file_path}: {e}"
            )
        main_file_text = main_file_text.replace(header_decl, "")

        # Add cstdint in case fixed-width integer types are used like int64_t
        # Add torch/torch.h in case parameters are of type torch::Tensor or c10::ScalarType
        header_file_text = (
            "#include <cstdint>\n#include <torch/torch.h>\n" + header_decl + "\n"
        )

        # Add the header include to both main file and CUDA file
        main_file_text = f'#include "cuda_model.cuh"\n{main_file_text}'
        cuda_file_text = f'#include "cuda_model.cuh"\n{cuda_file_text}'

        # Remove "inline" and 'extern "C"' because the linker will fail
        cuda_file_text = cuda_file_text.replace(
            "inline void launch_gpu_implementation", "void launch_gpu_implementation"
        ).replace('extern "C"', "")

    return main_file_text, header_file_text, cuda_file_text


class CompilationRequest(BaseModel):
    job_name: str
    main_file: str
    cuda_file: str
    sm_version: str

    # This flag allows the compilation server to save the CUDA source artifacts in a unique directory
    # that's only modified on shutdown.
    # This is useful when later commands need to reference the original CUDA source code e.g. NCU annotation.
    # Also, boolean flags are not supported in the REST API.
    persistent_artifacts: int = 0


class CompilationResult(BaseModel):
    job_name: str
    main_file: str
    cuda_file: str
    success: bool = False
    message: str = None
    output_path: str = None
    persistent_artifacts_dir: str = None


class CompilationError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

async def exec_compilation(
    job_name: str,
    main_file: str,
    cuda_file: str,
    sm_version: str,
    worker_id: int,
    output_path: Path,
    persistent_artifacts: bool,
    debug=False,
    timeout: int | None = None,
):
    """
    This function is used to compile a CUDA program.
    It will separate and move the GPU code declaration from the test file into its own header file.
    It will remove the header from the cuda file and instead add an includes statement to the header file.

    The structure is as follows:
    - main.cpp (copied from from <main_file> but with the kernel launch header removed)
    - cuda_model.cuh (solely the extracted kernel launch header)
    - cuda_model.cu (copied from <cuda_file> and with an additional #include "cuda_model.cuh" at the top)
    """
    assert not debug, "Debug compilation is not supported"

    if persistent_artifacts:
        work_dir = get_persistent_root(output_path.name)
    else:
        # Use the standard worker environment
        work_dir = get_cuda_env_root(worker_id)

    main_fp_out, header_fp_out, cuda_fp_out = write_compilation_files(
        work_dir, main_file, cuda_file
    )

    arch_version = extract_arch_version(sm_version)

    if timeout is None:
        timeout = int(getattr(args, "compile_timeout", 360))

    # this call is expensive, so only regenerate if the sm version is different
    sm_build_dir = work_dir / f"build_{sm_version}"
    if not sm_build_dir.exists():
        build_type = "Release"
        cmd = build_cmake_command(
            sm_build_dir,
            arch_version,
            build_type,
        )
        # Note: use central runner to capture stderr/stdout and handle timeouts.
        # CMake can hang when CUDA/toolchain discovery misbehaves.
        await run_subprocess_shell(
            stage=f"cmake_config:{sm_version}",
            cmd=cmd,
            cwd=work_dir,
            timeout_s=timeout,
            env=ENV_VARS,
            error_factory=CompilationError,
            logger=logger,
        )

    # Build step (make). Use central runner to ensure timeouts are debuggable.
    await run_subprocess_shell(
        stage=f"make:{sm_version}",
        cmd="make -j8",
        cwd=sm_build_dir,
        timeout_s=timeout,
        env=ENV_VARS,
        error_factory=CompilationError,
        logger=logger,
    )

    return sm_build_dir / "main"


# _cuda_compile_job removed in Phase E — was the legacy queue handler
# for the /compile endpoint. Callers now hit compile_server.py which
# invokes CUDACompileStrategy directly.


