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
from fastapi import FastAPI, HTTPException
from pathlib import Path
from pydantic import BaseModel
import logging
import shutil
import tempfile
import uuid
import uvicorn
import uvicorn.config
import re
import sysconfig
from torch.utils import cmake_prefix_path

from .server_logging import get_log_config
from .utils.process_management import safe_kill_process
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


# Start worker tasks in the background
@asynccontextmanager
async def lifespan(app):
    logger.info(
        f"Started compilation server on {args.host}:{args.port} with {args.num_workers} workers"
    )
    # Start worker tasks on startup
    _ = asyncio.create_task(start_workers(args.num_workers, args.compile_debug))
    yield
    free_cuda_envs()


APP = FastAPI(lifespan=lifespan)


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
    timeout=360,
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

    # this call is expensive, so only regenerate if the sm version is different
    sm_build_dir = work_dir / f"build_{sm_version}"
    if not sm_build_dir.exists():
        build_type = "Release"
        cmd = build_cmake_command(
            sm_build_dir,
            arch_version,
            build_type,
        )
        logger.info(f"Running cmake command: {cmd}")
        proc = await asyncio.subprocess.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        assert (
            proc.returncode == 0
        ), f"Failed to run cmake config for {work_dir}: stderr:\n{stderr.decode()}"

    proc = await asyncio.create_subprocess_shell(
        "make -j8",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=sm_build_dir,
        start_new_session=True,
        env=ENV_VARS,
    )
    try:
        # Wait for the process with timeout
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        if proc.returncode != 0:
            raise CompilationError(
                f"stdout:\n{stdout.decode()}\nstderr:\n{stderr.decode()}"
            )
    except asyncio.TimeoutError:
        # Kill the process if it times out
        await safe_kill_process(proc, logger)
        raise CompilationError(
            f"Timeout: Compilation process timed out after {timeout} seconds"
        )

    return sm_build_dir / "main"


async def compilation_worker(worker_id: int, debug: bool = False):
    """Process files from the compilation queue"""
    while True:
        (
            job_name,
            main_file,
            cuda_file,
            sm_version,
            persistent_artifacts,
            output_path,
            completion_future,
        ) = await QUEUE.get()
        try:
            logger.info(f"[Worker {worker_id}]: Compiling {job_name}")
            if persistent_artifacts:
                logger.info(
                    f"[Worker {worker_id}]: Using persistent_artifacts mode for {job_name}"
                )

            tmp_path = await exec_compilation(
                job_name,
                main_file,
                cuda_file,
                sm_version,
                worker_id,
                output_path,
                persistent_artifacts,
                debug=debug,
            )
            output_path.write_bytes(tmp_path.read_bytes())
            output_path.chmod(0o755)  # make this file executable
            logger.info(
                f"[Worker {worker_id}]: Successfully compiled {job_name} and saved to {output_path}"
            )
            completion_future.set_result(True)
        except CompilationError as e:
            logger.info(f"[Worker {worker_id}]: Error compiling {job_name}")
            completion_future.set_exception(e)
        except FileNotFoundError as e:
            logger.error(f"[Worker {worker_id}]: File not found: {e}")
            completion_future.set_exception(e)
        except Exception as e:
            error_msg = f"[Worker {worker_id}]: Unhandled exception compiling {job_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Wrap in CompilationError so it's handled properly
            completion_future.set_exception(CompilationError(error_msg))
            # Don't re-raise - let the worker continue processing other jobs
        finally:
            QUEUE.task_done()


@APP.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "compile-server"}


@APP.get("/compile", response_model=CompilationResult)
async def process_compilation_request(
    job_name: str,
    main_file: str,
    cuda_file: str,
    sm_version: str,
    persistent_artifacts: int = 0,
):
    logger.info(f"/compile request received: job_name={job_name}, main_file={main_file}, cuda_file={cuda_file}, sm_version={sm_version}, backlog: {QUEUE.qsize()}")
    
    try:
        if not Path(main_file).exists():
            error_msg = f"File {main_file} not found"
            logger.error(f"/compile error: {error_msg}")
            return CompilationResult(
                job_name=job_name,
                main_file=main_file,
                cuda_file=cuda_file,
                success=False,
                message=error_msg,
            )

        if cuda_file and not Path(cuda_file).exists():
            error_msg = f"File {cuda_file} not found"
            logger.error(f"/compile error: {error_msg}")
            return CompilationResult(
                job_name=job_name,
                main_file=main_file,
                cuda_file=cuda_file,
                success=False,
                message=error_msg,
            )

        # Create a future to track completion
        completion_future = asyncio.Future()

        # Calculate the output path that will be used
        with tempfile.NamedTemporaryFile(delete=False, dir=OUT_DIR) as f:
            output_path = Path(f.name)

        logger.info(f"Queueing compilation: {job_name} -> {output_path}")

        # Create a special queue item with the future
        await QUEUE.put(
            (
                job_name,
                main_file,
                cuda_file,
                sm_version,
                bool(persistent_artifacts),
                output_path,
                completion_future,
            )
        )

        # Wait for the compilation to complete
        try:
            await completion_future

            result = CompilationResult(
                job_name=job_name,
                main_file=main_file,
                cuda_file=cuda_file,
                success=True,
                message="Compilation successful",
                output_path=str(output_path),
            )

            if persistent_artifacts:
                persistent_artifacts_dir = (
                    args.artifacts_dir / "persistent_artifacts" / output_path.name
                )
                result.persistent_artifacts_dir = str(persistent_artifacts_dir)

            logger.info(f"/compile success: {job_name} -> {output_path}")
            return result
        except CompilationError as e:
            error_msg = str(e)
            logger.error(f"/compile CompilationError for {job_name}: {error_msg}")
            result = CompilationResult(
                job_name=job_name,
                main_file=main_file,
                cuda_file=cuda_file,
                success=False,
                message=error_msg,
            )

            if persistent_artifacts:
                persistent_artifacts_dir = (
                    args.artifacts_dir / "persistent_artifacts" / output_path.name
                )
                result.persistent_artifacts_dir = str(persistent_artifacts_dir)

            return result
        except asyncio.CancelledError:
            logger.warning(f"Compilation {job_name} was cancelled")
            raise HTTPException(status_code=500, detail="Compilation was cancelled")
        except Exception as e:
            error_msg = f"Unexpected error during compilation: {str(e)}"
            logger.error(f"/compile unexpected error for {job_name}: {error_msg}", exc_info=True)
            raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Error processing compilation request: {str(e)}"
        logger.error(f"/compile request processing error for {job_name}: {error_msg}", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


async def start_workers(num_workers: int, debug: bool = False):
    """Start the compilation worker tasks"""
    workers = [
        asyncio.create_task(compilation_worker(worker_id, debug))
        for worker_id in range(num_workers)
    ]
    try:
        await asyncio.gather(*workers)
    except Exception as e:
        logger.error(f"Worker exception: {e}")
        # Re-raise the exception to crash the server
        raise


def run_compilation_server(host: str, port: int):
    """
    Run the compilation server with REST API

    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        num_workers: Number of parallel compilation workers
        debug: Whether to compile in debug mode
    """

    # Run the FastAPI server
    log_config = get_log_config()
    uvicorn.run(
        APP, host=host, port=port, log_config=log_config, timeout_graceful_shutdown=0.1
    )


def main():
    # Run the REST API compilation server
    run_compilation_server(
        args.host,
        args.port,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=2001)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--compile-debug", action="store_true")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="Path to directory to store artifacts",
        default=Path("/tmp/kernelblaster"),
    )
    args = parser.parse_args()

    ENV_DIR = args.artifacts_dir / str(uuid.uuid4())
    OUT_DIR = ENV_DIR / "out"
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Store artifacts_dir in module-level variable for use in endpoint handlers
    import src.kernelblaster.servers.compile as compile_module
    compile_module._ARTIFACTS_DIR = args.artifacts_dir

    main()
