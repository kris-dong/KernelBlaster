"""
OpenCL compilation server for Qualcomm Adreno targets.

Compiles OpenCL C kernels (.cl) + host driver code (.c) into executables.
Supports two modes:
  - Local: compile on the build host (x86) for syntax validation / linking against ICD loader
  - Remote (SSH): compile on the target Adreno board (ARM64) for actual GPU execution

The remote mode is the primary path for generating runnable binaries.
"""
import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
from pydantic import BaseModel
import logging
import shutil
import tempfile
import uuid
import time

from .utils.subprocess import run_subprocess_shell

logger = logging.getLogger("uvicorn")

QUEUE = asyncio.Queue()
OPENCL_ENV_PATH = Path(__file__).parent / "opencl_env"
ENV_VARS = os.environ.copy()

# Module-level variables set during startup
_ARTIFACTS_DIR = None
_BOARD_HOST = None  # SSH target for remote compilation (e.g., "root@192.0.2.201")


class OpenCLCompilationRequest(BaseModel):
    job_name: str
    main_file: str  # Host-side C driver code
    kernel_file: str  # OpenCL kernel (.cl)
    opencl_version: str = "opencl_2.0"
    remote: bool = True  # Compile on the board via SSH


class OpenCLCompilationResult(BaseModel):
    job_name: str
    main_file: str
    kernel_file: str
    success: bool = False
    message: str = None
    output_path: str = None
    remote_binary_path: str = None  # Path on the board where the binary lives


class OpenCLCompilationError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def get_opencl_env_root(thread_id: int) -> Path:
    path = ENV_DIR / f"opencl_eval_{thread_id}"
    if not path.exists():
        shutil.copytree(OPENCL_ENV_PATH, path)
        logger.info(f"Set up OpenCL environment at {path}")
    return path


async def exec_local_compilation(
    job_name: str,
    main_file: str,
    kernel_file: str,
    opencl_version: str,
    worker_id: int,
    output_path: Path,
    timeout: int = 360,
):
    """Compile OpenCL host code locally (x86). Used for syntax checking."""
    work_dir = get_opencl_env_root(worker_id)

    # Copy source files into work dir
    shutil.copy2(main_file, work_dir / "main.c")
    shutil.copy2(kernel_file, work_dir / "kernel.cl")

    cl_version = opencl_version.replace("opencl_", "").replace(".", "") + "0"
    if len(cl_version) < 3:
        cl_version = cl_version + "0"

    build_dir = work_dir / "build"
    if not build_dir.exists():
        cmake_cmd = (
            f"mkdir -p {build_dir} && cd {build_dir} && "
            f"cmake -DCMAKE_BUILD_TYPE=Release "
            f'-DOPENCL_VERSION="{cl_version}" '
            f".."
        )
        await run_subprocess_shell(
            stage=f"cmake_config:opencl",
            cmd=cmake_cmd,
            cwd=work_dir,
            timeout_s=timeout,
            env=ENV_VARS,
            error_factory=OpenCLCompilationError,
            logger=logger,
        )

    await run_subprocess_shell(
        stage=f"make:opencl",
        cmd="make -j4",
        cwd=build_dir,
        timeout_s=timeout,
        env=ENV_VARS,
        error_factory=OpenCLCompilationError,
        logger=logger,
    )

    return build_dir / "main"


async def exec_remote_compilation(
    job_name: str,
    main_file: str,
    kernel_file: str,
    opencl_version: str,
    board_host: str,
    output_path: Path,
    timeout: int = 360,
) -> str:
    """
    Compile OpenCL kernel + host on the remote Adreno board via SSH.
    Returns the path to the compiled binary on the remote board.
    """
    remote_work_dir = f"/tmp/kernelblaster_compile/{job_name}_{uuid.uuid4().hex[:8]}"
    ssh_opts = "-o StrictHostKeyChecking=no -o ConnectTimeout=10"

    # Create remote work dir
    mkdir_cmd = f"ssh {ssh_opts} {board_host} 'mkdir -p {remote_work_dir}'"
    await run_subprocess_shell(
        stage="ssh_mkdir",
        cmd=mkdir_cmd,
        cwd=Path("/tmp"),
        timeout_s=30,
        env=ENV_VARS,
        error_factory=OpenCLCompilationError,
        logger=logger,
    )

    # Copy source files to board
    scp_cmd = (
        f"scp {ssh_opts} -q "
        f"{main_file} {kernel_file} "
        f"{board_host}:{remote_work_dir}/"
    )
    await run_subprocess_shell(
        stage="scp_sources",
        cmd=scp_cmd,
        cwd=Path("/tmp"),
        timeout_s=60,
        env=ENV_VARS,
        error_factory=OpenCLCompilationError,
        logger=logger,
    )

    # Compile on board
    main_basename = os.path.basename(main_file)
    compile_cmd = (
        f"ssh {ssh_opts} {board_host} "
        f"'cd {remote_work_dir} && "
        f"gcc -o main {main_basename} "
        f"-I/usr/include -L/usr/lib -lOpenCL -lm "
        f"-DCL_TARGET_OPENCL_VERSION=200'"
    )
    await run_subprocess_shell(
        stage=f"remote_compile:{job_name}",
        cmd=compile_cmd,
        cwd=Path("/tmp"),
        timeout_s=timeout,
        env=ENV_VARS,
        error_factory=OpenCLCompilationError,
        logger=logger,
    )

    remote_binary = f"{remote_work_dir}/main"

    # Also copy the binary back to the local output_path for the compilation result
    scp_back_cmd = (
        f"scp {ssh_opts} -q "
        f"{board_host}:{remote_binary} {output_path}"
    )
    await run_subprocess_shell(
        stage="scp_binary_back",
        cmd=scp_back_cmd,
        cwd=Path("/tmp"),
        timeout_s=60,
        env=ENV_VARS,
        error_factory=OpenCLCompilationError,
        logger=logger,
    )

    return remote_binary


# _opencl_compile_job, lifespan, endpoints, run_server, and __main__
# removed in Phase E — callers now hit compile_server.py which invokes
# OpenCLCompileStrategy directly.

