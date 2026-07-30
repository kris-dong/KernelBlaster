"""
OpenCL compilation server for Qualcomm Adreno targets.

Compiles OpenCL C kernels (.cl) + host driver code (.c) into executables.
Supports two modes:
  - Local: compile on the build host (x86) for syntax validation / linking against ICD loader
  - Remote (SSH): compile on the target Adreno board (ARM64) for actual GPU execution

The remote mode is the primary path for generating runnable binaries.
"""
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
import time

from .server_logging import get_log_config
from .utils.queue_server import queue_worker_loop, worker_pool
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


async def _opencl_compile_job(worker_id: int, job_args: tuple) -> str:
    """Single OpenCL compile job — delegates to :class:`OpenCLCompileStrategy`.

    Phase C extracted the body into the strategy. Returns the strategy's
    ``remote_binary_path`` (or the local path on x86 fallback) as the
    string that the ``/compile_opencl`` endpoint surfaces.
    """
    from .strategies import get_compile_strategy

    (
        job_name,
        main_file,
        kernel_file,
        opencl_version,
        remote,
        output_path,
    ) = job_args

    result = await get_compile_strategy("opencl").compile(
        worker_id=worker_id,
        job_name=job_name,
        main_file=main_file,
        source_file=kernel_file,
        backend_version=opencl_version,
        backend_flag=remote,
        output_path=output_path,
        artifacts_dir=_ARTIFACTS_DIR,
        board_host=_BOARD_HOST,
    )
    return result if result is not None else str(output_path)


@asynccontextmanager
async def lifespan(app):
    logger.info(
        f"Started OpenCL compilation server on {args.host}:{args.port} "
        f"with {args.num_workers} workers, board={args.board_host}"
    )

    def _cleanup_env_dir():
        if ENV_DIR.exists():
            shutil.rmtree(ENV_DIR, ignore_errors=True)

    async with worker_pool(
        num_workers=args.num_workers,
        queue=QUEUE,
        handler=_opencl_compile_job,
        domain_error=OpenCLCompilationError,
        logger=logger,
        on_shutdown=_cleanup_env_dir,
    ):
        yield


APP = FastAPI(lifespan=lifespan)


@APP.get("/health")
async def health_check():
    return {"status": "healthy", "service": "opencl-compile-server", "board": _BOARD_HOST}


@APP.get("/compile_opencl", response_model=OpenCLCompilationResult)
async def process_compilation_request(
    job_name: str,
    main_file: str,
    kernel_file: str,
    opencl_version: str = "opencl_2.0",
    remote: int = 1,
):
    logger.info(
        f"/compile_opencl request: job_name={job_name}, main_file={main_file}, "
        f"kernel_file={kernel_file}, remote={remote}, backlog={QUEUE.qsize()}"
    )

    if not Path(main_file).exists():
        return OpenCLCompilationResult(
            job_name=job_name, main_file=main_file, kernel_file=kernel_file,
            success=False, message=f"File {main_file} not found",
        )
    if not Path(kernel_file).exists():
        return OpenCLCompilationResult(
            job_name=job_name, main_file=main_file, kernel_file=kernel_file,
            success=False, message=f"File {kernel_file} not found",
        )

    completion_future = asyncio.Future()
    with tempfile.NamedTemporaryFile(delete=False, dir=OUT_DIR) as f:
        output_path = Path(f.name)

    await QUEUE.put((
        job_name, main_file, kernel_file, opencl_version,
        bool(remote), output_path, completion_future, time.time(),
    ))

    try:
        result_path = await completion_future
        return OpenCLCompilationResult(
            job_name=job_name, main_file=main_file, kernel_file=kernel_file,
            success=True, message="Compilation successful",
            output_path=str(output_path),
            remote_binary_path=result_path if bool(remote) else None,
        )
    except OpenCLCompilationError as e:
        return OpenCLCompilationResult(
            job_name=job_name, main_file=main_file, kernel_file=kernel_file,
            success=False, message=str(e),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str, port: int):
    log_config = get_log_config()
    uvicorn.run(APP, host=host, port=port, log_config=log_config, timeout_graceful_shutdown=0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2003)
    parser.add_argument("--num-workers", type=int, default=4)
    from ..backends.opencl import default_board_host
    parser.add_argument(
        "--board-host", type=str,
        default=default_board_host(),
        help="SSH target for remote compilation on Adreno board",
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("/tmp/kernelblaster"),
    )
    args = parser.parse_args()

    _BOARD_HOST = args.board_host
    ENV_DIR = args.artifacts_dir / f"opencl_{uuid.uuid4().hex[:8]}"
    OUT_DIR = ENV_DIR / "out"
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    run_server(args.host, args.port)
