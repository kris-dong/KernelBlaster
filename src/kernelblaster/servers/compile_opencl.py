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
from .utils.process_management import safe_kill_process

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


def _tail(s: bytes, limit: int = 8192) -> str:
    if not s:
        return ""
    try:
        txt = s.decode(errors="replace")
    except Exception:
        txt = repr(s)
    return txt[-limit:] if len(txt) > limit else txt


async def _run_subprocess_shell(
    *,
    stage: str,
    cmd: str,
    cwd: Path,
    timeout_s: float,
    env: dict | None = None,
) -> tuple[bytes, bytes]:
    logger.info(f"[{stage}] starting (timeout={timeout_s}s) cwd={cwd} cmd={cmd}")
    start = asyncio.get_running_loop().time()
    proc = await asyncio.subprocess.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
        start_new_session=True,
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        await safe_kill_process(proc, logger)
        elapsed = asyncio.get_running_loop().time() - start
        raise OpenCLCompilationError(
            f"[{stage}] Timeout after {timeout_s}s (elapsed={elapsed:.2f}s)\n"
            f"cmd: {cmd}\ncwd: {cwd}\n"
        )
    elapsed = asyncio.get_running_loop().time() - start
    rc = proc.returncode
    logger.info(f"[{stage}] finished rc={rc} elapsed={elapsed:.2f}s")
    if rc != 0:
        raise OpenCLCompilationError(
            f"[{stage}] Non-zero exit (rc={rc})\n"
            f"cmd: {cmd}\ncwd: {cwd}\n"
            f"stdout_tail:\n{_tail(stdout)}\n"
            f"stderr_tail:\n{_tail(stderr)}\n"
        )
    return stdout, stderr


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
        await _run_subprocess_shell(
            stage=f"cmake_config:opencl",
            cmd=cmake_cmd,
            cwd=work_dir,
            timeout_s=timeout,
            env=ENV_VARS,
        )

    await _run_subprocess_shell(
        stage=f"make:opencl",
        cmd="make -j4",
        cwd=build_dir,
        timeout_s=timeout,
        env=ENV_VARS,
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
    await _run_subprocess_shell(
        stage="ssh_mkdir",
        cmd=mkdir_cmd,
        cwd=Path("/tmp"),
        timeout_s=30,
        env=ENV_VARS,
    )

    # Copy source files to board
    scp_cmd = (
        f"scp {ssh_opts} -q "
        f"{main_file} {kernel_file} "
        f"{board_host}:{remote_work_dir}/"
    )
    await _run_subprocess_shell(
        stage="scp_sources",
        cmd=scp_cmd,
        cwd=Path("/tmp"),
        timeout_s=60,
        env=ENV_VARS,
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
    await _run_subprocess_shell(
        stage=f"remote_compile:{job_name}",
        cmd=compile_cmd,
        cwd=Path("/tmp"),
        timeout_s=timeout,
        env=ENV_VARS,
    )

    remote_binary = f"{remote_work_dir}/main"

    # Also copy the binary back to the local output_path for the compilation result
    scp_back_cmd = (
        f"scp {ssh_opts} -q "
        f"{board_host}:{remote_binary} {output_path}"
    )
    await _run_subprocess_shell(
        stage="scp_binary_back",
        cmd=scp_back_cmd,
        cwd=Path("/tmp"),
        timeout_s=60,
        env=ENV_VARS,
    )

    return remote_binary


async def compilation_worker(worker_id: int):
    """Process OpenCL compilation requests from the queue."""
    while True:
        (
            job_name,
            main_file,
            kernel_file,
            opencl_version,
            remote,
            output_path,
            completion_future,
            enqueue_ts,
        ) = await QUEUE.get()
        try:
            queue_wait_s = time.time() - enqueue_ts
            logger.info(
                f"[Worker {worker_id}]: dequeued {job_name} after queue_wait={queue_wait_s:.2f}s"
            )

            if remote and _BOARD_HOST:
                remote_binary = await exec_remote_compilation(
                    job_name,
                    main_file,
                    kernel_file,
                    opencl_version,
                    _BOARD_HOST,
                    output_path,
                )
                output_path.chmod(0o755)
                logger.info(
                    f"[Worker {worker_id}]: Remote compilation success: {job_name} -> {remote_binary}"
                )
                completion_future.set_result(remote_binary)
            else:
                tmp_path = await exec_local_compilation(
                    job_name,
                    main_file,
                    kernel_file,
                    opencl_version,
                    worker_id,
                    output_path,
                )
                output_path.write_bytes(tmp_path.read_bytes())
                output_path.chmod(0o755)
                logger.info(
                    f"[Worker {worker_id}]: Local compilation success: {job_name} -> {output_path}"
                )
                completion_future.set_result(str(output_path))
        except OpenCLCompilationError as e:
            logger.error(f"[Worker {worker_id}]: Compilation error for {job_name}: {e.message}")
            completion_future.set_exception(e)
        except Exception as e:
            error_msg = f"[Worker {worker_id}]: Unhandled exception compiling {job_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            completion_future.set_exception(OpenCLCompilationError(error_msg))
        finally:
            QUEUE.task_done()


@asynccontextmanager
async def lifespan(app):
    logger.info(
        f"Started OpenCL compilation server on {args.host}:{args.port} "
        f"with {args.num_workers} workers, board={args.board_host}"
    )
    for wid in range(args.num_workers):
        asyncio.create_task(compilation_worker(wid))
    yield
    if ENV_DIR.exists():
        shutil.rmtree(ENV_DIR, ignore_errors=True)


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
    parser.add_argument(
        "--board-host", type=str,
        default=os.getenv("KERNELBLASTER_ADRENO_BOARD_HOST", "root@192.0.2.201"),
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
