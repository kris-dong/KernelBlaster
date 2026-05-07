"""
GPU execution server for Qualcomm Adreno targets.

Executes compiled OpenCL binaries on the remote Adreno board via SSH.
Supports profiling via OpenCL event timing (--profile flag on binaries).
"""
import argparse
import asyncio
from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
import logging
from pydantic import BaseModel
from pathlib import Path
import uvicorn
import json
import tempfile
import uuid
from typing import Optional

from .server_logging import get_log_config
from .utils.process_management import safe_kill_process

logger = logging.getLogger("uvicorn")

QUEUE = asyncio.Queue()
_BOARD_HOST = None
_REMOTE_WORK_DIR = "/tmp/kernelblaster_gpu"
SSH_OPTS = "-o StrictHostKeyChecking=no -o ConnectTimeout=10"


class AdrenoExecutionResult(BaseModel):
    stdout: str | list[str] = []
    stderr: str | list[str] = []
    success: bool = False
    message: str = None


class AdrenoExecutionError(Exception):
    def __init__(self, error_message: str):
        self.error_message = error_message
        super().__init__(self.error_message)


async def exec_remote_command(
    cmd: str,
    board_host: str,
    timeout: float = 3600,
    cwd: str = None,
) -> tuple[str, str]:
    """Execute a command on the remote board via SSH."""
    remote_cmd = cmd
    if cwd:
        remote_cmd = f"cd {cwd} && {cmd}"
    full_cmd = f"ssh {SSH_OPTS} {board_host} '{remote_cmd}'"

    proc = await asyncio.create_subprocess_shell(
        full_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        stdout_str = stdout.decode()
        stderr_str = stderr.decode()
        if proc.returncode != 0:
            raise AdrenoExecutionError(
                f"stdout:\n{stdout_str}\nstderr:\n{stderr_str}"
            )
        return stdout_str, stderr_str
    except asyncio.TimeoutError:
        await safe_kill_process(proc, logger)
        raise AdrenoExecutionError(
            f"Timeout: Execution timed out after {timeout} seconds"
        )


async def upload_and_exec_binary(
    binary_data: bytes,
    filename: str,
    board_host: str,
    kernel_files: list[str] = None,
    args_str: str = "",
    timeout: float = 3600,
    n_runs: int = 1,
    profile: bool = False,
) -> tuple[list[str], list[str]]:
    """Upload a binary + kernel files to the board and execute it.

    Cleanup of ``remote_dir`` runs in a ``finally`` block so the board's
    /tmp (a 3.8 GiB tmpfs on the QRB5165) does NOT accumulate ~128 MiB
    reference_output.bin orphans every time a run times out / errors.
    """
    run_id = uuid.uuid4().hex[:8]
    remote_dir = f"{_REMOTE_WORK_DIR}/{run_id}"

    # Create remote dir
    await exec_remote_command(f"mkdir -p {remote_dir}", board_host, timeout=30)

    try:
        # Write binary to local temp and scp to board
        local_tmp = tempfile.NamedTemporaryFile(delete=False, prefix="adreno_bin_")
        try:
            local_tmp.write(binary_data)
            local_tmp.close()
            os.chmod(local_tmp.name, 0o755)

            scp_cmd = f"scp {SSH_OPTS} -q {local_tmp.name} {board_host}:{remote_dir}/main"
            proc = await asyncio.create_subprocess_shell(
                scp_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=60)

            # Make binary executable on board
            await exec_remote_command(f"chmod +x {remote_dir}/main", board_host, timeout=10)
        finally:
            os.unlink(local_tmp.name)

        # Copy kernel/extra files to the remote execution dir.
        # .cl files are renamed to kernel.cl (the driver opens "kernel.cl" at runtime).
        # Other files (e.g. reference_output.bin) are copied with their original basename.
        if kernel_files:
            for kf in kernel_files:
                if os.path.exists(kf):
                    remote_name = os.path.basename(kf)
                    if remote_name.endswith(".cl"):
                        remote_name = "kernel.cl"
                    scp_cmd = f"scp {SSH_OPTS} -q {kf} {board_host}:{remote_dir}/{remote_name}"
                    proc = await asyncio.create_subprocess_shell(
                        scp_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await asyncio.wait_for(proc.communicate(), timeout=60)

        # Execute
        exec_args = args_str
        if profile:
            exec_args = f"{exec_args} --profile".strip()

        stdout_list = []
        stderr_list = []

        for _ in range(n_runs):
            run_cmd = f"cd {remote_dir} && ./main {exec_args}"
            stdout, stderr = await exec_remote_command(run_cmd, board_host, timeout=timeout)
            stdout_list.append(stdout)
            stderr_list.append(stderr)

        if n_runs == 1:
            return stdout_list[0], stderr_list[0]
        return stdout_list, stderr_list
    finally:
        # Always wipe the remote dir, even on timeout / error / cancellation.
        # Best-effort: a stale orphan beats a re-raised exception that would
        # mask the original error from the caller.
        try:
            await exec_remote_command(f"rm -rf {remote_dir}", board_host, timeout=10)
        except Exception as cleanup_err:
            logger.warning(
                f"Failed to clean up remote dir {remote_dir} on board {board_host}: "
                f"{cleanup_err} (orphan will be reaped at next server start)"
            )


async def gpu_worker(worker_id: int):
    """Process Adreno GPU execution requests from the queue."""
    while True:
        queue_item = await QUEUE.get()
        completion_future = queue_item[-1]

        try:
            if len(queue_item) == 8:
                binary_data, filename, kernel_files, args_str, n_runs, profile, timeout, _ = queue_item
            else:
                raise ValueError(f"Invalid queue item format: {len(queue_item)} items")

            logger.info(
                f"[Worker {worker_id}]: Executing on Adreno board {_BOARD_HOST} "
                f"(file={filename}, n_runs={n_runs}, profile={profile})"
            )

            stdout, stderr = await upload_and_exec_binary(
                binary_data=binary_data,
                filename=filename,
                board_host=_BOARD_HOST,
                kernel_files=kernel_files,
                args_str=args_str,
                timeout=timeout,
                n_runs=n_runs,
                profile=profile,
            )

            logger.info(f"[Worker {worker_id}]: Execution successful")
            completion_future.set_result(
                AdrenoExecutionResult(success=True, stdout=stdout, stderr=stderr)
            )

        except AdrenoExecutionError as e:
            logger.error(f"[Worker {worker_id}]: Execution error: {e.error_message}")
            completion_future.set_result(
                AdrenoExecutionResult(success=False, message=e.error_message)
            )
        except Exception as e:
            logger.error(f"[Worker {worker_id}]: Unexpected error: {str(e)}")
            completion_future.set_result(
                AdrenoExecutionResult(success=False, message=f"Internal error: {str(e)}")
            )
        finally:
            QUEUE.task_done()


@asynccontextmanager
async def lifespan(app):
    global _BOARD_HOST
    _BOARD_HOST = args.board_host
    logger.info(f"Adreno GPU server starting: board={_BOARD_HOST}, workers={args.num_workers}")

    # Verify SSH connectivity to board
    try:
        stdout, _ = await exec_remote_command("echo ok && ls /dev/kgsl-3d0", _BOARD_HOST, timeout=15)
        logger.info(f"Board connectivity verified: {_BOARD_HOST}")
    except Exception as e:
        logger.warning(f"Board connectivity check failed: {e} (server will start anyway)")

    # Ensure remote work dir exists AND wipe any orphan run dirs left behind
    # by previously crashed processes. Without this, the board's /tmp (a 3.8
    # GiB tmpfs) stays full of stale reference_output.bin files (~128 MiB
    # each) from old SIGKILL'd / network-dropped runs and the new run quickly
    # hits ENOSPC again. Pre-flight reset is safe: a fresh server start
    # implies any prior run is dead, so its remote_dir is guaranteed orphan.
    try:
        await exec_remote_command(
            f"rm -rf {_REMOTE_WORK_DIR} && mkdir -p {_REMOTE_WORK_DIR}",
            _BOARD_HOST,
            timeout=30,
        )
        # Also reap stragglers from the host-side `_generate_reference` path
        # (opt_opencl_rl.py uses /tmp/kernelblaster_refgen_<host_pid>) — those
        # leak when the host RL process dies before the post-step rm fires.
        await exec_remote_command(
            f"rm -rf /tmp/kernelblaster_refgen_*",
            _BOARD_HOST,
            timeout=15,
        )
        # Probe free space now so the log shows what we recovered.
        df_out, _ = await exec_remote_command(
            "df -h /tmp | tail -1", _BOARD_HOST, timeout=10
        )
        logger.info(f"Pre-flight cleanup of {_BOARD_HOST}:/tmp done: {df_out.strip()}")
    except Exception as e:
        logger.warning(
            f"Pre-flight cleanup failed (continuing anyway): {e}"
        )

    for wid in range(args.num_workers):
        asyncio.create_task(gpu_worker(wid))
    yield


APP = FastAPI(lifespan=lifespan)


@APP.get("/health")
async def health_check():
    return {"status": "healthy", "service": "adreno-gpu-server", "board": _BOARD_HOST}


@APP.post("/gpu/binary", response_model=AdrenoExecutionResult)
async def execute_gpu_binary(
    binary: UploadFile = File(..., description="Binary executable to run on Adreno GPU"),
    args: Optional[str] = Form("", description="Command line arguments for the binary"),
    env_vars: Optional[str] = Form(None, description="Environment variables (JSON)"),
    prefix_command: Optional[str] = Form(None, description="Unused for Adreno (ignored)"),
    n_runs: Optional[int] = Form(1, description="Number of times to run the binary"),
    timeout: Optional[float] = Form(3600, description="Timeout in seconds"),
    kernel_files: Optional[str] = Form(None, description="JSON list of .cl kernel file paths to copy"),
    profile: Optional[bool] = Form(False, description="Enable OpenCL event profiling"),
):
    """Execute a binary on the remote Adreno GPU board via SSH."""
    logger.info(
        f"/gpu/binary (Adreno) - Binary: {binary.filename}, Args: {args}, "
        f"n_runs: {n_runs}, profile: {profile}, Queue backlog: {QUEUE.qsize()}"
    )

    try:
        binary_data = await binary.read()
        if not binary_data:
            raise HTTPException(status_code=400, detail="Empty binary file provided")

        parsed_kernel_files = None
        if kernel_files:
            try:
                parsed_kernel_files = json.loads(kernel_files)
            except json.JSONDecodeError:
                parsed_kernel_files = [kernel_files]

        completion_future = asyncio.Future()
        await QUEUE.put((
            binary_data,
            binary.filename or "adreno_executable",
            parsed_kernel_files,
            args or "",
            n_runs,
            profile or False,
            timeout,
            completion_future,
        ))

        await completion_future
        return completion_future.result()

    except asyncio.CancelledError:
        raise HTTPException(status_code=500, detail="Request was cancelled")
    except Exception as e:
        logger.error(f"Error processing Adreno execution request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def run_server(host: str, port: int, log_filepath: str = None):
    log_config = get_log_config(log_filepath=log_filepath)
    uvicorn.run(APP, host=host, port=port, log_config=log_config, timeout_graceful_shutdown=0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2004)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--board-host", type=str,
        default=os.getenv("KERNELBLASTER_ADRENO_BOARD_HOST", "root@192.0.2.201"),
        help="SSH target for the Adreno dev board",
    )
    parser.add_argument(
        "--log_path", type=Path, default=Path("/tmp/kernelblaster/adreno_gpu_server.log"),
    )
    args = parser.parse_args()

    if args.log_path:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)

    run_server(args.host, args.port, log_filepath=str(args.log_path) if args.log_path else None)
