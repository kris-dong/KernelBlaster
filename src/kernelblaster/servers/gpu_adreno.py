"""
GPU execution server for Qualcomm Adreno targets.

Executes compiled OpenCL binaries on the remote Adreno board via SSH.
Supports profiling via OpenCL event timing (--profile flag on binaries).
"""
import asyncio
import os
import time
import logging
from pydantic import BaseModel
from pathlib import Path
import json
import tempfile
import uuid
from typing import Optional

from .utils.subprocess import run_subprocess_shell

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
    """Execute a command on the remote Adreno board via SSH.

    Timeout + safe-kill plumbing is shared with the rest of the servers via
    ``run_subprocess_shell``. Raises ``AdrenoExecutionError`` on timeout or
    non-zero exit (error_factory contract).
    """
    remote_cmd = cmd
    if cwd:
        remote_cmd = f"cd {cwd} && {cmd}"
    full_cmd = f"ssh {SSH_OPTS} {board_host} '{remote_cmd}'"

    stdout, stderr = await run_subprocess_shell(
        stage=f"ssh:{cmd[:40]}",
        cmd=full_cmd,
        cwd=Path("/tmp"),
        timeout_s=timeout,
        error_factory=AdrenoExecutionError,
        logger=logger,
    )
    return stdout.decode(), stderr.decode()


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
            await run_subprocess_shell(
                stage="scp_binary",
                cmd=scp_cmd,
                cwd=Path("/tmp"),
                timeout_s=60,
                error_factory=AdrenoExecutionError,
                logger=logger,
            )

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
                    await run_subprocess_shell(
                        stage=f"scp_kernel:{remote_name}",
                        cmd=scp_cmd,
                        cwd=Path("/tmp"),
                        timeout_s=60,
                        error_factory=AdrenoExecutionError,
                        logger=logger,
                    )

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



# FastAPI parts (_adreno_binary_job, lifespan, APP, endpoints, run_server,
# __main__) deleted in the Exec unification arc — callers now hit
# servers.exec_server which invokes RemoteExecStrategy internally.
