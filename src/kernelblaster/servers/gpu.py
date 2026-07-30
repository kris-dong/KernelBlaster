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
import time
import psutil
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
import logging
from pydantic import BaseModel
from pathlib import Path
import uvicorn
import json
import tempfile
import stat
from typing import Optional

from .server_logging import get_log_config
from .utils.queue_server import worker_pool
from .utils.subprocess import run_subprocess_shell

env = None

QUEUE = asyncio.Queue()

logger = logging.getLogger("uvicorn")

# Common temporary directory for all operations
WORKING_DIR = None

# Multi-GPU worker configuration (populated at startup)
GPU_IDS: list[str] | None = None


def get_temp_dir():
    """Get or create a common temporary directory for all GPU operations"""
    global WORKING_DIR
    if WORKING_DIR is None or not os.path.exists(WORKING_DIR):
        WORKING_DIR = tempfile.mkdtemp(prefix="kernelblaster_gpu_")
    return WORKING_DIR


# Start worker tasks in the background
@asynccontextmanager
async def lifespan(app):
    global logger, env, GPU_IDS

    # Base environment for subprocesses launched by this server.
    # NOTE: per-worker GPU pinning is applied at execution time via env vars.
    env = os.environ.copy()
    env.setdefault("NVIDIA_TF32_OVERRIDE", "0")

    # Determine which GPUs (and how many workers) to use.
    # Examples:
    #   KERNELBLASTER_GPU_SERVER_GPU_IDS="0,1,2,3"
    #   KERNELBLASTER_GPU_SERVER_NUM_WORKERS=4
    gpu_ids_raw = os.getenv("KERNELBLASTER_GPU_SERVER_GPU_IDS", "").strip()
    if gpu_ids_raw:
        GPU_IDS = [s.strip() for s in gpu_ids_raw.split(",") if s.strip()]
    else:
        num_workers = int(os.getenv("KERNELBLASTER_GPU_SERVER_NUM_WORKERS", "1"))
        GPU_IDS = [str(i) for i in range(max(1, num_workers))]

    logger.info(
        f"GPU Server worker config: num_workers={len(GPU_IDS)} GPU_IDS={GPU_IDS} "
        f"(override via KERNELBLASTER_GPU_SERVER_NUM_WORKERS / KERNELBLASTER_GPU_SERVER_GPU_IDS)"
    )

    # Print the current user (whoami) at server startup
    logger.info(f"GPU Server running as user: {os.getuid()}")
    logger.info(f"GPU Server running as user: {os.geteuid()}")

    # Diagnostic-only — don't crash startup if the container's user/group DB
    # has holes (e.g. a mounted GID that isn't in /etc/group; `groups` exits
    # non-zero in that case, which used to abort the whole lifespan).
    try:
        stdout, stderr = await exec_command("whoami")
        logger.info(f"GPU Server running as user: {stdout}\n{stderr}")
    except Exception as diag_err:
        logger.warning(f"Diagnostic `whoami` failed (non-fatal): {diag_err}")

    try:
        stdout, stderr = await exec_command("groups")
        logger.info(f"User groups: {stdout}\n{stderr}")
    except Exception as diag_err:
        logger.warning(f"Diagnostic `groups` failed (non-fatal): {diag_err}")
    
    # Print nvidia-smi information before starting the server
    await print_nvidia_smi(logger)

    # Check for pre-existing GPU processes
    await check_gpu_processes()
    # Start worker tasks on startup (one per GPU id). ``worker_pool``
    # cancels the workers on lifespan exit — no more leaked coroutines.
    async with worker_pool(
        num_workers=len(GPU_IDS),
        queue=QUEUE,
        handler=_gpu_binary_job,
        domain_error=GpuCommandError,
        logger=logger,
    ):
        yield


APP = FastAPI(lifespan=lifespan)


class GpuExecutionRequest(BaseModel):
    """Request model for GPU binary execution"""

    args: Optional[str] = ""  # Command line arguments for the binary


class GpuCommandResult(BaseModel):
    stdout: str | list[str] = []
    stderr: str | list[str] = []
    success: bool = False
    message: str = None


class GpuCommandError(Exception):
    def __init__(self, error_message: str):
        self.error_message = error_message
        super().__init__(self.error_message)


async def print_nvidia_smi(logger):
    """Print nvidia-smi information"""
    try:
        nvidia_smi_stdout, nvidia_smi_stderr = await exec_command("nvidia-smi")
        logger.info(f"GPU Server Startup - nvidia-smi output:\n{nvidia_smi_stdout}")
        if nvidia_smi_stderr:
            logger.warning(
                f"GPU Server Startup - nvidia-smi stderr:\n{nvidia_smi_stderr}"
            )
    except Exception as nvidia_smi_error:
        logger.warning(
            f"GPU Server Startup - Failed to execute nvidia-smi: {str(nvidia_smi_error)}"
        )


async def _resolve_assigned_gpu_uuids() -> set[str] | None:
    """Return the UUIDs of GPUs this server is assigned to (via ``GPU_IDS``).

    Returns ``None`` if we can't map indices to UUIDs (nvidia-smi silent /
    parse fails), signalling to the caller "fall back to the global check
    to stay safe."
    """
    if not GPU_IDS:
        return None
    try:
        stdout, _ = await exec_command(
            "nvidia-smi --query-gpu=index,uuid --format=csv,noheader"
        )
    except Exception:
        return None

    index_to_uuid: dict[str, str] = {}
    for raw_line in stdout.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        index_to_uuid[parts[0]] = parts[1]

    assigned = {index_to_uuid[i] for i in GPU_IDS if i in index_to_uuid}
    return assigned or None


async def check_gpu_processes():
    """Check for pre-existing processes on the GPUs this server is assigned to.

    Filters out stale or non-existent PIDs and entries where the process name is
    reported as "[Not Found]" by nvidia-smi, to avoid false positives.

    When ``GPU_IDS`` restricts the server to a subset of GPUs (via
    ``KERNELBLASTER_GPU_SERVER_GPU_IDS``), only processes running on THAT
    subset are treated as conflicts — a shared cluster where the operator
    explicitly pins to an idle GPU should not be blocked by unrelated
    workloads on other GPUs. The scope-narrowing is opt-in: without a
    ``GPU_IDS`` override the check remains global (safe default).
    """
    try:
        assigned_uuids = await _resolve_assigned_gpu_uuids()
        scoped = assigned_uuids is not None
        query_fields = (
            "gpu_uuid,pid,process_name" if scoped else "pid,process_name"
        )
        stdout, _ = await exec_command(
            f"nvidia-smi --query-compute-apps={query_fields} --format=csv,noheader"
        )

        active_processes: list[str] = []
        for raw_line in stdout.split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(",")]
            if not parts:
                continue

            if scoped:
                if len(parts) < 3:
                    continue
                gpu_uuid, pid_str, proc_name = parts[0], parts[1], parts[2]
                if gpu_uuid not in assigned_uuids:
                    continue
            else:
                pid_str = parts[0]
                proc_name = parts[1] if len(parts) > 1 else ""

            # Skip entries with invalid PID format
            try:
                pid = int(pid_str)
            except ValueError:
                continue

            # Ignore stale entries or where process name cannot be resolved
            if proc_name == "[Not Found]" or not psutil.pid_exists(pid):
                continue

            active_processes.append(f"{pid}, {proc_name or '[Unknown]'}")

        if active_processes:
            scope_note = (
                f" (scoped to assigned GPUs {sorted(GPU_IDS)})" if scoped else ""
            )
            raise RuntimeError(
                f"Found pre-existing GPU processes{scope_note}:\n"
                f"{json.dumps(active_processes, indent=2)}"
            )

    except Exception as e:
        if "nvidia-smi: not found" in str(e):
            raise RuntimeError(
                "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
            )
        raise e


async def exec_command(
    cmd: str,
    timeout=3600,
    env_vars: Optional[dict] = None,
    n_runs: Optional[int] = 1,
) -> tuple[list[str], list[str]] | tuple[str, str]:
    """Execute a shell command (possibly multiple times).

    Subprocess plumbing — timeout, SIGKILL-on-timeout, non-zero-exit
    diagnostics — is handled by ``run_subprocess_shell``. ``GpuCommandError``
    is preserved as the raised exception type via the ``error_factory`` kwarg.
    """
    process_env = env.copy() if env else os.environ.copy()
    if env_vars:
        process_env.update(env_vars)

    working_dir = Path(get_temp_dir())

    stdout_list: list[str] = []
    stderr_list: list[str] = []

    for run_idx in range(n_runs):
        stdout, stderr = await run_subprocess_shell(
            stage=f"gpu_exec:{cmd[:60]}",
            cmd=cmd,
            cwd=working_dir,
            timeout_s=timeout,
            env=process_env,
            error_factory=GpuCommandError,
            logger=logger,
        )
        stdout_list.append(stdout.decode())
        stderr_list.append(stderr.decode())

    if n_runs == 1:
        return stdout_list[0], stderr_list[0]
    return stdout_list, stderr_list


async def exec_binary(
    binary_path: str,
    args: str = "",
    timeout=3600,
    env_vars: Optional[dict] = None,
    prefix_command: Optional[str] = None,
    n_runs: Optional[int] = 1,
) -> tuple[list[str], list[str]] | tuple[str, str]:
    """Execute a binary file with optional arguments, environment variables, and prefix command"""
    if prefix_command:
        cmd = f"{prefix_command} {binary_path} {args}".strip()
    else:
        cmd = f"{binary_path} {args}".strip()

    return await exec_command(cmd, timeout, env_vars, n_runs)


def save_binary_to_temp(binary_data: bytes, filename: str = "gpu_executable") -> str:
    """Save binary data to a temporary file and make it executable"""
    # Use common temp directory
    temp_dir = get_temp_dir()
    # IMPORTANT: never write to a path derived solely from the client-provided filename.
    # We can receive concurrent requests (and clients may retry the same request),
    # which would otherwise cause:
    # - [Errno 26] Text file busy (overwrite while executing)
    # - "does not exist" (another worker cleans up the shared path)
    safe_name = os.path.basename(filename) if filename else "gpu_executable"
    fd, binary_path = tempfile.mkstemp(prefix=f"{safe_name}_", dir=temp_dir)

    # Write binary data
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(binary_data)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        cleanup_temp_file(binary_path)
        raise

    # Make executable
    os.chmod(
        binary_path,
        stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH,
    )

    return binary_path


def cleanup_temp_file(binary_path: str):
    """Clean up temporary binary file"""
    try:
        if os.path.exists(binary_path):
            os.remove(binary_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary file: {e}")


async def _gpu_binary_job(worker_id: int, job_args: tuple) -> GpuCommandResult:
    """Handler for one ``/gpu/binary`` request.

    Called by :func:`worker_pool` for each item dequeued from
    ``QUEUE``. The handler catches ``GpuCommandError`` internally and
    returns a ``GpuCommandResult(success=False)`` on failure so the
    HTTP client sees a valid JSON body — matches the pre-Phase-B
    behaviour where the endpoint always returned 200 with an inline
    success flag.

    ``job_args`` shape (6-tuple):
    ``(binary_path, args, env_vars, prefix_command, n_runs, timeout)``
    """
    binary_path, cmd_args, env_vars, prefix_command, n_runs, timeout = job_args

    # Pin this worker to a specific GPU by injecting CUDA_VISIBLE_DEVICES.
    # If the caller explicitly passed CUDA_VISIBLE_DEVICES, respect it.
    eff_env_vars = dict(env_vars or {})
    if "CUDA_VISIBLE_DEVICES" not in eff_env_vars:
        gpu_id = str(worker_id)
        if GPU_IDS and worker_id < len(GPU_IDS):
            gpu_id = str(GPU_IDS[worker_id])
        eff_env_vars["CUDA_VISIBLE_DEVICES"] = gpu_id
    # Ensure TF32 override is stable unless caller requested otherwise.
    eff_env_vars.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    gpu_visible = eff_env_vars.get("CUDA_VISIBLE_DEVICES", "<unset>")

    logger.info(
        f"[Worker {worker_id}]: Assigned GPU CUDA_VISIBLE_DEVICES="
        f"{gpu_visible} for binary {binary_path}"
    )
    logger.info(
        f"[Worker {worker_id}]: Executing binary {binary_path} with "
        f"args: {cmd_args}, env_vars: {eff_env_vars}, "
        f"prefix: {prefix_command}, n_runs: {n_runs}, timeout: {timeout}"
    )

    try:
        stdout_list, stderr_list = await exec_binary(
            binary_path,
            cmd_args,
            timeout=timeout,
            env_vars=eff_env_vars,
            prefix_command=prefix_command,
            n_runs=n_runs,
        )
        logger.info(
            f"[Worker {worker_id}]: Successfully executed binary on "
            f"CUDA_VISIBLE_DEVICES={gpu_visible}: "
            f"{f'{prefix_command} ' if prefix_command else ''}{binary_path} "
            f"with {n_runs} runs"
        )
        return GpuCommandResult(success=True, stdout=stdout_list, stderr=stderr_list)
    except GpuCommandError as e:
        logger.error(
            f"[Worker {worker_id}]: Error executing binary {binary_path} "
            f"on CUDA_VISIBLE_DEVICES={gpu_visible}: {e.error_message}"
        )
        return GpuCommandResult(success=False, message=e.error_message)
    except Exception as e:
        logger.error(f"[Worker {worker_id}]: Unexpected error: {e}")
        return GpuCommandResult(success=False, message=f"Internal error: {e}")
    finally:
        # Always clean up the temp binary — success or failure.
        cleanup_temp_file(binary_path)


@APP.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "gpu-server"}


@APP.post("/gpu/binary", response_model=GpuCommandResult)
async def execute_gpu_binary(
    binary: UploadFile = File(..., description="Binary executable to run on GPU"),
    args: Optional[str] = Form("", description="Command line arguments for the binary"),
    env_vars: Optional[str] = Form(
        None, description="Environment variables in JSON format"
    ),
    prefix_command: Optional[str] = Form(
        None,
        description="Command to prefix before the binary (e.g., 'ncu', 'nsys profile')",
    ),
    n_runs: Optional[int] = Form(
        1,
        description="Number of times to run the binary",
    ),
    timeout: Optional[float] = Form(
        3600,
        description="Timeout in seconds for command execution",
    ),
):
    """Execute a binary file on the GPU server"""

    logger.info(
        f"/gpu/binary - Binary: {binary.filename}, Args: {args}, Env vars: {env_vars}, Prefix: {prefix_command}, Timeout: {timeout}s, Queue backlog: {QUEUE.qsize()}"
    )

    try:
        # Read binary data
        binary_data = await binary.read()
        if not binary_data:
            raise HTTPException(status_code=400, detail="Empty binary file provided")

        # Parse environment variables if provided
        parsed_env_vars = None
        if env_vars:
            try:
                parsed_env_vars = json.loads(env_vars)
                if not isinstance(parsed_env_vars, dict):
                    raise ValueError("Environment variables must be a JSON object")
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid environment variables JSON: {str(e)}",
                )

        # Save binary to temporary location
        binary_path = save_binary_to_temp(
            binary_data, binary.filename or "gpu_executable"
        )

        # Create a future to track completion
        completion_future = asyncio.Future()

        # Queue-item shape matches ``queue_worker_loop``'s convention:
        # ``(*job_args, completion_future, enqueue_ts)``. ``_gpu_binary_job``
        # unpacks the 6-tuple job_args.
        await QUEUE.put(
            (
                binary_path,
                args,
                parsed_env_vars,
                prefix_command,
                n_runs,
                timeout,
                completion_future,
                time.time(),
            )
        )

        # Wait for completion
        await completion_future
        return completion_future.result()

    except asyncio.CancelledError:
        logger.info(f"Request for binary {binary.filename} was cancelled")
        raise HTTPException(status_code=500, detail="Request was cancelled")
    except Exception as e:
        logger.error(f"Error processing binary execution request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# /gpu/cmd endpoint deleted in Phase B — zero in-tree callers per
# ``notes/server_unification_audit.md``. The unified exec server keeps
# only /gpu/binary.


def run_server(host: str, port: int, log_filepath: str = None):
    """
    Run the compilation server with REST API

    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        log_filepath: Optional path to log file for uvicorn logging
    """
    # Run the FastAPI server
    log_config = get_log_config(log_filepath=log_filepath)
    uvicorn.run(
        APP, host=host, port=port, log_config=log_config, timeout_graceful_shutdown=0.1
    )


def main(args):
    # Ensure log directory exists if log path is provided
    if args.log_path:
        log_dir = args.log_path.parent
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the REST API compilation server
    run_server(
        args.host,
        args.port,
        log_filepath=str(args.log_path) if args.log_path else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2002)
    parser.add_argument(
        "--log_path", type=Path, default=Path("/tmp/kernelblaster/gpu_server.log")
    )
    args = parser.parse_args()

    # Define base environment variables for GPU subprocesses.
    # Per-worker CUDA_VISIBLE_DEVICES pinning is applied in gpu_worker().
    env = os.environ.copy()
    env.setdefault("NVIDIA_TF32_OVERRIDE", "0")

    main(args)
