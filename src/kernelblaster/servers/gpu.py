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
import asyncio
from contextlib import asynccontextmanager
import os
import time
import psutil
import logging
from pydantic import BaseModel
from pathlib import Path
import json
import tempfile
import stat
from typing import Optional

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



# FastAPI parts (lifespan, APP, endpoints, worker handler, run_server, main,
# __main__) deleted in the Exec unification arc — callers now hit
# servers.exec_server which invokes LocalExecStrategy internally.
