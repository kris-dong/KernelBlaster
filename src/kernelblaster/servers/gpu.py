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
from .utils import safe_kill_process

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

    stdout, stderr = await exec_command("whoami")
    logger.info(f"GPU Server running as user: {stdout}\n{stderr}")

    stdout, stderr = await exec_command("groups")
    logger.info(f"User groups: {stdout}\n{stderr}")
    
    # Print nvidia-smi information before starting the server
    await print_nvidia_smi(logger)

    # Check for pre-existing GPU processes
    await check_gpu_processes()
    # Start worker tasks on startup (one per GPU id)
    for wid in range(len(GPU_IDS)):
        _ = asyncio.create_task(gpu_worker(wid))
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


async def check_gpu_processes():
    """Check for any pre-existing processes on NVIDIA GPUs.

    Filters out stale or non-existent PIDs and entries where the process name is
    reported as "[Not Found]" by nvidia-smi, to avoid false positives.
    """
    try:
        stdout, _ = await exec_command(
            "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader"
        )

        active_processes: list[str] = []
        for raw_line in stdout.split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(",")]
            if not parts:
                continue

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
            raise RuntimeError(
                f"Found pre-existing GPU processes:\n{json.dumps(active_processes, indent=2)}"
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
    """Execute a shell command"""
    # Prepare environment
    process_env = env.copy() if env else os.environ.copy()
    if env_vars:
        process_env.update(env_vars)

    # Use common temp directory as working directory
    working_dir = get_temp_dir()

    stdout_list = []
    stderr_list = []

    for _ in range(n_runs):
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=process_env,
            start_new_session=True,
            cwd=working_dir,
        )
        try:
            # Wait for the process with timeout
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            stdout_list.append(stdout.decode())
            stderr_list.append(stderr.decode())
            if proc.returncode != 0:
                raise GpuCommandError(
                    f"stdout:\n{stdout.decode()}\nstderr:\n{stderr.decode()}"
                )
        except asyncio.TimeoutError:
            # Kill the process if it times out
            logger.error(f"TIMEOUT: {cmd}")
            await safe_kill_process(proc, logger)
            raise GpuCommandError(
                f"Timeout: Execution timed out after {timeout} seconds"
            )

    if n_runs == 1:
        return stdout_list[0], stderr_list[0]
    else:
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


async def gpu_worker(worker_id: int) -> GpuCommandResult:
    """Process GPU execution requests from the queue"""
    while True:
        queue_item = await QUEUE.get()
        completion_future = queue_item[-1]  # Future is always the last item

        try:
            if len(queue_item) == 7:
                # Binary execution with prefix: (binary_path, args, env_vars, prefix_command, n_runs, timeout, completion_future)
                binary_path, args, env_vars, prefix_command, n_runs, timeout, _ = queue_item
            elif len(queue_item) == 6:
                # Backward compatibility: (binary_path, args, env_vars, prefix_command, n_runs, completion_future)
                binary_path, args, env_vars, prefix_command, n_runs, _ = queue_item
                timeout = 3600  # Default timeout
            
            # Common execution code for both 6-item and 7-item formats
            if len(queue_item) in (6, 7):
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
                    f"[Worker {worker_id}]: Assigned GPU CUDA_VISIBLE_DEVICES={gpu_visible} for binary {binary_path}"
                )
                logger.info(
                    f"[Worker {worker_id}]: Executing binary {binary_path} with args: {args}, env_vars: {eff_env_vars}, prefix: {prefix_command}, n_runs: {n_runs}, timeout: {timeout}"
                )

                stdout_list, stderr_list = await exec_binary(
                    binary_path,
                    args,
                    timeout=timeout,
                    env_vars=eff_env_vars,
                    prefix_command=prefix_command,
                    n_runs=n_runs,
                )
                logger.info(
                    f"[Worker {worker_id}]: Successfully executed binary on CUDA_VISIBLE_DEVICES={gpu_visible}: {f'{prefix_command} ' if prefix_command else ''}{binary_path} with {n_runs} runs"
                )

                # Clean up temporary binary file
                cleanup_temp_file(binary_path)

            elif len(queue_item) == 2:
                # CLI command execution: (command, completion_future)
                command, _ = queue_item
                logger.info(f"[Worker {worker_id}]: Executing CLI command: {command}")
                stdout_list, stderr_list = await exec_command(command)
                logger.info(
                    f"[Worker {worker_id}]: Successfully executed CLI command: {command}"
                )

            else:
                raise ValueError(f"Invalid queue item format: {len(queue_item)} items")

            completion_future.set_result(
                GpuCommandResult(success=True, stdout=stdout_list, stderr=stderr_list)
            )

        except GpuCommandError as e:
            if len(queue_item) == 6:
                binary_path = queue_item[0]
                # Best-effort GPU attribution for failures too
                gpu_visible = None
                try:
                    gpu_visible = (env_vars or {}).get("CUDA_VISIBLE_DEVICES")
                except Exception:
                    gpu_visible = None
                logger.error(
                    f"[Worker {worker_id}]: Error executing binary {binary_path}"
                    f"{' on CUDA_VISIBLE_DEVICES=' + str(gpu_visible) if gpu_visible is not None else ''}: {e.error_message}"
                )
                # Clean up on error too
                cleanup_temp_file(binary_path)
            else:
                command = queue_item[0]
                logger.error(
                    f"[Worker {worker_id}]: Error executing CLI command {command}: {e.error_message}"
                )

            completion_future.set_result(
                GpuCommandResult(success=False, message=e.error_message)
            )
        except Exception as e:
            logger.error(f"[Worker {worker_id}]: Unexpected error: {str(e)}")

            # Clean up binary file if this was a binary execution
            if len(queue_item) >= 4:
                cleanup_temp_file(queue_item[0])

            completion_future.set_result(
                GpuCommandResult(success=False, message=f"Internal error: {str(e)}")
            )
        finally:
            QUEUE.task_done()


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

        # Add to execution queue (7-item tuple format: binary_path, args, env_vars, prefix_command, n_runs, timeout, completion_future)
        await QUEUE.put(
            (
                binary_path,
                args,
                parsed_env_vars,
                prefix_command,
                n_runs,
                timeout,
                completion_future,
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


# Keep the old endpoint for backward compatibility (deprecated)
@APP.get("/gpu/cmd", response_model=GpuCommandResult)
async def process_command_request(command: str):
    """Execute a CLI command on the GPU server"""
    logger.info(f"/gpu/cmd - Command: {command}, Queue backlog: {QUEUE.qsize()}")

    # Create a future to track completion
    completion_future = asyncio.Future()

    # Add CLI command to the queue (2-item tuple format)
    await QUEUE.put((command, completion_future))

    try:
        await completion_future
        return completion_future.result()
    except asyncio.CancelledError:
        logger.info(f"Request {command} was cancelled")
        raise HTTPException(status_code=500, detail="Request was cancelled")


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
    parser.add_argument("--host", type=str, default="0.0.0.0")
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
