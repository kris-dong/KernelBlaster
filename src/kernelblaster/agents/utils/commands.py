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
import aiohttp
import asyncio
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import os
import json

from loguru import logger
from .error import FeedbackError
from ...config import config, GPUType
from ...resources import TCPClient

__all__ = [
    "run_gpu_executable",
    "compile_cu",
    "compile_and_run_cu_file",
    "compile_opencl",
    "run_adreno_executable",
    "compile_and_run_opencl",
    "BatchExecJob",
    "BatchExecJobResult",
    "run_gpu_batch",
    # RISC-V + Zephyr + spike/FPGA
    "compile_riscv",
    "run_riscv_executable",
    "compile_and_run_riscv",
]


async def _run_gpu_binary(
    binary_path,
    url,
    timeout,
    job_name,
    env_vars=None,
    prefix_command=None,
    n_runs=1,
    attempt=0,
):
    """Execute a binary via the new GPU binary upload endpoint."""
    try:
        # Read the binary file first to get size info
        with open(binary_path, "rb") as f:
            binary_data = f.read()
        
        binary_size = len(binary_data)
        binary_filename = os.path.basename(binary_path)
        
        # Check if this is an init.cu file for additional context
        is_init_cu = "init.cu" in str(job_name) if job_name else False
        init_cu_note = " (init.cu file)" if is_init_cu else ""
        
        logger.info(
            f"Uploading and executing binary to {url}/gpu/binary{init_cu_note} - "
            f"binary_path: {binary_path}, binary_filename: {binary_filename}, "
            f"binary_size: {binary_size} bytes, prefix_command: {prefix_command}, "
            f"n_runs: {n_runs}, job_name: {job_name}, attempt: {attempt}"
        )

        # Prepare the form data
        data = aiohttp.FormData()
        data.add_field(
            "binary",
            binary_data,
            filename=os.path.basename(binary_path),
            content_type="application/octet-stream",
        )
        data.add_field("n_runs", str(n_runs))
        data.add_field("timeout", str(timeout))
        if env_vars:
            data.add_field("env_vars", json.dumps(env_vars))
        if prefix_command:
            data.add_field("prefix_command", prefix_command)

        if url and "api.nvcf.nvidia.com" in url and os.getenv("NVCF_API_KEY"):
            headers = {"Authorization": f"Bearer {os.getenv('NVCF_API_KEY')}"}
        else:
            headers = {}
        # The server enforces the per-execution timeout (passed in the form
        # data). The HTTP client timeout must be larger to account for time
        # the request spends waiting in the server's execution queue.
        client_timeout = aiohttp.ClientTimeout(total=timeout + 3600)
        async with TCPClient.get_session().post(
            f"{url}/gpu/binary", data=data, timeout=client_timeout, headers=headers
        ) as response:
            if response.status != 200:
                response_text = await response.text()
                logger.warning(
                    f"GPU server returned status {response.status}, response: {response_text}"
                )
                raise FeedbackError(
                    f"GPU execution failed for {job_name}: {response_text}"
                )

            try:
                result = await response.json()
            except Exception as json_error:
                # If JSON parsing fails, try to get the text response
                response_text = await response.text()
                raise FeedbackError(
                    f"GPU server returned invalid JSON for {job_name}: {json_error}. Response: {response_text[:500]}"
                )
            
            success = result.get("success", False)
            if not success:
                error_message = result.get("message", result.get("detail", "Unknown error"))
                raise FeedbackError(
                    f"Execution failed for {job_name}: {error_message}"
                )
            return result.get("stdout", ""), result.get("stderr", "")
    except aiohttp.ClientError as e:
        error_msg = str(e).lower()
        # Get binary info for error logging
        try:
            binary_size = os.path.getsize(binary_path) if os.path.exists(binary_path) else "unknown"
            binary_filename = os.path.basename(binary_path)
        except:
            binary_size = "unknown"
            binary_filename = os.path.basename(binary_path) if binary_path else "unknown"
        
        # Retry transient connection errors (like "can not write request body")
        if ("can not write request body" in error_msg or "connection" in error_msg) and attempt == 0:
            # Check if this is an init.cu file for additional context
            is_init_cu = "init.cu" in str(job_name) if job_name else False
            init_cu_note = " (init.cu file)" if is_init_cu else ""
            
            logger.warning(
                f"Transient connection error for {job_name}{init_cu_note}: {e}. "
                f"Attempting to write binary_path: {binary_path}, binary_filename: {binary_filename}, "
                f"binary_size: {binary_size} bytes, url: {url}/gpu/binary, "
                f"prefix_command: {prefix_command}, n_runs: {n_runs}. Retrying once after 1s..."
            )
            await asyncio.sleep(1.0)
            # Retry the entire operation once
            return await _run_gpu_binary(binary_path, url, timeout, job_name, env_vars, prefix_command, n_runs, attempt=1)
        # Check if this is an init.cu file for additional context
        is_init_cu = "init.cu" in str(job_name) if job_name else False
        init_cu_note = " (init.cu file)" if is_init_cu else ""
        
        raise FeedbackError(
            f"Error connecting to GPU server for {job_name}{init_cu_note}: {e}. "
            f"binary_path: {binary_path}, binary_filename: {binary_filename}, "
            f"binary_size: {binary_size} bytes, url: {url}/gpu/binary, prefix_command: {prefix_command}"
        )
    except asyncio.TimeoutError as e:
        try:
            binary_size = os.path.getsize(binary_path) if os.path.exists(binary_path) else "unknown"
            binary_filename = os.path.basename(binary_path)
        except:
            binary_size = "unknown"
            binary_filename = os.path.basename(binary_path) if binary_path else "unknown"
        
        # Check if this is an init.cu file for additional context
        is_init_cu = "init.cu" in str(job_name) if job_name else False
        init_cu_note = " (init.cu file)" if is_init_cu else ""
        
        if attempt == 0:
            logger.warning(
                f"Timeout for {job_name}{init_cu_note}. binary_path: {binary_path}, binary_filename: {binary_filename}, "
                f"binary_size: {binary_size} bytes, url: {url}/gpu/binary, prefix_command: {prefix_command}, "
                f"n_runs: {n_runs}. Retrying once after 1s..."
            )
            await asyncio.sleep(1.0)
            return await _run_gpu_binary(binary_path, url, timeout, job_name, env_vars, prefix_command, n_runs, attempt=1)
        raise FeedbackError(
            f"Timeout: failed to receive a result for {job_name}{init_cu_note} after {timeout} seconds. "
            f"binary_path: {binary_path}, binary_filename: {binary_filename}, binary_size: {binary_size} bytes, "
            f"url: {url}/gpu/binary, prefix_command: {prefix_command}, n_runs: {n_runs}"
        )
    except IOError as e:
        logger.error(f"Error reading binary file {binary_path}: {e}")
        exit(1)


async def run_gpu_executable(
    executable_path: Path,
    gpu: GPUType,
    timeout: float,
    job_name: str,
    prefix_command: Optional[str] = None,
    n_runs: int = 1,
) -> tuple[list[str], list[str]]:
    url = config.get_gpu_server_url(gpu)
    return await _run_gpu_binary(
        executable_path,
        url,
        timeout,
        job_name,
        prefix_command=prefix_command,
        n_runs=n_runs,
    )


# ---------------------------------------------------------------------------
# Batched exec — POST /gpu/batch
#
# For strategies with ``supports_batching = True`` (e.g. FPGAExecStrategy)
# this amortises the fixed per-batch cost (bitstream flash, board reset)
# across ``len(jobs)`` binaries. Strategies without batching support just
# execute the jobs serially inside the server — the endpoint still works,
# so callers can opt into batching uniformly regardless of the active
# target and let the server handle it.
#
# Companion to :class:`agents.utils.batch_coordinator.BatchCoordinator`:
# most callers should never invoke ``run_gpu_batch`` directly and instead
# route single-request calls through a coordinator that flushes to this
# helper on size/time trigger. Direct callers are for cases where the
# workload is naturally batch-shaped (e.g. re-running a suite of N kernels
# against one bitstream).
# ---------------------------------------------------------------------------


@dataclass
class BatchExecJob:
    """One entry in a :func:`run_gpu_batch` call.

    Field set mirrors the per-job kwargs the server's ``/gpu/batch``
    endpoint parses (see :mod:`servers.exec_server`); the binary itself
    is a filesystem path so the coordinator doesn't hold megabytes of
    bytes in memory longer than necessary — we read + upload inside the
    HTTP call.
    """
    binary_path: Path
    args: str = ""
    env_vars: Optional[dict] = None
    prefix_command: Optional[str] = None
    n_runs: int = 1
    timeout: float = 3600
    kernel_files: Optional[list[str]] = None
    profile: bool = False
    job_name: str = ""     # informational — appears in logs, not sent


@dataclass
class BatchExecJobResult:
    """Result of one :class:`BatchExecJob`. ``success=False`` means the
    strategy raised for THIS specific job — the batch as a whole
    succeeded (otherwise :func:`run_gpu_batch` would raise instead of
    returning). Callers that treat any per-job failure as fatal should
    inspect ``success`` and raise :class:`FeedbackError` themselves —
    same pattern as the single-request path already uses."""
    stdout: str | list[str]
    stderr: str | list[str]
    success: bool
    message: Optional[str] = None


async def run_gpu_batch(
    jobs: list[BatchExecJob],
    gpu: GPUType,
    *,
    batch_name: str = "batch",
) -> list[BatchExecJobResult]:
    """POST ``jobs`` to the exec server's ``/gpu/batch`` endpoint.

    Returns per-job results in the same order as ``jobs``. Individual
    job failures are inline (``result.success = False``); this function
    only raises when the whole batch is rejected (transport error, HTTP
    non-200, response length mismatch, empty input).

    ``batch_name`` is used only for logging.
    """
    if not jobs:
        raise ValueError("run_gpu_batch: empty jobs list")

    url = config.get_gpu_server_url(gpu)
    max_timeout = max((j.timeout for j in jobs), default=3600)
    client_timeout = aiohttp.ClientTimeout(total=max_timeout + 3600)

    data = aiohttp.FormData()
    manifest: list[dict] = []
    for idx, job in enumerate(jobs):
        with open(job.binary_path, "rb") as f:
            binary_bytes = f.read()
        data.add_field(
            "binaries",
            binary_bytes,
            filename=os.path.basename(str(job.binary_path)),
            content_type="application/octet-stream",
        )
        manifest.append({
            "args": job.args,
            "env_vars": job.env_vars,
            "prefix_command": job.prefix_command,
            "n_runs": job.n_runs,
            "timeout": job.timeout,
            "kernel_files": job.kernel_files,
            "profile": job.profile,
        })
    data.add_field("manifest", json.dumps(manifest))

    logger.info(
        f"POST /gpu/batch [{batch_name}] - "
        f"count={len(jobs)} url={url} max_timeout={max_timeout}"
    )

    async with TCPClient.get_session().post(
        f"{url}/gpu/batch", data=data, timeout=client_timeout
    ) as response:
        if response.status != 200:
            body = await response.text()
            raise FeedbackError(
                f"batch exec [{batch_name}] failed: HTTP {response.status}: {body}"
            )
        payload = await response.json()

    raw_results = payload.get("results", [])
    if len(raw_results) != len(jobs):
        raise FeedbackError(
            f"batch exec [{batch_name}] length mismatch: "
            f"server returned {len(raw_results)} results for {len(jobs)} jobs"
        )

    return [
        BatchExecJobResult(
            stdout=r.get("stdout", ""),
            stderr=r.get("stderr", ""),
            success=bool(r.get("success", False)),
            message=r.get("message"),
        )
        for r in raw_results
    ]


async def _compile_cu(
    main_filepath: Path,
    cuda_filepath: Optional[Path],
    gpu: GPUType,
    url: str,
    timeout: float,
    job_name: str,
    persistent_artifacts: bool,
):
    try:
        # Ensure paths are absolute for compile server
        main_filepath_abs = main_filepath.resolve()
        cuda_filepath_abs = cuda_filepath.resolve() if cuda_filepath else None
        
        logger.info(f"Compile request - job_name: {job_name}")
        logger.info(f"  main_filepath (original): {main_filepath}")
        logger.info(f"  main_filepath (resolved): {main_filepath_abs}")
        if cuda_filepath:
            logger.info(f"  cuda_filepath (original): {cuda_filepath}")
            logger.info(f"  cuda_filepath (resolved): {cuda_filepath_abs}")
        logger.info(f"  persistent_artifacts: {persistent_artifacts}")
        logger.info(f"  sm_version: {gpu.sm}")
        
        logger.info(f"Submitted {job_name} to {url}/compile?backend=cuda")
        client_timeout = aiohttp.ClientTimeout(total=timeout + 3600)
        # Phase E: unified compile endpoint. ``backend_flag`` carries
        # ``persistent_artifacts`` for CUDA. Response shape:
        # ``UnifiedCompilationResult`` (``output_path`` + role-keyed ``extras``).
        async with TCPClient.get_session().get(
            f"{url}/compile",
            params={
                "backend": "cuda",
                "job_name": job_name,
                "main_file": str(main_filepath_abs),
                "source_file": str(cuda_filepath_abs) if cuda_filepath_abs else "",
                "backend_version": gpu.sm,
                "backend_flag": int(persistent_artifacts),
            },
            timeout=client_timeout,
        ) as response:
            if response.status != 200:
                response_text = await response.text()
                logger.warning(
                    f"Compilation server returned status {response.status}, response: {response_text}"
                )
                raise FeedbackError(
                    f"Failed to compile the file {job_name}: {response_text}"
                )

            result = await response.json()
            if not result["success"]:
                raise FeedbackError(
                    f"Failed to compile the file {job_name}: {result['message']}"
                )
            return result["output_path"]
    except aiohttp.ClientError as e:
        raise FeedbackError(f"Error connecting to compilation server: {e}")
    except asyncio.TimeoutError as e:
        raise FeedbackError(
            f"Timeout: failed to compile {job_name} after {timeout} seconds"
        )


async def compile_cu(
    main_filepath: Path,
    cuda_filepath: Optional[Path],
    gpu: GPUType,
    timeout: float = 120,
    job_name: str = "",
    persistent_artifacts: bool = False,
) -> str:
    """
    Compile a CUDA file via the compilation server.

    Args:
        job_name: Name of the job
        main_file: Path to the main .cu file
        cuda_file: Path to the CUDA .cuh file
        gpu: GPU type to compile the file on
        timeout: Timeout in seconds
        url: URL of the compilation server. If None, the default URL will be used.
        persistent_artifacts: If True, the compilation server will save the CUDA source artifacts in a unique directory, so that they're not overwritten by other threads compiling files in parallel.

    Returns:
        Path to the compiled binary
    """
    return await _compile_cu(
        main_filepath,
        cuda_filepath,
        gpu,
        config.COMPILE_SERVER_URL,
        timeout,
        job_name,
        persistent_artifacts,
    )


async def compile_and_run_cu_file(
    main_filepath: Path,
    cuda_filepath: Path,
    gpu: GPUType,
    timer,
    logger,
    persistent_artifacts=False,
    timeout=1200,
    num_runs=5,
    passed_keyword=None,
    prefix_command: Optional[str] = None,
) -> tuple[list[str], list[str], Path, bool]:
    """
    Compile and run a CUDA file multiple times using the new binary upload approach.

    Args:
        main_filepath: Path to the main CUDA file
        cuda_filepath: Path to the CUDA header file
        timer: Timer object
        logger: Logger object
        persistent_artifacts: If True, the compilation server will save the CUDA source artifacts in a unique directory, so that they're not overwritten by other threads compiling files in parallel.
        timeout: Timeout in seconds
        num_runs: Number of times to run the kernel
        passed_keyword: If provided, check if this keyword is in the stdout of each run. Stop running if not found.
        prefix_command: Command to prefix before the binary (e.g., 'ncu', 'nsys profile')

    Returns:
        A tuple containing:
        - A list of stdout strings from each run
        - A list of stderr strings from each run
        - The path to the compiled binary
        - Whether the kernel execution was successful
    """
    job_name = str(main_filepath) if cuda_filepath is None else str(cuda_filepath)
    timer.start("compilation")

    compiled_path = await compile_cu(
        main_filepath, cuda_filepath, gpu, timeout, job_name, persistent_artifacts
    )

    duration = timer.stop("compilation")
    logger.info(f"File compilation completed in {duration:0.2f} seconds")

    success = True

    timer.start("kernel_executions")

    if num_runs == 1:
        stdout, stderr = await run_gpu_executable(
            executable_path=Path(compiled_path),
            gpu=gpu,
            timeout=timeout,
            job_name=job_name,
            prefix_command=prefix_command,
            n_runs=num_runs,
        )
        stdout_list = [stdout]
        stderr_list = [stderr]
    else:
        stdout_list, stderr_list = await run_gpu_executable(
            executable_path=Path(compiled_path),
            gpu=gpu,
            timeout=timeout,
            job_name=job_name,
            prefix_command=prefix_command,
            n_runs=num_runs,
        )

    logger.info(
        f"Kernel execution of {num_runs} runs completed in {duration:0.2f} seconds"
    )

    duration = timer.stop("kernel_executions")

    # Stop early if passed_keyword is provided and not found in stdout
    for i, stdout in enumerate(stdout_list):
        if passed_keyword is not None and passed_keyword.lower() not in stdout.lower():
            logger.info(
                f"Keyword '{passed_keyword}' not found in run {i+1}, stopping further runs"
            )
            success = False
            break

    logger.info(
        f"{len(stdout_list)} kernel executions completed in {duration:0.2f} seconds. Success: {success}"
    )

    return stdout_list, stderr_list, compiled_path, success


# ---------------------------------------------------------------------------
# Qualcomm Adreno / OpenCL compilation and execution
# ---------------------------------------------------------------------------


async def compile_opencl(
    main_filepath: Path,
    kernel_filepath: Path,
    gpu: GPUType,
    timeout: float = 120,
    job_name: str = "",
    remote: bool = True,
) -> str:
    """Compile an OpenCL kernel + host driver via the OpenCL compilation server.

    When remote=True, the compile server SSH's to the Adreno board and produces
    an ARM64 binary there.
    """
    # Phase E: unified compile server. Legacy env var name preserved for
    # deployment continuity; the URL now points at the unified endpoint.
    opencl_compile_url = os.getenv(
        "KERNELBLASTER_OPENCL_COMPILE_SERVER_URL",
        config.get("OPENCL_COMPILE_SERVER_URL", "http://localhost:2003"),
    ) if hasattr(config, 'get') else os.getenv(
        "KERNELBLASTER_OPENCL_COMPILE_SERVER_URL", "http://localhost:2003"
    )

    try:
        main_filepath_abs = main_filepath.resolve()
        kernel_filepath_abs = kernel_filepath.resolve()

        logger.info(f"OpenCL compile request - job_name: {job_name}")
        logger.info(f"  main_filepath: {main_filepath_abs}")
        logger.info(f"  kernel_filepath: {kernel_filepath_abs}")
        logger.info(f"  opencl_version: {gpu.opencl_version}")
        logger.info(f"  remote: {remote}")

        # The HTTP client timeout must be larger than the per-execution timeout
        # passed in the params, to account for queue wait time on the server.
        client_timeout = aiohttp.ClientTimeout(total=timeout + 3600)
        async with TCPClient.get_session().get(
            f"{opencl_compile_url}/compile",
            params={
                "backend": "opencl",
                "job_name": job_name,
                "main_file": str(main_filepath_abs),
                "source_file": str(kernel_filepath_abs),
                "backend_version": gpu.opencl_version,
                "backend_flag": int(remote),
            },
            timeout=client_timeout,
        ) as response:
            if response.status != 200:
                response_text = await response.text()
                raise FeedbackError(
                    f"OpenCL compilation failed for {job_name}: {response_text}"
                )
            result = await response.json()
            if not result["success"]:
                raise FeedbackError(
                    f"OpenCL compilation failed for {job_name}: {result['message']}"
                )
            return result["output_path"]
    except aiohttp.ClientError as e:
        raise FeedbackError(f"Error connecting to OpenCL compilation server: {e}")
    except asyncio.TimeoutError:
        raise FeedbackError(
            f"Timeout: failed to compile OpenCL {job_name} after {timeout} seconds"
        )


async def run_adreno_executable(
    executable_path: Path,
    gpu: GPUType,
    timeout: float,
    job_name: str,
    kernel_files: list[str] = None,
    extra_files: list[str] = None,
    n_runs: int = 1,
    profile: bool = False,
    extra_args: str = "",
) -> tuple[list[str], list[str]]:
    """Execute a compiled binary on the Adreno GPU board via the Adreno GPU server."""
    adreno_gpu_url = os.getenv(
        "KERNELBLASTER_ADRENO_GPU_SERVER_URL", "http://localhost:2004"
    )

    try:
        with open(executable_path, "rb") as f:
            binary_data = f.read()

        logger.info(
            f"Adreno execution - job_name: {job_name}, binary_size: {len(binary_data)} bytes, "
            f"n_runs: {n_runs}, profile: {profile}, extra_args: {extra_args}"
        )

        data = aiohttp.FormData()
        data.add_field(
            "binary", binary_data,
            filename=os.path.basename(executable_path),
            content_type="application/octet-stream",
        )
        data.add_field("n_runs", str(n_runs))
        data.add_field("timeout", str(timeout))
        data.add_field("profile", str(profile).lower())
        if extra_args:
            data.add_field("args", extra_args)
        all_files = list(kernel_files or []) + list(extra_files or [])
        if all_files:
            data.add_field("kernel_files", json.dumps(all_files))

        # The server enforces the per-execution timeout (passed in the form
        # data).  The HTTP client timeout must be larger to account for time
        # the request spends waiting in the server's execution queue.
        client_timeout = aiohttp.ClientTimeout(total=timeout + 3600)
        async with TCPClient.get_session().post(
            f"{adreno_gpu_url}/gpu/binary", data=data, timeout=client_timeout
        ) as response:
            if response.status != 200:
                response_text = await response.text()
                raise FeedbackError(
                    f"Adreno GPU execution failed for {job_name}: {response_text}"
                )
            result = await response.json()
            if not result.get("success", False):
                error_message = result.get("message", "Unknown error")
                raise FeedbackError(
                    f"Adreno execution failed for {job_name}: {error_message}"
                )
            return result.get("stdout", ""), result.get("stderr", "")
    except aiohttp.ClientError as e:
        raise FeedbackError(f"Error connecting to Adreno GPU server: {e}")
    except asyncio.TimeoutError:
        raise FeedbackError(
            f"Timeout: Adreno execution for {job_name} after {timeout} seconds"
        )


async def compile_and_run_opencl(
    main_filepath: Path,
    kernel_filepath: Path,
    gpu: GPUType,
    timer,
    logger,
    timeout=1200,
    num_runs=5,
    passed_keyword=None,
    profile: bool = False,
    extra_files: list[str] = None,
    extra_args: str = "",
) -> tuple[list[str], list[str], str, bool]:
    """Compile and run an OpenCL kernel on Adreno; analogous to compile_and_run_cu_file."""
    job_name = str(kernel_filepath)
    timer.start("compilation")

    compiled_path = await compile_opencl(
        main_filepath, kernel_filepath, gpu, timeout, job_name, remote=True
    )

    duration = timer.stop("compilation")
    logger.info(f"OpenCL compilation completed in {duration:0.2f} seconds")

    success = True
    timer.start("kernel_executions")

    kernel_files = [str(kernel_filepath.resolve())]

    if num_runs == 1:
        stdout, stderr = await run_adreno_executable(
            executable_path=Path(compiled_path),
            gpu=gpu,
            timeout=timeout,
            job_name=job_name,
            kernel_files=kernel_files,
            extra_files=extra_files,
            n_runs=num_runs,
            profile=profile,
            extra_args=extra_args,
        )
        stdout_list = [stdout]
        stderr_list = [stderr]
    else:
        stdout_list, stderr_list = await run_adreno_executable(
            executable_path=Path(compiled_path),
            gpu=gpu,
            timeout=timeout,
            job_name=job_name,
            kernel_files=kernel_files,
            extra_files=extra_files,
            n_runs=num_runs,
            profile=profile,
            extra_args=extra_args,
        )

    duration = timer.stop("kernel_executions")

    for i, stdout in enumerate(stdout_list):
        if passed_keyword is not None and passed_keyword.lower() not in stdout.lower():
            logger.info(
                f"Keyword '{passed_keyword}' not found in run {i+1}, stopping further runs"
            )
            success = False
            break

    logger.info(
        f"{len(stdout_list)} OpenCL kernel executions completed in {duration:0.2f} seconds. Success: {success}"
    )

    return stdout_list, stderr_list, compiled_path, success


# ---------------------------------------------------------------------------
# RISC-V + Zephyr + spike/FPGA path
# ---------------------------------------------------------------------------


async def compile_riscv(
    main_filepath: Path,
    kernel_filepath: Path,
    gpu: GPUType,
    timeout: float = 600,
    job_name: str = "",
    *,
    link_as_lib: bool = False,
) -> str:
    """Compile a RISC-V C kernel via the unified compile server.

    Analogous to :func:`compile_opencl`. The compile server dispatches
    to :class:`ZephyrCompileStrategy` via ``?backend=riscv``. Returns
    the path to the produced Zephyr ELF (or static lib when
    ``link_as_lib=True`` for batched-exec fusing).

    ``gpu`` must be a RISC-V FPGA target (``GPUType.RISCV_FPGA_ZEPHYR``);
    its ``zephyr_board`` selects the Zephyr board (spike_riscv64,
    chipyard_riscv64/... for FireSim).
    """
    riscv_compile_url = os.getenv(
        "KERNELBLASTER_RISCV_COMPILE_SERVER_URL", "http://localhost:2001"
    )
    try:
        main_abs = main_filepath.resolve()
        kernel_abs = kernel_filepath.resolve()

        logger.info(f"RISC-V compile request - job_name: {job_name}")
        logger.info(f"  main: {main_abs}")
        logger.info(f"  kernel: {kernel_abs}")
        logger.info(f"  board: {gpu.zephyr_board}")
        logger.info(f"  link_as_lib: {link_as_lib}")

        client_timeout = aiohttp.ClientTimeout(total=timeout + 3600)
        async with TCPClient.get_session().get(
            f"{riscv_compile_url}/compile",
            params={
                "backend": "riscv",
                "job_name": job_name,
                "main_file": str(main_abs),
                "source_file": str(kernel_abs),
                "backend_version": gpu.zephyr_board,
                "backend_flag": int(link_as_lib),
            },
            timeout=client_timeout,
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise FeedbackError(
                    f"RISC-V compilation failed for {job_name}: HTTP "
                    f"{response.status}: {text}"
                )
            result = await response.json()
            if not result.get("success", False):
                raise FeedbackError(
                    f"RISC-V compilation failed for {job_name}: "
                    f"{result.get('message', 'unknown')}"
                )
            return result["output_path"]
    except aiohttp.ClientError as e:
        raise FeedbackError(f"Error connecting to RISC-V compile server: {e}")
    except asyncio.TimeoutError:
        raise FeedbackError(
            f"Timeout: RISC-V compilation for {job_name} exceeded {timeout}s"
        )


async def run_riscv_executable(
    executable_path: Path,
    gpu: GPUType,
    timeout: float,
    job_name: str,
    io_npz_path: Optional[Path] = None,
    n_runs: int = 1,
    args_str: str = "",
) -> tuple[list[str] | str, list[str] | str]:
    """Execute a Zephyr ELF via the RISC-V exec server (spike strategy).

    ``io_npz_path`` is the modelblaster golden — flows through
    ``kernel_files`` on the exec endpoint so SpikeExecStrategy can pass
    it as ``--io`` to spike_runner. Missing io = verify-only mode
    (spike_runner relies on the in-binary MODELBLASTER_VERIFY marker).
    ``args_str`` is a comma-list of spike args (e.g.
    ``"isa=rv64gcv,pmpregions=0"``) forwarded via ``--spike-arg=...``.

    Returns the raw spike output (with MODELBLASTER_WALL_CYCLES and
    per-op [PROFILE] markers) so :meth:`RiscvZephyrBackend.parse_profile`
    can pull cycles.
    """
    url = config.get_gpu_server_url(gpu)
    kernel_files = [str(io_npz_path.resolve())] if io_npz_path else None
    return await _run_gpu_binary(
        executable_path,
        url,
        timeout,
        job_name,
        prefix_command=None,
        n_runs=n_runs,
    ) if kernel_files is None else await _run_gpu_binary_with_kernel_files(
        executable_path=executable_path,
        url=url,
        timeout=timeout,
        job_name=job_name,
        kernel_files=kernel_files,
        args_str=args_str,
        n_runs=n_runs,
    )


async def _run_gpu_binary_with_kernel_files(
    *,
    executable_path: Path,
    url: str,
    timeout: float,
    job_name: str,
    kernel_files: list[str],
    args_str: str,
    n_runs: int,
) -> tuple[list[str] | str, list[str] | str]:
    """Variant of :func:`_run_gpu_binary` that carries ``kernel_files``
    (used by the RISC-V spike strategy to pass io.npz) and ``args``
    (comma-list of --spike-arg values). Split out because the CUDA/
    OpenCL path never uses these fields."""
    with open(executable_path, "rb") as f:
        binary_data = f.read()
    data = aiohttp.FormData()
    data.add_field(
        "binary",
        binary_data,
        filename=os.path.basename(str(executable_path)),
        content_type="application/octet-stream",
    )
    data.add_field("n_runs", str(n_runs))
    data.add_field("timeout", str(timeout))
    if args_str:
        data.add_field("args", args_str)
    data.add_field("kernel_files", json.dumps(kernel_files))
    client_timeout = aiohttp.ClientTimeout(total=timeout + 3600)
    async with TCPClient.get_session().post(
        f"{url}/gpu/binary", data=data, timeout=client_timeout,
    ) as response:
        if response.status != 200:
            body = await response.text()
            raise FeedbackError(
                f"RISC-V exec {job_name} failed: HTTP {response.status}: {body}"
            )
        result = await response.json()
        if not result.get("success", False):
            raise FeedbackError(
                f"RISC-V exec {job_name} failed: "
                f"{result.get('message', 'unknown')}"
            )
        return result.get("stdout", ""), result.get("stderr", "")


async def compile_and_run_riscv(
    main_filepath: Path,
    kernel_filepath: Path,
    gpu: GPUType,
    timer,
    logger,
    *,
    timeout: int = 3600,
    num_runs: int = 1,
    io_npz_path: Optional[Path] = None,
    spike_args_str: str = "",
    passed_keyword: Optional[str] = None,
) -> tuple[list[str], list[str], str, bool]:
    """Compile + run a RISC-V C kernel through the unified servers.

    Mirrors :func:`compile_and_run_opencl`: returns
    ``(stdout_list, stderr_list, compiled_elf_path, success)``.

    ``success`` follows the same convention as the OpenCL path — True
    unless ``passed_keyword`` is supplied and missing from a run's
    output.
    """
    job_name = str(kernel_filepath)
    timer.start("compilation")
    compiled_path = await compile_riscv(
        main_filepath, kernel_filepath, gpu, timeout, job_name,
    )
    logger.info(
        f"RISC-V compilation completed in {timer.stop('compilation'):.2f}s"
    )

    timer.start("kernel_executions")
    stdout, stderr = await run_riscv_executable(
        executable_path=Path(compiled_path),
        gpu=gpu,
        timeout=timeout,
        job_name=job_name,
        io_npz_path=io_npz_path,
        n_runs=num_runs,
        args_str=spike_args_str,
    )
    logger.info(
        f"RISC-V execution completed in {timer.stop('kernel_executions'):.2f}s"
    )

    stdout_list = stdout if isinstance(stdout, list) else [stdout]
    stderr_list = stderr if isinstance(stderr, list) else [stderr]

    success = True
    if passed_keyword is not None:
        for i, s in enumerate(stdout_list):
            if passed_keyword.lower() not in s.lower():
                logger.info(
                    f"Keyword '{passed_keyword}' missing in RISC-V run {i+1}"
                )
                success = False
                break

    return stdout_list, stderr_list, compiled_path, success


async def compile_and_run_riscv_batched(
    main_filepath: Path,
    kernel_filepath: Path,
    gpu: GPUType,
    timer,
    logger,
    batch_client,
    *,
    timeout: int = 3600,
    num_runs: int = 1,
    io_npz_path: Optional[Path] = None,
    spike_args_str: str = "",
    passed_keyword: Optional[str] = None,
) -> tuple[list[str], list[str], str, bool]:
    """Batched variant of :func:`compile_and_run_riscv`.

    Compilation stays per-item (there's nothing to amortise on the
    compile side — the compile server already runs N-way parallel via
    ``--num-workers``). Only the exec phase routes through
    ``batch_client``: N concurrent callers coalesce into one
    ``/gpu/batch`` HTTP call whose per-job cost is one boot / one flash
    amortised across ``N`` kernels.

    ``batch_client`` is an :class:`ExecBatchClient` (see
    :mod:`agents.utils.exec_batch_client`); pass ``None`` to disable
    batching (functionally identical to :func:`compile_and_run_riscv`).

    Returns the same ``(stdout_list, stderr_list, compiled_elf_path,
    success)`` 4-tuple as the single-item path so callers can swap
    seamlessly. Failure semantics also match: any per-job
    ``success=False`` re-raises as :class:`FeedbackError`, letting the
    RL fix-loop catch it unchanged.
    """
    if batch_client is None:
        return await compile_and_run_riscv(
            main_filepath, kernel_filepath, gpu, timer, logger,
            timeout=timeout,
            num_runs=num_runs,
            io_npz_path=io_npz_path,
            spike_args_str=spike_args_str,
            passed_keyword=passed_keyword,
        )

    job_name = str(kernel_filepath)
    timer.start("compilation")
    compiled_path = await compile_riscv(
        main_filepath, kernel_filepath, gpu, timeout, job_name,
    )
    logger.info(
        f"RISC-V compilation completed in {timer.stop('compilation'):.2f}s"
    )

    # ``kernel_id`` = the ELF stem. spike/firesim strategies key
    # per-kernel WALL_CYCLES markers off ``Path(job.filename).stem``,
    # and the compile server already generates UUID-based ELF paths
    # (compile_server.py:222), so this is unique across concurrent
    # rollouts by construction.
    kernel_id = Path(compiled_path).stem

    # Same-problem batching: derive base stage + model id from the
    # io.npz path (canonical layout ``<stage>/<mid>/<quant>/generated/
    # io.npz``). Passed through the batch client so the strategy's
    # fuse script can invoke harness_shared_input with the shared
    # MODEL_DIR. Skipped when io_npz_path is absent — degrades
    # gracefully to per-ELF batching.
    base_stage_dir: Optional[Path] = None
    mid: Optional[str] = None
    target: str = "rvv"
    if io_npz_path is not None:
        _io_abs = io_npz_path.resolve()
        # io.npz lives at <stage>/generated/io.npz, so:
        #   parent = <stage>/generated
        #   parent.parent = <stage> = kb_<name>/<quant>
        #   parent.parent.parent.name = kb_<name>
        base_stage_dir = _io_abs.parent.parent  # <stage>/generated/rvv
        # Actually we want the per-target generated dir. Try to infer
        # target from the kernel source path (KernelBlaster writes
        # RL candidates alongside `<stage>/generated/<target>/`).
        try:
            # kernel_filepath is the LLM's out .c, not under stage.
            # target comes from GPU.zephyr_board via the compile server
            # dispatch — fall back to 'rvv' which is the default target
            # for the current RISC-V flows.
            target = gpu.zephyr_board.split("/")[0].replace(
                "_riscv64", ""
            ).replace("spike_", "").replace("chipyard_", "")
            # Normalise: "" / unexpected → 'rvv'
            if not target or target in ("chipyard", "spike"):
                target = "rvv"
        except Exception:
            target = "rvv"
        base_stage_dir = _io_abs.parent / target  # <stage>/generated/<target>
        # mid = the kb_<name> segment two levels up from <stage>/generated
        try:
            mid = _io_abs.parent.parent.parent.name  # kb_<name>
        except Exception:
            mid = None

    board: Optional[str] = None
    try:
        board = gpu.zephyr_board
    except Exception:
        board = None

    timer.start("kernel_executions")
    result = await batch_client.submit_riscv(
        binary_path=Path(compiled_path),
        io_npz_path=io_npz_path,
        timeout=timeout,
        spike_args_str=spike_args_str,
        kernel_id=kernel_id,
        n_runs=num_runs,
        source_c_path=kernel_filepath,
        base_stage_dir=base_stage_dir,
        mid=mid,
        target=target,
        board=board,
    )
    logger.info(
        f"RISC-V batched execution completed in "
        f"{timer.stop('kernel_executions'):.2f}s (job={kernel_id})"
    )

    if not result.success:
        raise FeedbackError(
            f"RISC-V batched exec {job_name} failed: "
            f"{result.message or 'unknown'}"
        )

    stdout_list = (
        result.stdout if isinstance(result.stdout, list) else [result.stdout]
    )
    stderr_list = (
        result.stderr if isinstance(result.stderr, list) else [result.stderr]
    )

    success = True
    if passed_keyword is not None:
        for i, s in enumerate(stdout_list):
            if passed_keyword.lower() not in s.lower():
                logger.info(
                    f"Keyword '{passed_keyword}' missing in RISC-V run {i+1}"
                )
                success = False
                break

    return stdout_list, stderr_list, compiled_path, success
