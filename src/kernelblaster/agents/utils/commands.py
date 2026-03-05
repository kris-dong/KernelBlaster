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
from typing import Optional
from pathlib import Path
import os
import json

from loguru import logger
from .error import FeedbackError
from ...config import config, GPUType
from ...resources import TCPClient

__all__ = ["run_gpu_executable", "compile_cu", "compile_and_run_cu_file"]


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
        async with TCPClient.get_session().post(
            f"{url}/gpu/binary", data=data, timeout=timeout, headers=headers
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
        
        logger.info(f"Submitted {job_name} to {url}/compile")
        async with TCPClient.get_session().get(
            f"{url}/compile",
            params={
                "job_name": job_name,
                "main_file": str(main_filepath_abs),
                "cuda_file": str(cuda_filepath_abs) if cuda_filepath_abs else "",
                "persistent_artifacts": int(persistent_artifacts),
                "sm_version": gpu.sm,
            },
            timeout=timeout,
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
