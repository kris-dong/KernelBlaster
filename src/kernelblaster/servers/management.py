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
import requests
import subprocess
from pathlib import Path
try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)
from torch.utils import cmake_prefix_path
from io import TextIOWrapper
import time
import psutil
import socket
from ..config import GPUType, config
from typing import Optional


def test_server_connection(process, url, timeout: int = 5):
    """Test connection to a server and return whether it's available."""

    poll_interval = 0.25
    start_time = time.time()
    warned_exited = False
    while time.time() - start_time < timeout:
        # NOTE: We intentionally do NOT fail immediately if the launching process
        # appears to have exited. In practice this can happen with stale/zombie
        # processes or wrapper launchers, while the server endpoint is still
        # reachable (e.g., an already-running server on that port).
        if process is not None and process.poll() is not None and not warned_exited:
            warned_exited = True
            try:
                rc = process.poll()
            except Exception:
                rc = None
            logger.warning(
                f"⚠️ Server process for {url} appears to have exited (returncode={rc}); "
                f"continuing to probe {url}/health for up to {timeout}s."
            )
            # Stop checking the process object to avoid repeated warnings.
            process = None

        try:
            # Just try to connect to the server without making a real request
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code < 500:
                # Root path might not exist, but server is up.
                # Or, any non-server error means server is up
                logger.info(f"✅ Server at {url} is running")
                return True
            else:
                logger.error(
                    f"❌ Server at {url} returned error: {response.status_code}"
                )
                return False
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            logger.error(f"❌ Error connecting to {url}: {e}")
            return False
        time.sleep(poll_interval)

    logger.error(f"❌ Server at {url} failed to start after {timeout} seconds")
    return False


def initialize_compiler_server(
    log_file: TextIOWrapper,
    compile_server_url: str | None,
    artifacts_dir: Path,
    port: int | None,
):
    """Initialize the servers and return the URLs."""

    if compile_server_url is not None:
        logger.info(f"Using existing compile server at {compile_server_url}")
        return None, compile_server_url

    if port is None:
        port = find_free_port(start_port=2001)
        logger.info(f"🎯 Auto-assigned compiler server port: {port}")

    # Start the compile server
    compiler_server_process = None
    # Check that libtorch exists
    if not Path(cmake_prefix_path).exists():
        logger.error(
            f"Libtorch CMake prefix path {cmake_prefix_path} does not exist! Please install pytorch."
        )
        return False

    compiler_server_cmd = [
        "python",
        "-m",
        "src.kernelblaster.servers.compile",
        "--port",
        str(port),
        "--num-workers",
        str(psutil.cpu_count(logical=False) - 1),  # physical CPU cores
        "--artifacts-dir",
        str(artifacts_dir),
    ]

    # Use a single file for both stdout and stderr
    compiler_server_process = subprocess.Popen(
        compiler_server_cmd,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )

    compile_server_url = f"http://localhost:{port}"
    logger.info(
        f"Starting the compilation server at {compile_server_url}: {' '.join(compiler_server_cmd)}"
    )
    return compiler_server_process, compile_server_url


def initialize_gpu_server(
    log_file: TextIOWrapper,
    gpu: Optional[GPUType],
    port: int | None,
):
    """Initialize the GPU server and return the URL."""

    if gpu is None:
        gpu = GPUType.current()

    gpu_server_url = config.get_gpu_server_url(gpu)

    if gpu_server_url is not None:
        logger.info(f"Using existing GPU server at {gpu_server_url} for {gpu}")
        return None, gpu_server_url

    if port is None:
        port = find_free_port(start_port=2002)
        logger.info(f"🎯 Auto-assigned GPU server port: {port}")

    gpu_server_cmd = [
        "python",
        "-m",
        "src.kernelblaster.servers.gpu",
        "--port",
        str(port),
    ]

    # Use a single file for both stdout and stderr
    gpu_server_process = subprocess.Popen(
        gpu_server_cmd,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )

    gpu_server_url = f"http://localhost:{port}"
    logger.info(
        f"Starting the GPU command server at {gpu_server_url}: {' '.join(gpu_server_cmd)}"
    )
    config.set_gpu_server_url(gpu, gpu_server_url)
    return gpu_server_process, gpu_server_url


def find_free_port(start_port: int = 2001) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find free port starting from {start_port}")


def start_standalone_gpu_server(port: int = None, log_file_path: str = None) -> tuple[subprocess.Popen, str]:
    """Start a standalone GPU server and return the process and URL."""
    
    if port is None:
        port = find_free_port(start_port=2002)
        
    gpu_server_cmd = [
        "python",
        "-m",
        "src.kernelblaster.servers.gpu",
        "--port",
        str(port),
    ]
    
    # Set up logging
    if log_file_path:
        # Ensure log directory exists
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Pass log file path to GPU server so uvicorn can configure file logging
        gpu_server_cmd.extend(["--log_path", str(log_file_path)])
        # Also redirect stdout/stderr as backup with line buffering
        log_file = open(log_file_path, 'a', buffering=1)  # Line buffering
        stdout_file = log_file
        stderr_file = log_file
    else:
        stdout_file = subprocess.PIPE
        stderr_file = subprocess.PIPE
    
    gpu_server_process = subprocess.Popen(
        gpu_server_cmd,
        stdout=stdout_file,
        stderr=stderr_file,
        start_new_session=True,
    )
    
    gpu_server_url = f"http://localhost:{port}"
    logger.info(f"Starting standalone GPU server at {gpu_server_url}: {' '.join(gpu_server_cmd)}")
    
    # Test the connection
    if not test_server_connection(gpu_server_process, gpu_server_url, timeout=10):
        logger.error(f"Failed to start GPU server at {gpu_server_url}")
        gpu_server_process.terminate()
        raise RuntimeError(f"GPU server failed to start at {gpu_server_url}")
    
    return gpu_server_process, gpu_server_url
