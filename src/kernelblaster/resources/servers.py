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
from pathlib import Path
from typing import Optional
from ..servers.management import (
    initialize_compiler_server,
    initialize_gpu_server,
    test_server_connection,
)
from ..config import config
from ..config import GPUType


class ManagedServer:
    def __init__(self, logger, log_path: Path):
        self.logger = logger
        self.log_path = log_path
        self.log_file_handle = open(log_path, "w")
        self.process = None
        self.url = None

    def __del__(self):
        self.cleanup()

    @property
    def is_managed(self):
        return self.process is not None

    def cleanup(self):
        if self.is_managed:
            self.process.terminate()
            self.process.wait()
            self.process = None
        if self.log_file_handle:
            self.log_file_handle.close()

    def _log_error_output(self):
        """Log error output from the server process"""
        try:
            log_content = self.log_path.read_text()
            self.logger.error(f"Server Logs:\n{log_content}")
        except Exception as e:
            self.logger.error(f"Failed to read log file: {e}")

    def wait_for_connection(self, timeout: int = 5):
        """Wait for the server to start"""
        test_result = test_server_connection(self.process, self.url, timeout)
        if not test_result:
            self._log_error_output()
            raise RuntimeError(f"Failed to initialize server at {self.url}")


class CompileServer(ManagedServer):
    def __init__(
        self,
        logger,
        experiment_dir: Path,
        artifacts_dir: str = config.TEMP_DIRECTORY,
        port: int = None,
    ):
        """
        Create a new compile server.
        If port is not None, the server will be initialized with a new port.
        """
        super().__init__(logger, experiment_dir / "compile_server.log")
        self.artifacts_dir = artifacts_dir
        self.__initialize(port)

    def __initialize(self, port: int = None):
        self.process, self.url = initialize_compiler_server(
            self.log_file_handle,
            config.COMPILE_SERVER_URL,
            Path(self.artifacts_dir),
            port,
        )


class GPUServer(ManagedServer):
    def __init__(
        self,
        logger,
        experiment_dir: Path,
        gpu: Optional[GPUType] = None,
        port: int = None,
    ):
        super().__init__(logger, experiment_dir / "gpu_server.log")
        self.__initialize(gpu, port)

    def __initialize(self, gpu: Optional[GPUType], port: int = None):
        self.logger.info(
            f"Initializing GPU server for {gpu if gpu else 'current GPU'}..."
        )
        self.process, self.url = initialize_gpu_server(
            self.log_file_handle,
            gpu,
            port,
        )
