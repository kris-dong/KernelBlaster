# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""OpenCL compile strategy — thin wrapper over
:func:`compile_opencl.exec_remote_compilation` and
:func:`compile_opencl.exec_local_compilation`.

Selects remote (SSH+scp) vs local (x86 gcc) based on ``backend_flag``
+ presence of ``board_host``. Preserves the historical fallback:
without a board host, remote requests fall through to local x86
compile.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..utils.compile_strategy import CompileStrategy


class OpenCLCompileStrategy(CompileStrategy):
    """SSH+scp remote compile (or local x86 fallback) for OpenCL."""

    name = "opencl"
    kernel_ext = ".cl"

    async def compile(
        self,
        *,
        worker_id: int,
        job_name: str,
        main_file: str,
        source_file: str,
        backend_version: str,          # opencl_version, e.g. "opencl_2.0"
        backend_flag: bool,            # remote
        output_path: Path,
        artifacts_dir: Path,           # unused for OpenCL — placeholder for uniform ABI
        board_host: Optional[str] = None,
        debug: bool = False,
    ) -> Optional[str]:
        from ..compile_opencl import exec_local_compilation, exec_remote_compilation

        logger = logging.getLogger("uvicorn")

        if backend_flag and board_host:
            remote_binary = await exec_remote_compilation(
                job_name,
                main_file,
                source_file,
                backend_version,
                board_host,
                output_path,
            )
            output_path.chmod(0o755)
            logger.info(
                f"[Worker {worker_id}]: Remote compilation success: "
                f"{job_name} -> {remote_binary}"
            )
            return remote_binary

        tmp_path = await exec_local_compilation(
            job_name,
            main_file,
            source_file,
            backend_version,
            worker_id,
            output_path,
        )
        output_path.write_bytes(tmp_path.read_bytes())
        output_path.chmod(0o755)
        logger.info(
            f"[Worker {worker_id}]: Local compilation success: {job_name} -> {output_path}"
        )
        return str(output_path)
