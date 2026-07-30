# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""CUDA compile strategy — thin wrapper over :func:`compile.exec_compilation`.

The heavy lifting (CMake + make invocation, artifact copy, persistent-
artifacts directory management) still lives in :mod:`servers.compile`.
This strategy just adapts the ``compile()`` call shape to the historical
positional args of ``exec_compilation``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..utils.compile_strategy import CompileStrategy


class CUDACompileStrategy(CompileStrategy):
    """CMake + make (via ``exec_compilation``) for CUDA driver + kernel."""

    name = "cuda"
    kernel_ext = ".cu"

    async def compile(
        self,
        *,
        worker_id: int,
        job_name: str,
        main_file: str,
        source_file: str,
        backend_version: str,          # sm_version, e.g. "sm_89"
        backend_flag: bool,            # persistent_artifacts
        output_path: Path,
        artifacts_dir: Path,           # unused for CUDA — placeholder for uniform ABI
        board_host: Optional[str] = None,   # unused for CUDA
        debug: bool = False,
    ) -> Optional[str]:
        # Lazy import — avoids pulling the FastAPI ``APP`` object at
        # strategies-package import time.
        from ..compile import exec_compilation

        logger = logging.getLogger("uvicorn")
        logger.info(f"[Worker {worker_id}]: Compiling {job_name}")
        if backend_flag:
            logger.info(
                f"[Worker {worker_id}]: Using persistent_artifacts mode for {job_name}"
            )
        try:
            main_sz = Path(main_file).stat().st_size
        except Exception:
            main_sz = -1
        try:
            cuda_sz = Path(source_file).stat().st_size if source_file else -1
        except Exception:
            cuda_sz = -1
        logger.info(
            f"[Worker {worker_id}]: input file sizes bytes "
            f"main={main_sz} cuda={cuda_sz} sm={backend_version}"
        )

        tmp_path = await exec_compilation(
            job_name,
            main_file,
            source_file,
            backend_version,
            worker_id,
            output_path,
            backend_flag,
            debug=debug,
        )
        output_path.write_bytes(tmp_path.read_bytes())
        output_path.chmod(0o755)
        logger.info(
            f"[Worker {worker_id}]: Successfully compiled {job_name} and saved to {output_path}"
        )
        # CUDA has no separate ``remote_binary_path`` return field —
        # ``persistent_artifacts_dir`` is populated by ``exec_compilation``
        # itself when applicable, but the current handler doesn't
        # surface it. Preserve that behaviour.
        return None
