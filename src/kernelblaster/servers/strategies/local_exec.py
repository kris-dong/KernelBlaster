# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Local subprocess exec strategy.

Wraps the pre-refactor ``gpu.py::exec_binary`` semantics: write binary
bytes to a temp file, ``chmod +x``, run via subprocess with the
per-worker ``CUDA_VISIBLE_DEVICES`` pinning + caller-supplied
``env_vars`` and ``prefix_command`` (e.g. ``ncu`` / ``nsys profile``).

``kernel_files`` and ``profile`` params from the unified endpoint are
silently ignored — Adreno-specific.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from ..utils.exec_strategy import ExecStrategy


class LocalExecStrategy(ExecStrategy):
    """Run binary in a local subprocess (NVIDIA GPU or CPU host)."""

    name = "local"

    def __init__(self, *, gpu_ids: list[str]):
        """
        Args:
            gpu_ids: List of GPU device indices this server is assigned to.
                Worker ``i`` gets pinned to ``CUDA_VISIBLE_DEVICES=gpu_ids[i]``
                unless the caller passes CUDA_VISIBLE_DEVICES in env_vars.
        """
        self._gpu_ids = gpu_ids
        self._logger = logging.getLogger("uvicorn")

    async def exec(
        self,
        *,
        worker_id: int,
        binary_data: bytes,
        filename: str,
        args: str = "",
        env_vars: Optional[dict] = None,
        prefix_command: Optional[str] = None,
        n_runs: int = 1,
        timeout: float = 3600,
        kernel_files: Optional[list[str]] = None,  # ignored
        profile: bool = False,                     # ignored
    ):
        # Lazy import — avoid pulling the FastAPI ``APP`` object at
        # strategies-package import time.
        from ..gpu import save_binary_to_temp, cleanup_temp_file, exec_binary

        # Per-worker CUDA pinning. Caller override wins if set.
        eff_env_vars = dict(env_vars or {})
        if "CUDA_VISIBLE_DEVICES" not in eff_env_vars:
            gpu_id = str(worker_id)
            if self._gpu_ids and worker_id < len(self._gpu_ids):
                gpu_id = str(self._gpu_ids[worker_id])
            eff_env_vars["CUDA_VISIBLE_DEVICES"] = gpu_id
        eff_env_vars.setdefault("NVIDIA_TF32_OVERRIDE", "0")

        binary_path = save_binary_to_temp(binary_data, filename or "gpu_executable")
        try:
            self._logger.info(
                f"[Worker {worker_id}]: local exec binary={binary_path} "
                f"args={args!r} prefix={prefix_command!r} n_runs={n_runs} "
                f"CUDA_VISIBLE_DEVICES={eff_env_vars['CUDA_VISIBLE_DEVICES']}"
            )
            stdout, stderr = await exec_binary(
                binary_path,
                args,
                timeout=timeout,
                env_vars=eff_env_vars,
                prefix_command=prefix_command,
                n_runs=n_runs,
            )
            return stdout, stderr
        finally:
            cleanup_temp_file(binary_path)
