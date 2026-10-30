# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""SSH-remote binary-exec strategy (Adreno board target).

Wraps ``gpu_adreno.py::upload_and_exec_binary`` — scp binary + kernel
files to the board, chmod +x, run N times, always ``rm -rf`` the
remote dir in a finally clause. ``profile=True`` appends ``--profile``
to the args.

``env_vars`` and ``prefix_command`` are silently ignored — no way to
transport them through SSH cleanly (they aren't used by any known
caller either).
"""
from __future__ import annotations

import logging
from typing import Optional

from ..utils.exec_strategy import ExecStrategy


class RemoteExecStrategy(ExecStrategy):
    """SSH scp + remote exec (Adreno board)."""

    name = "remote"

    def __init__(self, *, board_host: str):
        """
        Args:
            board_host: SSH target for the Adreno board (e.g. ``root@10.44.120.201``).
        """
        self._board_host = board_host
        self._logger = logging.getLogger("uvicorn")

    async def exec(
        self,
        *,
        worker_id: int,
        binary_data: bytes,
        filename: str,
        args: str = "",
        env_vars: Optional[dict] = None,           # ignored
        prefix_command: Optional[str] = None,      # ignored
        n_runs: int = 1,
        timeout: float = 3600,
        kernel_files: Optional[list[str]] = None,
        profile: bool = False,
    ):
        from ..gpu_adreno import upload_and_exec_binary

        self._logger.info(
            f"[Worker {worker_id}]: remote exec on board={self._board_host} "
            f"file={filename} args={args!r} n_runs={n_runs} profile={profile}"
        )
        return await upload_and_exec_binary(
            binary_data=binary_data,
            filename=filename or "adreno_executable",
            board_host=self._board_host,
            kernel_files=kernel_files,
            args_str=args,
            timeout=timeout,
            n_runs=n_runs,
            profile=profile,
        )
