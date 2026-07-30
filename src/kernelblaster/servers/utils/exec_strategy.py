# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Local vs remote binary-exec strategies for the unified exec server.

Companion to :mod:`servers.utils.compile_strategy`. Where compile
dispatches per-request on ``?backend=cuda|opencl``, exec dispatches
per-server based on startup config: with ``--board-host`` set the
server picks :class:`RemoteExecStrategy` (SSH scp + remote run), else
:class:`LocalExecStrategy` (local subprocess).

Strategy contract:

  ``async def exec(*, worker_id, binary_data, filename, args, env_vars,
                    prefix_command, n_runs, timeout, kernel_files,
                    profile) -> (stdout, stderr)``

The two response fields are always ``str`` for ``n_runs == 1`` and
``list[str]`` for ``n_runs > 1`` — matches the pre-refactor return
shapes of both ``exec_binary`` and ``upload_and_exec_binary``.

Fields not applicable to a given strategy (e.g. ``kernel_files`` for
local, ``env_vars`` for remote) are accepted but silently ignored —
same behaviour the pre-refactor endpoints already had.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class ExecStrategy(ABC):
    """One-shot binary-exec strategy (local subprocess or remote SSH)."""

    name: str = ""      # "local" | "remote"

    @abstractmethod
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
        kernel_files: Optional[list[str]] = None,
        profile: bool = False,
    ) -> tuple[list[str] | str, list[str] | str]:
        """Execute the binary N times and return ``(stdout, stderr)``.

        Return shape:
          - ``n_runs == 1``: ``(str, str)``.
          - ``n_runs > 1``:  ``(list[str], list[str])`` of length n_runs.
        """
