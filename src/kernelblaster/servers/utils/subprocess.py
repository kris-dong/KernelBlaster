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
"""Shared subprocess helpers for compile/profile/exec servers.

Centralizes timeout-aware shell execution so each backend server gets the same
diagnostics (cmd, cwd, stdout/stderr tails) without re-implementing the
asyncio + safe-kill dance. Callers pass their own exception class via
``error_factory`` so domain-specific exception types are preserved at API
boundaries.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable

from .process_management import safe_kill_process


class SubprocessRunError(Exception):
    """Default error raised by ``run_subprocess_shell`` when no factory is provided."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def tail_bytes(s: bytes, limit: int = 8192) -> str:
    """Decode and tail bytes for logging/error messages."""
    if not s:
        return ""
    try:
        txt = s.decode(errors="replace")
    except Exception:
        txt = repr(s)
    if len(txt) <= limit:
        return txt
    return txt[-limit:]


async def run_subprocess_shell(
    *,
    stage: str,
    cmd: str,
    cwd: Path,
    timeout_s: float,
    env: dict | None = None,
    error_factory: Callable[[str], Exception] = SubprocessRunError,
    logger: logging.Logger | None = None,
) -> tuple[bytes, bytes]:
    """Run a subprocess command and return (stdout, stderr) or raise ``error_factory(msg)``.

    Centralizes timeout handling so we always capture useful diagnostics:
    on timeout, the process group is killed via ``safe_kill_process``; on
    non-zero exit, stdout/stderr tails are included in the raised exception.
    """
    log = logger or logging.getLogger("uvicorn")
    log.info(f"[{stage}] starting (timeout={timeout_s}s) cwd={cwd} cmd={cmd}")
    start = asyncio.get_running_loop().time()
    proc = await asyncio.subprocess.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
        start_new_session=True,
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        await safe_kill_process(proc, log)
        elapsed = asyncio.get_running_loop().time() - start
        raise error_factory(
            f"[{stage}] Timeout after {timeout_s}s (elapsed={elapsed:.2f}s)\n"
            f"cmd: {cmd}\n"
            f"cwd: {cwd}\n"
            "stdout/stderr unavailable (process killed on timeout)\n"
        )
    elapsed = asyncio.get_running_loop().time() - start
    rc = proc.returncode
    log.info(f"[{stage}] finished rc={rc} elapsed={elapsed:.2f}s")
    if rc != 0:
        raise error_factory(
            f"[{stage}] Non-zero exit (rc={rc})\n"
            f"cmd: {cmd}\n"
            f"cwd: {cwd}\n"
            f"stdout_tail:\n{tail_bytes(stdout)}\n"
            f"stderr_tail:\n{tail_bytes(stderr)}\n"
        )
    return stdout, stderr
