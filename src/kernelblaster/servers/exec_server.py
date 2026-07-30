# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Unified exec server (Adreno + L40S).

Single FastAPI app exposing ``POST /gpu/binary``. The strategy is
picked ONCE at startup based on ``--board-host``:

  - ``--board-host root@10.44.120.201`` → :class:`RemoteExecStrategy`
    (SSH scp + remote run against the Adreno board).
  - No ``--board-host`` → :class:`LocalExecStrategy` (local
    subprocess with per-worker CUDA_VISIBLE_DEVICES pinning).

Endpoint form fields are the UNION of the two legacy schemas — fields
not applicable to the active strategy are silently ignored (matches
the pre-Phase-A behaviour of both endpoints):

  binary          UploadFile   both strategies
  args            str          both
  n_runs          int          both
  timeout         float        both
  env_vars        JSON dict    local only (remote ignores)
  prefix_command  str          local only (remote ignores)
  kernel_files    JSON list    remote only (local ignores)
  profile         bool         remote only (local ignores)

Response: :class:`ExecResult` — same shape as the legacy
``GpuCommandResult`` / ``AdrenoExecutionResult``.

Runs alongside the legacy ``gpu.py`` / ``gpu_adreno.py`` in Phase B —
Phase C migrates callers + retires them.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import psutil
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .server_logging import get_log_config
from .strategies import ExecStrategy, LocalExecStrategy, RemoteExecStrategy
from .utils.queue_server import worker_pool


logger = logging.getLogger("uvicorn")
QUEUE: asyncio.Queue = asyncio.Queue()

# Populated in ``main`` before ``lifespan`` fires.
_STRATEGY: Optional[ExecStrategy] = None
_GPU_IDS: Optional[list[str]] = None
_BOARD_HOST: Optional[str] = None
args: argparse.Namespace = argparse.Namespace()


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

class ExecResult(BaseModel):
    """Unified exec response.

    Matches the pre-refactor ``GpuCommandResult`` /
    ``AdrenoExecutionResult`` shape byte-for-byte so migrated clients
    don't need to change their JSON destructuring.
    """
    stdout: str | list[str] = []
    stderr: str | list[str] = []
    success: bool = False
    message: Optional[str] = None


class ExecError(Exception):
    """Umbrella domain error for the exec server. Both underlying
    ``GpuCommandError`` (local) and ``AdrenoExecutionError`` (remote)
    are caught by the handler and converted into
    ``ExecResult(success=False, ...)`` — this class is just the
    ``worker_pool`` ``domain_error`` sentinel."""


# ---------------------------------------------------------------------------
# Startup checks (local-only)
# ---------------------------------------------------------------------------

async def _resolve_assigned_gpu_uuids() -> set[str] | None:
    """Map ``_GPU_IDS`` (indices) → GPU UUIDs via ``nvidia-smi``.

    Returns ``None`` if the mapping can't be built (nvidia-smi
    missing, parse fails) — caller falls back to the global check.
    """
    if not _GPU_IDS:
        return None
    from .gpu import exec_command  # reuse the subprocess wrapper
    try:
        stdout, _ = await exec_command(
            "nvidia-smi --query-gpu=index,uuid --format=csv,noheader"
        )
    except Exception:
        return None
    index_to_uuid: dict[str, str] = {}
    for raw_line in stdout.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        index_to_uuid[parts[0]] = parts[1]
    assigned = {index_to_uuid[i] for i in _GPU_IDS if i in index_to_uuid}
    return assigned or None


async def _check_gpu_processes():
    """Local-only pre-flight: refuse to start if pre-existing compute
    apps are running on our assigned GPUs. Scoped by GPU UUID when
    ``KERNELBLASTER_GPU_SERVER_GPU_IDS`` is set. Matches ``gpu.py``'s
    check exactly — we just call through to it here."""
    from .gpu import check_gpu_processes as _local_check
    await _local_check()


async def _remote_preflight():
    """Remote-only pre-flight: SSH connectivity probe + /tmp cleanup.

    Mirrors ``gpu_adreno.py::lifespan`` — verifies the board reachable
    and wipes any orphaned run dirs / refgen caches from a prior
    process that died without cleanup.
    """
    from .gpu_adreno import exec_remote_command, _REMOTE_WORK_DIR

    try:
        stdout, _ = await exec_remote_command(
            "echo ok && ls /dev/kgsl-3d0", _BOARD_HOST, timeout=15
        )
        logger.info(f"Board connectivity verified: {_BOARD_HOST}")
    except Exception as e:
        logger.warning(f"Board connectivity check failed: {e} (server will start anyway)")

    # /tmp cleanup — a fresh server start implies any prior run is
    # dead, so its remote_dir is guaranteed orphan.
    try:
        await exec_remote_command(
            f"rm -rf {_REMOTE_WORK_DIR} && mkdir -p {_REMOTE_WORK_DIR}",
            _BOARD_HOST, timeout=30,
        )
        await exec_remote_command(
            "rm -rf /tmp/kernelblaster_refgen_*", _BOARD_HOST, timeout=15,
        )
        df_out, _ = await exec_remote_command(
            "df -h /tmp | tail -1", _BOARD_HOST, timeout=10,
        )
        logger.info(f"Pre-flight cleanup of {_BOARD_HOST}:/tmp done: {df_out.strip()}")
    except Exception as e:
        logger.warning(f"Pre-flight cleanup failed (continuing anyway): {e}")


# ---------------------------------------------------------------------------
# Worker handler
# ---------------------------------------------------------------------------

async def _exec_job(worker_id: int, job_args: tuple) -> ExecResult:
    """Handler for :func:`worker_pool` — routes to the active strategy.

    ``job_args`` shape: single-element tuple carrying a dict with the
    per-request kwargs the strategies accept.
    """
    (payload,) = job_args
    assert _STRATEGY is not None, "exec strategy not initialised"

    try:
        stdout, stderr = await _STRATEGY.exec(worker_id=worker_id, **payload)
        return ExecResult(success=True, stdout=stdout, stderr=stderr)
    except Exception as e:
        # Both GpuCommandError (local) and AdrenoExecutionError
        # (remote) carry ``error_message``; other exceptions get their
        # str repr. Convert to inline success=False so the HTTP client
        # sees a 200-OK JSON body (matches pre-refactor contract).
        message = getattr(e, "error_message", None) or str(e)
        logger.error(f"[Worker {worker_id}]: exec failed: {message}")
        return ExecResult(success=False, message=message)


# ---------------------------------------------------------------------------
# FastAPI wiring
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app):
    global _STRATEGY

    if _BOARD_HOST:
        _STRATEGY = RemoteExecStrategy(board_host=_BOARD_HOST)
        logger.info(
            f"Started unified exec server on {args.host}:{args.port} "
            f"with {args.num_workers} workers, strategy=remote, board={_BOARD_HOST}"
        )
        await _remote_preflight()
        num_workers = args.num_workers
    else:
        assert _GPU_IDS is not None, "GPU_IDS not initialised for local strategy"
        _STRATEGY = LocalExecStrategy(gpu_ids=_GPU_IDS)
        logger.info(
            f"Started unified exec server on {args.host}:{args.port} "
            f"strategy=local, GPU_IDS={_GPU_IDS}"
        )
        # Pin LD_LIBRARY_PATH to venv libtorch (fix from commit 1e6b40c).
        # Only relevant for local strategy — remote binaries run on the
        # board's own runtime and don't care about our host paths.
        _pin_libtorch_ld_path()
        # Diagnostic-only: whoami / groups. Non-fatal on container
        # environments with incomplete /etc/group.
        from .gpu import exec_command as _local_exec_command
        for diag in ("whoami", "groups"):
            try:
                stdout, stderr = await _local_exec_command(diag)
                logger.info(f"Exec server startup {diag}: {stdout}\n{stderr}")
            except Exception as diag_err:
                logger.warning(f"Diagnostic `{diag}` failed (non-fatal): {diag_err}")
        # nvidia-smi + GPU pre-flight
        from .gpu import print_nvidia_smi as _print_nvidia_smi
        await _print_nvidia_smi(logger)
        await _check_gpu_processes()
        num_workers = len(_GPU_IDS)

    async with worker_pool(
        num_workers=num_workers,
        queue=QUEUE,
        handler=_exec_job,
        domain_error=ExecError,
        logger=logger,
    ):
        yield


def _pin_libtorch_ld_path():
    """Prepend the venv's ``torch/lib`` to ``LD_LIBRARY_PATH`` in the
    subprocess env used by the local strategy. Idempotent — see
    ``gpu.py::lifespan`` for the same code path + rationale (commit
    1e6b40c). Without this, CUDA drivers compiled against the venv's
    libtorch crash at load if the system torch is on LD_LIBRARY_PATH
    first.
    """
    from . import gpu as _gpu_mod

    env = _gpu_mod.env if _gpu_mod.env else os.environ.copy()
    env.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    try:
        import torch as _torch
        torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            existing = env.get("LD_LIBRARY_PATH", "")
            if not existing.startswith(torch_lib):
                env["LD_LIBRARY_PATH"] = (
                    f"{torch_lib}:{existing}" if existing else torch_lib
                )
            logger.info(f"Exec server LD_LIBRARY_PATH pinned to venv libtorch: {torch_lib}")
    except Exception as e:
        logger.warning(
            f"Failed to pin LD_LIBRARY_PATH (local binaries may hit ABI mismatch): {e}"
        )
    _gpu_mod.env = env


APP = FastAPI(lifespan=lifespan)


@APP.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "exec-server",
        "strategy": _STRATEGY.name if _STRATEGY else "<not-started>",
        "board_host": _BOARD_HOST,
    }


@APP.post("/gpu/binary", response_model=ExecResult)
async def execute_gpu_binary(
    binary: UploadFile = File(..., description="Binary executable to run"),
    args: Optional[str] = Form("", description="Command line arguments"),
    env_vars: Optional[str] = Form(None, description="JSON dict (local only)"),
    prefix_command: Optional[str] = Form(None, description="Prefix e.g. 'ncu' (local only)"),
    n_runs: Optional[int] = Form(1, description="Number of times to run"),
    timeout: Optional[float] = Form(3600, description="Timeout seconds"),
    kernel_files: Optional[str] = Form(None, description="JSON list of paths (remote only)"),
    profile: Optional[bool] = Form(False, description="Enable profiling (remote only)"),
):
    """Execute an uploaded binary. Strategy picked at server startup."""
    logger.info(
        f"/gpu/binary [{_STRATEGY.name if _STRATEGY else '?'}] - "
        f"file={binary.filename} args={args!r} n_runs={n_runs} "
        f"profile={profile} backlog={QUEUE.qsize()}"
    )
    try:
        binary_data = await binary.read()
        if not binary_data:
            raise HTTPException(status_code=400, detail="Empty binary file provided")

        parsed_env_vars: Optional[dict] = None
        if env_vars:
            try:
                parsed_env_vars = json.loads(env_vars)
                if not isinstance(parsed_env_vars, dict):
                    raise ValueError("env_vars must be a JSON object")
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid env_vars JSON: {e}"
                )

        parsed_kernel_files: Optional[list[str]] = None
        if kernel_files:
            try:
                parsed_kernel_files = json.loads(kernel_files)
                if not isinstance(parsed_kernel_files, list):
                    parsed_kernel_files = [kernel_files]
            except json.JSONDecodeError:
                parsed_kernel_files = [kernel_files]

        payload = dict(
            binary_data=binary_data,
            filename=binary.filename or "gpu_executable",
            args=args or "",
            env_vars=parsed_env_vars,
            prefix_command=prefix_command,
            n_runs=n_runs,
            timeout=timeout,
            kernel_files=parsed_kernel_files,
            profile=profile or False,
        )

        completion_future: asyncio.Future = asyncio.Future()
        await QUEUE.put((payload, completion_future, time.time()))
        await completion_future
        return completion_future.result()
    except asyncio.CancelledError:
        raise HTTPException(status_code=500, detail="Request was cancelled")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_server(host: str, port: int, log_filepath: str | None = None):
    log_config = get_log_config(log_filepath=log_filepath)
    uvicorn.run(
        APP, host=host, port=port,
        log_config=log_config, timeout_graceful_shutdown=0.1,
    )


def main():
    global args, _BOARD_HOST, _GPU_IDS

    parser = argparse.ArgumentParser(description="Unified GPU binary exec server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2006)
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Remote-strategy worker count. Local strategy "
                             "sizes workers from KERNELBLASTER_GPU_SERVER_GPU_IDS / "
                             "KERNELBLASTER_GPU_SERVER_NUM_WORKERS.")
    parser.add_argument("--board-host", type=str, default=None,
                        help="SSH target for Adreno board. When set, the "
                             "server picks the RemoteExecStrategy.")
    parser.add_argument(
        "--log_path", type=Path,
        default=Path("/tmp/kernelblaster/exec_server.log"),
    )
    args = parser.parse_args()

    _BOARD_HOST = args.board_host
    if _BOARD_HOST is None:
        # Local: read GPU_IDS from the same env vars the legacy gpu.py used.
        gpu_ids_raw = os.getenv("KERNELBLASTER_GPU_SERVER_GPU_IDS", "").strip()
        if gpu_ids_raw:
            _GPU_IDS = [s.strip() for s in gpu_ids_raw.split(",") if s.strip()]
        else:
            n = int(os.getenv("KERNELBLASTER_GPU_SERVER_NUM_WORKERS", "1"))
            _GPU_IDS = [str(i) for i in range(max(1, n))]

    # Push _BOARD_HOST into gpu_adreno's module state — its
    # ``upload_and_exec_binary`` reads it as ``_BOARD_HOST`` at request
    # time. Same setter pattern as the compile server.
    if _BOARD_HOST:
        from . import gpu_adreno as _adreno
        _adreno._BOARD_HOST = _BOARD_HOST

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    run_server(args.host, args.port, log_filepath=str(args.log_path))


if __name__ == "__main__":
    main()
