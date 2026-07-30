# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Unified compile server (Phase D).

Single FastAPI app that dispatches on ``?backend=cuda|opencl``. Both
strategies live in :mod:`servers.strategies` and share the queue +
worker plumbing from :mod:`servers.utils.queue_server`.

Runs alongside the legacy ``compile.py`` / ``compile_opencl.py`` for
Phase D — Phase E migrates clients and retires the legacy modules.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .server_logging import get_log_config
from .utils.queue_server import worker_pool
from .strategies import get_compile_strategy

# Import the two legacy modules for their domain-error classes + env-dir
# constants so this server can share cleanup semantics without duplicating
# the CMake toolchain plumbing. Phase E will move ``CompilationError`` /
# ``OpenCLCompilationError`` into ``servers.utils`` when the legacy files
# are retired.
from .compile import CompilationError, free_cuda_envs
from .compile_opencl import OpenCLCompilationError


logger = logging.getLogger("uvicorn")
QUEUE: asyncio.Queue = asyncio.Queue()
_ARTIFACTS_DIR: Optional[Path] = None
_BOARD_HOST: Optional[str] = None


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class UnifiedCompilationResult(BaseModel):
    """Unified response for both CUDA + OpenCL compilations.

    Backend-specific extras (persistent_artifacts_dir for CUDA,
    remote_binary_path for OpenCL) land in ``extras`` — a
    role-keyed dict rather than a per-backend field, mirroring the
    :class:`data.sources.Problem` design.
    """
    backend: str
    job_name: str
    main_file: str
    source_file: str
    success: bool = False
    message: Optional[str] = None
    output_path: Optional[str] = None
    extras: dict[str, Optional[str]] = {}


@dataclass
class _CompileJob:
    """Internal job spec passed through the queue to workers."""
    backend: str
    job_name: str
    main_file: str
    source_file: str
    backend_version: str
    backend_flag: bool
    output_path: Path
    artifacts_dir: Path
    board_host: Optional[str]
    debug: bool


# ---------------------------------------------------------------------------
# Worker handler
# ---------------------------------------------------------------------------

async def _dispatch_compile_job(
    worker_id: int, job_args: tuple
) -> UnifiedCompilationResult:
    """Handler for :func:`worker_pool`. Dispatches to the right strategy."""
    (job,) = job_args  # single-element job_args: _CompileJob
    assert isinstance(job, _CompileJob)

    strategy = get_compile_strategy(job.backend)
    extras: dict[str, Optional[str]] = {}
    try:
        extra_path = await strategy.compile(
            worker_id=worker_id,
            job_name=job.job_name,
            main_file=job.main_file,
            source_file=job.source_file,
            backend_version=job.backend_version,
            backend_flag=job.backend_flag,
            output_path=job.output_path,
            artifacts_dir=job.artifacts_dir,
            board_host=job.board_host,
            debug=job.debug,
        )
    except (CompilationError, OpenCLCompilationError) as e:
        return UnifiedCompilationResult(
            backend=job.backend,
            job_name=job.job_name,
            main_file=job.main_file,
            source_file=job.source_file,
            success=False,
            message=getattr(e, "message", None) or str(e),
        )
    except Exception as e:
        return UnifiedCompilationResult(
            backend=job.backend,
            job_name=job.job_name,
            main_file=job.main_file,
            source_file=job.source_file,
            success=False,
            message=f"Internal error: {e}",
        )

    if job.backend == "opencl":
        extras["remote_binary_path"] = extra_path
    elif job.backend == "cuda" and extra_path is not None:
        extras["persistent_artifacts_dir"] = extra_path

    return UnifiedCompilationResult(
        backend=job.backend,
        job_name=job.job_name,
        main_file=job.main_file,
        source_file=job.source_file,
        success=True,
        output_path=str(job.output_path),
        extras=extras,
    )


# ---------------------------------------------------------------------------
# FastAPI wiring
# ---------------------------------------------------------------------------

# Placeholder — ``args`` is populated by ``__main__`` before ``lifespan`` fires.
args: argparse.Namespace = argparse.Namespace()


@asynccontextmanager
async def lifespan(app):
    logger.info(
        f"Started unified compile server on {args.host}:{args.port} "
        f"with {args.num_workers} workers "
        f"(board_host={args.board_host}, artifacts_dir={args.artifacts_dir})"
    )
    # Rare edge — worker_pool catches its own errors; we route both
    # domain errors here so the queue's log line has the right tag.
    async with worker_pool(
        num_workers=args.num_workers,
        queue=QUEUE,
        handler=_dispatch_compile_job,
        # Two domain errors accepted; use CompilationError as the
        # umbrella since both handlers already catch internally.
        domain_error=CompilationError,
        logger=logger,
        on_shutdown=lambda: (free_cuda_envs(), _cleanup_env_dir()),
    ):
        yield


def _cleanup_env_dir() -> None:
    global _ARTIFACTS_DIR
    if _ARTIFACTS_DIR is not None and _ARTIFACTS_DIR.exists():
        shutil.rmtree(_ARTIFACTS_DIR, ignore_errors=True)


APP = FastAPI(lifespan=lifespan)


@APP.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "compile-server",
        "board_host": _BOARD_HOST,
    }


@APP.get("/compile", response_model=UnifiedCompilationResult)
async def compile_endpoint(
    backend: str = Query(..., description="Backend: 'cuda' or 'opencl'"),
    job_name: str = Query(...),
    main_file: str = Query(..., description="Driver source path (.c or .cpp)"),
    source_file: str = Query(..., description="Kernel source path (.cu or .cl)"),
    backend_version: str = Query(
        ..., description="sm_XX for CUDA, opencl_X.Y for OpenCL"
    ),
    backend_flag: int = Query(
        0, description="CUDA: persistent_artifacts; OpenCL: remote"
    ),
):
    """Compile a driver + kernel pair. Dispatches on ``backend``."""
    # Validate backend up front so the client gets a 400, not a 500
    # deep inside the worker.
    try:
        get_compile_strategy(backend)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    assert _ARTIFACTS_DIR is not None, "artifacts_dir not initialised"
    output_path = _ARTIFACTS_DIR / str(uuid.uuid4()) / "out" / f"tmp{uuid.uuid4().hex}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    job = _CompileJob(
        backend=backend,
        job_name=job_name,
        main_file=main_file,
        source_file=source_file,
        backend_version=backend_version,
        backend_flag=bool(backend_flag),
        output_path=output_path,
        artifacts_dir=_ARTIFACTS_DIR,
        board_host=_BOARD_HOST,
        debug=bool(getattr(args, "compile_debug", False)),
    )

    completion_future: asyncio.Future = asyncio.Future()
    await QUEUE.put((job, completion_future, time.time()))

    try:
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
        APP,
        host=host,
        port=port,
        log_config=log_config,
        timeout_graceful_shutdown=0.1,
    )


def main():
    global args, _ARTIFACTS_DIR, _BOARD_HOST

    parser = argparse.ArgumentParser(description="Unified compile server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2005)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("/tmp/kernelblaster_compile"),
        help="Root directory for scratch build artifacts.",
    )
    parser.add_argument(
        "--board-host",
        type=str,
        default=None,
        help="SSH target for remote OpenCL compile (e.g. root@10.44.120.201).",
    )
    parser.add_argument("--compile-debug", action="store_true")
    parser.add_argument(
        "--log_path",
        type=Path,
        default=Path("/tmp/kernelblaster/compile_server.log"),
    )

    args = parser.parse_args()
    _ARTIFACTS_DIR = args.artifacts_dir
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    _BOARD_HOST = args.board_host

    # Ensure log dir exists before uvicorn tries to open the file handler.
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    run_server(args.host, args.port, log_filepath=str(args.log_path))


if __name__ == "__main__":
    main()
