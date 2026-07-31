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
from .strategies import (
    ExecJob,
    ExecJobResult,
    ExecStrategy,
    LocalExecStrategy,
    RemoteExecStrategy,
    get_exec_strategy_cls,
)
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

async def _exec_job(worker_id: int, job_args: tuple):
    """Handler for :func:`worker_pool` — routes to the active strategy.

    ``job_args`` shape is a discriminated tuple flattened for
    :func:`queue_worker_loop`'s ``*job_args`` unpacking:

      * ``("single", payload_dict)`` — strategy.exec(**payload) →
        :class:`ExecResult`.
      * ``("batch", list[ExecJob])`` — strategy.batch_exec(jobs=...) →
        ``list[ExecJobResult]`` wrapped as ``list[ExecResult]``.

    Both paths convert underlying strategy exceptions into
    ``ExecResult(success=False, ...)`` — clients see 200-OK JSON with
    the failure flag, matching the pre-refactor contract.
    """
    kind, payload = job_args[0], job_args[1]
    assert _STRATEGY is not None, "exec strategy not initialised"

    if kind == "single":
        try:
            stdout, stderr = await _STRATEGY.exec(worker_id=worker_id, **payload)
            return ExecResult(success=True, stdout=stdout, stderr=stderr)
        except Exception as e:
            message = getattr(e, "error_message", None) or str(e)
            logger.error(f"[Worker {worker_id}]: exec failed: {message}")
            return ExecResult(success=False, message=message)

    if kind == "batch":
        jobs: list[ExecJob] = payload
        try:
            results = await _STRATEGY.batch_exec(worker_id=worker_id, jobs=jobs)
            return [_job_result_to_exec_result(r) for r in results]
        except Exception as e:
            # Whole-batch failure (e.g. bitstream flash refused). All
            # jobs get the same error — clients that want per-job
            # granularity should catch inside batch_exec instead.
            message = getattr(e, "error_message", None) or str(e)
            logger.error(f"[Worker {worker_id}]: batch_exec failed: {message}")
            return [ExecResult(success=False, message=message) for _ in jobs]

    raise ValueError(f"Unknown exec job kind: {kind!r}")


def _job_result_to_exec_result(r: ExecJobResult) -> "ExecResult":
    return ExecResult(
        stdout=r.stdout,
        stderr=r.stderr,
        success=r.success,
        message=r.message,
    )


# ---------------------------------------------------------------------------
# FastAPI wiring
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app):
    global _STRATEGY

    strategy_name = _resolve_strategy_name()

    if strategy_name == "remote":
        _STRATEGY = RemoteExecStrategy(board_host=_BOARD_HOST)
        logger.info(
            f"Started unified exec server on {args.host}:{args.port} "
            f"with {args.num_workers} workers, strategy=remote, board={_BOARD_HOST}"
        )
        await _remote_preflight()
        num_workers = args.num_workers
    elif strategy_name == "local":
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
    else:
        # Registry-dispatched strategies (currently: fpga). Instantiation
        # kwargs come from CLI flags translated into a kwargs dict below.
        strategy_cls = get_exec_strategy_cls(strategy_name)
        init_kwargs = _strategy_init_kwargs(strategy_name)
        _STRATEGY = strategy_cls(**init_kwargs)
        logger.info(
            f"Started unified exec server on {args.host}:{args.port} "
            f"strategy={strategy_name} init_kwargs={list(init_kwargs)}"
        )
        # Optional per-strategy pre-flight hook (real FPGAExecStrategy
        # uses this to check board connectivity + bitstream freshness).
        preflight = getattr(_STRATEGY, "preflight", None)
        if preflight is not None:
            await preflight()
        num_workers = args.num_workers

    async with worker_pool(
        num_workers=num_workers,
        queue=QUEUE,
        handler=_exec_job,
        domain_error=ExecError,
        logger=logger,
    ):
        yield


def _resolve_strategy_name() -> str:
    """Pick the strategy name from CLI flags, preserving pre-flag
    behaviour: ``--strategy`` explicit > ``--board-host`` implies
    ``remote`` > default ``local``. Central so tests can reason about
    resolution without a full server boot."""
    explicit = getattr(args, "strategy", None)
    if explicit:
        return explicit
    if _BOARD_HOST:
        return "remote"
    return "local"


def _strategy_init_kwargs(strategy_name: str) -> dict:
    """Translate CLI namespace → strategy __init__ kwargs. Kept here
    (not on the strategies themselves) so ``exec_server`` owns the CLI
    surface end-to-end and strategy classes remain plain data holders.
    Extend when a new strategy needs new flags."""
    if strategy_name == "fpga":
        return dict(
            board_host=_BOARD_HOST,
            bitstream_path=getattr(args, "bitstream_path", None),
            batch_runner_template=getattr(args, "batch_runner_template", None),
        )
    if strategy_name == "spike":
        spike_args_raw = getattr(args, "spike_extra_args", None) or ""
        default_spike_args = tuple(
            a for a in spike_args_raw.split(",") if a.strip()
        )
        return dict(
            spike_binary=getattr(args, "spike_binary", None),
            modelblaster_root=getattr(args, "modelblaster_root", None),
            multi_link_script=getattr(args, "multi_link_script", None),
            default_spike_args=default_spike_args,
        )
    if strategy_name == "firesim":
        # queue_timeout=None means "let the workload run to natural
        # completion" — the runner + queue both interpret that as
        # "no daemon SIGTERM watchdog"; add a per-job override at
        # submit time when a cap is needed.
        return dict(
            firesim_root=getattr(args, "firesim_root", None),
            firesim_env=getattr(args, "firesim_env", None),
            modelblaster_root=getattr(args, "modelblaster_root", None),
            # Shares the ``--multi-link-script`` flag with spike since
            # the fusion protocol (stdin manifest, $FUSED_OUT env,
            # BATCH_RELOC_ERROR_RE) is identical. Absent = batches of
            # >1 fall back to per-item via BatchTooLargeError.
            multi_link_script=getattr(args, "multi_link_script", None),
            queue_enabled=not getattr(args, "no_firesim_queue", False),
            queue_root=getattr(args, "firesim_queue_root", None),
            queue_bin=getattr(args, "firesim_queue_bin", None),
            queue_priority=getattr(args, "firesim_queue_priority", 5),
            queue_timeout_s=getattr(args, "firesim_queue_timeout", None),
            default_timeout=float(getattr(args, "firesim_default_timeout", 900)),
            python_bin=getattr(args, "firesim_python_bin", None),
        )
    return {}


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
        "supports_batching": bool(_STRATEGY and _STRATEGY.supports_batching),
        "board_host": _BOARD_HOST,
    }


# ---------------------------------------------------------------------------
# Request-parsing helpers (shared by /gpu/binary and /gpu/batch)
# ---------------------------------------------------------------------------

def _parse_env_vars(env_vars: Optional[str]) -> Optional[dict]:
    if not env_vars:
        return None
    try:
        parsed = json.loads(env_vars)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid env_vars JSON: {e}")
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="env_vars must be a JSON object")
    return parsed


def _parse_kernel_files(kernel_files: Optional[str]) -> Optional[list[str]]:
    if not kernel_files:
        return None
    try:
        parsed = json.loads(kernel_files)
    except json.JSONDecodeError:
        return [kernel_files]
    if not isinstance(parsed, list):
        return [kernel_files]
    return parsed


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

        payload = dict(
            binary_data=binary_data,
            filename=binary.filename or "gpu_executable",
            args=args or "",
            env_vars=_parse_env_vars(env_vars),
            prefix_command=prefix_command,
            n_runs=n_runs,
            timeout=timeout,
            kernel_files=_parse_kernel_files(kernel_files),
            profile=profile or False,
        )

        completion_future: asyncio.Future = asyncio.Future()
        # Flat shape: ("single", payload, fut, ts). queue_worker_loop's
        # ``*job_args, fut, ts = item`` unpacks to job_args=("single", payload).
        await QUEUE.put(("single", payload, completion_future, time.time()))
        await completion_future
        return completion_future.result()
    except asyncio.CancelledError:
        raise HTTPException(status_code=500, detail="Request was cancelled")


class BatchExecResponse(BaseModel):
    """Wrapper for :meth:`ExecStrategy.batch_exec` — per-job results in
    the same order as the incoming binaries. Individual failures are
    inline (``results[i].success = False``); the enclosing HTTP call
    only fails if the whole batch is rejected (invalid input, no
    strategy, ...).
    """
    results: list[ExecResult] = []


@APP.post("/gpu/batch", response_model=BatchExecResponse)
async def execute_gpu_batch(
    binaries: list[UploadFile] = File(..., description="Binaries in the batch"),
    manifest: str = Form(
        ...,
        description=(
            "JSON list of per-job kwargs, aligned with `binaries` by index. "
            "Each entry may set: args, env_vars, prefix_command, n_runs, "
            "timeout, kernel_files, profile. Missing keys default to the "
            "same defaults /gpu/binary uses."
        ),
    ),
):
    """Execute a batch of binaries in one queue slot.

    Dispatches to :meth:`ExecStrategy.batch_exec`. For strategies with
    ``supports_batching = False`` (Local, Remote SSH) this just runs
    them serially; the endpoint still works and returns the same shape,
    which lets client code opt into batching uniformly. For batching-
    aware strategies (FPGA) this amortises the fixed per-batch cost
    (bitstream flash, board reset) across ``len(binaries)`` jobs.
    """
    if len(binaries) == 0:
        raise HTTPException(status_code=400, detail="Empty batch")

    try:
        manifest_parsed = json.loads(manifest)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid manifest JSON: {e}")
    if not isinstance(manifest_parsed, list):
        raise HTTPException(status_code=400, detail="manifest must be a JSON list")
    if len(manifest_parsed) != len(binaries):
        raise HTTPException(
            status_code=400,
            detail=(
                f"manifest length ({len(manifest_parsed)}) does not match "
                f"binaries length ({len(binaries)})"
            ),
        )

    jobs: list[ExecJob] = []
    for idx, (upload, entry) in enumerate(zip(binaries, manifest_parsed)):
        if not isinstance(entry, dict):
            raise HTTPException(
                status_code=400, detail=f"manifest[{idx}] must be a JSON object"
            )
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"binaries[{idx}] is empty")

        # env_vars / kernel_files may arrive as JSON strings (form-encoded
        # style) OR native dicts/lists (JSON manifest style) — accept both.
        env_vars_raw = entry.get("env_vars")
        if isinstance(env_vars_raw, str):
            env_vars_val = _parse_env_vars(env_vars_raw)
        elif env_vars_raw is None or isinstance(env_vars_raw, dict):
            env_vars_val = env_vars_raw
        else:
            raise HTTPException(
                status_code=400,
                detail=f"manifest[{idx}].env_vars must be a dict or JSON string",
            )

        kernel_files_raw = entry.get("kernel_files")
        if isinstance(kernel_files_raw, str):
            kernel_files_val = _parse_kernel_files(kernel_files_raw)
        elif kernel_files_raw is None or isinstance(kernel_files_raw, list):
            kernel_files_val = kernel_files_raw
        else:
            raise HTTPException(
                status_code=400,
                detail=f"manifest[{idx}].kernel_files must be a list or JSON string",
            )

        jobs.append(ExecJob(
            binary_data=data,
            filename=upload.filename or f"gpu_executable_{idx}",
            args=entry.get("args", "") or "",
            env_vars=env_vars_val,
            prefix_command=entry.get("prefix_command"),
            n_runs=int(entry.get("n_runs", 1)),
            timeout=float(entry.get("timeout", 3600)),
            kernel_files=kernel_files_val,
            profile=bool(entry.get("profile", False)),
        ))

    logger.info(
        f"/gpu/batch [{_STRATEGY.name if _STRATEGY else '?'}] - "
        f"count={len(jobs)} supports_batching="
        f"{_STRATEGY.supports_batching if _STRATEGY else '?'} "
        f"backlog={QUEUE.qsize()}"
    )

    try:
        completion_future: asyncio.Future = asyncio.Future()
        # Flat shape (same convention as /gpu/binary above).
        await QUEUE.put(("batch", jobs, completion_future, time.time()))
        await completion_future
        results = completion_future.result()
        # The handler returns list[ExecResult] for the batch shape.
        return BatchExecResponse(results=results)
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
                        help="SSH target for Adreno board (or FPGA host). "
                             "Absent + no --strategy = LocalExecStrategy; "
                             "set + no --strategy = RemoteExecStrategy.")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Explicit exec strategy name (e.g. 'local', "
                             "'remote', 'fpga'). If unset, resolved from "
                             "--board-host (present = 'remote', absent = "
                             "'local').")
    parser.add_argument("--bitstream-path", type=Path, default=None,
                        help="Path to the FPGA bitstream (fpga strategy).")
    parser.add_argument("--batch-runner-template", type=Path, default=None,
                        help="Path to the batch-runner Zephyr app template "
                             "used by FPGAExecStrategy to build one boot ELF "
                             "from N kernel binaries.")
    parser.add_argument("--spike-binary", type=str, default=None,
                        help="Path to the spike executable (spike strategy). "
                             "Absent = look up on PATH.")
    parser.add_argument("--modelblaster-root", type=Path, default=None,
                        help="Root dir of modelblaster (contains "
                             "modelblaster/validation/). Required by the "
                             "spike strategy to find spike_runner.")
    parser.add_argument("--multi-link-script", type=Path, default=None,
                        help="Path to a script that fuses N kernel ELFs into "
                             "one multi-model ELF for batched spike runs. "
                             "Absent = spike batches devolve to per-item.")
    parser.add_argument("--spike-extra-args", type=str, default=None,
                        help="Comma-separated list of --spike-arg values "
                             "injected into every spike run (e.g. 'isa=rv64gcv').")

    # FireSim strategy (--strategy firesim). Reuses --modelblaster-root
    # from the spike group above (modelblaster.validation.firesim_runner
    # needs the same PYTHONPATH root).
    parser.add_argument("--firesim-root", type=str, default=None,
                        help="firesim install root "
                             "(<chipyard>/sims/firesim). Overrides FIRESIM_ROOT.")
    parser.add_argument("--firesim-env", type=str, default=None,
                        help="chipyard env.sh path. Overrides FIRESIM_ENV.")
    parser.add_argument("--no-firesim-queue", action="store_true",
                        help="Skip the on-host firesim queue and drive "
                             "firesim directly. Single-user dev only.")
    parser.add_argument("--firesim-queue-root", type=str, default=None,
                        help="FIRESIM_QUEUE_ROOT (e.g. "
                             "/scratch/dima/firesim_queue).")
    parser.add_argument("--firesim-queue-bin", type=str, default=None,
                        help="firesim-queue CLI path.")
    parser.add_argument("--firesim-queue-priority", type=int, default=5,
                        help="FIRESIM_QUEUE_PRIORITY (default 5 = middle).")
    parser.add_argument("--firesim-queue-timeout", type=int, default=None,
                        help="FIRESIM_QUEUE_TIMEOUT — hard daemon-side cap "
                             "in seconds. Absent = let the workload run "
                             "to natural completion.")
    parser.add_argument("--firesim-default-timeout", type=int, default=900,
                        help="Default runner timeout when the ExecJob "
                             "doesn't specify. Should exceed one firesim "
                             "runworkload (few minutes).")
    parser.add_argument("--firesim-python-bin", type=str, default=None,
                        help="Python interpreter to invoke "
                             "modelblaster.validation.firesim_runner with "
                             "(defaults to sys.executable; on garden use "
                             "the miniforge zephyr env).")

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
