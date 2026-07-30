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
"""Shared FastAPI worker-pool scaffolding for compile/profile/exec servers.

Each backend server enqueues work tuples of the form
``(*job_args, completion_future, enqueue_ts)`` and supplies a handler that
performs the backend-specific work. This module owns the worker loop that
sits between the queue and the handler: dispatch, error routing onto the
future, ``task_done`` bookkeeping.

Convention: the LAST two slots of every queue item are
``(completion_future: asyncio.Future, enqueue_ts: float)``. By convention,
``job_args[0]`` is the job name (used for log lines), but the handler is
free to interpret the remaining args however it wants.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Optional


JobHandler = Callable[[int, tuple], Awaitable[Any]]
"""Async ``(worker_id, job_args) -> result`` callable. ``result`` is set on the future."""


@asynccontextmanager
async def worker_pool(
    *,
    num_workers: int,
    queue: asyncio.Queue,
    handler: JobHandler,
    domain_error: type[Exception],
    logger: logging.Logger | None = None,
    on_shutdown: Optional[Callable[[], Any]] = None,
):
    """Async context manager that spawns N workers around
    :func:`queue_worker_loop` and cancels them on exit.

    Replaces the ``create_task``+``cancel``-in-``finally`` boilerplate
    that each compile/exec server used to hand-roll. Optional
    ``on_shutdown`` fires after workers are cancelled — CUDA uses this
    to release GPU envs; OpenCL uses it to rm the compile-scratch dir.

    Both compile servers (compile.py, compile_opencl.py) and both exec
    servers (gpu.py, gpu_adreno.py) now use this helper. Step 5
    tactical cleanup for the eventual endpoint unification into a
    single compile server + single exec server (each parameterised
    on a backend strategy).
    """
    tasks = [
        asyncio.create_task(
            queue_worker_loop(
                worker_id=wid,
                queue=queue,
                handler=handler,
                domain_error=domain_error,
                logger=logger,
            )
        )
        for wid in range(num_workers)
    ]
    try:
        yield tasks
    finally:
        for t in tasks:
            t.cancel()
        if on_shutdown is not None:
            try:
                result = on_shutdown()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                log = logger or logging.getLogger("uvicorn")
                log.warning(f"worker_pool on_shutdown raised: {e}")


async def queue_worker_loop(
    *,
    worker_id: int,
    queue: asyncio.Queue,
    handler: JobHandler,
    domain_error: type[Exception],
    logger: logging.Logger | None = None,
) -> None:
    """Generic worker: pull items, dispatch to handler, route result/exception to the future.

    Items must end with ``(completion_future, enqueue_ts)``. Exceptions raised by
    ``handler`` are routed onto ``completion_future``; ``domain_error`` is preserved
    as-is, ``FileNotFoundError`` is preserved as-is, and any other exception is
    wrapped in ``domain_error``.
    """
    log = logger or logging.getLogger("uvicorn")
    while True:
        item = await queue.get()
        *job_args, completion_future, enqueue_ts = item
        job_name = job_args[0] if job_args else "<unknown>"
        try:
            queue_wait_s = time.time() - enqueue_ts
            log.info(
                f"[Worker {worker_id}]: dequeued {job_name} after "
                f"queue_wait={queue_wait_s:.2f}s (backlog_now={queue.qsize()})"
            )
            result = await handler(worker_id, tuple(job_args))
            completion_future.set_result(result)
        except domain_error as e:
            log.info(f"[Worker {worker_id}]: Error processing {job_name}: {e}")
            completion_future.set_exception(e)
        except FileNotFoundError as e:
            log.error(f"[Worker {worker_id}]: File not found while processing {job_name}: {e}")
            completion_future.set_exception(e)
        except Exception as e:
            msg = f"[Worker {worker_id}]: Unhandled exception processing {job_name}: {e}"
            log.error(msg, exc_info=True)
            completion_future.set_exception(domain_error(msg))
        finally:
            queue.task_done()
