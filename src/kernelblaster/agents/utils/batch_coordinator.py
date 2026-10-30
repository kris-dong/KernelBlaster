# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Client-side batching helper for the unified exec server.

The exec server exposes ``POST /gpu/batch`` (see
:mod:`servers.exec_server`); targets whose strategy sets
``supports_batching = True`` (e.g. the FPGA path) amortise a fixed
per-batch cost (bitstream flash, board reset) across N jobs. But most
callers naturally produce work one request at a time (each RL rollout
step calls :func:`run_gpu_executable` once).

:class:`BatchCoordinator` bridges the two: callers submit single jobs,
the coordinator buffers concurrent submits and flushes to
``/gpu/batch`` on **either** a size trigger (``max_batch_size`` items
queued) **or** a time trigger (``max_wait_ms`` elapsed since the first
queued item). Each :meth:`submit` returns a future that resolves with
that job's result — so submitting N jobs from N asyncio coroutines
looks the same as N single :func:`run_gpu_executable` calls, but hits
the server as one batched HTTP request.

The pattern (buffer + flush-on-size-or-time) is standard in inference
servers (vLLM, Triton, TGI, ...). This is a minimal in-process version
tailored to the exec-server contract.

Concurrency notes:
    * Single event loop only — the coordinator holds ``asyncio``
      state, not thread-safe.
    * ``submit_fn`` gets a ``list[TJob]`` and must return a matching
      ``list[TResult]`` in the same order. Length mismatch raises on
      every buffered future.
    * If ``submit_fn`` raises, every buffered future in that batch
      receives the exception. Subsequent submits open a fresh batch.

The coordinator is decoupled from HTTP: ``submit_fn`` is just an async
callable, which makes it trivial to unit-test with an in-process
recorder. See ``run_gpu_batch`` in :mod:`commands` for the wired-up
HTTP variant.
"""
from __future__ import annotations

import asyncio
import os
from typing import Awaitable, Callable, Generic, TypeVar

TJob = TypeVar("TJob")
TResult = TypeVar("TResult")


DEFAULT_MAX_BATCH_SIZE = int(os.getenv("KERNELBLASTER_EXEC_BATCH_SIZE", "32"))
DEFAULT_MAX_WAIT_MS = int(os.getenv("KERNELBLASTER_EXEC_BATCH_MAX_WAIT_MS", "1000"))


class BatchCoordinator(Generic[TJob, TResult]):
    """Buffer + flush-on-size-or-time coordinator.

    Args:
        submit_fn: Async callable that takes a batch (``list[TJob]``)
            and returns aligned results (``list[TResult]``). The
            coordinator does not care what the transport is — HTTP,
            in-process, RPC — it just awaits the callable.
        max_batch_size: Flush when this many items are queued.
            Reads ``KERNELBLASTER_EXEC_BATCH_SIZE`` env var by default.
        max_wait_ms: Flush this many milliseconds after the first
            queued item, even if the size trigger hasn't fired.
            Reads ``KERNELBLASTER_EXEC_BATCH_MAX_WAIT_MS`` env var by
            default. Tune down for latency-sensitive workloads, up for
            targets where the amortised cost dominates (FPGA flash).
    """

    def __init__(
        self,
        submit_fn: Callable[[list[TJob]], Awaitable[list[TResult]]],
        *,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_wait_ms: int = DEFAULT_MAX_WAIT_MS,
    ):
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        if max_wait_ms < 0:
            raise ValueError(f"max_wait_ms must be >= 0, got {max_wait_ms}")
        self._submit_fn = submit_fn
        self._max_batch_size = max_batch_size
        self._max_wait_s = max_wait_ms / 1000.0
        self._pending: list[tuple[TJob, asyncio.Future[TResult]]] = []
        self._flush_timer: asyncio.Task | None = None

    async def submit(self, job: TJob) -> TResult:
        """Queue ``job`` and await its result.

        The result becomes available when the batch containing this
        job flushes to :attr:`_submit_fn` and that call returns. If the
        batch call raises, ``submit`` re-raises the same exception.
        """
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[TResult] = loop.create_future()
        self._pending.append((job, fut))

        if len(self._pending) >= self._max_batch_size:
            self._flush_now()
        elif self._flush_timer is None or self._flush_timer.done():
            self._flush_timer = asyncio.create_task(self._deferred_flush())

        return await fut

    async def flush(self) -> None:
        """Force-flush any pending items now. Safe to call with an
        empty buffer (no-op). Useful for callers that want to
        deterministically drain the coordinator at end-of-work."""
        self._flush_now()

    async def _deferred_flush(self) -> None:
        """Timer arm: sleep for ``max_wait_s`` then flush. Cancelled
        when a size-triggered flush fires first."""
        try:
            await asyncio.sleep(self._max_wait_s)
        except asyncio.CancelledError:
            return
        self._flush_now()

    def _flush_now(self) -> None:
        """Snapshot the pending buffer + spawn the batch task. All
        state mutation happens synchronously (no ``await``) so
        concurrent :meth:`submit` calls can't interleave a partial
        buffer with the reset."""
        if not self._pending:
            return
        batch = self._pending
        self._pending = []
        if self._flush_timer is not None and not self._flush_timer.done():
            self._flush_timer.cancel()
        self._flush_timer = None
        asyncio.create_task(self._run_batch(batch))

    async def _run_batch(
        self, batch: list[tuple[TJob, asyncio.Future[TResult]]]
    ) -> None:
        jobs = [j for j, _ in batch]
        futures = [f for _, f in batch]
        try:
            results = await self._submit_fn(jobs)
        except Exception as e:
            for f in futures:
                if not f.done():
                    f.set_exception(e)
            return

        if len(results) != len(batch):
            err = RuntimeError(
                f"BatchCoordinator submit_fn returned {len(results)} "
                f"results for {len(batch)} jobs"
            )
            for f in futures:
                if not f.done():
                    f.set_exception(err)
            return

        for f, r in zip(futures, results):
            if not f.done():
                f.set_result(r)


__all__ = [
    "BatchCoordinator",
    "DEFAULT_MAX_BATCH_SIZE",
    "DEFAULT_MAX_WAIT_MS",
]
