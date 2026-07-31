# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Binary-exec strategies for the unified exec server.

Companion to :mod:`servers.utils.compile_strategy`. Two selection
modes are supported:

* **Per-server** (default): the exec server picks a strategy ONCE at
  startup based on CLI flags — historically ``--board-host`` selected
  :class:`RemoteExecStrategy` (SSH scp + remote run) and its absence
  fell back to :class:`LocalExecStrategy` (local subprocess). The new
  ``--strategy`` flag lets callers ask for a named strategy explicitly
  (``local`` / ``remote`` / ``fpga``).
* **Per-request**: dispatched through :func:`get_exec_strategy` — used
  by targets whose selection can vary per-request (rare; the registry
  is mostly for the FPGA path to be pluggable).

Two execution shapes:

* :meth:`ExecStrategy.exec` — single binary, single result (unchanged
  contract). The path CUDA + OpenCL + reprofile-nsys have always used.
* :meth:`ExecStrategy.batch_exec` — N binaries, N results. Default impl
  loops over :meth:`exec` (no throughput gain, just contract fulfilment).
  Targets with expensive fixed setup — e.g. an FPGA bitstream that takes
  minutes to flash — override to amortise that cost across the batch by
  linking N binaries into one boot ELF, flashing once, then dispatching
  each and parsing per-job output.

Return-shape convention for :meth:`exec`:

  - ``n_runs == 1``: ``(str, str)``.
  - ``n_runs > 1``:  ``(list[str], list[str])`` of length n_runs.

Fields not applicable to a given strategy (e.g. ``kernel_files`` for
local, ``env_vars`` for remote) are accepted but silently ignored —
same behaviour the pre-refactor endpoints already had.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecJob:
    """One entry in a :meth:`ExecStrategy.batch_exec` call.

    Same field set as :meth:`ExecStrategy.exec` (minus ``worker_id``,
    which is supplied per-batch, not per-item). Kept as a dataclass so
    the HTTP handler can build a homogeneous ``list[ExecJob]`` before
    dispatch.
    """
    binary_data: bytes
    filename: str
    args: str = ""
    env_vars: Optional[dict] = None
    prefix_command: Optional[str] = None
    n_runs: int = 1
    timeout: float = 3600
    kernel_files: Optional[list[str]] = None
    profile: bool = False


@dataclass
class ExecJobResult:
    """Result of one :class:`ExecJob` inside a batch.

    ``success=False`` + ``message`` on failure — matches
    :class:`servers.exec_server.ExecResult`'s per-request semantics so
    the HTTP handler can straight-line into a
    ``list[ExecResult]`` response.
    """
    stdout: str | list[str] = ""
    stderr: str | list[str] = ""
    success: bool = True
    message: Optional[str] = None


class BatchTooLargeError(Exception):
    """Raised by a strategy's link/build step when a batched artifact
    can't be produced at the current batch size — for RISC-V ELFs this
    hits the ``R_RISCV_PCREL_HI20`` relocation truncation once the
    fused ``.text``/``.rodata`` grows past ~2 GB, but the same failure
    mode is common on other embedded targets (image-size caps, EEPROM
    limits, ...).

    Carries ``suggested_split`` so the base ``batch_exec`` loop can
    subdivide sensibly. ``suggested_split=None`` means "halve it";
    strategies that can be smarter (e.g. "your accumulated size hit X;
    the current batch is Y items so keep it under X * items / Y")
    populate the field to skip the ceiling search.

    ``strategy.batch_exec`` catches this and splits the batch rather
    than failing everything — see :meth:`ExecStrategy.batch_exec` for
    the recursion.
    """

    def __init__(
        self,
        message: str,
        *,
        suggested_split: Optional[int] = None,
    ):
        super().__init__(message)
        self.error_message = message
        self.suggested_split = suggested_split


class ExecStrategy(ABC):
    """Binary-exec strategy (local subprocess, remote SSH, FPGA batch, ...)."""

    name: str = ""              # "local" | "remote" | "fpga" | "spike"

    #: Whether the strategy's ``batch_exec`` amortises fixed cost across
    #: the batch. False (default) = ``batch_exec`` is just a serial loop
    #: over :meth:`exec` and callers gain nothing by batching. Targets
    #: that DO gain (FPGA bitstream flash, remote board resets, batched
    #: multi-model ELF link, ...) override this to ``True`` so
    #: client-side coordinators know whether to buffer.
    supports_batching: bool = False

    #: Cheapest batch size the strategy is known to be able to link.
    #: ``batch_exec``'s split-and-retry falls back to this floor before
    #: giving up per-job. Targets with no known floor leave the default
    #: (1 == degenerate per-job).
    min_batch_size: int = 1

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

    async def batch_exec(
        self,
        *,
        worker_id: int,
        jobs: list[ExecJob],
    ) -> list[ExecJobResult]:
        """Execute a batch of jobs and return per-job results in order.

        Default implementation loops :meth:`exec` sequentially — correct
        for strategies with no meaningful batching gain (Local / Remote
        SSH). Targets that CAN amortise fixed cost (e.g. FPGA bitstream
        flash, one linked ELF for N kernels) override this and set
        :attr:`supports_batching` = True. Overrides that raise
        :class:`BatchTooLargeError` from the link step get split-and-
        retry for free — see :meth:`_batch_exec_impl`.

        Any exception raised by :meth:`exec` for a single job is caught
        and packaged as ``ExecJobResult(success=False, message=...)`` —
        one failing job does not abort the batch. This matches the
        single-request path in :mod:`servers.exec_server`.
        """
        try:
            return await self._batch_exec_impl(worker_id=worker_id, jobs=jobs)
        except BatchTooLargeError as e:
            return await self._split_and_retry(
                worker_id=worker_id, jobs=jobs, error=e,
            )

    async def _batch_exec_impl(
        self,
        *,
        worker_id: int,
        jobs: list[ExecJob],
    ) -> list[ExecJobResult]:
        """Actual batched execution — override this in subclasses.

        The base implementation is a serial loop over :meth:`exec` that
        gains nothing from batching. Strategies that link N binaries
        into one artifact override this and may raise
        :class:`BatchTooLargeError` if the link step fails at the
        current size.
        """
        results: list[ExecJobResult] = []
        for job in jobs:
            try:
                stdout, stderr = await self.exec(
                    worker_id=worker_id,
                    binary_data=job.binary_data,
                    filename=job.filename,
                    args=job.args,
                    env_vars=job.env_vars,
                    prefix_command=job.prefix_command,
                    n_runs=job.n_runs,
                    timeout=job.timeout,
                    kernel_files=job.kernel_files,
                    profile=job.profile,
                )
                results.append(ExecJobResult(stdout=stdout, stderr=stderr, success=True))
            except Exception as e:
                message = getattr(e, "error_message", None) or str(e)
                results.append(ExecJobResult(success=False, message=message))
        return results

    async def _split_and_retry(
        self,
        *,
        worker_id: int,
        jobs: list[ExecJob],
        error: "BatchTooLargeError",
    ) -> list[ExecJobResult]:
        """Handle :class:`BatchTooLargeError` from :meth:`_batch_exec_impl`
        by subdividing and recursing.

        Termination: each recursive call either succeeds or halves the
        job count. At len==1 the strategy has no way to shrink further,
        so we short-circuit with a per-job failure carrying the
        original ``error.error_message``.

        Split heuristic: prefer ``error.suggested_split`` when set (the
        strategy's own estimate of "how many fit"). Otherwise halve.
        This is a base-class implementation; strategies that want a
        different policy (e.g. dispatch remaining jobs to other
        workers) override :meth:`batch_exec` directly.
        """
        if len(jobs) <= 1:
            return [
                ExecJobResult(
                    stdout="", stderr="",
                    success=False,
                    message=(
                        f"{error.error_message} (batch was already 1 item — "
                        f"strategy can't shrink further)"
                    ),
                )
                for _ in jobs
            ]

        split = error.suggested_split
        if split is None or split < 1 or split >= len(jobs):
            split = max(1, len(jobs) // 2)

        head, tail = jobs[:split], jobs[split:]
        head_results = await self.batch_exec(worker_id=worker_id, jobs=head)
        tail_results = await self.batch_exec(worker_id=worker_id, jobs=tail)
        return head_results + tail_results


# ---------------------------------------------------------------------------
# Registry — parallel to compile_strategy._REGISTRY.
#
# Exec strategies are usually picked ONCE per server startup (unlike
# compile strategies, which dispatch per-request on ?backend=). The
# registry exists so new strategies (fpga) can be plugged in by name
# from a CLI flag, and so tests can look them up without importing
# every module.
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[ExecStrategy]] = {}


def register_exec_strategy(strategy_cls: type[ExecStrategy]) -> None:
    """Register an exec-strategy CLASS by its ``name`` attribute.

    Unlike :func:`servers.utils.compile_strategy.register_compile_strategy`
    (which stores instances because compile strategies are
    parameter-free), exec strategies carry per-server config (GPU IDs,
    board host, bitstream path) so we store the class and let the
    caller instantiate it once at startup.
    """
    if not strategy_cls.name:
        raise ValueError(f"{strategy_cls!r} has empty .name — cannot register")
    _REGISTRY[strategy_cls.name] = strategy_cls


def get_exec_strategy_cls(strategy_name: str) -> type[ExecStrategy]:
    """Look up a registered exec-strategy CLASS by name.

    Raises ``ValueError`` if unknown — the caller (exec_server main)
    converts this into a fatal startup error, not an HTTP response.
    """
    try:
        return _REGISTRY[strategy_name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none registered>"
        raise ValueError(
            f"Unknown exec strategy: {strategy_name!r}. Known: {known}"
        ) from None


__all__ = [
    "ExecStrategy",
    "ExecJob",
    "ExecJobResult",
    "BatchTooLargeError",
    "register_exec_strategy",
    "get_exec_strategy_cls",
]
