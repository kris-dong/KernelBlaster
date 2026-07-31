# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""FPGA exec strategy (RISC-V SoC + Zephyr + FPGA emulation).

The core architectural distinction from :class:`LocalExecStrategy` and
:class:`RemoteExecStrategy`: **bitstream flash is expensive** (minutes)
and **must be amortised across many kernel executions**. To make that
economically viable this strategy overrides
:meth:`ExecStrategy.batch_exec` and sets ``supports_batching = True``.

Batched flow (the primary path):

1. Receive N jobs (kernel binaries — either standalone ELFs or the
   static libraries produced by :class:`ZephyrCompileStrategy` in
   ``link_as_lib`` mode).
2. Link them into a single **batch-runner ELF** using the template
   at :attr:`_batch_runner_template`. The batch runner is a small
   Zephyr app that:
      * Boots on the RISC-V SoC.
      * Reads a queue of ``(kernel_id, args)`` records from UART.
      * Dispatches each embedded kernel, sampling the ``mcycle`` CSR
        before and after.
      * Emits ``[PROFILE] <kernel_id>: <cycles> cycles`` on UART.
3. Ensure the FPGA bitstream is loaded (skipped if the cached hash
   matches). Flash + reset if it isn't (~minutes).
4. Upload the linked ELF over JTAG / xmodem.
5. Run each embedded kernel by writing its dispatch record to UART
   and reading back the profile line.
6. Parse per-kernel UART output into per-job ``ExecJobResult``.

Single-request path (:meth:`exec`) is a degenerate 1-item
:meth:`batch_exec` — no separate code path, just a wrapper. That
guarantees ``supports_batching = True`` also means "single-request
callers pay the batch-runner link cost per request" — deployments
where the RL loop only ever sends one job at a time should stick with
a coordinator or accept the overhead.

Sections marked ``# TODO(fpga-hw)`` are hardware-side placeholders —
they need real device / build-flow integration (openOCD, xmodem, a
JTAG bridge). Everything else (batch-runner-template layout,
bitstream cache key computation, UART parsing) is structurally
correct and testable against a fake board.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

from ..utils.exec_strategy import (
    BatchTooLargeError,
    ExecJob,
    ExecJobResult,
    ExecStrategy,
)


# UART line format the batch runner emits (must match
# RiscvZephyrBackend._PROFILE_LINE_RE). Kept as a class-level regex
# so tests can override the shape without patching the strategy.
_PROFILE_LINE_RE = re.compile(
    r"\[PROFILE\]\s+(\S+):\s+([0-9]+)\s*cycles", re.IGNORECASE,
)

# RISC-V linker signature for the classic "batched ELF too big" failure —
# ``.text`` / ``.rodata`` grew past what the PC-relative HI20/LO12 pair
# can address. Detected in the batch-runner link stderr so the strategy
# can raise :class:`BatchTooLargeError` and let the base class split-and-
# retry. Also catches the plainer ``relocation truncated`` wording emitted
# by ld for oversize sections. Deliberately loose — the exact wording
# varies across binutils versions.
BATCH_RELOC_ERROR_RE = re.compile(
    r"(R_RISCV_PCREL_HI20|relocation truncated|section .+ is too large)",
    re.IGNORECASE,
)


class FPGAExecError(Exception):
    """Domain error for :class:`FPGAExecStrategy` — bitstream flash
    failure, UART timeout, batch-runner link failure. Carries the
    reason on ``error_message`` for the exec server's generic handler."""

    def __init__(self, message: str):
        super().__init__(message)
        self.error_message = message


class FPGAExecStrategy(ExecStrategy):
    """RISC-V SoC on FPGA — batched exec via a shared bitstream + one
    boot ELF containing N problem instances."""

    name = "fpga"
    supports_batching = True

    def __init__(
        self,
        *,
        board_host: Optional[str] = None,
        bitstream_path: Optional[Path] = None,
        batch_runner_template: Optional[Path] = None,
    ):
        """
        Args:
            board_host: SSH target for the host machine physically
                connected to the FPGA (openOCD / JTAG proxy). May be
                ``None`` for a local FPGA on the same box as the exec
                server.
            bitstream_path: Path to the FPGA ``.bit`` file to flash.
                Required at request time; strategies without a
                configured bitstream fail their first batch fast.
            batch_runner_template: Directory holding the Zephyr batch-
                runner app template (source + CMakeLists). Required at
                request time — same fail-fast semantics.
        """
        self._board_host = board_host
        self._bitstream_path = (
            Path(bitstream_path) if bitstream_path is not None else None
        )
        self._batch_runner_template = (
            Path(batch_runner_template)
            if batch_runner_template is not None
            else None
        )
        self._logger = logging.getLogger("uvicorn")
        # Content hash of the currently-flashed bitstream, or ``None``
        # if the FPGA state is unknown. Used to skip the multi-minute
        # flash when the same bitstream would be loaded again. Guarded
        # by ``_flash_lock`` so parallel batches don't double-flash.
        self._loaded_bitstream_hash: Optional[str] = None
        self._flash_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # ExecStrategy contract
    # ------------------------------------------------------------------

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
    ):
        """Single-request path — degenerate 1-item batch.

        Returns the per-job stdout / stderr as ``(str, str)`` when
        ``n_runs == 1``, mirroring the pre-refactor return shape of
        ``exec_binary`` / ``upload_and_exec_binary``. For ``n_runs > 1``
        each rerun becomes an extra element in the returned lists.
        """
        job = ExecJob(
            binary_data=binary_data,
            filename=filename,
            args=args,
            env_vars=env_vars,
            prefix_command=prefix_command,
            n_runs=n_runs,
            timeout=timeout,
            kernel_files=kernel_files,
            profile=profile,
        )
        [result] = await self.batch_exec(worker_id=worker_id, jobs=[job])
        if not result.success:
            raise FPGAExecError(result.message or "FPGA exec failed")
        return result.stdout, result.stderr

    async def _batch_exec_impl(
        self,
        *,
        worker_id: int,
        jobs: list[ExecJob],
    ) -> list[ExecJobResult]:
        """Primary batched path: link, (maybe) flash, run, parse.

        The four-stage flow is decomposed into small methods so tests
        can substitute in a fake board (patch :meth:`_flash_bitstream`
        + :meth:`_upload_and_run`) without touching the rest.

        Raises :class:`BatchTooLargeError` when the link step's stderr
        matches :data:`BATCH_RELOC_ERROR_RE` — the base
        :meth:`ExecStrategy.batch_exec` catches this and splits + retries
        so callers see per-job success/failure instead of an all-or-
        nothing batch failure.
        """
        if self._bitstream_path is None:
            return _all_failed(jobs, "FPGAExecStrategy: bitstream_path not configured")
        if self._batch_runner_template is None:
            return _all_failed(jobs, "FPGAExecStrategy: batch_runner_template not configured")

        try:
            batch_runner_elf = await self._link_batch_runner(worker_id, jobs)
            await self._ensure_bitstream_loaded()
            uart_output = await self._upload_and_run(
                worker_id, batch_runner_elf, jobs,
            )
        except BatchTooLargeError:
            # Bubble up unchanged — the base class catches it and
            # subdivides. Log with the size for post-hoc tuning of
            # the client-side max_batch_size.
            self._logger.warning(
                f"[Worker {worker_id}]: batched ELF too large for "
                f"{len(jobs)} jobs — splitting"
            )
            raise
        except FPGAExecError as e:
            self._logger.error(f"[Worker {worker_id}]: batch failed: {e}")
            return _all_failed(jobs, e.error_message)
        except Exception as e:
            self._logger.exception(f"[Worker {worker_id}]: unexpected batch failure")
            return _all_failed(jobs, str(e))

        return self._parse_batch_output(jobs, uart_output)

    # ------------------------------------------------------------------
    # Optional lifespan hook (called from ``exec_server.lifespan``).
    # ------------------------------------------------------------------

    async def preflight(self) -> None:
        """Startup smoke test: verify openOCD / JTAG reachable + the
        batch-runner template exists on disk.

        Failures here surface as warnings (not fatals) so a server can
        start up and diagnose without the FPGA physically connected —
        the first real batch will error out with a specific message
        when the board is actually needed.
        """
        if self._batch_runner_template is not None and not self._batch_runner_template.exists():
            self._logger.warning(
                f"FPGAExecStrategy: batch_runner_template does not exist "
                f"at {self._batch_runner_template}. Batches will fail until "
                f"this path resolves."
            )
        if self._bitstream_path is not None and not self._bitstream_path.exists():
            self._logger.warning(
                f"FPGAExecStrategy: bitstream_path does not exist "
                f"at {self._bitstream_path}. Batches will fail until this "
                f"path resolves."
            )
        # TODO(fpga-hw): probe openOCD / JTAG bridge here; log board id.

    # ------------------------------------------------------------------
    # Internal steps — factored so tests can patch one without the rest
    # ------------------------------------------------------------------

    async def _link_batch_runner(
        self,
        worker_id: int,
        jobs: list[ExecJob],
    ) -> Path:
        """Link N kernel binaries into one boot ELF.

        Not yet wired — the real implementation copies
        :attr:`_batch_runner_template` into a per-worker scratch dir,
        drops each ``job.binary_data`` in as a static library or object
        file, updates the CMakeLists' ``target_link_libraries`` list,
        and runs ``west build``. Kernel-id ↔ index mapping (needed by
        :meth:`_parse_batch_output`) is derived from ``job.filename``.
        """
        # TODO(fpga-hw): materialize a proper linked ELF. For now, a
        # clear NotImplementedError makes the intent obvious in logs.
        raise FPGAExecError(
            f"FPGAExecStrategy._link_batch_runner not yet implemented "
            f"(worker {worker_id}, {len(jobs)} jobs). Wire the "
            f"west-build-with-embedded-kernels flow here."
        )

    async def _ensure_bitstream_loaded(self) -> None:
        """Flash the FPGA if the desired bitstream isn't already loaded.

        Guards on :attr:`_loaded_bitstream_hash` — the multi-minute
        flash only fires when the content hash changes. The lock
        serialises parallel batches so concurrent workers don't both
        try to flash at the same time.
        """
        assert self._bitstream_path is not None
        desired = self._bitstream_hash()
        if self._loaded_bitstream_hash == desired:
            self._logger.debug(
                f"Bitstream cache hit ({desired[:12]}...) — skipping flash"
            )
            return
        async with self._flash_lock:
            # Re-check under the lock — another coroutine may have
            # flashed while we waited.
            if self._loaded_bitstream_hash == desired:
                return
            self._logger.info(
                f"Flashing bitstream {self._bitstream_path} "
                f"(hash={desired[:12]}...) — this can take several minutes"
            )
            await self._flash_bitstream()
            self._loaded_bitstream_hash = desired

    def _bitstream_hash(self) -> str:
        """SHA-256 of the bitstream file, memoised per instance.

        Compact + collision-resistant. Reading the file each call is
        fine — bitstreams are large-ish (~100MB) but flashing takes
        minutes so a one-time streaming hash is negligible.
        """
        assert self._bitstream_path is not None
        h = hashlib.sha256()
        with self._bitstream_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    async def _flash_bitstream(self) -> None:
        """Actually flash the bitstream and reset the SoC.

        Not yet wired — the real implementation shells out to openOCD
        or an equivalent JTAG loader.
        """
        # TODO(fpga-hw): openOCD -f board.cfg -c "load_bitstream ..." etc.
        raise FPGAExecError(
            "FPGAExecStrategy._flash_bitstream not yet implemented — "
            "wire openOCD / vendor JTAG loader here."
        )

    async def _upload_and_run(
        self,
        worker_id: int,
        batch_runner_elf: Path,
        jobs: list[ExecJob],
    ) -> str:
        """Upload the ELF, kick off the batch runner, capture UART.

        Returns the raw UART output as a string — parsing is handled
        separately by :meth:`_parse_batch_output`.
        """
        # TODO(fpga-hw): xmodem / JTAG upload, then read UART until an
        # end-of-batch sentinel appears (batch runner emits e.g.
        # "[BATCH_DONE]" after the last kernel).
        raise FPGAExecError(
            f"FPGAExecStrategy._upload_and_run not yet implemented "
            f"(worker {worker_id}, elf={batch_runner_elf}, "
            f"n_jobs={len(jobs)})."
        )

    def _parse_batch_output(
        self,
        jobs: list[ExecJob],
        uart_output: str,
    ) -> list[ExecJobResult]:
        """Distribute the UART output back to per-job results.

        Each job's UART slice is the run of lines between its
        ``[PROFILE] <filename>`` marker and the next one; the profile
        line itself becomes ``stdout`` because that's what
        :class:`RiscvZephyrBackend.parse_profile` expects. Any job that
        didn't emit its expected marker gets an inline failure.
        """
        # Split the UART output into per-kernel slices by marker.
        markers = list(_PROFILE_LINE_RE.finditer(uart_output))
        found: dict[str, str] = {}
        for i, m in enumerate(markers):
            kernel_id = m.group(1)
            start = m.start()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(uart_output)
            found[kernel_id] = uart_output[start:end].strip()

        results: list[ExecJobResult] = []
        for job in jobs:
            key = _kernel_id_for(job)
            if key in found:
                results.append(ExecJobResult(
                    stdout=found[key],
                    stderr="",
                    success=True,
                ))
            else:
                results.append(ExecJobResult(
                    stdout="",
                    stderr=uart_output,
                    success=False,
                    message=(
                        f"batch runner did not emit [PROFILE] {key} — "
                        f"kernel likely faulted or timed out"
                    ),
                ))
        return results


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------


def _kernel_id_for(job: ExecJob) -> str:
    """Derive the kernel identifier used in UART markers from a job.

    The batch runner tags each dispatch with the ELF's declared symbol
    name — we use the binary's stem (path minus extension) as the
    canonical id. Clients that supply meaningful filenames (e.g.
    ``problem_42.o``) get meaningful markers; anonymous binaries get
    ``gpu_executable`` which is fine for single-job debug batches.
    """
    stem = Path(job.filename).stem if job.filename else "gpu_executable"
    return stem


def _all_failed(jobs: list[ExecJob], message: str) -> list[ExecJobResult]:
    """Convenience for whole-batch failure: every result gets the same
    error, keeping the per-job shape uniform."""
    return [
        ExecJobResult(stdout="", stderr="", success=False, message=message)
        for _ in jobs
    ]
