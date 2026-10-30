"""Client-side batching adapter for the exec server.

Bridges single-request callers (each RL rollout step calls
:func:`compile_and_run_riscv` once) to the exec server's ``/gpu/batch``
endpoint. When N concurrent rollouts each ``submit()`` a candidate ELF,
the underlying :class:`BatchCoordinator` buffers them and flushes to one
batched HTTP call on either a size trigger or a time trigger.

Amortises the per-boot overhead: for FireSim in particular, the
bitstream flash + Zephyr boot are minutes-scale fixed costs that would
otherwise fire once per candidate; batching N candidates behind one
boot recovers ~N× throughput.

The client is transport-thin — it just wires
:func:`agents.utils.commands.run_gpu_batch` into
:class:`BatchCoordinator` and translates the client-side
:class:`BatchExecJob` schema. All fusion / parsing / fallback lives on
the server side (see
:class:`servers.strategies.spike_exec.SpikeExecStrategy._batch_exec_impl`
and :class:`servers.strategies.fpga_exec.FPGAExecStrategy._batch_exec_impl`).
When the active strategy has ``supports_batching = False`` or fires
``BatchTooLargeError`` mid-batch, the server transparently subdivides
and the results the client sees are indistinguishable from per-item
execs (just with the coalescing latency).

Concurrency: one :class:`ExecBatchClient` per ``(GPUType, event loop)``
tuple. The module-level :func:`get_exec_batch_client` factory caches by
``GPUType`` since a single RL run pins to one target.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from src.kernelblaster.config import GPUType
from src.kernelblaster.agents.utils.batch_coordinator import (
    BatchCoordinator,
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_MAX_WAIT_MS,
)
from src.kernelblaster.agents.utils.commands import (
    BatchExecJob,
    BatchExecJobResult,
    run_gpu_batch,
)

logger = logging.getLogger(__name__)


class ExecBatchClient:
    """Coalescing client for ``/gpu/batch``.

    One instance per ``GPUType``. Callers use :meth:`submit_riscv` (or
    the generic :meth:`submit`) instead of talking to
    :func:`run_gpu_binary` directly; concurrent submits from N
    coroutines are transparently coalesced into fewer HTTP calls.

    The env knobs :envvar:`KERNELBLASTER_EXEC_BATCH_SIZE` and
    :envvar:`KERNELBLASTER_EXEC_BATCH_MAX_WAIT_MS` (parsed by
    :class:`BatchCoordinator`) tune size / latency. For a spike smoke,
    the default 32 / 1000ms window is fine; for FireSim RL runs with
    ``num_rl_iterations=16`` and multi-minute boot cost, raise both.
    """

    def __init__(
        self,
        gpu: GPUType,
        *,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_wait_ms: int = DEFAULT_MAX_WAIT_MS,
    ):
        self._gpu = gpu
        self._coordinator: BatchCoordinator[BatchExecJob, BatchExecJobResult] = (
            BatchCoordinator(
                submit_fn=self._submit_batch,
                max_batch_size=max_batch_size,
                max_wait_ms=max_wait_ms,
            )
        )

    async def submit_riscv(
        self,
        *,
        binary_path: Path,
        io_npz_path: Optional[Path],
        timeout: float,
        spike_args_str: str,
        kernel_id: str,
        n_runs: int = 1,
        source_c_path: Optional[Path] = None,
        base_stage_dir: Optional[Path] = None,
        mid: Optional[str] = None,
        target: str = "rvv",
        board: Optional[str] = None,
    ) -> BatchExecJobResult:
        """RISC-V convenience wrapper: builds the :class:`BatchExecJob`
        with the ``kernel_files=[io.npz]`` + ``args=<spike-comma-list>``
        shape the spike/firesim strategies expect.

        Same-problem batching kwargs (opt-in): ``source_c_path`` +
        ``base_stage_dir`` + ``mid``. When set, the strategy's fusion
        path (harness_shared_input via multi_link.sh) uses the source
        .c directly (bypassing the per-candidate ELF that was only a
        early-error-check) and stages against the shared model dir.
        Batches of >1 same-problem candidates fuse into one Zephyr
        boot per FireSim runworkload.
        """
        # kernel_files carries the io.npz for verify AND (opt-in) the
        # source .c so the fusion path can read the LLM's kernel without
        # re-parsing the compiled ELF's debug info. Strategy searches
        # kernel_files for `.npz` and `.c` respectively.
        kernel_files: list[str] = []
        if io_npz_path is not None:
            kernel_files.append(str(io_npz_path.resolve()))
        if source_c_path is not None:
            kernel_files.append(str(source_c_path.resolve()))

        # env_vars propagates the shared-input harness config to the
        # strategy's _link_batch_elf → multi_link.sh chain. All jobs
        # in a same-problem batch share these values; the fuse script
        # reads them from its subprocess env.
        env_vars: Optional[dict] = None
        if base_stage_dir is not None or mid is not None:
            env_vars = {}
            if base_stage_dir is not None:
                env_vars["KB_MULTI_MODEL_DIR"] = str(base_stage_dir.resolve())
            if mid is not None:
                env_vars["KB_MULTI_MID"] = mid
            env_vars["KB_MULTI_TARGET"] = target
            if board is not None:
                env_vars["KB_MULTI_BOARD"] = board

        job = BatchExecJob(
            binary_path=binary_path,
            args=spike_args_str,
            env_vars=env_vars,
            prefix_command=None,
            n_runs=n_runs,
            timeout=timeout,
            kernel_files=kernel_files or None,
            profile=False,
            job_name=kernel_id,
        )
        return await self.submit(job)

    async def submit(self, job: BatchExecJob) -> BatchExecJobResult:
        """Queue a raw :class:`BatchExecJob` and await its result.

        Callers that need per-candidate control over ``args`` /
        ``env_vars`` / ``prefix_command`` use this directly; the
        RISC-V convenience shape is :meth:`submit_riscv`.
        """
        return await self._coordinator.submit(job)

    async def flush(self) -> None:
        """Force-flush any buffered items now. Use at end-of-work if
        you can't wait for the time-trigger."""
        await self._coordinator.flush()

    async def _submit_batch(
        self, jobs: list[BatchExecJob]
    ) -> list[BatchExecJobResult]:
        """The coordinator's flush callback — forwards to
        :func:`run_gpu_batch`."""
        batch_name = (
            f"rl-batch-n={len(jobs)}"
            if len(jobs) > 1 else f"rl-single ({jobs[0].job_name or 'unnamed'})"
        )
        logger.info(
            "ExecBatchClient flush: gpu=%s size=%d names=%s",
            self._gpu.value if hasattr(self._gpu, "value") else str(self._gpu),
            len(jobs),
            ",".join(j.job_name or "?" for j in jobs[:8])
            + ("..." if len(jobs) > 8 else ""),
        )
        return await run_gpu_batch(jobs, self._gpu, batch_name=batch_name)


_CLIENTS: dict[tuple[GPUType, int], ExecBatchClient] = {}
_CLIENTS_LOCK = asyncio.Lock()


async def get_exec_batch_client(gpu: GPUType) -> ExecBatchClient:
    """Return the process-scoped singleton client for ``gpu``.

    Keyed by ``(gpu, event_loop_id)`` — asyncio pins each coordinator
    to one loop, so switching loops (e.g. across test harnesses) gets
    a fresh client rather than a cross-loop future crash. In normal
    single-loop runs the ``event_loop_id`` half is stable and behaves
    like a plain per-GPU singleton.
    """
    loop_id = id(asyncio.get_event_loop())
    key = (gpu, loop_id)
    async with _CLIENTS_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            client = ExecBatchClient(gpu)
            _CLIENTS[key] = client
            logger.info(
                "ExecBatchClient: created for gpu=%s loop=%d (batch_size=%s, "
                "wait_ms=%s)",
                gpu.value if hasattr(gpu, "value") else str(gpu),
                loop_id,
                os.getenv("KERNELBLASTER_EXEC_BATCH_SIZE", str(DEFAULT_MAX_BATCH_SIZE)),
                os.getenv(
                    "KERNELBLASTER_EXEC_BATCH_MAX_WAIT_MS",
                    str(DEFAULT_MAX_WAIT_MS),
                ),
            )
        return client


__all__ = [
    "ExecBatchClient",
    "get_exec_batch_client",
]
