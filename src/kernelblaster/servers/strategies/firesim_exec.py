# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""FireSim exec strategy — RL-loop entry into the FPGA FireSim install.

Symmetric with :class:`SpikeExecStrategy`, but the transport is
``python -m modelblaster.validation.firesim_runner`` instead of a
direct ``spike`` invocation. modelblaster's runner owns the FireSim
lifecycle (infrasetup → runworkload → parse WALL_CYCLES → kill), which
is what we already validated via the P5.16 smoke on 019_ReLU (U250
FPGA, ``*** PASSED *** after 314M target cycles``).

Design:

* ``FIRESIM_QUEUE=1`` mode is the default; the runner routes the
  submission through the on-host queue daemon, which owns the FPGA
  under an flock and serialises across concurrent users. That's the
  path proven to work end-to-end. ``queue_enabled=False`` skips the
  queue and drives firesim directly — dev / single-user only.

* Single-item exec today (T2). ``supports_batching = False``, which
  means the client-side :class:`ExecBatchClient` sees a fallback path:
  the base :meth:`ExecStrategy.batch_exec` loops :meth:`exec`
  sequentially. T3 flips this flag and implements ``_batch_exec_impl``
  to fuse N kernel ELFs into one Zephyr boot per firesim runworkload
  — that's where the multi-minute per-boot overhead finally amortises.

* All env plumbing (``FIRESIM_ROOT``, ``FIRESIM_ENV``,
  ``FIRESIM_QUEUE*``) is set explicitly on the child process. We do
  NOT inherit uncontrolled env from the exec-server parent — the
  strategy is the authoritative source of what the runner sees. This
  matches how the standalone smoke wrapper set them.

* Return shape matches :class:`SpikeExecStrategy.exec` verbatim:
  ``(stdout, stderr)`` with the modelblaster runner's own log (which
  includes the uartlog block wrapped in
  ``=== MODELBLASTER_RAW_FIRESIM_{BEGIN,END} ===``). Downstream
  :meth:`RiscvZephyrBackend.parse_profile` reads WALL_CYCLES + per-op
  PROFILE from that block, same as the spike path.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from ..utils.exec_strategy import (
    BatchTooLargeError,
    ExecJob,
    ExecJobResult,
    ExecStrategy,
)
# The multi-link fusion protocol + WALL_CYCLES parse regex are shared
# with SpikeExecStrategy — same modelblaster harness convention, same
# per-model marker shape. Rather than duplicate 100 lines of tricky
# regex + subprocess plumbing, import the pieces we can reuse. If we
# ever grow a third RISC-V exec target, promote these to a shared
# ``strategies/utils/riscv_batching.py``.
from .spike_exec import _WALL_CYCLES_RE, _kernel_id_for


class FireSimError(Exception):
    """Domain error for :class:`FireSimExecStrategy` — runner missing,
    FIRESIM_ROOT unset, timeout, non-zero rc. ``error_message`` is
    what the exec-server generic handler surfaces to the client."""

    def __init__(self, message: str):
        super().__init__(message)
        self.error_message = message


def _resolve_modelblaster_root() -> Path:
    """Same convention as SpikeExecStrategy — env var wins; error out
    clearly when unset, since the runner import needs a valid PYTHONPATH."""
    from_env = os.getenv("KERNELBLASTER_MODELBLASTER_ROOT", "")
    if not from_env:
        raise FireSimError(
            "FireSimExecStrategy: KERNELBLASTER_MODELBLASTER_ROOT unset "
            "and no modelblaster_root passed. Set the env var or "
            "--modelblaster-root on the exec server."
        )
    p = Path(from_env)
    if not (p / "modelblaster" / "validation" / "firesim_runner.py").exists():
        raise FireSimError(
            f"FireSimExecStrategy: {p} doesn't look like a modelblaster "
            f"root (missing modelblaster/validation/firesim_runner.py)."
        )
    return p


class FireSimExecStrategy(ExecStrategy):
    """FireSim FPGA target — runs kernels via
    ``modelblaster.validation.firesim_runner``.

    The runner internally handles the ``FIRESIM_QUEUE=1`` submission
    path (queue daemon owns kill → infrasetup → runworkload → parse
    → kill atomically), so this strategy is a thin transport wrapper.
    """

    name = "firesim"
    # T3: fuse N kernel drop-ins into one Zephyr boot per firesim
    # runworkload. Amortises the multi-minute infrasetup + boot cost
    # across the whole batch — the actual FPGA-overhead payoff.
    # ``_multi_link_script`` must be configured for a batch of >1 to
    # go through the fused path; absent = every batch fires
    # BatchTooLargeError and the base ExecStrategy falls back to
    # per-item, so this defaults to safe-but-slow when the fusion
    # plumbing isn't wired.
    supports_batching = True

    def __init__(
        self,
        *,
        firesim_root: Optional[str] = None,
        firesim_env: Optional[str] = None,
        modelblaster_root: Optional[Path] = None,
        multi_link_script: Optional[Path] = None,
        queue_enabled: bool = True,
        queue_root: Optional[str] = None,
        queue_bin: Optional[str] = None,
        queue_priority: int = 5,
        queue_timeout_s: Optional[int] = None,
        default_timeout: float = 900.0,
        python_bin: Optional[str] = None,
        extra_env: Optional[dict] = None,
    ):
        """
        Args:
            firesim_root: ``<chipyard>/sims/firesim``. ``None`` falls
                back to the ``FIRESIM_ROOT`` env var (which the runner
                also honours). One of the two MUST be set.
            firesim_env: The ``env.sh`` under the chipyard root.
                ``None`` falls back to ``FIRESIM_ENV``.
            modelblaster_root: Repo dir containing
                ``modelblaster/validation/firesim_runner.py``. ``None``
                = env var ``KERNELBLASTER_MODELBLASTER_ROOT``.
            queue_enabled: Route through the on-host firesim queue
                (recommended — serialises FPGA access). Default True.
            queue_root: ``FIRESIM_QUEUE_ROOT`` (e.g. dima's is at
                ``/scratch/dima/firesim_queue``). ``None`` = runner
                default (agustin's shared queue on garden).
            queue_bin: ``FIRESIM_QUEUE_BIN`` — the ``firesim-queue``
                CLI. ``None`` = runner default (agustin's install).
            queue_priority: ``FIRESIM_QUEUE_PRIORITY`` — 5 is the
                middle bucket.
            queue_timeout_s: ``FIRESIM_QUEUE_TIMEOUT`` — hard cap the
                daemon uses to SIGTERM a runaway workload. ``None``
                = let the workload run to natural completion (the
                default in the runner).
            default_timeout: Fallback overall wall-clock timeout for
                the runner subprocess when the job doesn't specify.
            python_bin: The Python interpreter to invoke
                ``modelblaster.validation.firesim_runner`` with. ``None``
                = ``sys.executable``, but on garden you almost certainly
                want the miniforge env
                (``.../tools/miniforge3/envs/zephyr/bin/python``) that
                already has aiohttp / boto3 / firesim's fabric.
            extra_env: Additional env vars to inject into the runner
                subprocess (e.g. bespoke ``FIRESIM_SLOT`` for a run
                cohort tag). Merged on top of the strategy's derived
                env.
        """
        self._firesim_root = firesim_root or os.getenv("FIRESIM_ROOT", "")
        self._firesim_env = firesim_env or os.getenv("FIRESIM_ENV", "")
        self._modelblaster_root = (
            Path(modelblaster_root)
            if modelblaster_root
            else _resolve_modelblaster_root()
        )
        self._multi_link_script = (
            Path(multi_link_script) if multi_link_script is not None else None
        )
        self._queue_enabled = queue_enabled
        self._queue_root = queue_root
        self._queue_bin = queue_bin
        self._queue_priority = queue_priority
        self._queue_timeout_s = queue_timeout_s
        self._default_timeout = default_timeout
        self._python_bin = python_bin or sys.executable
        self._extra_env = dict(extra_env or {})
        self._logger = logging.getLogger("uvicorn")

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
    ) -> tuple[list[str] | str, list[str] | str]:
        """Run one Zephyr ELF on FireSim via the modelblaster runner.

        ``kernel_files[0]`` is expected to be the ``io.npz`` for
        verify — same contract as :class:`SpikeExecStrategy.exec`.
        Absent → the runner falls back to in-binary
        ``MODELBLASTER_VERIFY``.

        ``n_runs > 1`` re-invokes the runner sequentially. Each
        firesim ``runworkload`` is minutes-scale, so callers should
        pass ``n_runs=1`` and rely on the batched path (T3) to
        amortise instead.
        """
        io_npz = self._pick_io_npz(kernel_files)
        with tempfile.TemporaryDirectory(
            prefix=f"kb_firesim_w{worker_id}_"
        ) as td:
            elf_path = Path(td) / (filename or "zephyr.elf")
            elf_path.write_bytes(binary_data)
            elf_path.chmod(0o755)

            cmd = self._build_cmd(
                elf_path=elf_path,
                io_npz=io_npz,
                extra_args=self._parse_firesim_args(args),
                prefix_command=prefix_command,
                timeout=timeout,
            )
            env = self._build_env(env_vars)

            if n_runs == 1:
                stdout, stderr = await self._run_cmd(
                    cmd, env=env, timeout=timeout,
                )
                return stdout, stderr

            outs: list[str] = []
            errs: list[str] = []
            for _ in range(n_runs):
                stdout, stderr = await self._run_cmd(
                    cmd, env=env, timeout=timeout,
                )
                outs.append(stdout)
                errs.append(stderr)
            return outs, errs

    # ------------------------------------------------------------------
    # Optional lifespan hook (called from ``exec_server.lifespan``).
    # ------------------------------------------------------------------

    async def preflight(self) -> None:
        """Startup smoke test: firesim_runner reachable, FIRESIM_ROOT
        set. Non-fatal — real runs surface specific errors, but a
        warning here catches obvious misconfigs early."""
        runner_py = (
            self._modelblaster_root
            / "modelblaster" / "validation" / "firesim_runner.py"
        )
        if not runner_py.exists():
            self._logger.warning(
                f"FireSimExecStrategy: firesim_runner.py not found at "
                f"{runner_py}. Runs will fail."
            )
        if not self._firesim_root and not os.getenv("FIRESIM_ROOT"):
            self._logger.warning(
                "FireSimExecStrategy: FIRESIM_ROOT is unset. Runs will "
                "fall through to the runner's built-in default, which "
                "may not match this host."
            )
        if self._queue_enabled and self._queue_bin:
            if not Path(self._queue_bin).exists():
                self._logger.warning(
                    f"FireSimExecStrategy: queue_bin {self._queue_bin} "
                    f"does not exist. Falling back to direct firesim."
                )
        if self._python_bin != sys.executable and not shutil.which(
            self._python_bin
        ):
            if not Path(self._python_bin).exists():
                self._logger.warning(
                    f"FireSimExecStrategy: python_bin {self._python_bin} "
                    f"not executable. Runs will fail — did the miniforge "
                    f"env path change?"
                )
        if self._multi_link_script is not None and not self._multi_link_script.exists():
            self._logger.warning(
                f"FireSimExecStrategy: multi_link_script "
                f"{self._multi_link_script} does not exist. Batches of >1 "
                f"will fall back to per-item runs (BatchTooLargeError → "
                f"split_and_retry)."
            )

    # ------------------------------------------------------------------
    # Batched exec — fuse N kernel ELFs into one Zephyr boot, submit ONE
    # firesim runworkload, parse per-model markers.
    # ------------------------------------------------------------------

    async def _batch_exec_impl(
        self,
        *,
        worker_id: int,
        jobs: list[ExecJob],
    ) -> list[ExecJobResult]:
        """Fuse N Zephyr ELFs into one multi-model boot ELF, run it
        through ``firesim_runner --models n1,n2,...``, parse per-model
        output.

        Contract identical to :meth:`SpikeExecStrategy._batch_exec_impl`
        — same manifest into ``_multi_link_script``, same WALL_CYCLES
        marker convention on the way back out. Only the transport
        differs (firesim_runner vs. spike_runner).

        Raises :class:`BatchTooLargeError` when
        ``_multi_link_script`` is unset OR the link step overflows
        RISC-V PC-relative relocations — the base
        :meth:`ExecStrategy.batch_exec` catches that and subdivides,
        so callers see per-job success/failure inline instead of an
        all-or-nothing batch failure.
        """
        if len(jobs) == 1:
            # Degenerate batch — one boot per candidate anyway, so
            # skip fusion entirely. Handled BEFORE the multi_link_script
            # check because a batch of 1 needs no fusion — the coordinator
            # from T1 flushes even a lone job through /gpu/batch, and it
            # would be wrong to reject those when the whole point of the
            # single-item path is that no fusion is needed.
            [j] = jobs
            try:
                stdout, stderr = await self.exec(
                    worker_id=worker_id,
                    binary_data=j.binary_data,
                    filename=j.filename,
                    args=j.args,
                    env_vars=j.env_vars,
                    prefix_command=j.prefix_command,
                    n_runs=j.n_runs,
                    timeout=j.timeout,
                    kernel_files=j.kernel_files,
                    profile=j.profile,
                )
                return [ExecJobResult(stdout=stdout, stderr=stderr, success=True)]
            except Exception as e:
                msg = getattr(e, "error_message", None) or str(e)
                return [ExecJobResult(success=False, message=msg)]
        if self._multi_link_script is None:
            # No multi-link plumbing for a batch of N>1 → force per-item
            # fallback via base ExecStrategy._split_and_retry.
            raise BatchTooLargeError(
                "FireSimExecStrategy: multi_link_script not configured "
                "(cannot fuse batch)",
                suggested_split=1,
            )

        with tempfile.TemporaryDirectory(
            prefix=f"kb_firesim_fuse_w{worker_id}_",
        ) as td:
            fused_elf = await self._link_batch_elf(
                worker_id=worker_id, td=Path(td), jobs=jobs,
            )
            # Model names sent to firesim_runner. When same-problem
            # batching is active (jobs carry KB_MULTI_MID env var),
            # harness_shared_input emits per-variant markers under
            # ``<mid>@<tag>`` — construct the model list to match so
            # the runner's ``--models`` parses cleanly. Otherwise fall
            # back to the plain kernel-id-per-job shape used by the
            # cross-problem path.
            tags = [_kernel_id_for(j) for j in jobs]
            mid = None
            if jobs and jobs[0].env_vars:
                mid = jobs[0].env_vars.get("KB_MULTI_MID")
            if mid:
                model_names = [f"{mid}@{t}" for t in tags]
            else:
                model_names = tags
            io_paths = self._collect_io_paths(jobs, model_names)

            # Longest per-job timeout dominates the fused run.
            timeout = max(
                (j.timeout for j in jobs), default=self._default_timeout,
            )

            cmd = self._build_multi_cmd(
                elf_path=fused_elf,
                model_names=model_names,
                io_paths=io_paths,
                extra_args=self._parse_firesim_args(""),
                timeout=timeout,
            )
            env = self._build_env(per_job_env=None)
            self._logger.info(
                f"[Worker {worker_id}]: firesim batch of {len(jobs)}: "
                f"models={','.join(model_names)}"
            )
            stdout, stderr = await self._run_cmd(
                cmd, env=env, timeout=timeout,
            )

        return self._parse_batch_output(jobs, model_names, stdout, stderr)

    async def _link_batch_elf(
        self,
        *,
        worker_id: int,
        td: Path,
        jobs: list[ExecJob],
    ) -> Path:
        """Invoke ``_multi_link_script`` (typically
        ``scripts/riscv/multi_link.sh``) to fuse N kernel drop-ins into
        one ``fused.elf`` via ``modelblaster/harness_shared_input``.

        Protocol:

        * Each ``jobs[i]``'s ``binary_data`` IS the LLM's kernels.c
          source bytes (not a compiled ELF). We write it as
          ``<staged_dir>/kernels.c`` for the fuse script.
        * Manifest lines ``<tag> <staged_dir>`` on stdin — one per job.
        * Shared-input env vars propagated from job.env_vars:
          ``KB_MULTI_MODEL_DIR`` (base per-problem generated dir),
          ``KB_MULTI_TARGET`` (backend), ``KB_MULTI_MID`` (mangled
          model name — used later for the ``--models <mid>@<tag>``
          spike/firesim_runner call), ``KB_MULTI_BOARD`` (Zephyr
          board id).
        * Writes fused ELF to ``$FUSED_OUT``.
        * Non-zero rc + reloc marker in stderr →
          :class:`BatchTooLargeError`.
        """
        assert self._multi_link_script is not None
        model_names = [_kernel_id_for(j) for j in jobs]

        manifest_lines: list[str] = []
        for j, name in zip(jobs, model_names):
            staged = td / name
            staged.mkdir()
            # Prefer the source .c from kernel_files (new same-problem
            # batching path) so the fuse script can build variants
            # against the shared base stage. Falls back to writing the
            # binary_data as-is when no .c is provided (legacy per-ELF
            # path — kept working for spike_exec parity).
            src_c = self._pick_source_c(j.kernel_files)
            if src_c is not None and src_c.exists():
                (staged / "kernels.c").write_bytes(src_c.read_bytes())
            else:
                (staged / "prebuilt.elf").write_bytes(j.binary_data)
            manifest_lines.append(f"{name} {staged}")

        fused_out = td / "fused.elf"
        env = os.environ.copy()
        env["FUSED_OUT"] = str(fused_out)
        env["FUSE_WORKER_ID"] = str(worker_id)
        env["FUSE_TARGET"] = "firesim"
        # Propagate the shared-input harness env vars from the FIRST
        # job (all jobs in a batch share a base problem). The fuse
        # script keys on these to invoke harness_shared_input's
        # west build with the right MODEL_DIR + backend + board.
        if jobs and jobs[0].env_vars:
            for k, v in jobs[0].env_vars.items():
                if k.startswith("KB_MULTI_"):
                    env[k] = str(v)

        proc = await asyncio.create_subprocess_exec(
            str(self._multi_link_script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout_b, stderr_b = await proc.communicate(
            input="\n".join(manifest_lines).encode(),
        )
        stderr = stderr_b.decode(errors="replace")

        if proc.returncode != 0:
            from .fpga_exec import BATCH_RELOC_ERROR_RE
            if BATCH_RELOC_ERROR_RE.search(stderr):
                raise BatchTooLargeError(
                    f"multi_link_script hit reloc limit at N={len(jobs)}. "
                    f"stderr tail: {stderr[-500:]}",
                    suggested_split=max(1, len(jobs) // 2),
                )
            raise FireSimError(
                f"multi_link_script failed rc={proc.returncode}. "
                f"stderr tail: {stderr[-500:]}"
            )

        if not fused_out.exists():
            raise FireSimError(
                f"multi_link_script exited 0 but did not produce {fused_out}"
            )
        return fused_out

    def _build_multi_cmd(
        self,
        *,
        elf_path: Path,
        model_names: list[str],
        io_paths: Optional[dict[str, str]],
        extra_args: tuple[str, ...],
        timeout: float,
    ) -> list[str]:
        """Multi-model firesim_runner invocation.

        ``firesim_runner`` accepts ``--models n1,n2,...`` and the
        parallel ``--io-paths n1=<npz1>,n2=<npz2>`` shape (see
        firesim_runner.py:807-816); the response format is the same
        WALL_CYCLES-per-model that spike_runner emits, so the parser
        below is shared.
        """
        cmd = [
            self._python_bin, "-m", "modelblaster.validation.firesim_runner",
            "--elf", str(elf_path),
            "--models", ",".join(model_names),
            "--timeout", str(int(timeout)),
        ]
        if io_paths:
            cmd += [
                "--io-paths",
                ",".join(f"{k}={v}" for k, v in io_paths.items()),
            ]
        cmd.extend(extra_args)
        return cmd

    def _collect_io_paths(
        self,
        jobs: list[ExecJob],
        model_names: list[str],
    ) -> Optional[dict[str, str]]:
        """Per-model io.npz mapping for ``--io-paths``. Absent io.npz
        means verify-only for that model (relies on the in-binary
        MODELBLASTER_VERIFY marker)."""
        out: dict[str, str] = {}
        for j, name in zip(jobs, model_names):
            io = self._pick_io_npz(j.kernel_files)
            if io is not None:
                out[name] = str(io)
        return out or None

    def _parse_batch_output(
        self,
        jobs: list[ExecJob],
        model_names: list[str],
        stdout: str,
        stderr: str,
    ) -> list[ExecJobResult]:
        """Distribute firesim_runner stdout back to per-job results.

        The runner emits ``=== MODELBLASTER_WALL_CYCLES [<name>@<quant>] === N``
        once per model (wrapped in the
        ``=== MODELBLASTER_RAW_FIRESIM_{BEGIN,END} ===`` markers the
        runner adds around the uartlog). Same shape as spike, so we
        share the regex and slice by WALL-to-WALL boundaries.

        Missing markers → per-job failure with a clear message.
        """
        matches: list[tuple[int, int, str]] = []
        for m in _WALL_CYCLES_RE.finditer(stdout):
            tag = m.group("name") or ""
            base = tag.split("@", 1)[0]
            matches.append((m.start(), m.end(), base))

        per_name: dict[str, str] = {}
        prev_wall_end = 0
        for start, end, base in matches:
            window = stdout[prev_wall_end:end]
            per_name[base] = window
            prev_wall_end = end

        results: list[ExecJobResult] = []
        for j, name in zip(jobs, model_names):
            block = per_name.get(name)
            if block is None:
                results.append(ExecJobResult(
                    stdout="", stderr=stderr,
                    success=False,
                    message=(
                        f"firesim output for model {name} missing "
                        f"MODELBLASTER_WALL_CYCLES marker"
                    ),
                ))
                continue
            results.append(ExecJobResult(
                stdout=block.strip(), stderr="", success=True,
            ))
        return results

    # ------------------------------------------------------------------
    # Internal helpers — decomposed so tests can substitute individual
    # steps.
    # ------------------------------------------------------------------

    def _build_cmd(
        self,
        *,
        elf_path: Path,
        io_npz: Optional[Path],
        extra_args: tuple[str, ...],
        prefix_command: Optional[str],
        timeout: float,
    ) -> list[str]:
        cmd: list[str] = []
        if prefix_command:
            cmd.extend(shlex.split(prefix_command))
        cmd += [
            self._python_bin, "-m", "modelblaster.validation.firesim_runner",
            "--elf", str(elf_path),
            "--timeout", str(int(timeout)),
        ]
        if io_npz is not None:
            cmd += ["--io", str(io_npz)]
        # Runner accepts extra pass-through args (e.g. --profile-out-root
        # for post-run reporting).
        cmd.extend(extra_args)
        return cmd

    def _build_env(self, per_job_env: Optional[dict]) -> dict:
        """Assemble the child process env from strategy config +
        per-job overrides. We start from the current process env so
        PATH / LD_LIBRARY_PATH / SSH_AUTH_SOCK propagate (fabric-to-
        localhost inside the queue needs SSH_AUTH_SOCK)."""
        env = os.environ.copy()
        if self._firesim_root:
            env["FIRESIM_ROOT"] = self._firesim_root
        if self._firesim_env:
            env["FIRESIM_ENV"] = self._firesim_env
        # modelblaster.validation.* needs its parent on PYTHONPATH.
        pypath = str(self._modelblaster_root)
        env["PYTHONPATH"] = (
            f"{pypath}:{env['PYTHONPATH']}"
            if env.get("PYTHONPATH") else pypath
        )
        if self._queue_enabled:
            env["FIRESIM_QUEUE"] = "1"
            if self._queue_root:
                env["FIRESIM_QUEUE_ROOT"] = self._queue_root
            if self._queue_bin:
                env["FIRESIM_QUEUE_BIN"] = self._queue_bin
            env["FIRESIM_QUEUE_PRIORITY"] = str(self._queue_priority)
            if self._queue_timeout_s is not None:
                env["FIRESIM_QUEUE_TIMEOUT"] = str(self._queue_timeout_s)
        else:
            env.pop("FIRESIM_QUEUE", None)

        env.update(self._extra_env)
        if per_job_env:
            env.update({str(k): str(v) for k, v in per_job_env.items()})
        return env

    def _pick_io_npz(
        self, kernel_files: Optional[list[str]]
    ) -> Optional[Path]:
        """The convention we share with SpikeExecStrategy: the first
        ``.npz`` in ``kernel_files`` is the modelblaster golden."""
        if not kernel_files:
            return None
        for kf in kernel_files:
            if kf.endswith(".npz") and Path(kf).exists():
                return Path(kf)
        return None

    def _pick_source_c(
        self, kernel_files: Optional[list[str]]
    ) -> Optional[Path]:
        """The first ``.c`` in ``kernel_files`` is the LLM-emitted
        kernel source (same-problem batching path). Absent = legacy
        per-ELF batching where ``binary_data`` carries a compiled ELF."""
        if not kernel_files:
            return None
        for kf in kernel_files:
            if kf.endswith(".c") and Path(kf).exists():
                return Path(kf)
        return None

    def _parse_firesim_args(self, args: str) -> tuple[str, ...]:
        """Parse the comma-list ``args`` field into extra tokens for
        the runner CLI. Empty → no extras. Mirrors
        ``SpikeExecStrategy._parse_spike_args`` but the tokens aren't
        wrapped in ``--spike-arg=`` — they're passed to firesim_runner
        as-is."""
        if not args:
            return ()
        return tuple(a.strip() for a in args.split(",") if a.strip())

    async def _run_cmd(
        self,
        cmd: list[str],
        *,
        env: dict,
        timeout: float,
    ) -> tuple[str, str]:
        """Fork the runner as a subprocess; capture stdout/stderr;
        surface non-zero rc as :class:`FireSimError`."""
        self._logger.info(
            f"firesim exec: {' '.join(shlex.quote(c) for c in cmd)}"
        )
        # Include some slack for post-run reporting (profile.csv write,
        # etc.) beyond the raw runworkload timeout the runner enforces.
        overall_timeout = float(timeout) + 300.0
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=overall_timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise FireSimError(
                f"firesim runner exceeded overall timeout "
                f"({overall_timeout:.0f}s)"
            )
        stdout = stdout_b.decode(errors="replace")
        stderr = stderr_b.decode(errors="replace")
        if proc.returncode != 0:
            # The runner may still have written useful output (the
            # uartlog markers we care about often appear before the
            # non-zero exit — chipyard's copy-back can raise after a
            # successful FPGA run). Surface both so backend.parse_profile
            # can decide if the profile is salvageable.
            raise FireSimError(
                f"firesim runner rc={proc.returncode}: "
                f"{stderr[-1500:] or stdout[-1500:] or '(no output)'}"
            )
        return stdout, stderr


__all__ = ["FireSimExecStrategy", "FireSimError"]
