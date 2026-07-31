# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Spike exec strategy — wraps modelblaster's ``spike_runner`` for the
RISC-V ISA simulator.

Terminology map to modelblaster (``zephyr-chipyard-sw/modelblaster``):

* One kernel run  ↔  ``python -m modelblaster.validation.spike_runner
  --elf <path> --io <io.npz>``
* Batched run     ↔  ``... --models name1,name2,... --io-paths <name>=<io.npz>``

The batching model here is fundamentally the same as :class:`FPGAExecStrategy`
— link N kernels into one boot ELF, run it once, parse per-kernel
tagged output — but:

* No bitstream flash (spike is a functional simulator).
* No JTAG upload (spike takes a filesystem path to the ELF).
* No board_host indirection (spike runs on the host that owns the exec server).

The **linking** step (fusing N Zephyr apps into one) is delegated to
:class:`servers.strategies.ZephyrCompileStrategy` — the actual west
build already knows how to produce a multi-model ELF via
``modelblaster/pipeline/generate_multi_main.py``. The exec strategy
just receives the finished ELF as ``binary_data`` (from the caller)
or, in batch mode, links N pre-compiled kernel drop-ins into one ELF
via its :meth:`_link_batch_elf` hook.

Output-line parsing:

* Wall cycles  →  ``MODELBLASTER_WALL_CYCLES: <int>`` (per model in
  multi-model mode; one line at end in single-model mode).
* Verify       →  ``MODELBLASTER_VERIFY: model=<name> ... status=<ok|fail>``.
* Per-kernel   →  ``MODELBLASTER_PROFILE_BEGIN`` / ``_END`` sections
  (mapped through the modelblaster IREE profile shape; kept as raw
  stdout for now — the RiscvZephyrBackend's ``parse_profile`` re-uses
  the ``[PROFILE] name: N cycles`` convention that already lives in
  modelblaster's benchmarks path).

Failure handling: the linker failure mode this strategy has to worry
about is :class:`BatchTooLargeError` when the fused multi-model ELF
exceeds ``R_RISCV_PCREL_HI20`` — same failure mode
:class:`FPGAExecStrategy` sees, handled the same way (base class
:meth:`ExecStrategy.batch_exec` split-and-retry).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import shutil
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


# modelblaster output markers — kept in sync with
# modelblaster/validation/runner_common.py._WALL_RE / _VERIFY_RE.
# Wire format (single-model + multi-model):
#     === MODELBLASTER_WALL_CYCLES === 12345678
#     === MODELBLASTER_WALL_CYCLES [kb_19_ReLU@fp32] === 12345678
#     === MODELBLASTER_VERIFY [kb_19_ReLU@fp32] === max_abs_err=0.0 max_rel_err=0.0 n=1024
# The optional bracket carries the tag as ``<model>@<quant>`` (or
# ``<model>@p<pool_size>`` for pool sweeps). Bucketing below strips the
# ``@`` suffix so a job whose filename stem is ``kb_19_ReLU`` matches
# ``kb_19_ReLU@fp32``.
_WALL_CYCLES_RE = re.compile(
    r"=== MODELBLASTER_WALL_CYCLES(?: \[(?P<name>[^\]]+)\])? === (?P<cycles>\d+)"
)
_VERIFY_RE = re.compile(
    r"=== MODELBLASTER_VERIFY(?: \[(?P<name>[^\]]+)\])? === "
    r"max_abs_err=(?P<abs>\S+) max_rel_err=(?P<rel>\S+) n=(?P<n>\d+)"
)
# Section brackets — used by _parse_batch_output to slice per-model blocks.
_OUTPUT_END_RE = re.compile(
    r"=== MODELBLASTER_OUTPUT_END(?: \[(?P<name>[^\]]+)\])? ===",
)


class SpikeError(Exception):
    """Domain error for :class:`SpikeExecStrategy` — spike missing on PATH,
    ELF missing/malformed, timeout, unparseable output. Wraps the
    underlying reason on ``error_message`` for the exec server's
    generic handler."""

    def __init__(self, message: str):
        super().__init__(message)
        self.error_message = message


class SpikeExecStrategy(ExecStrategy):
    """RISC-V ISA simulator target — runs kernels via ``spike``.

    Two shapes:
    * Single :meth:`exec`: one ELF, one run. Returns raw spike stdout
      (with WALL_CYCLES + VERIFY + PROFILE markers) so the caller's
      backend (:class:`RiscvZephyrBackend.parse_profile`) can pull out
      cycles.
    * Batched :meth:`_batch_exec_impl`: N kernel drop-ins fused into
      one multi-model ELF, one spike run, per-model output parsed
      back. ``batch_exec`` (base class) wraps this with split-and-
      retry so an oversize fused ELF doesn't kill the whole batch.
    """

    name = "spike"
    supports_batching = True

    def __init__(
        self,
        *,
        spike_binary: Optional[str] = None,
        modelblaster_root: Optional[Path] = None,
        multi_link_script: Optional[Path] = None,
        default_spike_args: tuple[str, ...] = (),
        default_timeout: float = 600.0,
    ):
        """
        Args:
            spike_binary: Path to the ``spike`` executable. ``None`` =
                look it up on PATH (matches ``spike_runner.find_spike``).
            modelblaster_root: Root of the modelblaster package (the
                repo dir containing ``modelblaster/validation/...``).
                Required for the ``python -m modelblaster.validation.
                spike_runner`` invocation; ``None`` falls back to the
                env var ``KERNELBLASTER_MODELBLASTER_ROOT``.
            multi_link_script: Optional path to a wrapper script that
                fuses N staged kernel dirs into one Zephyr ELF (see
                modelblaster's ``run_multi.sh``). Absent = single-item
                batches only (batch of >1 raises BatchTooLargeError
                with ``suggested_split=1`` so the base class devolves
                to per-item).
            default_spike_args: Extra ``--spike-arg`` values injected
                into every run (e.g. ``--isa=rv64gcv`` for RVV).
            default_timeout: Fallback timeout when the job doesn't
                specify one.
        """
        self._spike_binary = spike_binary
        self._modelblaster_root = (
            Path(modelblaster_root)
            if modelblaster_root
            else _resolve_modelblaster_root()
        )
        self._multi_link_script = (
            Path(multi_link_script) if multi_link_script is not None else None
        )
        self._default_spike_args = tuple(default_spike_args)
        self._default_timeout = default_timeout
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
        env_vars: Optional[dict] = None,             # ignored — spike env is fixed
        prefix_command: Optional[str] = None,        # e.g. "spike-vpi trace"; passed through
        n_runs: int = 1,
        timeout: float = 3600,
        kernel_files: Optional[list[str]] = None,    # ignored — kernels bundled in the ELF
        profile: bool = False,
    ):
        """Run one ELF through spike_runner.

        ``binary_data`` is the compiled ``zephyr.elf`` (from
        :class:`ZephyrCompileStrategy`). ``args`` is passed as
        ``--spike-arg`` values (comma-delimited; spike_runner accepts
        the flag repeatably).

        For ``n_runs > 1``, invokes spike n_runs times and returns
        list-of-strings for stdout / stderr — matches the pre-refactor
        return contract used by NCU/OpenCL callers.
        """
        if self._modelblaster_root is None:
            raise SpikeError(
                "SpikeExecStrategy: modelblaster_root not configured "
                "(set KERNELBLASTER_MODELBLASTER_ROOT env var or pass at init)"
            )

        # Materialise the ELF to a temp file — spike_runner takes a
        # filesystem path, not bytes on stdin.
        with tempfile.TemporaryDirectory(prefix=f"kb_spike_w{worker_id}_") as td:
            elf_path = Path(td) / (filename or "zephyr.elf")
            elf_path.write_bytes(binary_data)
            elf_path.chmod(0o755)
            # Runner needs an io.npz — the caller supplies it via
            # kernel_files[0] in single-mode. Absent = we synthesise one
            # by asking the runner to skip verification (see runner_common).
            io_npz = self._pick_io_npz(kernel_files)

            base_cmd = self._build_single_cmd(
                elf_path=elf_path,
                io_npz=io_npz,
                extra_spike_args=self._parse_spike_args(args),
                prefix_command=prefix_command,
                timeout=timeout,
            )

            if n_runs == 1:
                stdout, stderr = await self._run_cmd(base_cmd, timeout=timeout)
                return stdout, stderr

            outs: list[str] = []
            errs: list[str] = []
            for _ in range(n_runs):
                stdout, stderr = await self._run_cmd(base_cmd, timeout=timeout)
                outs.append(stdout)
                errs.append(stderr)
            return outs, errs

    async def _batch_exec_impl(
        self,
        *,
        worker_id: int,
        jobs: list[ExecJob],
    ) -> list[ExecJobResult]:
        """Fuse N Zephyr ELFs into one multi-model ELF, run once, parse
        per-model output.

        The fuse step delegates to :attr:`_multi_link_script` — this
        strategy stays transport-only. When the fuse script is not
        configured, callers get a single-item ``BatchTooLargeError``
        immediately so the base class splits down to per-item exec.

        Raises :class:`BatchTooLargeError` on the RISC-V reloc failure
        so the base ``batch_exec`` wraps this in split-and-retry.
        """
        if self._modelblaster_root is None:
            return _all_failed(jobs, "SpikeExecStrategy: modelblaster_root not configured")
        if len(jobs) == 1:
            # Degenerate batch — no reason to fuse. Handled BEFORE the
            # multi_link_script check because a batch of 1 needs no
            # fusion; the T1 coordinator can flush a lone job through
            # /gpu/batch and we should serve it as a single-item run
            # rather than rejecting for missing fusion plumbing.
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
                "SpikeExecStrategy: multi_link_script not configured "
                "(cannot fuse batch)",
                suggested_split=1,
            )

        # 1) Link N ELFs into one — hands off to the user-configured
        #    fuse script. Names the output ``fused.elf`` in a per-worker
        #    tmpdir so parallel workers don't collide.
        with tempfile.TemporaryDirectory(prefix=f"kb_spike_fuse_w{worker_id}_") as td:
            fused_elf = await self._link_batch_elf(
                worker_id=worker_id, td=Path(td), jobs=jobs,
            )
            # Same-problem batching (harness_shared_input): jobs carry
            # KB_MULTI_MID; emit ``<mid>@<tag>`` model names to match
            # the harness's marker format. Cross-problem path falls
            # back to plain per-job kernel ids.
            tags = [_kernel_id_for(j) for j in jobs]
            mid = None
            if jobs and jobs[0].env_vars:
                mid = jobs[0].env_vars.get("KB_MULTI_MID")
            if mid:
                model_names = [f"{mid}@{t}" for t in tags]
            else:
                model_names = tags
            io_paths = self._collect_io_paths(jobs, model_names)

            # Longest per-job timeout dominates.
            timeout = max((j.timeout for j in jobs), default=self._default_timeout)

            cmd = self._build_multi_cmd(
                elf_path=fused_elf,
                model_names=model_names,
                io_paths=io_paths,
                extra_spike_args=self._default_spike_args,
                timeout=timeout,
            )
            self._logger.info(
                f"[Worker {worker_id}]: spike batch of {len(jobs)}: "
                f"models={','.join(model_names)}"
            )
            stdout, stderr = await self._run_cmd(cmd, timeout=timeout)

        return self._parse_batch_output(jobs, model_names, stdout, stderr)

    # ------------------------------------------------------------------
    # Optional lifespan hook (called from ``exec_server.lifespan``).
    # ------------------------------------------------------------------

    async def preflight(self) -> None:
        """Startup smoke test: spike binary reachable, modelblaster
        root present. Non-fatal — real batches surface specific
        errors, but a warning here catches obvious misconfigs early."""
        if self._spike_binary and not Path(self._spike_binary).exists():
            self._logger.warning(
                f"SpikeExecStrategy: spike_binary {self._spike_binary} does "
                f"not exist. Runs will fail until this path resolves."
            )
        elif not self._spike_binary and shutil.which("spike") is None:
            self._logger.warning(
                "SpikeExecStrategy: spike not on PATH. Runs will fail until "
                "spike is installed or spike_binary is set."
            )
        if self._modelblaster_root and not (
            self._modelblaster_root / "modelblaster" / "validation"
        ).exists():
            self._logger.warning(
                f"SpikeExecStrategy: modelblaster_root {self._modelblaster_root} "
                f"does not contain modelblaster/validation/. Runs will fail."
            )

    # ------------------------------------------------------------------
    # Internal helpers — decomposed so tests can substitute individual
    # steps.
    # ------------------------------------------------------------------

    def _build_single_cmd(
        self,
        *,
        elf_path: Path,
        io_npz: Optional[Path],
        extra_spike_args: tuple[str, ...],
        prefix_command: Optional[str],
        timeout: float,
    ) -> list[str]:
        # ``python -m modelblaster.validation.spike_runner --elf ... --io ...
        #      --timeout ... --spike-arg=... [--spike-arg=... ...]``
        cmd: list[str] = []
        if prefix_command:
            cmd.extend(shlex.split(prefix_command))
        cmd += [
            sys.executable, "-m", "modelblaster.validation.spike_runner",
            "--elf", str(elf_path),
            "--timeout", str(timeout),
        ]
        if io_npz is not None:
            cmd += ["--io", str(io_npz)]
        if self._spike_binary:
            cmd += ["--spike", self._spike_binary]
        for a in (*self._default_spike_args, *extra_spike_args):
            cmd += [f"--spike-arg={a}"]
        return cmd

    def _build_multi_cmd(
        self,
        *,
        elf_path: Path,
        model_names: list[str],
        io_paths: Optional[dict[str, str]],
        extra_spike_args: tuple[str, ...],
        timeout: float,
    ) -> list[str]:
        cmd = [
            "python", "-m", "modelblaster.validation.spike_runner",
            "--elf", str(elf_path),
            "--models", ",".join(model_names),
            "--timeout", str(timeout),
        ]
        if io_paths:
            cmd += ["--io-paths", ",".join(f"{k}={v}" for k, v in io_paths.items())]
        if self._spike_binary:
            cmd += ["--spike", self._spike_binary]
        for a in extra_spike_args:
            cmd += [f"--spike-arg={a}"]
        return cmd

    async def _run_cmd(
        self,
        cmd: list[str],
        *,
        timeout: float,
    ) -> tuple[str, str]:
        env = os.environ.copy()
        # Ensure the modelblaster package is importable — same
        # PYTHONPATH prepend the modelblaster launcher scripts use.
        if self._modelblaster_root is not None:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{self._modelblaster_root}:{existing}"
                if existing else str(self._modelblaster_root)
            )
        self._logger.info(f"spike cmd: {' '.join(shlex.quote(c) for c in cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout + 30,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise SpikeError(
                f"spike_runner exceeded outer timeout={timeout + 30:.0f}s "
                f"(inner spike timeout={timeout:.0f}s)"
            )
        stdout = stdout_b.decode(errors="replace")
        stderr = stderr_b.decode(errors="replace")
        if proc.returncode != 0:
            raise SpikeError(
                f"spike_runner exited rc={proc.returncode}. "
                f"stderr tail: {stderr[-500:]!r}"
            )
        return stdout, stderr

    async def _link_batch_elf(
        self,
        *,
        worker_id: int,
        td: Path,
        jobs: list[ExecJob],
    ) -> Path:
        """Invoke ``multi_link_script`` to fuse N kernel drop-ins into
        one ``fused.elf``. Raises :class:`BatchTooLargeError` on the
        RISC-V reloc failure. The script contract:

        * Reads a manifest of ``<model_name> <staged_dir>`` lines from
          stdin.
        * Writes the fused ELF to ``$FUSED_OUT``.
        * Non-zero rc + reloc marker in stderr → BatchTooLargeError.
        """
        assert self._multi_link_script is not None
        model_names = [_kernel_id_for(j) for j in jobs]

        # Drop each job's kernel source (or ELF, for legacy per-ELF
        # fuse scripts) into a per-model staging dir. New same-problem
        # batching (kernel_files carries a .c) writes kernels.c
        # directly for harness_shared_input's west build to consume;
        # older ELF-fuse scripts still work via the binary_data
        # fallback.
        manifest_lines: list[str] = []
        for j, name in zip(jobs, model_names):
            staged = td / name
            staged.mkdir()
            src_c = self._pick_source_c(j.kernel_files) if hasattr(self, "_pick_source_c") else None
            if src_c is not None and src_c.exists():
                (staged / "kernels.c").write_bytes(src_c.read_bytes())
            else:
                (staged / "prebuilt.elf").write_bytes(j.binary_data)
            manifest_lines.append(f"{name} {staged}")

        fused_out = td / "fused.elf"
        env = os.environ.copy()
        env["FUSED_OUT"] = str(fused_out)
        env["FUSE_WORKER_ID"] = str(worker_id)
        # Propagate the shared-input harness env vars from the FIRST
        # job (all jobs in a same-problem batch share these). Same
        # convention as firesim_exec.
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
            input="\n".join(manifest_lines).encode()
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
            raise SpikeError(
                f"multi_link_script failed rc={proc.returncode}. "
                f"stderr tail: {stderr[-500:]}"
            )

        if not fused_out.exists():
            raise SpikeError(
                f"multi_link_script exited 0 but did not produce {fused_out}"
            )
        return fused_out

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_batch_output(
        self,
        jobs: list[ExecJob],
        model_names: list[str],
        stdout: str,
        stderr: str,
    ) -> list[ExecJobResult]:
        """Distribute spike stdout back to per-job results.

        spike_runner emits ``=== MODELBLASTER_WALL_CYCLES [<name>@<quant>] === N``
        (and matching VERIFY / OUTPUT_END lines) — one set per model in
        multi-model mode. We locate each model's WALL_CYCLES occurrence
        and return the surrounding lines (a small window between the
        prior WALL_CYCLES and this one) as ``stdout`` so the caller's
        ``parse_profile`` sees a coherent block including any inline
        ``[PROFILE]`` markers.

        Missing markers → per-job failure with a specific message.
        """
        # Find every WALL_CYCLES occurrence — record its char span + name.
        # WALL_CYCLES is emitted AFTER everything else for its model
        # (OUTPUT / PROFILE / VERIFY all print earlier), so slicing on
        # WALL-to-WALL boundaries captures each model's full block.
        matches: list[tuple[int, int, str]] = []   # (start, end, name)
        for m in _WALL_CYCLES_RE.finditer(stdout):
            tag = m.group("name") or ""     # e.g. "kb_19_ReLU@fp32"
            base = tag.split("@", 1)[0]     # → "kb_19_ReLU"
            matches.append((m.start(), m.end(), base))

        # Window for model k spans [end_of_k-1_wall .. end_of_k_wall];
        # the first model's window starts at 0.
        per_name: dict[str, str] = {}
        prev_wall_end = 0
        for start, end, base in matches:
            window = stdout[prev_wall_end:end]
            # Later windows overwrite earlier ones for a given name so
            # retries / reruns report the most recent metric.
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
                        f"spike output for model {name} missing "
                        f"MODELBLASTER_WALL_CYCLES marker"
                    ),
                ))
                continue
            results.append(ExecJobResult(
                stdout=block.strip(), stderr="", success=True,
            ))
        return results

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------

    def _pick_io_npz(
        self, kernel_files: Optional[list[str]],
    ) -> Optional[Path]:
        """Single-mode io.npz picker — first ``.npz`` in kernel_files, if any.

        Callers signal "no verify" by not supplying an .npz; spike_runner
        then just captures stdout without the golden compare (relies on
        the in-binary VERIFY marker instead). Matches the shape our
        RiscvZephyrBackend expects.
        """
        if not kernel_files:
            return None
        for f in kernel_files:
            if f.endswith(".npz"):
                p = Path(f)
                if p.exists():
                    return p
        return None

    def _pick_source_c(
        self, kernel_files: Optional[list[str]],
    ) -> Optional[Path]:
        """First ``.c`` in ``kernel_files`` — the LLM's kernel source
        for same-problem batching. Absent = legacy per-ELF batching."""
        if not kernel_files:
            return None
        for f in kernel_files:
            if f.endswith(".c"):
                p = Path(f)
                if p.exists():
                    return p
        return None

    def _collect_io_paths(
        self,
        jobs: list[ExecJob],
        model_names: list[str],
    ) -> Optional[dict[str, str]]:
        """Batch-mode ``--io-paths`` builder — glues each job to its
        io.npz. Job carries the .npz via kernel_files[0]; missing io
        means no compare (verify-only).
        """
        out: dict[str, str] = {}
        for j, name in zip(jobs, model_names):
            io = self._pick_io_npz(j.kernel_files)
            if io is not None:
                out[name] = str(io)
        return out or None

    def _parse_spike_args(self, args: str) -> tuple[str, ...]:
        """``args`` on the exec endpoint is a free-form string; treat
        comma-separated tokens as individual ``--spike-arg=`` values.
        Empty/whitespace yields no extras (matches CUDA/OpenCL callers
        that don't need spike-specific flags)."""
        if not args:
            return ()
        return tuple(a.strip() for a in args.split(",") if a.strip())


# ---------------------------------------------------------------------------
# Module-private helpers (shared shape with fpga_exec.py)
# ---------------------------------------------------------------------------


def _kernel_id_for(job: ExecJob) -> str:
    stem = Path(job.filename).stem if job.filename else "gpu_executable"
    return stem


def _all_failed(jobs: list[ExecJob], message: str) -> list[ExecJobResult]:
    return [
        ExecJobResult(stdout="", stderr="", success=False, message=message)
        for _ in jobs
    ]


def _resolve_modelblaster_root() -> Optional[Path]:
    """Env-var-first resolution — matches how ``_run_lib.sh`` finds the
    package. ``None`` = strategy can start (preflight logs a warning)
    but any batch/exec fails fast with a clear message."""
    v = os.getenv("KERNELBLASTER_MODELBLASTER_ROOT")
    if v:
        return Path(v)
    return None
