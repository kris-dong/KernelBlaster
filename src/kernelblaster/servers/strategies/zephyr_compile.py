# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Zephyr compile strategy for the RISC-V target.

Compiles a KernelBlaster-generated kernel into a Zephyr ELF via the
existing **modelblaster** pipeline (``zephyr-chipyard-sw/modelblaster/
examples/_run_lib.sh``). KernelBlaster's job is to generate the C
kernel implementation; the pipeline's job is to produce the runnable
ELF around it.

**Integration point**: modelblaster's ``BACKEND=llm`` mode already
expects generated kernel source to appear at
``examples/<model>/<quant>/generated/<target>/kernels/*.c``. This
strategy drops the caller-supplied kernel there and shells out to a
wrapper script the user controls.

Command template placeholders (substituted with ``str.format``):

* ``{worker_id}`` — integer worker index (isolates parallel builds).
* ``{job_name}`` — job identifier (kernelbench basename like
  ``kb_19_ReLU``); flows into the modelblaster ``MODEL_NAME`` env var
  and thus into ``examples/<model>/...`` paths.
* ``{main_file}`` — path to the KernelBlaster-generated skeleton /
  driver stub. Rarely used for RISC-V (modelblaster's generate_skeleton
  owns the driver), but kept for signature-uniformity with CUDA/OpenCL.
* ``{source_file}`` — user kernel ``.c`` path (the RL-generated
  implementation). The template's script drops this into modelblaster's
  ``generated/<target>/kernels/`` and re-runs ``west build``.
* ``{output_path}`` — target path for the resulting ``zephyr.elf``.
* ``{artifacts_dir}`` — parent dir the caller wants build artefacts in.
* ``{board}`` — Zephyr board identifier (from ``backend_version`` —
  ``spike_riscv64`` for the simulator target, ``chipyard_riscv64/...``
  for FireSim).
* ``{link_as_lib}`` — ``"1"`` when ``backend_flag = True`` (produce a
  linkable object for :class:`SpikeExecStrategy` /
  :class:`FPGAExecStrategy` batch fusing), else ``"0"``.
* ``{modelblaster_root}`` — passed via env so the script can locate
  ``examples/_run_lib.sh``.

Failure detection: when the build fails with a RISC-V PC-relative
relocation error (``R_RISCV_PCREL_HI20``) — which happens when the
combined kernels/rodata section outgrows the ~2 GB reach of the HI20
imm — this strategy raises :class:`BatchTooLargeError`. Batched-exec
callers (``SpikeExecStrategy.batch_exec``) catch that and subdivide;
single-kernel builds propagate the error to the caller unchanged.

Deployments must set ``KERNELBLASTER_ZEPHYR_BUILD_CMD`` to a real
script or the default (``false ...``) fails immediately with a clear
message.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
from pathlib import Path
from typing import Optional

from ..utils.compile_strategy import CompileStrategy
from ..utils.exec_strategy import BatchTooLargeError
from .fpga_exec import BATCH_RELOC_ERROR_RE


DEFAULT_ZEPHYR_BUILD_CMD = os.getenv(
    "KERNELBLASTER_ZEPHYR_BUILD_CMD",
    "false 'ZephyrCompileStrategy: set KERNELBLASTER_ZEPHYR_BUILD_CMD to a build command template'",
)


class ZephyrBuildError(Exception):
    """Domain error raised by :class:`ZephyrCompileStrategy` on any
    non-zero return from the user-provided build command. Carries the
    truncated stderr on ``error_message`` for the compile server's
    generic error handler.

    :class:`BatchTooLargeError` is a distinct type (imported from the
    exec_strategy module) so batched-exec callers can catch it
    specifically for split-and-retry.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.error_message = message


class ZephyrCompileStrategy(CompileStrategy):
    """Compile a C kernel into a Zephyr application or static lib.

    The strategy is a thin process launcher; the real build logic lives
    in the user-provided ``KERNELBLASTER_ZEPHYR_BUILD_CMD`` template.
    That indirection means we don't hard-code the ``west build`` /
    ``cmake`` invocation shape — different SoCs / Zephyr forks / build
    infrastructures can plug in without any changes here.
    """

    name = "riscv"
    kernel_ext = ".c"

    def __init__(
        self,
        *,
        build_cmd_template: Optional[str] = None,
        default_board: str = "spike_riscv64",
        modelblaster_root: Optional[str] = None,
    ):
        self._build_cmd_template = build_cmd_template or DEFAULT_ZEPHYR_BUILD_CMD
        self._default_board = default_board
        self._modelblaster_root = (
            modelblaster_root
            or os.getenv("KERNELBLASTER_MODELBLASTER_ROOT", "")
        )
        self._logger = logging.getLogger("uvicorn")

    async def compile(
        self,
        *,
        worker_id: int,
        job_name: str,
        main_file: str,
        source_file: str,
        backend_version: str,          # Zephyr board id (e.g. "qemu_riscv32")
        backend_flag: bool,            # link_as_lib — True = .a for batch runner
        output_path: Path,
        artifacts_dir: Path,
        board_host: Optional[str] = None,  # accepted but unused; build is local
        debug: bool = False,
    ) -> Optional[str]:
        board = backend_version or self._default_board
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = self._build_cmd_template.format(
            worker_id=worker_id,
            job_name=job_name,
            main_file=main_file,
            source_file=source_file,
            output_path=str(output_path),
            artifacts_dir=str(artifacts_dir),
            board=board,
            link_as_lib="1" if backend_flag else "0",
            modelblaster_root=self._modelblaster_root,
        )
        env = os.environ.copy()
        if self._modelblaster_root:
            env["KERNELBLASTER_MODELBLASTER_ROOT"] = self._modelblaster_root
        self._logger.info(
            f"[Worker {worker_id}]: zephyr build job={job_name} board={board} "
            f"link_as_lib={backend_flag} cmd={cmd}"
        )

        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout_b, stderr_b = await proc.communicate()
        stdout = stdout_b.decode(errors="replace")
        stderr = stderr_b.decode(errors="replace")

        if proc.returncode != 0:
            # Distinct-type failure: reloc overflow means the fused
            # ELF is oversize. Batched callers catch this specifically
            # and subdivide; single builds see it as a normal build
            # error with a clear message.
            combined = f"{stdout}\n{stderr}"
            if BATCH_RELOC_ERROR_RE.search(combined):
                raise BatchTooLargeError(
                    f"Zephyr link failed for {job_name} — RISC-V "
                    f"PC-relative relocation overflow (fused .text/.rodata "
                    f"too large). stderr tail: {stderr[-500:]}",
                )
            msg = (
                f"Zephyr build failed for {job_name} (rc={proc.returncode}). "
                f"cmd={shlex.quote(cmd)}\n"
                f"stdout tail: {stdout[-1000:]}\n"
                f"stderr tail: {stderr[-1000:]}"
            )
            self._logger.error(msg)
            raise ZephyrBuildError(msg)

        if not output_path.exists():
            msg = (
                f"Zephyr build succeeded (rc=0) for {job_name} but "
                f"output_path {output_path} was not created. Fix the "
                f"KERNELBLASTER_ZEPHYR_BUILD_CMD template to write the "
                f"final artefact to {{output_path}}."
            )
            self._logger.error(msg)
            raise ZephyrBuildError(msg)

        output_path.chmod(0o755)
        self._logger.info(
            f"[Worker {worker_id}]: zephyr build success: "
            f"{job_name} -> {output_path} ({output_path.stat().st_size} bytes)"
        )

        # There's no "remote binary path" for Zephyr — the exec server
        # side handles all remote transfers via FPGAExecStrategy. Return
        # the local output path as a diagnostic breadcrumb; matches
        # OpenCLCompileStrategy's return convention.
        return str(output_path)
