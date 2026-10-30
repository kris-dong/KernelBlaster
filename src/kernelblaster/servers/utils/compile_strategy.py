# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Backend-specific compile strategies for the unified compile server (Phase C).

The compile server hands each dequeued job off to a
:class:`CompileStrategy` that owns the backend-specific compilation
work (CMake+make for CUDA; gcc+scp-over-SSH for OpenCL). Selection is
per-request — every job carries its ``backend_name`` and the server
looks up the strategy from a registry.

Strategy contract:

  ``async def compile(worker_id, job_name, main_file, source_file,
                       backend_version, backend_flag, output_path,
                       artifacts_dir, board_host) -> str | None``

  - ``main_file`` — driver source (C or C++).
  - ``source_file`` — kernel source (`.cu` or `.cl`).
  - ``backend_version`` — SM version for CUDA; OpenCL version string for OpenCL.
  - ``backend_flag`` — persistent-artifacts bool (CUDA) or remote bool (OpenCL).
  - Returns the extra output path field: ``persistent_artifacts_dir``
    for CUDA (or ``None``), ``remote_binary_path`` for OpenCL.

  Raises the backend's domain error (``CompilationError`` /
  ``OpenCLCompilationError``) on failure.

Historical bodies for these strategies live in
:mod:`servers.compile` and :mod:`servers.compile_opencl` as
``_cuda_compile_job`` / ``_opencl_compile_job`` (both still in use by
their respective legacy servers). Phase D lands the unified server;
Phase E retires the legacy files.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class CompileStrategy(ABC):
    """Backend-specific compilation strategy."""

    name: str = ""            # "cuda" | "opencl"
    kernel_ext: str = ""      # ".cu" | ".cl"
    """Extension of the source-file kwarg — used for logging/telemetry.
    Matches ``backends.Backend.kernel_ext``."""

    @abstractmethod
    async def compile(
        self,
        *,
        worker_id: int,
        job_name: str,
        main_file: str,
        source_file: str,
        backend_version: str,
        backend_flag: bool,
        output_path: Path,
        artifacts_dir: Path,
        board_host: Optional[str] = None,
        debug: bool = False,
    ) -> Optional[str]:
        """Compile ``main_file`` + ``source_file`` into ``output_path``.

        Returns the backend-specific "extra" path:
          - CUDA: ``persistent_artifacts_dir`` (or ``None``).
          - OpenCL: ``remote_binary_path`` (or the local path, on
            local fallback).

        Writes the binary bytes to ``output_path`` and chmods it 0o755
        before returning.

        Raises the strategy's domain error on failure — the caller
        (queue worker) routes it onto the request future.
        """


_REGISTRY: dict[str, CompileStrategy] = {}


def register_compile_strategy(strategy: CompileStrategy) -> None:
    """Register a strategy by its ``name`` (single-source-of-truth
    for the unified compile server's ``?backend=`` dispatch)."""
    _REGISTRY[strategy.name] = strategy


def get_compile_strategy(backend_name: str) -> CompileStrategy:
    """Look up a strategy by backend name.

    Raises ``ValueError`` if unknown — the unified server converts
    this to HTTP 400.
    """
    try:
        return _REGISTRY[backend_name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none registered>"
        raise ValueError(
            f"Unknown compile backend: {backend_name!r}. Known: {known}"
        ) from None
