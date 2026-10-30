# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""SOL-ExecBench PyTorch reference source (Item 2, Phase 3-followup / SOL split).

The SOL-ExecBench benchmark suite as PyTorch reference kernels. Split
out of :class:`KernelBenchSource` because SOL is a distinct benchmark
suite (from a different upstream project) that just happens to share
the ``data/kernelbench/`` tree layout for historical reasons.

Layout: ``data/kernelbench/sol-level{1,2}/**/*.py``. Backends run kgen
against the torch reference to produce a driver+kernel pair; SOL
problems also ship pre-generated CUDA artifacts (see
:class:`SOLExecBenchCUDASource`) so the CUDA path can skip kgen.

Tier taxonomy: ``sol-level1`` (equivalent complexity to KernelBench
level1) and ``sol-level2`` (equivalent to level2). No ``sol-level3``
today — the frozen suite from the upstream SOL-ExecBench release
covers only these two levels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from ._precision import ALLOWED_PRECISION, inject_precision
from .base import Problem, ProblemSource


_ALLOWED_TIERS = frozenset({"sol-level1", "sol-level2"})


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1]


class SOLExecBenchTorchSource(ProblemSource):
    """PyTorch SOL-ExecBench reference source (backend-agnostic).

    Supports both CUDA and OpenCL through kgen. When curated CUDA
    artifacts exist for the same problem (see
    :class:`SOLExecBenchCUDASource`), consumers targeting CUDA should
    prefer those to skip the kgen step.
    """

    name = "sol-execbench"

    def __init__(
        self,
        *,
        precision: str = "fp32",
        root_dir: str | Path | None = None,
    ):
        if precision not in ALLOWED_PRECISION:
            raise ValueError(
                f"Invalid precision: {precision!r}. Expected one of {sorted(ALLOWED_PRECISION)}."
            )
        self._precision = precision
        # Shares the ``data/kernelbench`` root with :class:`KernelBenchSource`;
        # SOL subtrees live under ``sol-level{1,2}/`` in the same tree.
        self._root = (
            Path(root_dir) if root_dir is not None else _data_dir() / "kernelbench"
        )

    # ---- ProblemSource contract ----
    def supports_backend(self, backend_name: str) -> bool:
        return backend_name in {"cuda", "opencl"}

    def curated_root_for(self, backend_name: str) -> Optional[Path]:
        # No curated artifacts on THIS source — SOLExecBenchCUDASource
        # owns the curated .cu/.cpp path.
        return None

    inject_precision = staticmethod(inject_precision)

    # ---- Loading ----
    def iter_problems(
        self,
        *,
        tier: Optional[str] = None,
        problem_numbers: Optional[list[int]] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Iterator[Problem]:
        if tier is not None and tier not in _ALLOWED_TIERS:
            raise ValueError(
                f"Invalid tier for sol-execbench: {tier!r}. "
                f"Expected one of {sorted(_ALLOWED_TIERS)}."
            )
        if not self._root.exists():
            raise FileNotFoundError(f"Dataset directory {self._root} not found")

        tiers = [tier] if tier else sorted(_ALLOWED_TIERS)
        entries: list[dict[str, Any]] = []
        for t in tiers:
            for path in self._root.glob(f"kernelbench/{t}/**/*.py"):
                # Only accept files immediately under ``kernelbench/<t>/``
                # (matches the pre-refactor scanner's parent-stem check).
                if path.parent.stem != t:
                    continue

                parts = list(filter(lambda f: f, path.stem.split("_")))
                try:
                    num = int(parts[0])
                except (ValueError, IndexError):
                    continue

                if problem_numbers is not None and num not in problem_numbers:
                    continue
                if start is not None and num < start:
                    continue
                if end is not None and num > end:
                    continue

                new_name = f'{num:03d}_{"_".join(parts[1:])}'
                reference_code = inject_precision(path.read_text(), self._precision)
                entries.append({
                    "id": f"{t}/{new_name}",
                    "problem_name": new_name,
                    "problem_num": num,
                    "level": t,
                    "reference_code": reference_code,
                    "filepath": path,
                    "precision": self._precision,
                })

        entries.sort(key=lambda e: e["id"])
        for e in entries:
            yield self._entry_to_problem(e)

    def _entry_to_problem(self, entry: dict[str, Any]) -> Problem:
        return Problem(
            id=f"{self.name}:{entry['id']}",
            source=self.name,
            tier=str(entry["level"]),
            problem_num=int(entry["problem_num"]),
            problem_name=str(entry["problem_name"]),
            curated_artifacts={"reference_py": entry["filepath"]},
            reference_code=entry["reference_code"],
            metadata={"precision": entry["precision"]},
            backends_supported=frozenset({"cuda", "opencl"}),
        )
