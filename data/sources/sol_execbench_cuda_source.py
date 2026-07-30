# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""SOL-ExecBench pre-generated CUDA source (Item 2, SOL split).

The SOL-ExecBench suite's problems are generated once (via kgen batch
runs) into curated CUDA artifacts so downstream RL / optimize flows
can skip the torch -> kgen -> compile step per iteration.

Layout: ``data/kernelbench-cuda/sol-level{1,2}/<problem>/``, each
containing at minimum:

  - ``driver.cpp`` (or ``test_driver.cpp``)   — host-side test driver
  - ``final_cuda.cu`` (or ``init.cu`` / ``kernel.cu``) — CUDA kernel

The multiple candidate filenames match
``scripts/run_opt_ncu_rl_optimized.py::collect_problems`` (which is the
primary consumer today). The first existing candidate wins for each
role, matching that script's ordering.

This source is a *sibling* to :class:`KernelBenchCUDASource` — same
on-disk root (``data/kernelbench-cuda``) but sol-level tiers. They
were split because SOL is a distinct benchmark suite from a different
upstream (with its own tier taxonomy + kernel-filename conventions).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from .base import Problem, ProblemSource


_ALLOWED_TIERS = frozenset({"sol-level1", "sol-level2"})

# Kernel-file candidates, tried in order. ``final_cuda.cu`` is the
# canonical kgen output; ``init.cu`` matches the pre-optimization RL
# starting point; ``kernel.cu`` is a legacy alias.
_KERNEL_CANDIDATES: tuple[str, ...] = ("final_cuda.cu", "init.cu", "kernel.cu")
_DRIVER_CANDIDATES: tuple[str, ...] = ("driver.cpp", "test_driver.cpp")


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1]


class SOLExecBenchCUDASource(ProblemSource):
    """Curated CUDA artifacts for the SOL-ExecBench suite."""

    name = "sol-execbench-cuda"

    def __init__(
        self,
        *,
        root_dir: str | Path | None = None,
    ):
        self._root = (
            Path(root_dir) if root_dir is not None else self._default_root()
        )

    @staticmethod
    def _default_root() -> Path:
        return _data_dir() / "kernelbench-cuda"

    # ---- ProblemSource contract ----
    def supports_backend(self, backend_name: str) -> bool:
        return backend_name == "cuda"

    def curated_root_for(self, backend_name: str) -> Optional[Path]:
        if backend_name != "cuda":
            return None
        return self._root

    def artifact_filenames(self, backend_name: str) -> Mapping[str, str]:
        if backend_name != "cuda":
            return {}
        # First-preference filenames. Loader still tolerates the
        # alternates listed above at scan time.
        return {"driver": _DRIVER_CANDIDATES[0], "kernel": _KERNEL_CANDIDATES[0]}

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
                f"Invalid tier for sol-execbench-cuda: {tier!r}. "
                f"Expected one of {sorted(_ALLOWED_TIERS)}."
            )
        if not self._root.exists():
            # No curated tree at all — treat as empty. Matches
            # ``KernelBenchCUDASource``'s FileNotFoundError, but sol
            # data being absent is a common condition on release
            # snapshots (the sol-level* subdirs aren't shipped),
            # so treat "root exists but sol subdirs missing" the same
            # as "root missing" — return empty iterator, no raise.
            return

        tiers = [tier] if tier else sorted(_ALLOWED_TIERS)
        entries: list[dict[str, Any]] = []
        for t in tiers:
            level_dir = self._root / t
            if not level_dir.exists():
                continue
            self._scan_level_dir(
                level_dir, t, problem_numbers, start, end, entries,
            )
        entries.sort(key=lambda e: e["id"])
        for e in entries:
            yield self._entry_to_problem(e)

    def _scan_level_dir(
        self,
        level_dir: Path,
        tier: str,
        problem_numbers: Optional[list[int]],
        start: Optional[int],
        end: Optional[int],
        out: list[dict[str, Any]],
    ) -> None:
        for problem_dir in sorted(p for p in level_dir.iterdir() if p.is_dir()):
            try:
                num = int(problem_dir.name.split("_", 1)[0])
            except Exception:
                continue

            if problem_numbers is not None and num not in problem_numbers:
                continue
            if start is not None and num < start:
                continue
            if end is not None and num > end:
                continue

            kernel = _first_existing(problem_dir, _KERNEL_CANDIDATES)
            driver = _first_existing(problem_dir, _DRIVER_CANDIDATES)
            if kernel is None or driver is None:
                continue

            out.append({
                "id": f"{tier}/{problem_dir.name}",
                "problem_name": problem_dir.name,
                "problem_num": num,
                "level": tier,
                "driver": driver,
                "kernel": kernel,
            })

    def _entry_to_problem(self, entry: dict[str, Any]) -> Problem:
        return Problem(
            id=f"{self.name}:{entry['id']}",
            source=self.name,
            tier=str(entry["level"]),
            problem_num=int(entry["problem_num"]),
            problem_name=str(entry["problem_name"]),
            curated_artifacts={
                "driver": entry["driver"],
                "kernel": entry["kernel"],
            },
            backends_supported=frozenset({"cuda"}),
        )


def _first_existing(directory: Path, candidates: tuple[str, ...]) -> Optional[Path]:
    for name in candidates:
        p = directory / name
        if p.exists():
            return p
    return None
