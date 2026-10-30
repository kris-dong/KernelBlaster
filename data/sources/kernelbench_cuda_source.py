# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Curated CUDA-artifacts source (Item 2, Phase 3).

Layout:

  data/kernelbench-cuda/<level1|level2|level3>/<problem_name>/
    - driver.cpp        (host-side libtorch driver code)
    - init.cu           (CUDA kernel + launch wrapper — starting point for RL)

Curated CUDA problems ship a compilable driver + kernel pair per problem
and skip kgen. The tier taxonomy is ``level1``/``level2``/``level3`` only;
SOL-level artifacts under ``data/kernelbench-cuda/sol-level*`` were
historically rejected here (see the pre-Phase-3 assert in
``KernelBenchCUDADataset.__init__``). Phase 3 preserves that behaviour —
splitting out a ``SOLExecBenchCUDASource`` is a separate follow-up that
requires understanding the torch-reference SOL handling in
``kernelbench.py`` (Phase 4 territory).

The legacy ``data.kernelbench_cuda`` module is now a back-compat shim
over :class:`KernelBenchCUDASource` — see that module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from .base import Problem, ProblemSource


_ALLOWED_TIERS = frozenset({"level1", "level2", "level3"})


def _data_dir() -> Path:
    """``<repo>/data`` — locates the ``kernelbench-cuda`` root.

    ``__file__`` at ``data/sources/kernelbench_cuda_source.py``:
    ``parents[0]`` = ``data/sources``, ``parents[1]`` = ``data``.
    """
    return Path(__file__).resolve().parents[1]


class KernelBenchCUDASource(ProblemSource):
    """Curated CUDA artifacts source (Phase 3 — self-contained)."""

    name = "kernelbench-cuda"

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
        return {"driver": "driver.cpp", "kernel": "init.cu"}

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
                f"Invalid tier for kernelbench-cuda: {tier!r}. "
                f"Expected one of {sorted(_ALLOWED_TIERS)}."
            )
        if not self._root.exists():
            raise FileNotFoundError(f"Dataset directory {self._root} not found")

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

    # ---- internals ----
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

            driver_cpp = problem_dir / "driver.cpp"
            init_cu = problem_dir / "init.cu"
            if not driver_cpp.exists() or not init_cu.exists():
                continue

            out.append({
                "id": f"{tier}/{problem_dir.name}",
                "problem_name": problem_dir.name,
                "problem_num": num,
                "level": tier,
                "driver_cpp": driver_cpp,
                "init_cu": init_cu,
            })

    def _entry_to_problem(self, entry: dict[str, Any]) -> Problem:
        return Problem(
            id=f"{self.name}:{entry['id']}",
            source=self.name,
            tier=str(entry["level"]),
            problem_num=int(entry["problem_num"]),
            problem_name=str(entry["problem_name"]),
            curated_artifacts={
                "driver": entry["driver_cpp"],
                "kernel": entry["init_cu"],
            },
            backends_supported=frozenset({"cuda"}),
        )
