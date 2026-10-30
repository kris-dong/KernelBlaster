# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Curated OpenCL/Adreno-artifacts source (Item 2, Phase 2).

Layout:

  data/benchmark-opencl/<L1|L2|L3>/<problem_name>/
    - driver.c          (host-side OpenCL driver code)
    - kernel.cl         (OpenCL C kernel source)
    - reference.py      (optional, reference implementation for validation)

Fallback layout for sol-level tiers:

  data/kernelbench-opencl/<sol-level1|sol-level2>/<problem_name>/…

Phase 2 absorbed the loading logic, the tier canonicalization maps
(``SUBSET_TO_BENCHMARK_DIR``, ``RUN_FOLDER_PARENT_TO_BENCHMARK_DIR``),
and the two root-path helpers directly into this source. The legacy
``data.kernelbench_opencl`` module is now a back-compat shim that
re-exports from here — see that module for the deprecation contract.

Alternative filenames accepted (from the pre-refactor code):
  - ``kernel.cl`` OR ``kernel.opencl``
  - ``driver.c`` OR ``main.c``
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from .base import Problem, ProblemSource


# --- tier canonicalization ------------------------------------------------
# Subset flags (CLI) -> directory names under ``data/benchmark-opencl/<dir>/``.
# Also used as the "run-folder parent -> benchmark dir" mapping since the two
# maps had identical entries pre-Phase-2 (verified during the audit).
_SUBSET_TO_BENCHMARK_DIR: Mapping[str, str] = {
    "L1": "L1",
    "L2": "L2",
    "L3": "L3",
    "level1": "L1",
    "level2": "L2",
    "level3": "L3",
    "sol-level1": "sol-level1",
    "sol-level2": "sol-level2",
}


def _data_dir() -> Path:
    """``<repo>/data`` — parent of both OpenCL root candidates.

    ``__file__`` is at ``data/sources/kernelbench_opencl_source.py``, so
    ``parents[1]`` is ``data/`` (parents[0] = ``data/sources/``).
    """
    return Path(__file__).resolve().parents[1]


class KernelBenchOpenCLSource(ProblemSource):
    """Curated OpenCL/Adreno artifacts source (Phase 2 — self-contained)."""

    name = "kernelbench-opencl"

    # Class-level constant so shim callers (data.kernelbench_opencl) can
    # re-export without instantiating.
    TIER_MAP: Mapping[str, str] = _SUBSET_TO_BENCHMARK_DIR

    def __init__(
        self,
        *,
        root_dir: str | Path | None = None,
    ):
        # Primary root — kept as an instance attribute so tests/consumers
        # can override without touching module state.
        self._benchmark_root = (
            Path(root_dir) if root_dir is not None else self._default_benchmark_root()
        )

    # ---- root helpers (were module-level free functions pre-Phase-2) ----
    @staticmethod
    def _default_benchmark_root() -> Path:
        """Primary benchmark tree for OpenCL kernels."""
        return _data_dir() / "benchmark-opencl"

    @staticmethod
    def _port_root() -> Path:
        """Hand-curated OpenCL layout: ``data/kernelbench-opencl/<subset>/<problem>/``."""
        return _data_dir() / "kernelbench-opencl"

    # ---- ProblemSource contract ----
    def supports_backend(self, backend_name: str) -> bool:
        return backend_name == "opencl"

    def curated_root_for(self, backend_name: str) -> Optional[Path]:
        if backend_name != "opencl":
            return None
        return self._benchmark_root

    def artifact_filenames(self, backend_name: str) -> Mapping[str, str]:
        if backend_name != "opencl":
            return {}
        # Canonical names; the loader tolerates ``kernel.opencl`` / ``main.c``
        # as historical alternates.
        return {"driver": "driver.c", "kernel": "kernel.cl"}

    def tier_dir_for(self, backend_name: str, tier: str) -> str:
        """CLI subset -> on-disk directory name."""
        return _SUBSET_TO_BENCHMARK_DIR.get(tier, tier)

    def iter_problems(
        self,
        *,
        tier: Optional[str] = None,
        problem_numbers: Optional[list[int]] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Iterator[Problem]:
        if tier is not None and tier not in _SUBSET_TO_BENCHMARK_DIR:
            raise ValueError(f"Invalid level subset: {tier}")

        data_dir = self._benchmark_root
        if not data_dir.exists():
            # The pre-Phase-2 contract: for sol-level* tiers, tolerate a
            # missing primary root iff the port root has that tier.
            if tier in {"sol-level1", "sol-level2"} and self._port_root().is_dir():
                pass
            else:
                extra = ""
                if tier in {"sol-level1", "sol-level2"}:
                    extra = (
                        f"; for --subset {tier} without data/benchmark-opencl, create "
                        f"directory {self._port_root()} "
                        f"(e.g. …/{tier}/001_…/)"
                    )
                raise FileNotFoundError(
                    f"Dataset directory {data_dir} not found{extra}"
                )

        entries: list[dict[str, Any]] = []
        if tier is None:
            # No tier filter: walk all L1/L2/L3.
            for bench_level in ["L1", "L2", "L3"]:
                self._scan_level_dir(
                    data_dir / bench_level, bench_level,
                    problem_numbers, start, end, entries,
                )
        elif tier in {"sol-level1", "sol-level2"}:
            # SOL fallback pyramid: port_root -> dedicated -> fallback L*.
            fallback_bench_level = "L1" if tier == "sol-level1" else "L2"
            port_sol = self._port_root() / tier
            if port_sol.is_dir() and any(p.is_dir() for p in port_sol.iterdir()):
                self._scan_level_dir(
                    port_sol, tier, problem_numbers, start, end, entries,
                )
            if not entries:
                dedicated = data_dir / tier
                if dedicated.is_dir() and any(p.is_dir() for p in dedicated.iterdir()):
                    self._scan_level_dir(
                        dedicated, tier, problem_numbers, start, end, entries,
                    )
            if not entries:
                self._scan_level_dir(
                    data_dir / fallback_bench_level, tier,
                    problem_numbers, start, end, entries,
                )
        else:
            bench_level = _SUBSET_TO_BENCHMARK_DIR[tier]
            self._scan_level_dir(
                data_dir / bench_level, bench_level,
                problem_numbers, start, end, entries,
            )

        entries.sort(key=lambda e: e["id"])
        for e in entries:
            yield self._entry_to_problem(e)

    # ---- internals ----
    def _scan_level_dir(
        self,
        level_dir: Path,
        id_prefix: str,
        problem_numbers: Optional[list[int]],
        start: Optional[int],
        end: Optional[int],
        out: list[dict[str, Any]],
    ) -> None:
        if not level_dir.is_dir():
            return
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

            driver_c = problem_dir / "driver.c"
            kernel_cl = problem_dir / "kernel.cl"
            if not kernel_cl.exists():
                kernel_cl = problem_dir / "kernel.opencl"
            if not driver_c.exists():
                driver_c = problem_dir / "main.c"

            if not kernel_cl.exists():
                continue

            reference_py = problem_dir / "reference.py"
            entry: dict[str, Any] = {
                "id": f"{id_prefix}/{problem_dir.name}",
                "problem_name": problem_dir.name,
                "problem_num": num,
                "level": id_prefix,
                "driver_c_fp": str(driver_c) if driver_c.exists() else None,
                "kernel_cl_fp": str(kernel_cl),
            }
            if reference_py.exists():
                entry["reference_py_fp"] = str(reference_py)
            out.append(entry)

    def _entry_to_problem(self, entry: dict[str, Any]) -> Problem:
        artifacts: dict[str, Path] = {"kernel": Path(entry["kernel_cl_fp"])}
        if entry.get("driver_c_fp"):
            artifacts["driver"] = Path(entry["driver_c_fp"])
        if entry.get("reference_py_fp"):
            artifacts["reference_py"] = Path(entry["reference_py_fp"])

        return Problem(
            id=f"{self.name}:{entry['id']}",
            source=self.name,
            tier=str(entry["level"]),
            problem_num=int(entry["problem_num"]),
            problem_name=str(entry["problem_name"]),
            curated_artifacts=artifacts,
            backends_supported=frozenset({"opencl"}),
        )
