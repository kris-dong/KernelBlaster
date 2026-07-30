# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""KernelBench (PyTorch reference) source (Item 2, Phase 4).

Layout: ``data/kernelbench/kernelbench/**/*.py`` (the doubled directory
comes from the upstream repo layout). Also supports SOL-level tiers via
``data/kernelbench/sol-level1/`` and ``data/kernelbench/sol-level2/`` —
the two SOL tiers are handled inline here for now; the audit's proposed
``SOLExecBenchSource`` split is deferred to a follow-up.

This source has no on-disk curated artifacts — ``curated_artifacts`` is
empty and ``reference_code`` is the inlined PyTorch module text (with
precision snippet injected). Backends run kgen against
``reference_code`` to produce their driver+kernel pair.

Phase 4 pulled loading + precision injection out of
:class:`KernelBenchDataset`; the legacy module is now a back-compat
shim over this source. ``scripts/run_kgen_opencl.py::_inject_precision``
still exists as a script-local duplicate; Phase 6 deletes it in favour
of :meth:`KernelBenchSource.inject_precision`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from .base import Problem, ProblemSource


_ALLOWED_TIERS = frozenset({
    "level1", "level2", "level3", "sol-level1", "sol-level2",
})
_ALLOWED_PRECISION = frozenset({"fp32", "fp16", "bf16"})

# Levels considered "in scope" when no tier filter is given — historical
# behaviour from ``KernelBenchDataset._load_dataset`` (skip level3/level4
# without an explicit tier).
_DEFAULT_IN_SCOPE_LEVELS = frozenset({1, 2})

_DTYPE_SNIPPETS: Mapping[str, str] = {
    "fp32": "# Use fp32 datatype for all tensors\ntorch.set_default_dtype(torch.float32)",
    "fp16": "# Use fp16 datatype for all tensors\ntorch.set_default_dtype(torch.float16)",
    "bf16": "# Use bf16 datatype for all tensors\ntorch.set_default_dtype(torch.bfloat16)",
}


def _data_dir() -> Path:
    """``<repo>/data``."""
    return Path(__file__).resolve().parents[1]


class KernelBenchSource(ProblemSource):
    """PyTorch KernelBench reference source (backend-agnostic).

    Supports both CUDA and OpenCL — the graph invokes kgen to translate
    the torch reference into a backend-specific driver+kernel pair.
    """

    name = "kernelbench"

    def __init__(
        self,
        *,
        precision: str = "fp32",
        root_dir: str | Path | None = None,
    ):
        if precision not in _ALLOWED_PRECISION:
            raise ValueError(
                f"Invalid precision: {precision!r}. Expected one of {sorted(_ALLOWED_PRECISION)}."
            )
        self._precision = precision
        self._root = (
            Path(root_dir) if root_dir is not None else _data_dir() / "kernelbench"
        )

    # ---- ProblemSource contract ----
    def supports_backend(self, backend_name: str) -> bool:
        return backend_name in {"cuda", "opencl"}

    def curated_root_for(self, backend_name: str) -> Optional[Path]:
        # No curated artifacts — kgen produces them per problem.
        return None

    # ---- Precision-injection (Phase 4) ----
    @staticmethod
    def inject_precision(reference_code: str, precision: str) -> str:
        """Inject a ``torch.set_default_dtype`` snippet before ``class Model``.

        Robust version (matches the ``scripts/run_kgen_opencl.py`` copy;
        supersedes the CUDA-path variant in :class:`KernelBenchDataset`
        which had no idempotency guard and would corrupt the source if
        ``class Model`` was missing).

        - Returns input unchanged when ``precision`` isn't recognised
          (silent no-op — matches script behaviour for unknown
          precisions like ``"bf16"`` in the OpenCL loader).
        - Skips injection when the snippet is already present
          (idempotent — safe to call multiple times).
        - Falls back to prepending the snippet when ``class Model``
          isn't found (previously silently corrupted the source).
        """
        snippet = _DTYPE_SNIPPETS.get(precision)
        if not snippet:
            return reference_code
        if "set_default_dtype" in reference_code:
            return reference_code
        insertion_point = reference_code.find("class Model")
        if insertion_point == -1:
            return snippet + "\n\n" + reference_code
        return (
            reference_code[:insertion_point]
            + snippet
            + "\n\n"
            + reference_code[insertion_point:]
        )

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
                f"Invalid tier for kernelbench: {tier!r}. "
                f"Expected one of {sorted(_ALLOWED_TIERS)}."
            )
        if not self._root.exists():
            raise FileNotFoundError(f"Dataset directory {self._root} not found")

        is_sol = tier in {"sol-level1", "sol-level2"}
        if tier == "sol-level1":
            paths = self._root.glob("kernelbench/sol-level1/**/*.py")
        elif tier == "sol-level2":
            paths = self._root.glob("kernelbench/sol-level2/**/*.py")
        else:
            paths = self._root.glob("kernelbench/**/*.py")

        level_num = (
            int(tier.split("level")[1])
            if tier and tier.startswith("level")
            else None
        )

        entries: list[dict[str, Any]] = []
        for path in paths:
            # Skip SOL subtrees when no tier filter is given.
            rel_parts = path.relative_to(self._root).parts
            if (
                not is_sol
                and len(rel_parts) >= 2
                and rel_parts[1] in {"sol-level1", "sol-level2"}
            ):
                continue

            parts = list(filter(lambda f: f, path.stem.split("_")))
            try:
                num = int(parts[0])
            except (ValueError, IndexError):
                continue
            parent_stem = path.parent.stem

            if is_sol:
                expected_sol_level = tier  # sol-level1 / sol-level2
                if parent_stem != expected_sol_level:
                    continue
                new_name = f'{num:03d}_{"_".join(parts[1:])}'
                pid = f"{expected_sol_level}/{new_name}"
                level_tag = expected_sol_level
            else:
                try:
                    level = int(parent_stem.split("level")[1])
                except (ValueError, IndexError):
                    continue
                new_name = f'{num:03d}_{"_".join(parts[1:])}'
                pid = f"level{level}/{new_name}"
                level_tag = f"level{level}"
                if level_num is not None and level_num != level:
                    continue
                if level_num is None and level not in _DEFAULT_IN_SCOPE_LEVELS:
                    # historical: skip level3/level4 unless explicitly asked
                    continue

            if problem_numbers is not None and num not in problem_numbers:
                continue
            if start is not None and num < start:
                continue
            if end is not None and num > end:
                continue

            reference_code = self.inject_precision(path.read_text(), self._precision)
            entries.append({
                "id": pid,
                "problem_name": new_name,
                "problem_num": num,
                "level": level_tag,
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
