# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelBlaster partition problems (manifest ``mb-kb-problem/1``).

ModelBlaster's ``pipeline/kb_export.py`` writes one directory per graph
partition: a manifest, one editable source per kernel slot, a walker and a
self-contained driver. This source finds those directories under a root and
exposes each as a multi-slot :class:`Problem` (``slots`` set, ``driver`` and
``manifest`` artifacts, the manifest dict in ``metadata``).

    get_source("modelblaster", root="/path/to/exported/problems")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from .base import Problem, ProblemSource

FORMAT = "mb-kb-problem/1"


class ModelBlasterSource(ProblemSource):
    name = "modelblaster"

    def __init__(self, root: str | Path | None = None):
        import os
        root = root or os.getenv("KB_MODELBLASTER_PROBLEMS")
        if not root:
            raise ValueError("ModelBlasterSource needs root= (or KB_MODELBLASTER_PROBLEMS)")
        self.root = Path(root).resolve()

    def _dirs(self) -> list[Path]:
        if (self.root / "manifest.json").exists():
            return [self.root]
        return sorted(p.parent for p in self.root.rglob("manifest.json"))

    def iter_problems(self, *, tier: Optional[str] = None, problem_numbers: Optional[list[int]] = None,
                      start: Optional[int] = None, end: Optional[int] = None) -> Iterator[Problem]:
        for num, d in enumerate(self._dirs(), start=1):
            if problem_numbers is not None and num not in problem_numbers:
                continue
            if (start is not None and num < start) or (end is not None and num > end):
                continue
            man = json.loads((d / "manifest.json").read_text())
            if man.get("format") != FORMAT:
                continue
            t = man.get("ir_name", "modelblaster")
            if tier is not None and tier != t:
                continue
            yield Problem(
                id=f"{self.name}:{t}/{man['name']}", source=self.name, tier=t, problem_num=num,
                problem_name=man["name"],
                curated_artifacts={"driver": d / man["driver"], "manifest": d / "manifest.json"},
                metadata={"manifest": man, "root": str(d)},
                backends_supported=frozenset({man.get("target", "cuda")}),
                slots={s["id"]: d / s["file"] for s in man["slots"]},
            )

    def supports_backend(self, backend_name: str) -> bool:
        return backend_name == "cuda"

    def curated_root_for(self, backend_name: str) -> Optional[Path]:
        return self.root if backend_name == "cuda" else None

    def artifact_filenames(self, backend_name: str):
        return {"driver": "driver.cpp", "manifest": "manifest.json"}
