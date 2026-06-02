# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hardware backend abstraction — Phase 2 (adapter layer).

The ``Backend`` ABC is the unified contract for compile / run / profile / artifact
naming across CUDA (NVIDIA + NCU) and OpenCL (Adreno + event timing). Phase 2
implementations are pure *facades* over the existing free functions in
``kernelblaster.agents.utils`` and the FastAPI servers under
``kernelblaster.servers``; they preserve current behavior so consumers can
migrate one call site at a time.

Phase 3+ will progressively move consumers (database, RL agents, graph nodes)
to call methods on this ABC rather than the backend-specific free functions.

``ProfileResult`` is the key normalization: CUDA currently produces an
integer ``elapsed_cycles`` while OpenCL produces a float ``total_kernel_time_ms``.
The canonical metric across backends is ``ProfileResult.total_time_ms``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass
class ProfileResult:
    """Normalized profile data across backends.

    ``total_time_ms`` is the canonical metric used for speedup comparison
    (replaces the CUDA-cycles vs OpenCL-ms split). ``raw_metrics`` carries
    backend-specific extras (NCU cycles, occupancy, SM throughput; OpenCL
    per-kernel events) for prompts and dashboards.
    """

    total_time_ms: float
    per_kernel_ms: dict[str, float] = field(default_factory=dict)
    raw_metrics: dict[str, Any] = field(default_factory=dict)
    raw_log: str = ""


class Backend(ABC):
    """Contract every hardware backend satisfies.

    Subclasses must set the four class-level identity attributes (``name``,
    ``kernel_ext``, ``driver_filename``, ``compile_server_env``) and implement
    the abstract methods below.
    """

    # ---- identity (class attributes, set by subclasses) ----
    name: str = ""               # "cuda" | "opencl"
    kernel_ext: str = ""         # ".cu" | ".cl"
    driver_filename: str = ""    # "driver.cpp" | "driver.c"

    # ---- prompts / database assets ----
    @property
    @abstractmethod
    def technique_map(self) -> Mapping[str, str]:
        """Technique-ID -> prose description, consumed by RL prompts."""

    @property
    @abstractmethod
    def database_footer_path(self) -> Path:
        """Path to the optimization-database footer markdown for this backend."""

    # ---- compile + run ----
    # Note: Phase 2 keeps the per-backend signature shape (mirroring
    # ``compile_and_run_cu_file`` / ``compile_and_run_opencl``) so consumers
    # can migrate with a mechanical diff. Phase 4 will normalize on a single
    # signature.
    @abstractmethod
    async def compile_and_run(self, **kwargs):
        """Compile + execute. Returns ``(stdout_list, stderr_list, binary_path, success)``."""

    # ---- profile parsing (this IS unified now) ----
    @abstractmethod
    def parse_profile(self, raw_log: str) -> ProfileResult:
        """Extract a normalized ``ProfileResult`` from a backend-specific raw log."""

    # ---- artifact naming ----
    @abstractmethod
    def step_filename(self, trajectory: int, step: int, technique: str) -> str:
        """Filename used for a single RL step's generated kernel (e.g. ``step_3_tiling.cu``)."""

    @abstractmethod
    def best_filename(self) -> str:
        """Filename for the trajectory-best kernel (e.g. ``global_best_rl_optimization.cu``)."""

    # ---- LLM response handling ----
    @abstractmethod
    def extract_code_from_response(self, response_text: str) -> str | None:
        """Pull kernel code out of an LLM response, using the backend's expected
        code-block tags (CUDA: ``cpp``; OpenCL: ``c`` falling back to ``opencl``).

        Returns ``None`` if no code block matched — callers raise a
        ``FeedbackError`` so the LLM can be re-prompted.
        """

    # ---- result artifact formatting ----
    @abstractmethod
    def format_result_artifact(self, code: str, metric_value: float) -> str:
        """Append the backend's canonical performance footer to ``code``.

        Any existing footer of the same shape is stripped first so re-formatting
        a previously-annotated kernel doesn't produce double footers.

        CUDA footer:  ``// Elapsed Cycles: <int>``
        OpenCL footer: ``// Kernel time: <float> ms``
        """

    # ---- default optimizations (RL fallback catalog) ----
    @abstractmethod
    def get_default_optimizations(self) -> "Mapping[str, list[tuple[str, float]]]":
        """Per-backend fallback catalog: ``bottleneck -> [(technique_id, predicted_pct), ...]``.

        Used by the RL agents' ``_try_add_default_optimizations`` when the LLM
        produces a state name we have no recorded optimizations for. The
        technique IDs are backend-specific (CUDA uses generic names like
        ``memory_coalescing_optimization``; OpenCL uses entries from
        ``technique_map`` like ``1.1_coalesced_access``).
        """
