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
"""OpenCL/Adreno backend — facade over the existing remote-board flow."""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from .base import Backend, ProfileResult

if TYPE_CHECKING:
    from ..config import GPUType


# Fallback catalog: bottleneck -> [(technique_id, predicted_improvement_%), ...].
# Previously inline in ``opt_opencl_rl._try_add_default_optimizations``. The
# technique IDs reference entries in ``_OPENCL_TECHNIQUE_MAP`` below.
_OPENCL_DEFAULT_OPTIMIZATIONS: Mapping[str, list[tuple[str, float]]] = {
    "memory_bound": [
        ("1.1_coalesced_access", 20.0),
        ("2.1_local_memory_tiling", 25.0),
        ("1.1_vectorized_access", 15.0),
    ],
    "compute_bound": [
        ("3.1_work_per_item_increase", 30.0),
        ("3.3_mad_fma_usage", 20.0),
        ("3.4_half_precision", 25.0),
    ],
    "latency_bound": [
        ("1.1_occupancy_tuning", 35.0),
        ("2.2_work_group_size_tuning", 30.0),
        ("6.1_thread_coarsening", 25.0),
    ],
    "hybrid_bound": [
        ("2.1_local_memory_tiling", 40.0),
        ("2.1_register_tiling", 35.0),
        ("4.1_async_copy", 30.0),
    ],
}


# Adreno-tuned technique descriptions. Previously lived inline in
# ``opt_opencl_rl.generate_opencl_strategy_prompt``.
_OPENCL_TECHNIQUE_MAP: Mapping[str, str] = {
    "1.1_coalesced_access": "Ensure memory accesses are coalesced. Rearrange thread-to-data mapping so consecutive work-items access consecutive memory locations.",
    "1.1_occupancy_tuning": "Tune work-group size and local memory usage to maximise GPU occupancy on Adreno.",
    "1.1_vectorized_access": "Use OpenCL vector types (float4, int4) to widen memory transactions and improve bandwidth utilisation.",
    "2.1_local_memory_tiling": "Tile data into __local memory to exploit data reuse and reduce global memory traffic.",
    "2.1_register_tiling": "Increase work per work-item by computing multiple output elements, keeping intermediate results in private registers.",
    "2.2_work_group_size_tuning": "Experiment with different local work-group dimensions to match Adreno SP/TP architecture.",
    "2.2_texture_memory": "Use image/texture objects or read_only __global with __attribute__((nosvm)) for read-heavy data to exploit texture cache.",
    "2.3_data_layout_optimization": "Reorganise data layout (SoA vs AoS, padding) to improve cache line utilisation.",
    "3.1_work_per_item_increase": "Increase arithmetic intensity per work-item through loop unrolling or computing multiple outputs.",
    "3.2_barrier_reduction": "Minimise barrier() calls by restructuring algorithms to reduce synchronisation points.",
    "3.3_mad_fma_usage": "Use mad() / fma() and -cl-mad-enable to exploit fused multiply-add hardware.",
    "3.4_half_precision": "Use half/half4 types where precision allows to double throughput on Adreno ALUs.",
    "4.1_local_memory_bank_conflicts": "Pad or rearrange __local memory access patterns to avoid bank conflicts.",
    "4.1_async_copy": "Use async_work_group_copy for DMA-style transfers between global and local memory.",
    "6.1_thread_coarsening": "Assign multiple output elements to each work-item to amortise launch and synchronisation overhead.",
}


def _default_board_host() -> str:
    """Canonical source for the Adreno SSH target.

    Audit identified 5 hardcoded copies of this string across the repo
    (``utils/arguments.py``, ``opt_opencl_rl.py``, ``gpu_adreno.py``,
    ``compile_opencl.py``, ``run_single_kernelblaster_opencl.sh``). Phase 6
    will migrate those sites to read through ``OpenCLBackend.board_host``.
    """
    return os.getenv("KERNELBLASTER_ADRENO_BOARD_HOST", "root@192.0.2.201")


class OpenCLBackend(Backend):
    """Qualcomm Adreno + OpenCL event-time backend."""

    name = "opencl"
    kernel_ext = ".cl"
    driver_filename = "driver.c"

    def __init__(
        self,
        *,
        board_host: str | None = None,
        gpu: "GPUType | None" = None,
    ):
        self.board_host = board_host or _default_board_host()
        self.gpu = gpu

    # ---- assets ----
    @property
    def technique_map(self) -> Mapping[str, str]:
        return _OPENCL_TECHNIQUE_MAP

    @property
    def database_footer_path(self) -> Path:
        repo_root = Path(__file__).resolve().parents[3]
        return repo_root / "data" / "kernelblaster" / "optimization_database_footer_opencl.md"

    # ---- compile + run ----
    async def compile_and_run(
        self,
        main_filepath: Path,
        kernel_filepath: Path,
        gpu: "GPUType",
        timer,
        logger,
        *,
        timeout: int = 1200,
        num_runs: int = 5,
        passed_keyword: str | None = None,
        profile: bool = False,
        extra_files: list[str] | None = None,
        extra_args: str = "",
    ):
        """Facade over ``agents.utils.commands.compile_and_run_opencl``."""
        from ..agents.utils import compile_and_run_opencl

        return await compile_and_run_opencl(
            main_filepath,
            kernel_filepath,
            gpu,
            timer,
            logger,
            timeout=timeout,
            num_runs=num_runs,
            passed_keyword=passed_keyword,
            profile=profile,
            extra_files=extra_files,
            extra_args=extra_args,
        )

    # ---- profile parsing ----
    def parse_profile(self, raw_log: str) -> ProfileResult:
        """Extract per-kernel ms from ``[PROFILE] name: X ms`` markers."""
        # Avoid importing opt_opencl_rl here — it pulls the full agent surface.
        # Inline the parsing logic (kept in sync with parse_opencl_profile).
        import re

        timings: dict[str, float] = {}
        for m in re.finditer(r"\[PROFILE\]\s+(\S+):\s+([0-9]+(?:\.[0-9]+)?)\s*ms", raw_log):
            timings[m.group(1)] = float(m.group(2))

        total_ms = sum(timings.values()) if timings else 0.0
        return ProfileResult(
            total_time_ms=total_ms,
            per_kernel_ms=timings,
            raw_metrics={},
            raw_log=raw_log,
        )

    # ---- artifact naming ----
    def step_filename(self, trajectory: int, step: int, technique: str) -> str:
        return f"step_{step}_{technique}.cl"

    def best_filename(self) -> str:
        return "global_best_rl_optimization.cl"

    # ---- default optimizations ----
    def get_default_optimizations(self) -> Mapping[str, list[tuple[str, float]]]:
        return _OPENCL_DEFAULT_OPTIMIZATIONS

    # ---- primary metric ----
    @property
    def metric_name(self) -> str:
        return "ms"

    def format_metric(self, value, *, with_unit: bool = True) -> str:
        if isinstance(value, (int, float)):
            s = f"{float(value):.3f}"
        else:
            s = str(value)
        return f"{s} ms" if with_unit else s

    def extract_primary_metric(self, profile_result: ProfileResult) -> float:
        return float(profile_result.total_time_ms)

    # ---- LLM response handling ----
    def extract_code_from_response(self, response_text: str) -> str | None:
        """OpenCL: prefer ```c, fall back to ```opencl."""
        from ..agents.utils import extract_code_from_response as _extract
        code = _extract(response_text, tag="c")
        if code is None:
            code = _extract(response_text, tag="opencl")
        return code

    # ---- result artifact formatting ----
    _KERNEL_TIME_FOOTER_RE = None  # lazy-compiled in format_result_artifact

    def format_result_artifact(self, code: str, metric_value: float) -> str:
        """Append ``// Kernel time: <float> ms``; strip prior matching footer first."""
        import re

        if OpenCLBackend._KERNEL_TIME_FOOTER_RE is None:
            OpenCLBackend._KERNEL_TIME_FOOTER_RE = re.compile(
                r"\n*//\s*Kernel time:\s*[0-9]+(?:\.[0-9]+)?\s*ms\s*$",
                re.IGNORECASE | re.MULTILINE,
            )
        body = OpenCLBackend._KERNEL_TIME_FOOTER_RE.sub("", code).rstrip()
        return f"{body}\n\n// Kernel time: {metric_value:.3f} ms\n"
