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
"""CUDA backend — facade over the existing NCU+nsys flow.

Phase 2 adapter: methods delegate to ``kernelblaster.agents.utils`` and
``kernelblaster.servers`` to preserve current behavior. Later phases will
migrate consumers off the free functions onto this surface.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from .base import Backend, ProfileResult

if TYPE_CHECKING:
    from ..config import GPUType


# Fallback catalog: bottleneck -> [(technique_id, predicted_improvement_%), ...].
# Previously inline in ``opt_ncu_rl._try_add_default_optimizations``.
_CUDA_DEFAULT_OPTIMIZATIONS: Mapping[str, list[tuple[str, float]]] = {
    "memory_bound": [
        ("memory_coalescing_optimization", 20.0),
        ("shared_memory_tiling", 25.0),
        ("vectorized_processing", 15.0),
    ],
    "compute_bound": [
        ("instruction_level_parallelism", 30.0),
        ("fast_math_optimization", 20.0),
        ("vectorized_operations", 25.0),
    ],
    "latency_bound": [
        ("occupancy_optimization", 35.0),
        ("register_pressure_reduction", 30.0),
        ("work_per_thread_increase", 25.0),
    ],
    "hybrid_bound": [
        ("memory_compute_overlap", 40.0),
        ("algorithmic_optimization", 35.0),
        ("adaptive_tiling", 30.0),
    ],
}


# Technique IDs are conceptually shared with OpenCL but the prose is CUDA-tuned
# (mentions of warps, tensor cores, shared memory banks, etc.). This map
# previously lived inline in ``opt_ncu_rl.generate_strategy_guided_prompt``.
_CUDA_TECHNIQUE_MAP: Mapping[str, str] = {
    "1.1_coalesced_access": "Focus on ensuring memory accesses are coalesced. Rearrange thread-to-data mapping so consecutive threads access consecutive memory locations.",
    "1.1_occupancy_tuning": "Optimize occupancy by tuning threads per block and shared memory usage. Aim for high occupancy while avoiding resource bottlenecks.",
    "1.1_register_optimization": "Reduce register pressure by minimizing local variables and using shared memory for frequently accessed data.",
    "1.1_shared_memory_optimization": "Optimize shared memory usage by reducing bank conflicts and improving access patterns.",
    "1.1_block_size_tuning": "Experiment with different block sizes to maximize occupancy and resource utilization.",
    "2.1_shared_memory_tiling": "Implement tiling using shared memory to reduce global memory accesses through data reuse.",
    "2.1_tensor_core_utilization": "Modify the code to use tensor cores by ensuring proper data types (half, bfloat16) and matrix sizes.",
    "2.2_thread_data_mapping": "Rearrange how threads map to data elements to improve memory access patterns and reduce conflicts.",
    "2.2_functional_unit_optimization": "Balance the workload across different functional units (ALU, SFU, memory units).",
    "2.2_instruction_mix_optimization": "Optimize the instruction mix to better utilize available compute resources.",
    "2.3_data_layout_optimization": "Reorganize data layout in memory to improve cache utilization and memory bandwidth.",
    "2.3_constant_cache_usage": "Move read-only data to constant memory to leverage the constant cache.",
    "3.1_increase_thread_count": "Launch more threads by increasing grid size or using multiple kernels.",
    "3.1_thread_work_remapping": "Remap thread work assignment to reduce warp divergence.",
    "3.2_work_per_thread_increase": "Increase work per thread through loop unrolling or processing multiple elements per thread.",
    "3.2_data_layout_for_divergence": "Restructure data layout to minimize control flow divergence.",
    "3.3_vector_load_usage": "Use vector loads (float2, float4) to process multiple elements efficiently.",
    "3.4_maximum_occupancy_tuning": "Fine-tune launch parameters to achieve maximum theoretical occupancy.",
    "4.1_shared_memory_caching": "Cache frequently accessed global memory data in shared memory.",
    "4.1_shared_memory_bank_conflict_removal": "Eliminate shared memory bank conflicts by padding or restructuring access patterns.",
    "4.1_register_tiling": "Use register tiling to keep frequently accessed data in registers.",
    "6.1_thread_coarsening": "Assign multiple work items to each thread to amortize parallelization overhead.",
}


class CUDABackend(Backend):
    """NVIDIA + NCU/nsys backend."""

    name = "cuda"
    kernel_ext = ".cu"
    driver_filename = "driver.cpp"

    def __init__(self, *, gpu: "GPUType | None" = None):
        # GPU type is preserved for callers that still need SM version etc.
        # Once Phase 4 lands, callers will get this through the Backend rather
        # than handling GPUType directly.
        self.gpu = gpu

    # ---- assets ----
    @property
    def technique_map(self) -> Mapping[str, str]:
        return _CUDA_TECHNIQUE_MAP

    @property
    def database_footer_path(self) -> Path:
        repo_root = Path(__file__).resolve().parents[3]
        return repo_root / "data" / "kernelblaster" / "optimization_database_footer.md"

    # ---- compile + run ----
    async def compile_and_run(
        self,
        main_filepath: Path,
        cuda_filepath: Path,
        gpu: "GPUType",
        timer,
        logger,
        *,
        persistent_artifacts: bool = False,
        timeout: int = 1200,
        num_runs: int = 5,
        passed_keyword: str | None = None,
        prefix_command: str | None = None,
    ):
        """Facade over ``agents.utils.commands.compile_and_run_cu_file``.

        Signature mirrors the existing free function to keep the Phase 2
        diff at consumer sites mechanical.
        """
        # Lazy import: agents.utils brings in heavy deps (numpy, pandas) that
        # backend consumers shouldn't pay for at import time.
        from ..agents.utils import compile_and_run_cu_file

        return await compile_and_run_cu_file(
            main_filepath,
            cuda_filepath,
            gpu,
            timer,
            logger,
            persistent_artifacts=persistent_artifacts,
            timeout=timeout,
            num_runs=num_runs,
            passed_keyword=passed_keyword,
            prefix_command=prefix_command,
        )

    # ---- profile parsing ----
    def parse_profile(self, raw_log: str) -> ProfileResult:
        """Extract elapsed cycles from an NCU log.

        Raises ``RuntimeError`` if cycles cannot be parsed (preserves the
        existing ``get_elapsed_cycles_ncu_log`` contract — callers depend on
        the failure being loud).

        Phase 2 leaves ``total_time_ms`` as 0.0 — cycles aren't trivially
        convertible to wall time without the GPU's clock frequency. Phase 3
        (database collapse) will introduce a real ms estimate from nsys.
        """
        from ..agents.utils import get_elapsed_cycles_ncu_log

        cycles = get_elapsed_cycles_ncu_log(raw_log)
        return ProfileResult(
            total_time_ms=0.0,
            per_kernel_ms={},
            raw_metrics={"elapsed_cycles": cycles},
            raw_log=raw_log,
        )

    # ---- artifact naming ----
    def step_filename(self, trajectory: int, step: int, technique: str) -> str:
        return f"step_{step}_{technique}.cu"

    def best_filename(self) -> str:
        return "global_best_rl_optimization.cu"

    # ---- default optimizations ----
    def get_default_optimizations(self) -> Mapping[str, list[tuple[str, float]]]:
        return _CUDA_DEFAULT_OPTIMIZATIONS

    # ---- primary metric ----
    @property
    def metric_name(self) -> str:
        return "cycles"

    def format_metric(self, value, *, with_unit: bool = True) -> str:
        if isinstance(value, (int, float)):
            s = f"{int(value)}"
        else:
            s = str(value)
        return f"{s} cycles" if with_unit else s

    def extract_primary_metric(self, profile_result: ProfileResult) -> float:
        return float(profile_result.raw_metrics.get("elapsed_cycles", 0))

    # ---- State derivation glue (Phase 4f.3a) ----
    def derive_metrics_for_state(self, profile_result: ProfileResult) -> dict:
        """Parse NCU Speed-Of-Light metrics from the raw NCU log."""
        from ..agents.utils.parsing import parse_ncu_metrics
        return parse_ncu_metrics(profile_result.raw_log)

    def state_cycles_arg(self, profile_result: ProfileResult) -> int:
        return int(profile_result.raw_metrics.get("elapsed_cycles", 0))

    # ---- Prompt + DB glue (Phase 4f.3d.a) ----
    def build_strategy_prompt(
        self,
        optimization_entry,
        code: str,
        profile_result: ProfileResult,
        database_content: str,
        description: str = "",
    ) -> str:
        from ..agents.opt_ncu_rl import generate_strategy_guided_prompt

        annotated_ncu = profile_result.raw_metrics.get("annotated_ncu", "") or ""
        return generate_strategy_guided_prompt(
            optimization_entry,
            annotated_ncu,
            profile_result.raw_log,
            database_content,
            override_description=description or None,
            original_code=code,
            backend=self,
        )

    def build_fix_prompt(
        self,
        code: str,
        error_msg: str,
        database_footer: str = "",
    ) -> str:
        parts = [
            "The previously generated CUDA code failed to compile or run.\n\n",
            "COMPILER / RUNTIME ERROR LOG:\n```\n",
            f"{error_msg}\n",
            "```\n\n",
            "ORIGINAL CUDA CODE (for reference – please modify in place):\n```cpp\n",
            f"{code}\n",
            "```\n\n",
        ]
        if database_footer:
            parts.append("OPTIMIZATION DATABASE FOOTER (reference snippets):\n```\n")
            parts.append(database_footer)
            parts.append("\n```\n\n")
        parts.extend(
            [
                "Please provide a corrected, fully compilable version of the kernel. Return **complete CUDA code** in one ```cpp``` block.",
                " Please keep the code structure otherwise unchanged; it is compiled together with separate test code, so do NOT add a main function.\n\n",
                "Include ALL necessary components:\n",
                "   - #include statements (cuda_fp16.h, cuda_runtime.h, etc.)\n",
                "   - #define constants – DEFINE ALL CONSTANTS BEFORE USING THEM\n",
                "   - Complete __global__ kernel function with proper signature\n",
                "   - Complete launch_gpu_implementation(void*, void*, void*, int64_t) function\n",
            ]
        )
        return "".join(parts)

    # CUDA relies on the database's default baseline-file speedup parse,
    # so ``database_update_kwargs`` inherits the empty base default.

    # ---- Metric-shape glue (Phase 4f.3d.b) ----
    def parse_state_metrics(self, raw_log: str, current_metric) -> dict:
        from ..agents.utils.parsing import parse_ncu_metrics
        return parse_ncu_metrics(raw_log)

    def state_cycles_from_metric(self, current_metric) -> int:
        return int(current_metric or 0)

    def metric_to_traj_cycles(self, metric) -> int:
        return int(metric)

    def metric_from_traj_cycles(self, cycles: int) -> float:
        return float(cycles)

    # ---- RL graph-node config ----
    def rl_node_config(self):
        """Per-backend wiring for the unified RL-optimization graph node.

        Curated-artifact resolution now lives on ``ProblemSource`` — see
        :class:`data.sources.KernelBenchCUDASource` +
        :class:`data.sources.SOLExecBenchCUDASource`.
        """
        from .base import RLNodeConfig
        from ..agents.opt_ncu_rl import RLNCUAgent

        return RLNodeConfig(
            state_kernel_fp_input="cuda_fp",
            state_perf_fp_output="rl_ncu_cuda_fp",
            agent_class=RLNCUAgent,
            agent_kernel_fp_kwarg="code_to_optimize_fp",
            fb_config_agent_name="rl_ncu",
            num_pgen=4,
            final_filename="final_rl_cuda_perf.cu",
            use_global_best_preference=False,
        )

    # ---- LLM response handling ----
    def extract_code_from_response(self, response_text: str) -> str | None:
        """CUDA uses the ```cpp tag."""
        from ..agents.utils import extract_code_from_response as _extract
        return _extract(response_text, tag="cpp")

    # ---- result artifact formatting ----
    _CYCLES_FOOTER_RE = None  # lazy-compiled in format_result_artifact

    def format_result_artifact(self, code: str, metric_value) -> str:
        """Append ``// Elapsed Cycles: <value>`` and strip any prior matching footer.

        ``metric_value`` is typically an int (cycles) but accepts a string
        sentinel (e.g. ``"N/A"``) for failure-path artifacts — the prior
        inline call sites stringified directly via f-string, so we preserve
        that flexibility.
        """
        import re

        if CUDABackend._CYCLES_FOOTER_RE is None:
            # Match "Elapsed Cycles: <digits>" OR "Elapsed Cycles: N/A" (failure path).
            CUDABackend._CYCLES_FOOTER_RE = re.compile(
                r"\n*//\s*Elapsed Cycles:\s*(?:\d+|N/A)\s*$",
                re.IGNORECASE | re.MULTILINE,
            )
        body = CUDABackend._CYCLES_FOOTER_RE.sub("", code).rstrip()
        if isinstance(metric_value, (int, float)):
            metric_str = f"{int(metric_value)}"
        else:
            metric_str = str(metric_value)
        return f"{body}\n\n// Elapsed Cycles: {metric_str}\n"
