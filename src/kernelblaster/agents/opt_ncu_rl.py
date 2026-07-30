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
"""
Reinforcement Learning-based CUDA Optimization Agent.
Implements the LLM-based policy optimization via strategy-guided rollouts.
"""
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
import loguru
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import json
import asyncio
import sys
from ..config import config
from .feedback import FeedbackAgent, Feedback, FeedbackConfig
from .opt_rl_base import RLAgentBase
from .database import OptimizationDatabase, OptimizationEntry, CompositeOptimization
from .rl_agents import (
    ReplayBuffer, Trajectory, TrajectoryStep,
    PolicyEvaluationAgent, PerfGapAnalysisAgent, ParameterUpdateAgent
)
from .utils import (
    FeedbackError,
    compile_and_run_cu_file,
    run_gpu_executable,
    format_ncu_source_as_csv,
    format_ncu_details_as_csv,
    annotate_source,
    UTILIZATION_METRICS,
    find_kernel_names_ncu,
    get_elapsed_cycles_ncu_log,
    NamedTimer,
    parse_ncu_metrics,
)
from .database import LLMInterface

import os



# parse_ncu_metrics moved to agents/utils/parsing.py in Phase 4f.3a so it can
# be consumed by CUDABackend.derive_metrics_for_state without a circular import.
# Re-exported via .utils below; callers in this file use the re-export.


def generate_strategy_guided_prompt(
    optimization_entry: OptimizationEntry | CompositeOptimization,
    annotated_ncu: str,
    ncu_log: str,
    database_content: str = "",
    override_description: str | None = None,
    original_code: str | None = None,
    backend=None,
) -> str:
    """Generate a prompt that guides the LLM using the comprehensive optimization database.

    The CUDA technique map is sourced from ``backend.technique_map`` (single
    source of truth — see ``kernelblaster.backends.cuda``). The ``backend``
    parameter is optional for back-compat; if omitted, the default CUDA backend
    is constructed lazily. Phase 4b will fold this with the OpenCL prompt
    function into a single backend-agnostic template.
    """
    if backend is None:
        from ..backends import get_backend
        backend = get_backend("cuda")
    technique_descriptions = backend.technique_map

    # Handle composite optimizations differently
    if isinstance(optimization_entry, CompositeOptimization):
        # Composite optimization with multiple techniques
        techniques = [t for t in [optimization_entry.technique1, optimization_entry.technique2, optimization_entry.technique3] if t]
        technique_descs = []
        for tech in techniques:
            desc = technique_descriptions.get(tech, f"Apply {tech}")
            technique_descs.append(f"- {tech}: {desc}")
        
        composite_desc = "\n".join(technique_descs)
        order_desc = "\n".join(optimization_entry.order_of_techniques) if optimization_entry.order_of_techniques else "Apply techniques in the order listed above"
        
        params_desc = ""
        if optimization_entry.parameters_to_fine_tune:
            params_list = [f"- {k}: {v}" for k, v in optimization_entry.parameters_to_fine_tune.items()]
            params_desc = f"\n\nPARAMETER TUNING:\n" + "\n".join(params_list)
        
        side_effects_note = ""
        if optimization_entry.side_effects:
            side_effects_note = f"\n\nWARNING - POTENTIAL SIDE EFFECTS:\n{optimization_entry.side_effects}"
        
        # Use original code as fallback if annotated_ncu is empty
        source_code_display = annotated_ncu if annotated_ncu.strip() else (original_code or "// Source code not available")
        source_code_label = "ANNOTATED SOURCE CODE (with per-line analysis):" if annotated_ncu.strip() else "SOURCE CODE:"
        
        # Only include NCU profiling log section if there's meaningful content
        # (not just "Kernels: ..." which indicates extraction failed)
        ncu_section = ""
        ncu_log_stripped = ncu_log.strip()
        if ncu_log_stripped and not (ncu_log_stripped.startswith("Kernels:") and len(ncu_log_stripped.split('\n')) <= 2):
            ncu_section = f"""
RAW NCU PROFILING LOG (Speed Of Light Throughput Summary):
```
{ncu_log[:4000] if len(ncu_log) > 4000 else ncu_log}
```
"""
        
        return f"""You are a CUDA optimization expert with access to comprehensive optimization knowledge.

COMPREHENSIVE GPU OPTIMIZATION DATABASE:
```
{database_content[:6000] if database_content else "Database not available - using fallback descriptions"}
```

COMPOSITE OPTIMIZATION STRATEGY:
{optimization_entry.get_composite_id()}
PREDICTED IMPROVEMENT: {optimization_entry.predicted_improvement}%

TECHNIQUES TO APPLY:
{composite_desc}

APPLICATION ORDER:
{order_desc}{params_desc}{side_effects_note}

CURRENT KERNEL ANALYSIS:

{source_code_label}
```
{source_code_display}
```
{ncu_section}

OPTIMIZATION TASK:
You are an expert CUDA optimization agent, and you are provided an optimization plan. Your task is to apply the optimization plan to the current kernel. You will be provided with the annotated source code, the raw NCU profiling log, and the optimization plan.

CRITICAL REQUIREMENTS:
1. Reference the optimization plan for detailed implementation guidance
2. Apply ALL specified techniques in the given order: {' -> '.join(techniques)}
3. Use the specified parameter values for fine-tuning
4. Generate COMPLETE, COMPILABLE CUDA code
5. Include ALL necessary components:
   - #include statements (cuda_fp16.h, cuda_runtime.h, etc.)
   - #define constants - DEFINE ALL CONSTANTS BEFORE USING THEM
   - Complete __global__ kernel function with proper signature
   - Complete launch_gpu_implementation(void*, void*, void*, int64_t) function
6. Format ALL code in a single ```cpp code block
7. Consider potential side effects mentioned in the database
8. COMPILATION SAFETY: Ensure all constants are properly defined


You are a knowledgeable and efficient CUDA programming assistant, skilled in analyzing NSight Compute logs and optimizing the cuda kernels. Your task is to analyze the provided NSight Compute logs and generate optimized CUDA code based on the analysis. You should focus on finding the largest deficiencies from the NCU log and optimize those attributes first. 

For perf comparisons, please use the "Elapsed Cycles" metric in the GPU Speed of Light Throughput section of the NCU log. The lower the better. Please only write one kernel in the output.

Optimization Tips:
* For better memory bandwidth utilization, please try to use coalescing and coalesced memory access patterns. You can also use vectorized datatypes like float4, int4, uint4, __nv_bfloat162, etc.

APPROACH:
1. Analyze the profiling data to understand current bottlenecks
2. Consult the optimization database for best practices
3. Apply the composite strategy systematically
4. Generate optimized code addressing the identified performance issues"""
    
    else:
        # Single technique optimization
        technique_name = (
            optimization_entry.get_composite_id()
            if isinstance(optimization_entry, CompositeOptimization)
            else getattr(optimization_entry, "technique", str(optimization_entry))
        )
        technique_desc = (
            override_description
            if override_description
            else technique_descriptions.get(
                technique_name,
                f"Apply the {technique_name} optimization technique.",
            )
        )
        pred_impr = getattr(optimization_entry, "predicted_improvement", None)
        category = getattr(optimization_entry, "category", "general")
        pred_impr_str = f"{pred_impr}%" if pred_impr is not None else "N/A"
        
        # Use original code as fallback if annotated_ncu is empty
        source_code_display = annotated_ncu if annotated_ncu.strip() else (original_code or "// Source code not available")
        source_code_label = "ANNOTATED SOURCE CODE (with per-line analysis):" if annotated_ncu.strip() else "SOURCE CODE:"
        
        # Only include NCU profiling log section if there's meaningful content
        # (not just "Kernels: ..." which indicates extraction failed)
        ncu_section = ""
        ncu_log_stripped = ncu_log.strip()
        if ncu_log_stripped and not (ncu_log_stripped.startswith("Kernels:") and len(ncu_log_stripped.split('\n')) <= 2):
            ncu_section = f"""
RAW NCU PROFILING LOG (Speed Of Light Throughput Summary):
```
{ncu_log[:4000] if len(ncu_log) > 4000 else ncu_log}
```
"""
        
        return f"""OPTIMIZATION TASK:
You are an expert CUDA optimization agent, and you are provided an optimization plan. Your task is to apply the optimization plan to the current kernel. You will be provided with the annotated source code, the raw NCU profiling log, and the optimization plan.

OPTIMIZATION STRATEGY: {technique_name}
PREDICTED IMPROVEMENT: {pred_impr_str}
CATEGORY: {category}

STRATEGY DESCRIPTION:
{technique_desc}

CURRENT KERNEL ANALYSIS:

{source_code_label}
```
{source_code_display}
```
{ncu_section}

COMPREHENSIVE GPU OPTIMIZATION DATABASE:
```
{database_content}
```

CRITICAL REQUIREMENTS:
1. Reference the optimization database for detailed implementation guidance
2. Generate COMPLETE, COMPILABLE CUDA code
3. Include ALL necessary components:
   - #include statements (cuda_fp16.h, cuda_runtime.h, etc.)
   - #define constants - DEFINE ALL CONSTANTS BEFORE USING THEM
   - Complete __global__ kernel function with proper signature
   - Complete launch_gpu_implementation function
4. Format ALL code in a single ```cpp code block
5. Focus specifically on the technique described in the database
6. COMPILATION SAFETY: Ensure all constants are properly defined
7. Summarize the optimization technique applied and the reason for the improvement before the code.



APPROACH:
1. Apply requested the optimization technique systematically
2. If applying a new technique not yet attempted in the code, start with the most minimal example, focusing on correctness.
4. Please use the reference code provided in the prompt as helper functions for your optimized kernel.
3. Generate optimized code addressing the identified performance issues"""


@dataclass
class RLNCUFeedback(Feedback):
    # ``metric`` is the primary backend metric (NCU elapsed cycles for CUDA).
    # Renamed from ``elapsed_cycles`` in Phase 4d to harmonize with the
    # OpenCL feedback shape.
    metric: Optional[float] = None
    ncu_log: Optional[str] = None
    annotated_ncu: Optional[str] = None
    optimization_technique: Optional[str] = None
    predicted_improvement: Optional[float] = None
    actual_improvement: Optional[float] = None
    state: Optional[str] = None


class RLNCUAgent(RLAgentBase):
    """
    RL-based CUDA optimization agent implementing strategy-guided rollouts.
    """
    
    def __init__(
        self,
        fb_config: FeedbackConfig,
        code_to_optimize_fp: Path,
        database_path: Path,
        max_rollout_steps: int = 5,
        replay_buffer_size: int = 1000,
        update_frequency: int = 10,
        database: Optional[OptimizationDatabase] = None,
    ):
        # Phase 4f step 2: shared __init__ lives on RLAgentBase. The public
        # CUDA constructor keeps ``code_to_optimize_fp`` for back-compat;
        # internally it's passed under the canonical kernel_source_fp name.
        super().__init__(
            fb_config=fb_config,
            kernel_source_fp=code_to_optimize_fp,
            database_path=database_path,
            max_rollout_steps=max_rollout_steps,
            replay_buffer_size=replay_buffer_size,
            update_frequency=update_frequency,
            database=database,
        )

    # initialize() lifted to RLAgentBase in Phase 4f.3c. This subclass only
    # contributes the two per-backend hooks below (_write_init_artifact,
    # _handle_init_failure). _maybe_generate_reference inherits the no-op
    # default — CUDA's libtorch driver computes the reference in-process.

    def _write_init_artifact(self, profile_result) -> None:
        """Write ``0_init_annotated.cu`` with the NCU-annotated source CSV."""
        annotated = profile_result.raw_metrics.get("annotated_ncu", "") or ""
        (self.folder / "0_init_annotated.cu").write_text(annotated)

    def _handle_init_failure(self) -> None:
        """Populate fallback state via the database; write a placeholder
        ``0_init_annotated.cu`` (raw init.cu) so downstream steps can proceed."""
        try:
            # Compute fallback state for its database side-effects; the return
            # value isn't stored anywhere else, mirroring the legacy behavior.
            self.database._fallback_state_analysis("", {})
        except Exception as e:
            self.agent_logger.warning(f"Fallback state analysis failed: {e}")

        try:
            init_src = self.kernel_source_fp.read_text()
            (self.folder / "0_init_annotated.cu").write_text(init_src)
        except Exception as _e:
            self.agent_logger.warning(
                f"Failed to write fallback 0_init_annotated.cu: {_e}"
            )

    # run() lifted to RLAgentBase in Phase 4f.3d.f. This subclass only
    # contributes the ``_finalize_run`` hook below (baseline-recompute
    # fallback + backend.format_result_artifact for the failure file).
    async def _finalize_run(self, best_filename, best_metric) -> Path:
        ext = self.backend.kernel_ext

        if best_filename is not None:
            # Baseline recompute so we can judge improvement.
            try:
                if self.initial_metric is None:
                    init_fp = getattr(self, "kernel_source_fp", None)
                    if not init_fp or not init_fp.exists():
                        self.kernel_source_fp = self.folder / f"init{ext}"
                        self.kernel_source_fp.write_text(self.kernel_source)
                    pr = await self.gather_profile_result(self.kernel_source_fp)
                    self.initial_metric = self.backend.extract_primary_metric(pr)
            except Exception as e:
                self.agent_logger.warning(
                    f"Failed to obtain baseline cycles before finalizing result: {e}"
                )

            if self.initial_metric is not None and best_metric < self.initial_metric:
                final_filename = self.folder / f"success_rl_optimization{ext}"
                final_filename.write_text(best_filename.read_text())
                return final_filename

            failure_file = self.folder / f"failure_rl_optimization{ext}"
            baseline_str = self.initial_metric if self.initial_metric is not None else "N/A"
            try:
                failure_file.write_text(
                    self.backend.format_result_artifact(self.kernel_source, baseline_str)
                )
            except Exception:
                try:
                    init_fp = getattr(self, "kernel_source_fp", None)
                    if init_fp and init_fp.exists():
                        failure_file.write_text(
                            self.backend.format_result_artifact(init_fp.read_text(), baseline_str)
                        )
                except Exception:
                    pass
            self.agent_logger.error(
                f"RL did not produce an improvement; wrote failure_rl_optimization{ext} "
                f"with baseline (if available)"
            )
            return failure_file

        # No trajectory produced a candidate.
        try:
            if self.initial_metric is None:
                init_fp = getattr(self, "kernel_source_fp", None)
                if not init_fp or not init_fp.exists():
                    self.kernel_source_fp = self.folder / f"init{ext}"
                    self.kernel_source_fp.write_text(self.kernel_source)
                pr = await self.gather_profile_result(self.kernel_source_fp)
                cycles = self.backend.extract_primary_metric(pr)
                self.initial_metric = cycles
                self.best_metric = (
                    min(self.best_metric, cycles) if self.best_metric else cycles
                )
        except Exception as e:
            self.agent_logger.warning(
                f"Failed to obtain baseline cycles for original code: {e}"
            )

        fallback = self.initial_metric if self.initial_metric is not None else "N/A"
        failure_file = self.folder / f"failure_rl_optimization{ext}"
        try:
            failure_file.write_text(
                self.backend.format_result_artifact(self.kernel_source, fallback)
            )
        except Exception:
            try:
                init_fp = getattr(self, "kernel_source_fp", None)
                if init_fp and init_fp.exists():
                    failure_file.write_text(
                        self.backend.format_result_artifact(init_fp.read_text(), fallback)
                    )
            except Exception:
                pass
        self.agent_logger.error(
            f"All RL iterations failed; wrote failure_rl_optimization{ext} "
            f"with baseline (if available)"
        )
        return failure_file

    async def gather_profile_result(self, filepath: Path):
        """Phase 4f.3b: wrap the CUDA tuple return into a ProfileResult.

        Old callers (~5 sites in this file) still use the tuple directly;
        shared methods in RLAgentBase use this wrapper to stay backend-
        agnostic. ``raw_metrics`` carries the CUDA-specific extras
        (``annotated_ncu``, ``stderr``).
        """
        from ..backends import ProfileResult
        annotated_ncu, ncu_log, stderr, cycles = await self.gather_perf_metrics(filepath)
        return ProfileResult(
            total_time_ms=0.0,
            per_kernel_ms={},
            raw_metrics={
                "elapsed_cycles": cycles,
                "annotated_ncu": annotated_ncu,
                "stderr": stderr,
            },
            raw_log=ncu_log,
        )

    async def gather_perf_metrics(self, filepath: Path) -> Tuple[str, str, str, int]:
        """Gather performance metrics using NCU profiling."""
        # Reuse the existing profiling logic from opt_ncu_annot_fixed5.py
        # Use a single execution run to avoid non‐deterministic kernels causing spurious
        # verification failures across repeated runs.
        stdout_list, stderr_list, path, success = await compile_and_run_cu_file(
            self.test_code_fp,
            filepath,
            self.gpu,
            NamedTimer(),
            self.agent_logger,
            persistent_artifacts=True,
            timeout=3600,
            num_runs=1,
            passed_keyword="passed",
        )
        
        if not success:
            FeedbackAgent.raise_numerics_verification_error(stdout_list, stderr_list)

        # Optional: cycles-only mode to avoid including full NCU logs in the agentic flow.
        # Still runs NCU to get accurate cycle counts, but only returns the cycles (not full logs).
        cycles_only = os.getenv("KERNELAGENT_RL_NCU_CYCLES_ONLY", "0") in (
            "1",
            "true",
            "True",
            "yes",
            "YES",
            "y",
            "on",
            "ON",
        )
        if cycles_only:
            err_text = "\n".join(stderr_list or [])
            cycles = None
            try:
                # Still run NCU to get accurate cycle counts from the Speed Of Light section
                kernel_names = await find_kernel_names_ncu(path, filepath, self.gpu, 3600)
                
                if not kernel_names:
                    raise ValueError("No kernel names found for NCU profiling")
                
                # Run basic NCU profiling to get cycles (this includes Speed Of Light section)
                # Use the first kernel name (most kernels have one main kernel)
                kernel_name = kernel_names[0]
                ncu_stdout, ncu_stderr = await run_gpu_executable(
                    path, self.gpu, 3600,
                    job_name=f"{filepath} (ncu cycles-only)",
                    prefix_command=f"NVIDIA_TF32_OVERRIDE=0 ncu -k {kernel_name}",
                )
                
                if "No Kernels were profiled" in ncu_stdout:
                    raise ValueError("NCU did not profile any kernels")
                
                # Parse cycles from NCU output using the existing utility function
                cycles = get_elapsed_cycles_ncu_log(ncu_stdout)
                
                err_text += f"\nNCU stderr: {ncu_stderr}"
                
            except Exception as e:
                self.agent_logger.warning(
                    f"KERNELAGENT_RL_NCU_CYCLES_ONLY is set but failed to parse elapsed cycles from NCU output: {e}"
                )
                cycles = None  # Use None instead of 0 to indicate parsing failure
            # Return empty NCU logs/annotations so prompts stay small.
            # Use 0 if cycles is None (parsing failed) to maintain backward compatibility with int return type
            return "", "", err_text, cycles if cycles is not None else 0

        kernel_names = await find_kernel_names_ncu(path, filepath, self.gpu, 3600)
        
        # Debug: log kernel names being profiled
        self.agent_logger.info(f"Profiling {len(kernel_names)} kernel(s) from CUDA file: {kernel_names}")

        # Single NCU call for details CSV and raw logs
        # Using --csv flag to get CSV format for parsing, but the output still contains full text with CSV embedded
        # Build kernel filter: if single kernel, use -k flag; if multiple, profile all (no -k flag)
        if len(kernel_names) == 1:
            # Single kernel: use -k flag to filter
            kernel_filter = f"-k {kernel_names[0]}"
        else:
            # Multiple kernels: profile all (NCU doesn't support multiple -k flags)
            # We'll filter in post-processing to only process kernels from CUDA file
            kernel_filter = ""
            self.agent_logger.debug(f"Multiple kernels detected, profiling all and filtering to: {kernel_names}")
        
        details_command = (
            f"ncu {kernel_filter} --page details --section=SchedulerStats --section=Occupancy --section=SpeedOfLight --section=LaunchStats --section=WarpStateStats --section=InstructionStats --csv --metrics "
            + ",".join(UTILIZATION_METRICS)
        )

        # Profile kernels in a single NCU call
        # Get both details CSV (parsed from text) and raw logs from one call
        details_stdout, details_stderr = await run_gpu_executable(
            path, self.gpu, 3600,
            job_name=f"{filepath} (details)",
            prefix_command=f"NVIDIA_TF32_OVERRIDE=0 {details_command} ",
        )

        if "No Kernels were profiled" in details_stdout:
            self.agent_logger.warning(f"No kernels were profiled for {filepath}")
            return "", "", details_stderr, 0
        
        # Use details output for raw logs (it contains comprehensive profiling information)
        combined_ncu_logs = details_stdout
        
        stderr = f"details: {details_stderr}\n"
        
        # Parse the details CSV output and split by kernel
        try:
            all_details_df = format_ncu_details_as_csv(details_stdout)
        except ValueError as e:
            raise ValueError(f"Failed to extract CSV from NCU logs: {e}")

        # Split the details dataframe by kernel name
        details_dfs = []
        cycles = 0
        
        # For details CSV, split by "Kernel Name" column
        # Only process kernels found in the CUDA file (from find_kernel_names_ncu)
        if "Kernel Name" in all_details_df.columns:
            # Log what we found vs what we expect
            all_profiled_kernels = all_details_df["Kernel Name"].str.split("(").str[0].str.strip().unique().tolist()
            self.agent_logger.info(
                f"Found {len(all_profiled_kernels)} kernels in NCU CSV output: {all_profiled_kernels}"
            )
            self.agent_logger.info(
                f"Processing {len(kernel_names)} kernels from CUDA file: {kernel_names}"
            )
            
            # Only process kernels found in the CUDA file
            for kernel_name in kernel_names:
                # Filter rows for this kernel (handle kernel name with or without parameters)
                kernel_base_name = kernel_name.split("(")[0].strip()
                name_series = all_details_df["Kernel Name"].astype(str)
                base_series = name_series.str.split("(").str[0].str.strip()

                # First try exact base-name match
                kernel_mask = base_series == kernel_base_name

                # If no rows, fall back to fuzzy contains match to handle templates like
                # "void linear_bias_relu_kernel<1>" vs "linear_bias_relu_kernel"
                if not kernel_mask.any():
                    import re as _re

                    pattern = _re.escape(kernel_base_name)
                    kernel_mask = base_series.str.contains(pattern, case=False, regex=True)

                kernel_details_df = all_details_df[kernel_mask].copy()
                
                if len(kernel_details_df) > 0:
                    details_dfs.append(kernel_details_df)
                    
                    # Get cycles from details for this kernel
                    for _, row in kernel_details_df.iterrows():
                        if row["Metric Name"] == "Elapsed Cycles":
                            cycles += int(row["Metric Value"].replace(",", ""))
                    
                    self.agent_logger.debug(
                        f"Extracted {len(kernel_details_df)} metric rows for kernel '{kernel_name}'"
                    )
                else:
                    # No details found for this kernel - this can happen if source profiling was skipped
                    # or if the kernel wasn't actually executed, or kernel name doesn't match
                    # Try fuzzy matching to help diagnose
                    similar_kernels = [
                        k for k in all_profiled_kernels 
                        if kernel_base_name.lower() in k.lower() or k.lower() in kernel_base_name.lower()
                    ]
                    if similar_kernels:
                        self.agent_logger.warning(
                            f"Expected kernel '{kernel_name}' was not found in NCU details. "
                            f"Similar kernel names found: {similar_kernels}. "
                            f"This may indicate a kernel name mismatch or the kernel was not executed."
                        )
                    else:
                        self.agent_logger.warning(
                            f"Expected kernel '{kernel_name}' was not found in NCU details - "
                            f"may not have been executed. Profiled kernels: {all_profiled_kernels}"
                        )
                    # Add empty dataframe to maintain alignment
                    details_dfs.append(pd.DataFrame())
        else:
            # Fallback: if no Kernel Name column, assume single kernel
            self.agent_logger.warning("No 'Kernel Name' column in NCU CSV - assuming single kernel")
            details_dfs.append(all_details_df)
            for _, row in all_details_df.iterrows():
                if row["Metric Name"] == "Elapsed Cycles":
                    cycles += int(row["Metric Value"].replace(",", ""))

        # Create empty source dataframes (no source profiling needed - we have raw logs and details)
        # The annotate_source function expects source_dfs, but we'll pass empty ones since we don't need per-line annotations
        # Ensure source_dfs matches the number of details_dfs (which now includes all profiled kernels)
        source_dfs = [pd.DataFrame() for _ in range(len(details_dfs))]

        # Annotate source (will use details only, source annotations will be minimal/empty)
        # This will generate profile summaries for all kernels found in the CSV
        annotated_ncu = annotate_source(filepath, source_dfs, details_dfs)
        
        # Log summary of what was processed
        kernels_with_details = sum(1 for df in details_dfs if not df.empty)
        self.agent_logger.info(
            f"NCU profiling summary: {kernels_with_details}/{len(details_dfs)} kernels have detailed metrics"
        )

        # Extract only the GPU Speed Of Light Throughput section to reduce token usage
        # Similar to minimal agent - only include summary info, not full verbose logs
        combined_ncu_logs = self._extract_speed_of_light_section(combined_ncu_logs, kernel_names)
        
        return annotated_ncu, combined_ncu_logs, stderr, cycles
    
    def _extract_speed_of_light_section(self, ncu_output: str, kernel_names: list) -> str:
        """
        Extract only the GPU Speed Of Light Throughput section from NCU log.
        Returns simplified log with kernel names and just the summary tables for each kernel.
        This significantly reduces token usage while preserving essential performance metrics.
        """
        import re
        
        sections = []
        
        # Split by kernel markers if present
        kernel_blocks = []
        if "[Kernel:" in ncu_output:
            # Split by kernel markers (from our manual markers)
            kernel_pattern = r"\[Kernel: ([^\]]+)\]\n(.*?)(?=\[Kernel:|\Z)"
            for match in re.finditer(kernel_pattern, ncu_output, re.DOTALL):
                kernel_name = match.group(1)
                kernel_log = match.group(2)
                kernel_blocks.append((kernel_name, kernel_log))
        else:
            # No kernel markers - NCU outputs kernel info before each section
            # Look for kernel name patterns before "Section: GPU Speed Of Light Throughput"
            section_pattern = r"Section: GPU Speed Of Light Throughput"
            section_matches = list(re.finditer(section_pattern, ncu_output, re.MULTILINE))
            
            for i, section_match in enumerate(section_matches):
                # Look backwards from the section header to find the kernel name
                section_start = section_match.start()
                # Get the 50 lines before this section to find kernel name
                lines_before = ncu_output[max(0, section_start - 5000):section_start]
                
                # Try to find kernel name in the lines before the section
                kernel_name = None
                for known_kernel in kernel_names:
                    # Look for kernel name patterns: kernel_name@, kernel_name(, or [timestamp] kernel_name
                    # Escape special regex chars in kernel name
                    escaped_name = re.escape(known_kernel)
                    kernel_patterns = [
                        rf"{escaped_name}@",  # kernel_name@...
                        rf"{escaped_name}\(",  # kernel_name(...
                        rf"\[.*?\]\s+{escaped_name}",  # [timestamp] kernel_name
                        rf"==PROF==.*?{escaped_name}",  # ==PROF== ... kernel_name
                    ]
                    for pattern in kernel_patterns:
                        if re.search(pattern, lines_before, re.IGNORECASE | re.MULTILINE):
                            kernel_name = known_kernel
                            break
                    if kernel_name:
                        break
                
                # If we couldn't match, use index-based matching as fallback
                if kernel_name is None and i < len(kernel_names):
                    kernel_name = kernel_names[i]
                elif kernel_name is None:
                    kernel_name = f"kernel_{i}"
                
                # Extract the section content
                section_end = section_match.end()
                if i + 1 < len(section_matches):
                    next_section_start = section_matches[i + 1].start()
                    section_content = ncu_output[section_end:next_section_start]
                else:
                    section_content = ncu_output[section_end:]
                
                kernel_blocks.append((kernel_name, section_content))
        
        # Process each kernel block
        for kernel_name, kernel_log in kernel_blocks:
            # Find "Section: GPU Speed Of Light Throughput" sections in this kernel's log
            pattern = r"Section: GPU Speed Of Light Throughput\n(.*?)(?=\n\s+Section:|==PROF==|\Z|\[Kernel:)"
            matches = list(re.finditer(pattern, kernel_log, re.DOTALL | re.MULTILINE))
            
            for match in matches:
                table_content = match.group(1)
                # Extract lines until we hit the end of the table
                lines = table_content.split('\n')
                table_lines = []
                
                # Always add kernel name header
                table_lines.append(f"Kernel: {kernel_name}")
                table_lines.append("Section: GPU Speed Of Light Throughput")
                
                separator_count = 0
                found_metrics = False
                
                for line in lines:
                    # Check if this is a separator line (mostly dashes and spaces)
                    is_separator = bool(re.match(r'^[\s-]+$', line))
                    
                    if is_separator:
                        separator_count += 1
                        table_lines.append(line)
                        # After we've seen metrics and hit another separator, we're done
                        if found_metrics and separator_count >= 3:
                            break
                    elif separator_count >= 2:
                        # We're past the header separators, now in metrics
                        found_metrics = True
                        table_lines.append(line)
                        # Stop if we hit an empty line after metrics (end of table)
                        if not line.strip() and found_metrics:
                            break
                    elif separator_count == 1:
                        # Header row (Metric Name, Metric Unit, Metric Value)
                        table_lines.append(line)
                    else:
                        # Before first separator - skip any extra content
                        continue
                
                # Only add if we found the actual table content
                if len(table_lines) > 3:  # Header + at least 2 separator lines
                    sections.append('\n'.join(table_lines))
        
        if not sections:
            # Fallback: try simpler extraction - just get first 15 lines after each section header
            pattern = r"Section: GPU Speed Of Light Throughput"
            matches = list(re.finditer(pattern, ncu_output))
            for i, match in enumerate(matches):
                start_pos = match.end()
                # Get next 15 lines
                remaining = ncu_output[start_pos:]
                lines = remaining.split('\n')[:15]
                if lines:
                    kernel_label = f"Kernel: {kernel_names[i] if i < len(kernel_names) else 'unknown'}\n" if kernel_names else ""
                    sections.append(kernel_label + "Section: GPU Speed Of Light Throughput\n" + '\n'.join(lines))
        
        if not sections:
            # Last resort: return minimal info with cycles.
            # Downgrade to info – this is expected when running with --csv details output.
            self.agent_logger.info(
                "Could not extract Speed Of Light sections from NCU text; using minimal cycles-only summary"
            )
            simplified = []
            for kernel_name in kernel_names:
                # Try to find cycles for this kernel - escape special regex chars
                escaped_name = re.escape(kernel_name)
                cycles_pattern = rf"{escaped_name}.*?Elapsed Cycles\s+\w+\s+(\d+)"
                cycles_match = re.search(cycles_pattern, ncu_output, re.DOTALL | re.IGNORECASE)
                if cycles_match:
                    simplified.append(f"Kernel: {kernel_name}\nElapsed Cycles: {cycles_match.group(1)}")
            if simplified:
                return "\n\n".join(simplified)
            # If we still have nothing, return empty string to omit the section entirely
            return ""
        
        return "\n\n".join(sections)

    # run_rollout lifted to RLAgentBase in Phase 4f.3d.e. Backend hooks
    # (parse_state_metrics, state_cycles_from_metric, metric_to_traj_cycles,
    # database_update_kwargs) supply the CUDA-flavoured bits — no override
    # needed here.
    # ------------------------------------------------------------------
    # Helper to find an optimisation entry by its technique/composite ID
    # ------------------------------------------------------------------
    # _lookup_optim_entry_by_name lifted to RLAgentBase in Phase 4f.

    # apply_optimization lifted to RLAgentBase in Phase 4f.3d. Backend
    # picks the prompt via ``build_strategy_prompt`` / ``build_fix_prompt``.
    # The base method returns a float metric; ``run_rollout`` in this
    # class still casts to ``int`` before writing TrajectoryStep.cycles.

    # calculate_reward lifted to RLAgentBase in Phase 4f.

    async def policy_update_cycle(self):
        """Run the policy evaluation and update cycle."""
        if len(self.replay_buffer.trajectories) < 3:
            return  # Need some trajectories to analyze
        
        self.agent_logger.info("Running policy update cycle...")
        
        try:
            # Policy Evaluation
            evaluation_result = await self.policy_evaluation_agent.evaluate_policy(
                self.replay_buffer, self.database
            )
            
            # Collect recent failures for gap analysis
            recent_failures = []
            for traj in self.replay_buffer.get_recent_trajectories(5):
                for step in traj.steps:
                    if step.reward < 0 or step.actual_improvement < step.predicted_improvement * 0.5:
                        recent_failures.append(step)
            
            # Performance Gap Analysis
            gap_analysis = await self.perf_gap_analysis_agent.analyze_performance_gaps(
                evaluation_result, recent_failures
            )
            
            # Parameter Update
            updates = await self.parameter_update_agent.update_parameters(
                gap_analysis, self.database
            )
            
            # Save analysis results
            analysis_file = self.folder / f"analysis_iteration_{self.iteration_count}.json"
            analysis_data = {
                'iteration': self.iteration_count,
                'evaluation_result': evaluation_result,
                'gap_analysis': gap_analysis,
                'updates': updates,
                'buffer_stats': self.replay_buffer.get_statistics(),
                'database_stats': self.database.get_database_stats()
            }
            
            with open(analysis_file, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            self.agent_logger.info(f"Policy update completed. Analysis saved to {analysis_file}")
            
        except Exception as e:
            self.agent_logger.error(f"Error in policy update cycle: {e}")

    # get_feedback lifted to RLAgentBase in Phase 4f.3d. This subclass only
    # contributes the per-backend _build_feedback hook below (CUDA framing:
    # cycles, ncu_log, annotated_ncu). The trajectory-min best-step
    # selection is the base default (_get_verified_best).
    def _build_feedback(
        self,
        *,
        response,
        task_id,
        code: str,
        initial_metric: float,
        profile_result,
        initial_state: str,
        trajectory,
    ) -> Feedback:
        ncu_log = profile_result.raw_log
        annotated_ncu = profile_result.raw_metrics.get("annotated_ncu", "") or ""

        # Best-perf side-effect (preserved from the pre-refactor path).
        if trajectory.final_cycles < self.best_metric:
            self.best_metric = trajectory.final_cycles

        if trajectory.steps:
            best_step = min(trajectory.steps, key=lambda s: s.cycles)
            if self.initial_metric is not None and self.initial_metric > 0:
                improvement_pct = ((self.initial_metric - best_step.cycles) / self.initial_metric) * 100
            else:
                improvement_pct = 0.0

            feedback_msg = f"""Optimization trajectory completed with {len(trajectory.steps)} steps.

BEST RESULT:
- Cycles: {best_step.cycles} (vs initial: {self.initial_metric})
- Overall improvement: {improvement_pct:.1f}%
- Best technique: {best_step.action}
- Total reward: {trajectory.total_reward:.2f}

FINAL OPTIMIZED CODE:
```
{best_step.code}
```

The optimization process is learning and adapting. Continue with further optimizations."""

            best_file = self.folder / f"best_task_{task_id}.cu"
            best_file.write_text(best_step.code)

            return RLNCUFeedback(
                new_messages=[
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": feedback_msg},
                ],
                success=True,
                filename=str(best_file),
                contents=best_step.code,
                metric=best_step.cycles,
                ncu_log=ncu_log,
                annotated_ncu=annotated_ncu,
                optimization_technique=best_step.action,
                predicted_improvement=best_step.predicted_improvement,
                actual_improvement=best_step.actual_improvement,
                state=initial_state,
            )

        return RLNCUFeedback(
            new_messages=[
                {"role": "assistant", "content": response},
                {
                    "role": "user",
                    "content": "No successful optimization steps completed. Please try a different approach.",
                },
            ],
            success=False,
            metric=initial_metric,
            ncu_log=ncu_log,
            annotated_ncu=annotated_ncu,
            state=initial_state,
        )

    # _try_add_default_optimizations lifted to RLAgentBase in Phase 4f.

    # get_performance_summary skeleton lifted to RLAgentBase in Phase 4f;
    # this override only contributes the CUDA-named metric keys.
    def _perf_summary_extras(self) -> Dict[str, Any]:
        return {
            "initial_cycles": self.initial_metric,
            "best_cycles": self.best_metric,
        }
