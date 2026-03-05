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
)
from .database import LLMInterface

import os



def parse_ncu_metrics(ncu_log: str) -> Dict[str, float]:
    """Parse key metrics from NCU log for state determination."""
    metrics = {}

    
    # Nsight-Compute text tables do not always print a trailing '%' after the value.
    # Instead the column layout is: <Metric Name>  <Metric Unit>  <Metric Value>
    # We therefore search for the *name* and grab the **last numeric token on that line**.

    def _build_pattern(keyword: str) -> str:
        """Return a regex that captures the last number on the matching line."""
        # .*? non-greedy up to the final number  (handles variable columns / spacing)
        return rf"{keyword}.*?([0-9]+(?:\.[0-9]+)?)"

    patterns = {
        'memory_throughput'        : _build_pattern(r"Memory\s+Throughput"),
        'compute_throughput'       : _build_pattern(r"Compute\s*\(SM\)\s*Throughput"),
        'sm_efficiency'            : _build_pattern(r"SM\s+Efficiency"),
        'occupancy'                : _build_pattern(r"Achieved\s+Occupancy"),
        'coalescing_efficiency'    : _build_pattern(r"Global\s+Memory\s+Coalescing"),
        'cache_hit_rate'           : _build_pattern(r"L2\s+Cache\s+Hit\s+Rate"),
        'shared_memory_efficiency' : _build_pattern(r"Shared\s+Memory\s+Efficiency"),
        'tensor_core_usage'        : _build_pattern(r"Tensor\s+Core\s+Usage"),
        'register_usage'           : _build_pattern(r"Registers\s+Per\s+Thread"),
        'shared_memory_usage'      : _build_pattern(r"Shared\s+Memory\s+Usage"),
    }
    
    for metric_name, pattern in patterns.items():
        match = re.search(pattern, ncu_log, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                metrics[metric_name] = float(match.group(1))
            except ValueError:
                metrics[metric_name] = 0.0
        else:
            metrics[metric_name] = 0.0
    
    return metrics


def generate_strategy_guided_prompt(
    optimization_entry: OptimizationEntry | CompositeOptimization,
    annotated_ncu: str,
    ncu_log: str,
    database_content: str = "",
    override_description: str | None = None,
    original_code: str | None = None,
) -> str:
    """Generate a prompt that guides the LLM using the comprehensive optimization database."""
    
    technique_descriptions = {
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
        "6.1_thread_coarsening": "Assign multiple work items to each thread to amortize parallelization overhead."
    }
    
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
    elapsed_cycles: Optional[int] = None
    ncu_log: Optional[str] = None
    annotated_ncu: Optional[str] = None
    optimization_technique: Optional[str] = None
    predicted_improvement: Optional[float] = None
    actual_improvement: Optional[float] = None
    state: Optional[str] = None


class RLNCUAgent(FeedbackAgent):
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
        update_frequency: int = 10,  # Update database every N trajectories
        database: Optional[OptimizationDatabase] = None,
    ):
        # Initialize base feedback agent
        super().__init__(fb_config)
        
        self.test_code_fp = fb_config.test_code_fp
        self.test_code = fb_config.test_code_fp.read_text()
        self.code_to_optimize_fp = code_to_optimize_fp
        self.code_to_optimize = code_to_optimize_fp.read_text()
        
        # RL-specific components - Use enhanced database with GPU optimization report
        gpu_report_path = Path(__file__).parent.parent.parent.parent.parent / "algo-sol-modeling/algo-space/gpu_optimization_report.md"
        llm_interface = LLMInterface(self.model, self.agent_logger)
        # Use provided shared database if available; otherwise create a new one
        if database is not None:
            self.database = database
        else:
            self.database = OptimizationDatabase(database_path, gpu_report_path, llm_interface)
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
        self.max_rollout_steps = max_rollout_steps
        self.update_frequency = update_frequency
        
        # RL agents
        self.policy_evaluation_agent = PolicyEvaluationAgent()
        self.perf_gap_analysis_agent = PerfGapAnalysisAgent()
        self.parameter_update_agent = ParameterUpdateAgent()
        
        # Tracking
        self.iteration_count = 0
        self.total_trajectories = 0
        self.best_cycles = float('inf')
        self.initial_cycles = None
        
        # Concurrency helpers
        import asyncio as _asyncio
        self._trajectory_lock: _asyncio.Lock = _asyncio.Lock()
        
        # Current trajectory
        self.current_trajectory = None
        
        # Number of RL iterations to run (can be set by the workflow)
        self.num_rl_iterations = 50  # Default to 50 RL iterations

    async def initialize(self):
        """Initialize the agent by gathering initial profiling data."""
        # Copy init cu file to folder
        self.code_to_optimize_fp = self.folder / "init.cu"
        self.code_to_optimize_fp.write_text(self.code_to_optimize)

        self.agent_logger.info(f"Gathering initial NCU log...")
        try:
            annotated_ncu, init_ncu_log, _, cycles = await self.gather_perf_metrics(
                self.code_to_optimize_fp
            )
            self.initial_cycles = cycles
            self.best_cycles = cycles

            # Persist first NCU log so subsequent steps can perform analysis
            self.last_ncu_log = init_ncu_log
            
            # Save initial state
            init_metrics = parse_ncu_metrics(init_ncu_log)
            initial_state = await self.database.get_state_from_ncu_report(
                init_ncu_log, init_metrics, self.code_to_optimize, elapsed_cycles=cycles
            )
            
            self.agent_logger.info(f"Initial state: {initial_state}, cycles: {cycles}")
            
            # Save initial files
            (self.folder / "0_init_annotated.cu").write_text(annotated_ncu)
            
        except FeedbackError as e:
            # Log the failure but continue with a fallback analysis so the agent can proceed.
            self.agent_logger.warning(
                f"Initial profiling failed numeric verification; proceeding with fallback state. Details: {e}"
            )

            # Use basic fallback state; keep cycles as None so we do not report bogus values.
            init_metrics = {}
            initial_state_profile = self.database._fallback_state_analysis("", init_metrics)
            initial_state = initial_state_profile.state_name
            # Leave self.initial_cycles unchanged (None by default). Keep best_cycles as-is.
            # Persist placeholder NCU log for downstream steps
            self.last_ncu_log = ""

            # Fallback: write annotated file using raw init.cu so downstream steps can proceed
            try:
                init_src = self.code_to_optimize_fp.read_text()
                (self.folder / "0_init_annotated.cu").write_text(init_src)
            except Exception as _e:
                self.agent_logger.warning(f"Failed to write fallback 0_init_annotated.cu: {_e}")

    async def run(self) -> Path:
        """
        Override the base run method to implement RL-specific behavior.
        Runs multiple RL iterations **in parallel** and returns the best result.
        """
        import asyncio as _asyncio
        
        best_filename = None
        best_cycles = float('inf')

        # Ensure initial profiling data is available ONCE before spawning tasks
        if not hasattr(self, "last_ncu_log") or not self.last_ncu_log:
            await self.initialize()
        # Compute and share the initial state derived from the initial NCU log
        initial_state_shared = await self.database.get_state_from_ncu_report(
            self.last_ncu_log,
            parse_ncu_metrics(self.last_ncu_log),
            self.code_to_optimize,
            elapsed_cycles=self.initial_cycles,
        )

        async def _run_single_iteration(iteration_idx: int):
            """Helper that performs one rollout and returns its trajectory."""
            self.agent_logger.info(
                f"[Async] RL Iteration {iteration_idx + 1}/{self.num_rl_iterations}")
            try:
                # Initial state derived once and shared across iterations
                initial_state = initial_state_shared

                trajectory = await self.run_rollout(self.code_to_optimize, initial_state)
                return iteration_idx, trajectory
            except Exception as exc:
                self.agent_logger.error(
                    f"RL iteration {iteration_idx + 1} failed: {exc}")
                return iteration_idx, None

        # Launch all iterations concurrently
        tasks = [_asyncio.create_task(_run_single_iteration(i)) for i in range(self.num_rl_iterations)]

        for coro in _asyncio.as_completed(tasks):
            iteration_idx, trajectory = await coro
            if trajectory is None:
                continue

            # Process trajectory results
            if trajectory.steps:
                best_step = min(trajectory.steps, key=lambda s: s.cycles)
                if best_step.cycles < best_cycles:
                    best_cycles = best_step.cycles
                    best_filename_path = self.folder / f"rl_iter_{iteration_idx}_best.cu"
                    best_filename_path.write_text(
                        best_step.code + f"\n\n// Elapsed Cycles: {best_step.cycles}\n")
                    best_filename = best_filename_path
                    self.agent_logger.info(
                        f"[Async] New best result from iter {iteration_idx}: {best_cycles} cycles")

            # Update replay buffer & trajectory counters **after** trajectory is complete
            if trajectory:
                self.replay_buffer.add_trajectory(trajectory)
                self.total_trajectories += 1

        # After all tasks completed
        # Persist a numbered snapshot of the optimisation database JSON
        try:
            # Ensure current DB state is persisted
            self.database._persist_database()
            persist_fp = self.database._persist_json_fp
            snapshots_dir = persist_fp.parent / "snapshots"
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            existing = sorted(snapshots_dir.glob("optimization_database_*.json"))
            next_idx = len(existing)
            snapshot_fp = snapshots_dir / f"optimization_database_{next_idx}.json"
            snapshot_fp.write_text(persist_fp.read_text(encoding="utf-8"), encoding="utf-8")
            self.agent_logger.info(f"Saved database snapshot to {snapshot_fp}")
        except Exception as snap_exc:
            self.agent_logger.warning(f"Failed to write database snapshot: {snap_exc}")

        if best_filename is not None:
            # Ensure we have baseline cycles to judge improvement.
            try:
                if self.initial_cycles is None:
                    init_fp = getattr(self, "code_to_optimize_fp", None)
                    if not init_fp or not init_fp.exists():
                        self.code_to_optimize_fp = self.folder / "init.cu"
                        self.code_to_optimize_fp.write_text(self.code_to_optimize)
                    _, _, _, baseline_cycles = await self.gather_perf_metrics(self.code_to_optimize_fp)
                    self.initial_cycles = baseline_cycles
            except Exception as e:
                self.agent_logger.warning(
                    f"Failed to obtain baseline cycles before finalizing result: {e}"
                )

            # Decide success vs failure based on improvement over baseline.
            if self.initial_cycles is not None and best_cycles < self.initial_cycles:
                final_filename = self.folder / "success_rl_optimization.cu"
                final_filename.write_text(best_filename.read_text())
                return final_filename

            failure_file = self.folder / "failure_rl_optimization.cu"
            baseline_str = self.initial_cycles if self.initial_cycles is not None else "N/A"
            try:
                failure_file.write_text(
                    self.code_to_optimize + f"\n\n// Elapsed Cycles: {baseline_str}\n"
                )
            except Exception:
                try:
                    init_fp = getattr(self, "code_to_optimize_fp", None)
                    if init_fp and init_fp.exists():
                        failure_file.write_text(
                            init_fp.read_text() + f"\n\n// Elapsed Cycles: {baseline_str}\n"
                        )
                except Exception:
                    pass
            self.agent_logger.error(
                "RL did not produce an improvement; wrote failure_rl_optimization.cu with baseline (if available)"
            )
            return failure_file

        # No trajectory produced a candidate. Still write a failure file (with baseline if available).
        try:
            if self.initial_cycles is None:
                init_fp = getattr(self, "code_to_optimize_fp", None)
                if not init_fp or not init_fp.exists():
                    self.code_to_optimize_fp = self.folder / "init.cu"
                    self.code_to_optimize_fp.write_text(self.code_to_optimize)
                _, _, _, cycles = await self.gather_perf_metrics(self.code_to_optimize_fp)
                self.initial_cycles = cycles
                self.best_cycles = min(self.best_cycles, cycles) if self.best_cycles else cycles
        except Exception as e:
            self.agent_logger.warning(f"Failed to obtain baseline cycles for original code: {e}")

        fallback_cycles = self.initial_cycles if self.initial_cycles is not None else "N/A"
        failure_file = self.folder / "failure_rl_optimization.cu"
        try:
            failure_file.write_text(self.code_to_optimize + f"\n\n// Elapsed Cycles: {fallback_cycles}\n")
        except Exception:
            try:
                init_fp = getattr(self, "code_to_optimize_fp", None)
                if init_fp and init_fp.exists():
                    failure_file.write_text(init_fp.read_text() + f"\n\n// Elapsed Cycles: {fallback_cycles}\n")
            except Exception:
                pass
        self.agent_logger.error(
            "All RL iterations failed; wrote failure_rl_optimization.cu with baseline (if available)"
        )
        return failure_file

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

    async def run_rollout(self, initial_code: str, initial_state: str) -> Trajectory:
        """Run a single optimization rollout trajectory."""
        import json as _json, random, uuid as _uuid
        from dataclasses import asdict

        # --------------------------------------------------------------
        # Create per-trajectory folder for logs & artefacts
        # --------------------------------------------------------------
        async with self._trajectory_lock:
            self.total_trajectories += 1
            trajectory_index = self.total_trajectories  # Unique per agent instance

        # Use uuid suffix to avoid folder name collisions in concurrent runs
        _uid = _uuid.uuid4().hex[:8]
        trajectory_dir = self.folder / f"trajectory_{trajectory_index}_{_uid}"
        trajectory_dir.mkdir(parents=True, exist_ok=True)

        # Initialise trajectory container
        trajectory = Trajectory()

        current_code: str = initial_code
        current_state: str = initial_state
        current_cycles: int = self.initial_cycles
        last_ncu_log: str = getattr(self, "last_ncu_log", "")
        
        self.agent_logger.info(f"Starting rollout from state: {current_state}")
        

        for step in range(self.max_rollout_steps):
            # ----------------------------------------------------------
            # 1) Analyse current performance state using the LLM helper
            # ----------------------------------------------------------
            metrics = parse_ncu_metrics(last_ncu_log)
            # print("CURRENT CODE: ", current_code)
            # exit(0)
            try:
                profile = await self.database.analyze_performance_state(
                    last_ncu_log, metrics, current_code, elapsed_cycles=current_cycles
                )
                analysis_json_str = _json.dumps(asdict(profile), indent=2)

                # --------------------------------------------------
                # 2) Ask the DB to generate a ranked optimisation plan
                # --------------------------------------------------
                # Dynamic top_n based on rollout step (1-based)
                cur_iter = step + 1
                plan = await self.database.generate_optimization_plan(
                    analysis_json_str, current_code, top_n= max(4,(self.max_rollout_steps-int(cur_iter))))
                
            except Exception as exc:
                self.agent_logger.warning(f"Plan generation failed, falling back: {exc}")
                plan = []

            # ----------------------------------------------------------
            # 3) Pick one technique randomly weighted by relevance score
            # ----------------------------------------------------------
            optimization_entry = None
            if plan:
                def _safe_rel(x):
                    try:
                        r = float(x)
                    except (TypeError, ValueError):
                        r = 0.05
                    return min(max(r, 0.0), 1.0)
                # Optional deterministic selection for reproducibility/debugging.
                # If set, choose the single highest-relevance item instead of sampling.
                force_top1 = os.getenv("KERNELAGENT_DB_FALLBACK_TOP1", "0") in (
                    "1",
                    "true",
                    "True",
                    "yes",
                    "YES",
                    "y",
                    "on",
                    "ON",
                )
                if force_top1:
                    chosen_plan = max(
                        plan,
                        key=lambda p: _safe_rel(p.get("relevance_score", 0.05)),
                    )
                    self.agent_logger.info(
                        f"KERNELAGENT_DB_FALLBACK_TOP1 is set; deterministically selecting top-1 from LLM plan: "
                        f"{chosen_plan.get('technique')} (relevance {chosen_plan.get('relevance_score', 0.0)})"
                    )
                else:
                    # Cube the relevance to downweight low-relevance options
                    weights = [max(_safe_rel(p.get("relevance_score", 0.05)) ** 3, 0.001) for p in plan]
                    chosen_plan = random.choices(plan, weights=weights, k=1)[0]
                technique_name = chosen_plan.get("technique")

                # Helper to locate the corresponding entry in the DB
                optimization_entry = self._lookup_optim_entry_by_name(technique_name)
                strategy_description = chosen_plan.get("description", "")

                self.agent_logger.info(
                    f"Selected technique from optimisation plan: {technique_name} (relevance {chosen_plan.get('relevance_score', 0.0):.2f})"
                )

            # ----------------------------------------------------------
            # 4) Fallback to legacy database chooser if needed
            # ----------------------------------------------------------
            if optimization_entry is None:
                optimization_entry = self.database.select_best_optimization(current_state)
                if optimization_entry is None:
                    # Try to find unused optimizations
                    optimization_entry = self.database.select_best_optimization(current_state, exclude_used=True)
                    if optimization_entry is None:
                        # Try to find any optimization from all states as fallback
                        all_states_with_optimizations = [
                            state for state, state_data in self.database.optimization_strategies.items()
                            if len(state_data.get("optimizations", [])) > 0
                        ]
                        
                        for state_name in all_states_with_optimizations:
                            optimization_entry = self.database.select_best_optimization(state_name)
                            if optimization_entry is not None:
                                self.agent_logger.info(f"Using fallback optimization from state: {state_name}")
                                break
                        
                        if optimization_entry is None:
                            # Last resort: try to add default optimizations for the discovered state
                            if self._try_add_default_optimizations(current_state):
                                optimization_entry = self.database.select_best_optimization(current_state)
                                if optimization_entry is not None:
                                    self.agent_logger.info(f"Using default optimization for new state: {current_state}")
                    
                    if optimization_entry is None:
                        self.agent_logger.warning(f"No optimization found for state: {current_state}, stopping rollout at step {step}")
                        break
                else:
                    self.agent_logger.info(f"Using unused optimization for state: {current_state}")
            else:
                self.agent_logger.info(f"Using best optimization for state: {current_state}")

            # Get the technique name based on the optimization type
            if isinstance(optimization_entry, CompositeOptimization):
                technique_name = optimization_entry.get_composite_id()
            elif hasattr(optimization_entry, "technique"):
                technique_name = optimization_entry.technique
            else:
                technique_name = str(optimization_entry)
            
            # Safe predicted value for logging
            _pred_impr = getattr(optimization_entry, "predicted_improvement", None)
            pred_log = f" (predicted: {_pred_impr}%)" if _pred_impr is not None else ""
            self.agent_logger.info(
                f"Step {step}: Applying {technique_name}{pred_log} | entry_type={type(optimization_entry).__name__}"
            )
            
            try:
                # Apply optimization
                optimized_code, new_cycles, new_state, new_ncu_log = await self.apply_optimization(
                    current_code,
                    optimization_entry,
                    step,
                    trajectory_dir,
                    strategy_description if 'strategy_description' in locals() else "",
                )
                
                # Calculate actual improvement
                if current_cycles is not None and current_cycles > 0:
                    actual_improvement = ((current_cycles - new_cycles) / current_cycles) * 100
                else:
                    # Baseline unknown; treat improvement as 0 for reward/logging purposes
                    actual_improvement = 0.0
                reward = self.calculate_reward(
                    getattr(optimization_entry, "predicted_improvement", None), 
                    actual_improvement,
                    (current_cycles is not None and new_cycles < current_cycles)
                )
                
                # Create trajectory step
                action_name = (
                    optimization_entry.get_composite_id()
                    if isinstance(optimization_entry, CompositeOptimization)
                    else getattr(optimization_entry, "technique", str(optimization_entry))
                )
                
                traj_step = TrajectoryStep(
                    state=current_state,
                    action=action_name,
                    code=optimized_code,
                    cycles=new_cycles,
                    predicted_improvement=getattr(optimization_entry, "predicted_improvement", 0.0),
                    actual_improvement=actual_improvement,
                    reward=reward
                )
                self.agent_logger.info(f"Adding trajectory step: {traj_step}")
                trajectory.add_step(traj_step)

                self.agent_logger.info(f"Updating database with actual results for {technique_name} in state {current_state} with actual improvement {actual_improvement}")
                # Update database with actual results
                if isinstance(optimization_entry, CompositeOptimization):
                    self.database.update_composite_optimization_result(
                        current_state,
                        technique_name,
                        actual_improvement
                    )
                else:
                    # Log this with logger
                    self.agent_logger.info(f"Updating optimization result for {technique_name} in state {current_state} with actual improvement {actual_improvement}")
                    self.database.update_optimization_result(
                        current_state, 
                        technique_name,
                        actual_improvement
                    )
                
                self.agent_logger.info(
                    f"Step {step} result: {new_cycles} cycles "
                    f"({actual_improvement:.1f}% improvement, reward: {reward:.2f})"
                )
                
                # Update for next step
                current_code = optimized_code
                current_state = new_state
                current_cycles = new_cycles
                last_ncu_log = new_ncu_log or last_ncu_log  # keep for next iteration
                
                # Early stopping if severe degradation (relaxed from -20% to -50% to -500%)
                if actual_improvement < -500:  # More than 500% slower
                    self.agent_logger.warning(f"Stopping rollout due to severe degradation: {actual_improvement:.1f}%")
                    break
                    
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                # Log detailed traceback and context
                try:
                    self.agent_logger.error(
                        f"Error in step {step}: {e}\n"
                        f"Technique: {technique_name} | Entry type: {type(optimization_entry).__name__}\n"
                        f"Raw optimization entry: {optimization_entry}\n"
                        f"Traceback:\n{tb}"
                    )
                except Exception:
                    # Fallback if logger formatting fails
                    print(f"Error in step {step}: {e}\n{tb}")
                break
        
        return trajectory

    # ------------------------------------------------------------------
    # Helper to find an optimisation entry by its technique/composite ID
    # ------------------------------------------------------------------
    def _lookup_optim_entry_by_name(
        self, technique_name: str
    ) -> Optional[OptimizationEntry | CompositeOptimization]:
        # Search individual techniques
        for state_data in self.database.optimization_strategies.values():
            for opt in state_data.get("optimizations", []):
                if opt.technique == technique_name:
                    return opt

        # Search composite optimisations
        for comps in self.database.composite_optimizations.values():
            for comp in comps:
                if comp.get_composite_id() == technique_name:
                    return comp

        return None

    async def apply_optimization(
        self,
        code: str,
        optimization_entry: OptimizationEntry | CompositeOptimization,
        step: int,
        trajectory_dir: Path | None = None,
        strategy_description: str = "",
    ) -> Tuple[str, int, str, str]:
        """Apply a specific optimization and return the optimized code, cycles, new state."""
        # --------------------------------------------------------------
        # Helper to persist prompt/response pairs for agentic inspection
        # --------------------------------------------------------------
        def _save_agentic_log(label: str, prompt_text: str, response_text: str):
            if trajectory_dir is None:
                return  # Logging disabled if no directory provided
            log_fp = trajectory_dir / "agentic_steps_log.txt"
            with open(log_fp, "a", encoding="utf-8") as f:
                f.write(f"=== {label} ===\n")
                f.write("--- PROMPT ---\n")
                f.write(prompt_text.rstrip() + "\n")
                f.write("--- RESPONSE ---\n")
                f.write(response_text.rstrip() + "\n\n")
        
        # Create temporary file for this optimization attempt
        if isinstance(optimization_entry, CompositeOptimization):
            technique_name = optimization_entry.get_composite_id()
        else:
            technique_name = getattr(optimization_entry, "technique", str(optimization_entry))
        base_label = f"step_{step}_{technique_name}"
        # Route all per-step artifacts into the trajectory directory when available
        base_dir = trajectory_dir if trajectory_dir is not None else self.folder
        temp_file = base_dir / f"step_{step}_{technique_name}.cu"
        temp_file.write_text(code)
        
        # Gather current profiling data; tolerate numeric-verification failures
        try:
            annotated_ncu, ncu_log, _, _ = await self.gather_perf_metrics(temp_file)
        except FeedbackError as prof_err:
            self.agent_logger.warning(
                f"Profiling failed at step {step} with FeedbackError; using empty NCU log. Details: {prof_err}"
            )
            annotated_ncu, ncu_log = "", ""
        except Exception as prof_other:
            self.agent_logger.warning(
                f"Unexpected profiling error at step {step}: {prof_other}; continuing with empty logs."
            )
            annotated_ncu, ncu_log = "", ""
        
        # Generate strategy-guided prompt with full database content
        try:
            database_content = self.database.get_database_md_text()
            # Fallback to footer if full database is empty
            if not database_content or database_content.strip() == "":
                self.agent_logger.warning("Database markdown is empty, trying footer")
                database_content = self.database.get_database_footer_text()
                if not database_content or database_content.strip() == "":
                    self.agent_logger.warning("Database footer is also empty, using GPU optimization knowledge")
                    # Final fallback: use GPU optimization report
                    database_content = getattr(self.database, 'gpu_optimization_knowledge', '')[:6000] or ""
        except Exception as e:
            self.agent_logger.warning(f"Failed to load database content: {e}")
            try:
                database_content = self.database.get_database_footer_text()
            except Exception:
                # Final fallback
                database_content = getattr(self.database, 'gpu_optimization_knowledge', '')[:6000] or ""
        
        # Log database content size for debugging
        if database_content:
            self.agent_logger.debug(f"Using database content: {len(database_content)} characters")
        else:
            self.agent_logger.warning("No database content available for prompt")
        prompt = generate_strategy_guided_prompt(
            optimization_entry,
            annotated_ncu,
            ncu_log,
            database_content,
            override_description=strategy_description or None,
            original_code=code,  # Pass original code as fallback when annotated_ncu is empty
        )

        # Persist initial prompt/response
        # (logging occurs after LLM response is available)
        
        # Get optimized code from LLM
        from .utils import generate_code_retry
        response = await generate_code_retry(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            logger=self.agent_logger,
            max_retries=3
        )

        # Persist initial prompt/response
        _save_agentic_log(f"{base_label}_initial", prompt, response.generations[0])
        
        # Extract and test optimized code
        optimized_code, filepath = self.get_code_from_response(
            response.generations[0], step, 0, self.agent_logger
        )
        # Relocate the intermediate file produced by get_code_from_response into the trajectory folder to avoid collisions
        try:
            target_fp = base_dir / f"{base_label}_initial.cu"
            # If the source and destination differ, move contents
            if filepath != target_fp:
                # Prefer rename; fallback to rewrite if cross-device
                try:
                    filepath.rename(target_fp)
                except Exception:
                    target_fp.write_text(optimized_code)
                    try:
                        filepath.unlink()
                    except Exception:
                        pass
            filepath = target_fp
        except Exception:
            # Best-effort; continue even if relocation fails
            pass

        # --------------------------------------------------------------
        # 2) Compile / run profiling with automatic fix attempts
        # --------------------------------------------------------------
        MAX_FIX_ATTEMPTS = 4  # How many times to attempt auto-repairs

        attempt_idx = 0
        compile_success = False
        run_success = False
        new_cycles = 0
        new_ncu_log = ""

        while attempt_idx < MAX_FIX_ATTEMPTS:
            # Write the (potentially fixed) code to a unique file
            filepath = base_dir / f"{base_label}_attempt{attempt_idx}.cu"
            filepath.write_text(optimized_code)

            try:
                # Profile the optimized code (this implicitly compiles + runs it)
                _, new_ncu_log, _, new_cycles = await self.gather_perf_metrics(filepath)

                # If we reach here, compilation and run were successful
                compile_success = True
                run_success = True

                # Log compile / run success for inspection
                if trajectory_dir is not None:
                    log_fp = trajectory_dir / "agentic_steps_log.txt"
                    with open(log_fp, "a", encoding="utf-8") as f:
                        f.write(f"Compile success: {compile_success}\n")
                        f.write(f"Run success    : {run_success}\n")
                        f.write(f"Elapsed cycles  : {new_cycles}\n\n")

                break  # Exit retry loop – success

            except Exception as e:
                # Compilation or runtime failed – capture error message
                error_msg = str(e)

                # Append failure info to agentic log
                if trajectory_dir is not None:
                    log_fp = trajectory_dir / "agentic_steps_log.txt"
                    with open(log_fp, "a", encoding="utf-8") as f:
                        f.write(f"Compile/Run failed on attempt {attempt_idx}: {error_msg}\n\n")

                attempt_idx += 1
                if attempt_idx >= MAX_FIX_ATTEMPTS:
                    # Give up and propagate the error – outer caller will handle
                    raise

                # ------------------------------------------------------
                # Build a fix prompt for the LLM using the error message
                # ------------------------------------------------------
                # Try to include the Optimization Database footer (contains useful code snippets)
                db_footer_text = ""
                try:
                    if hasattr(self.database, "optimization_db_footer_path") and self.database.optimization_db_footer_path.exists():
                        db_footer_text = self.database.optimization_db_footer_path.read_text(encoding="utf-8")
                except Exception:
                    db_footer_text = ""

                fix_prompt_parts = [
                    "The previously generated CUDA code failed to compile or run.\n\n",
                    "COMPILER / RUNTIME ERROR LOG:\n```\n",
                    f"{error_msg}\n",
                    "```\n\n",
                    "ORIGINAL CUDA CODE (for reference – please modify in place):\n```cpp\n",
                    f"{optimized_code}\n",
                    "```\n\n",
                ]
                if db_footer_text:
                    fix_prompt_parts.append("OPTIMIZATION DATABASE FOOTER (reference snippets):\n```\n")
                    fix_prompt_parts.append(db_footer_text)
                    fix_prompt_parts.append("\n```\n\n")
                fix_prompt_parts.extend([
                    "Please provide a corrected, fully compilable version of the kernel. Return **complete CUDA code** in one ```cpp``` block.",
                    " Please keep the code structure otherwise unchanged; it is compiled together with separate test code, so do NOT add a main function.\n\n",
                    "Include ALL necessary components:\n",
                    "   - #include statements (cuda_fp16.h, cuda_runtime.h, etc.)\n",
                    "   - #define constants – DEFINE ALL CONSTANTS BEFORE USING THEM\n",
                    "   - Complete __global__ kernel function with proper signature\n",
                    "   - Complete launch_gpu_implementation(void*, void*, void*, int64_t) function\n",
                ])
                fix_prompt = "".join(fix_prompt_parts)

                # Ask the LLM to fix the code
                fix_response = await generate_code_retry(
                    messages=[{"role": "user", "content": fix_prompt}],
                    model=self.model,
                    logger=self.agent_logger,
                    max_retries=2,
                )

                # Log the fix attempt prompt/response
                _save_agentic_log(
                    f"{base_label}_fix_attempt_{attempt_idx}",
                    fix_prompt,
                    fix_response.generations[0],
                )

                # Extract new code for next iteration
                optimized_code, fix_fp = self.get_code_from_response(
                    fix_response.generations[0], step, attempt_idx, self.agent_logger
                )
                # Relocate the intermediate fix file to the trajectory directory to avoid collisions
                try:
                    fix_target_fp = base_dir / f"{base_label}_attempt{attempt_idx}_llm.cu"
                    if fix_fp != fix_target_fp:
                        try:
                            fix_fp.rename(fix_target_fp)
                        except Exception:
                            fix_target_fp.write_text(optimized_code)
                            try:
                                fix_fp.unlink()
                            except Exception:
                                pass
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # 3) Determine new state (only if compilation/run succeeded)
        # ------------------------------------------------------------------
        new_metrics = parse_ncu_metrics(new_ncu_log)
        # new_state = await self.database.get_state_from_ncu_report(new_ncu_log, new_metrics)
        new_state = None  # TODO: Temp Disable state update
        return optimized_code, new_cycles, new_state, new_ncu_log

    def calculate_reward(self, predicted_improvement: Optional[float], actual_improvement: float, 
                        is_faster: bool) -> float:
        """Calculate reward based on prediction accuracy and actual performance.
        Safely handles None/zero predicted_improvement by skipping accuracy bonus.
        """
        
        # Base reward for improvement
        base_reward = actual_improvement / 100.0  # Convert percentage to fraction
        
        # Bonus for prediction accuracy
        try:
            safe_predicted = float(predicted_improvement) if predicted_improvement is not None else 0.0
        except (TypeError, ValueError):
            safe_predicted = 0.0
        
        if safe_predicted > 0.0:
            accuracy = min(actual_improvement / safe_predicted, 2.0)
            if 0.8 <= accuracy <= 1.2:  # Good prediction
                accuracy_bonus = 0.2
            else:  # Poor prediction
                accuracy_bonus = -0.1 * abs(accuracy - 1.0)
        else:
            accuracy_bonus = 0.0
        
        # Penalty for making things worse
        penalty = -0.5 if not is_faster else 0.0
        
        return base_reward + accuracy_bonus + penalty

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

    async def get_feedback(self, response, attempt_id, task_id, logger) -> Feedback:
        """Main feedback loop implementing the RL algorithm."""
        
        # Initialize if this is the first call
        if self.initial_cycles is None:
            await self.initialize()
        
        # Start a new trajectory for this task
        logger.info(f"Starting RL optimization trajectory for task {task_id}")
        
        # Get initial state
        temp_file = self.folder / f"temp_task_{task_id}.cu"
        code, filepath = self.get_code_from_response(response, attempt_id, task_id, logger)
        
        try:
            # Profile initial code to determine state
            annotated_ncu, ncu_log, _, cycles = await self.gather_perf_metrics(filepath)
            metrics = parse_ncu_metrics(ncu_log)
            initial_state = await self.database.get_state_from_ncu_report(ncu_log, metrics, code, elapsed_cycles=cycles)
            
            # Run optimization rollout
            trajectory = await self.run_rollout(code, initial_state)
            
            # Add trajectory to replay buffer
            self.replay_buffer.add_trajectory(trajectory)
            self.total_trajectories += 1
            
            # Update best performance
            if trajectory.final_cycles < self.best_cycles:
                self.best_cycles = trajectory.final_cycles
                is_faster = True
            else:
                is_faster = False
            
            # Prepare feedback messages
            if trajectory.steps:
                best_step = min(trajectory.steps, key=lambda s: s.cycles)
                if self.initial_cycles is not None and self.initial_cycles > 0:
                    improvement_pct = ((self.initial_cycles - best_step.cycles) / self.initial_cycles) * 100
                else:
                    improvement_pct = 0.0
                
                feedback_msg = f"""Optimization trajectory completed with {len(trajectory.steps)} steps.

BEST RESULT:
- Cycles: {best_step.cycles} (vs initial: {self.initial_cycles})
- Overall improvement: {improvement_pct:.1f}%
- Best technique: {best_step.action}
- Total reward: {trajectory.total_reward:.2f}

FINAL OPTIMIZED CODE:
```
{best_step.code}
```

The optimization process is learning and adapting. Continue with further optimizations."""
                
                new_messages = [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": feedback_msg}
                ]
                
                # Save best result
                best_file = self.folder / f"best_task_{task_id}.cu"
                best_file.write_text(best_step.code)
                
                return RLNCUFeedback(
                    new_messages=new_messages,
                    success=True,  # Consider successful if we completed a trajectory
                    filename=str(best_file),
                    contents=best_step.code,
                    elapsed_cycles=best_step.cycles,
                    ncu_log=ncu_log,
                    annotated_ncu=annotated_ncu,
                    optimization_technique=best_step.action,
                    predicted_improvement=best_step.predicted_improvement,
                    actual_improvement=best_step.actual_improvement,
                    state=initial_state
                )
            else:
                # No successful optimization steps
                return RLNCUFeedback(
                    new_messages=[
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": "No successful optimization steps completed. Please try a different approach."}
                    ],
                    success=False,
                    elapsed_cycles=cycles,
                    ncu_log=ncu_log,
                    annotated_ncu=annotated_ncu,
                    state=initial_state
                )
                
        except FeedbackError as e:
            logger.error(f"Error in RL optimization: {e}")
            return Feedback(
                new_messages=[
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": f"Optimization failed: {str(e)}. Please fix the issues and try again."}
                ],
                success=False,
                feedback=e.feedback if hasattr(e, 'feedback') else str(e)
            )

    def _try_add_default_optimizations(self, current_state: str) -> bool:
        """
        Try to add default optimizations for a discovered state based on its characteristics.
        
        This is a fallback mechanism when no optimizations are found.
        """
        try:
            # Define default optimizations based on common patterns
            default_optimizations = {
                "memory_bound": [
                    ("memory_coalescing_optimization", 20.0),
                    ("shared_memory_tiling", 25.0),
                    ("vectorized_processing", 15.0)
                ],
                "compute_bound": [
                    ("instruction_level_parallelism", 30.0),
                    ("fast_math_optimization", 20.0),
                    ("vectorized_operations", 25.0)
                ],
                "latency_bound": [
                    ("occupancy_optimization", 35.0),
                    ("register_pressure_reduction", 30.0),
                    ("work_per_thread_increase", 25.0)
                ],
                "hybrid_bound": [
                    ("memory_compute_overlap", 40.0),
                    ("algorithmic_optimization", 35.0),
                    ("adaptive_tiling", 30.0)
                ]
            }
            
            # Extract the primary bottleneck from the state name
            primary_bottleneck = None
            for bottleneck in default_optimizations.keys():
                if bottleneck in current_state:
                    primary_bottleneck = bottleneck
                    break
            
            if primary_bottleneck and primary_bottleneck in default_optimizations:
                # Add default optimizations for this state
                for technique, improvement in default_optimizations[primary_bottleneck]:
                    self.database.add_new_optimization(current_state, technique, improvement)
                
                self.agent_logger.info(f"Added {len(default_optimizations[primary_bottleneck])} default optimizations for state: {current_state}")
                return True
            
        except Exception as e:
            self.agent_logger.error(f"Error adding default optimizations: {e}")
        
        return False

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'total_trajectories': self.total_trajectories,
            'iteration_count': self.iteration_count,
            'initial_cycles': self.initial_cycles,
            'best_cycles': self.best_cycles,
            'overall_improvement': ((self.initial_cycles - self.best_cycles) / self.initial_cycles * 100) if self.initial_cycles else 0,
            'buffer_stats': self.replay_buffer.get_statistics(),
            'database_stats': self.database.get_database_stats()
        } 