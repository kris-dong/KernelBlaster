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
Minimal Single-Shot Optimization Agent.

Simple flow for each iteration:
1. Take starting code
2. Get profile results (or profile inline)
3. Pack: code + profile + database knowledge -> prompt
4. Request optimized version from LLM
5. Get new perf results from optimized code
6. Repeat for next iteration

No state matching, no strategy selection - just the most basic agent flow.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json
import loguru
import uuid
import asyncio
from dataclasses import dataclass, asdict

from ..config import GPUType
from .feedback import FeedbackAgent, FeedbackConfig, Feedback
from .database import OptimizationDatabase, LLMInterface
from .utils import (
    FeedbackError,
    compile_and_run_cu_file,
    find_kernel_names_ncu,
    get_elapsed_cycles_ncu_log,
    NamedTimer,
    format_ncu_details_as_csv,
    format_ncu_source_as_csv,
    annotate_source,
    UTILIZATION_METRICS,
    run_gpu_executable,
    extract_code_from_response,
    write_code_to_file,
)
from .opt_ncu_rl import parse_ncu_metrics, generate_strategy_guided_prompt
import os


@dataclass
class ProfilingData:
    """Profiling data from reprofile agent."""
    ncu_log: str
    annotated_ncu: Optional[str] = None
    cycles: int = 0
    metrics: Dict[str, float] = None
    kernel_names: list = None
    
    @classmethod
    def from_profiling_result(cls, result_path: Path) -> "ProfilingData":
        """Load profiling data from a reprofile result directory."""
        summary_file = result_path / "summary.json"
        ncu_log_file = result_path / "ncu_log.txt"
        annotated_file = result_path / "annotated_source.txt"
        metrics_file = result_path / "metrics.json"
        
        if not summary_file.exists():
            raise ValueError(f"Summary file not found: {summary_file}")
        
        summary = json.loads(summary_file.read_text())
        ncu_log = ncu_log_file.read_text() if ncu_log_file.exists() else ""
        annotated_ncu = annotated_file.read_text() if annotated_file.exists() else None
        metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
        
        return cls(
            ncu_log=ncu_log,
            annotated_ncu=annotated_ncu,
            cycles=summary.get("cycles", 0),
            metrics=metrics,
            kernel_names=summary.get("kernel_names", []),
        )


SYSTEM_PROMPT = """You are a knowledgeable and efficient CUDA programming assistant, skilled in analyzing NSight Compute logs and optimizing CUDA kernels. Your task is to generate optimized CUDA code based on the provided optimization strategy and profiling data.

CRITICAL REQUIREMENTS:
1. Generate COMPLETE, COMPILABLE CUDA code
2. Include ALL necessary components:
   - #include statements (cuda_fp16.h, cuda_runtime.h, etc.)
   - #define constants - DEFINE ALL CONSTANTS BEFORE USING THEM
   - Complete __global__ kernel function with proper signature
   - Complete launch_gpu_implementation() function
3. Format ALL code in a single ```cpp code block
4. Apply the optimization strategy systematically
5. Ensure all constants are properly defined before use
"""

# Configuration constants
MAX_FIX_RETRIES = 4  # Max retries for LLM fix attempts (set to 1 to disable retries)
MAX_FIX_BUDGET = 4  # Total fix budget shared across all iterations in a trajectory (set to 1 to disable retries)
MAX_FIX_ATTEMPTS_PER_ITERATION = 4  # Maximum fix attempts per iteration (caps the budget usage per iteration)


class MinimalOptimizationAgent(FeedbackAgent):
    """
    Minimal single-shot optimization agent with outer loop support.
    
    Uses kernel code and optimization database to select strategy and generate optimized code.
    
    Simple flow:
    1. Initialize: Profile initial code inline (or use provided profiling data)
    2. Each iteration:
       - Pack: code + profile + database knowledge -> prompt
       - Get optimized code from LLM
       - Profile optimized code to get new perf results
       - Repeat with new profiling data
    
    Key features:
    - Simple flow: code + profile + database -> optimized code (no state matching, no strategy selection)
    - Single-shot optimization within each iteration (one LLM call per attempt)
    - Supports outer loops: iterations (max_attempts) and rollouts (num_pgen)
    - Error correction: retries on compilation/execution errors or performance issues
    - Uses base FeedbackAgent.run() for iteration/rollout management
    - Final solution selection: Best performance (lowest cycles) across all iterations/rollouts
    """
    
    def __init__(
        self,
        fb_config: FeedbackConfig,
        code_to_optimize_fp: Path,
        profiling_data: Optional[ProfilingData] = None,  # Optional - will profile inline if not provided
        database_path: Optional[Path] = None,
        database: Optional[OptimizationDatabase] = None,
        timeout: int = 3600,
    ):
        # Set system prompt
        if fb_config.system_prompt is None:
            fb_config.system_prompt = SYSTEM_PROMPT
        
        super().__init__(fb_config)
        
        self.test_code_fp = fb_config.test_code_fp
        self.test_code = fb_config.test_code_fp.read_text()
        self.code_to_optimize_fp = code_to_optimize_fp
        self.code_to_optimize = code_to_optimize_fp.read_text()
        self.profiling_data = profiling_data  # Optional - for backward compatibility
        self.timeout = timeout
        
        # Debug logging - check what we got from FeedbackAgent
        self.agent_logger.info(f"DEBUG: After super().__init__, self.num_pgen={self.num_pgen}, self.max_attempts={self.max_attempts}")
        
        # Track trajectories and best results (matching RL agent structure)
        self.total_trajectories = 0
        self._trajectory_lock = asyncio.Lock()
        self.trajectory_dirs = {}  # task_id -> trajectory_dir
        self.best_cycles = None  # Track best cycles across iterations
        self.last_ncu_log = None  # Store NCU log for database queries
        
        # Initialize database
        if database is not None:
            self.database = database
        else:
            gpu_report_path = Path(__file__).parent.parent.parent.parent.parent / "algo-sol-modeling/algo-space/gpu_optimization_report.md"
            llm_interface = LLMInterface(self.model, self.agent_logger)
            if database_path is None:
                # Try to find database in common locations
                database_path = self.base_folder.parent.parent.parent / "optimization_database.md"
            self.database = OptimizationDatabase(database_path, gpu_report_path, llm_interface)
    
    async def initialize(self):
        """Initialize by profiling code inline (simple flow: code + profile + database -> optimized code)."""
        # Copy init cu file to folder
        original_code_path = self.code_to_optimize_fp
        self.code_to_optimize_fp = self.folder / "init.cu"
        self.agent_logger.info(f"Initializing with code from: {original_code_path}")
        self.agent_logger.info(f"Code will be copied to: {self.code_to_optimize_fp}")
        self.agent_logger.info(f"Code length: {len(self.code_to_optimize)} characters")
        self.code_to_optimize_fp.write_text(self.code_to_optimize)
        
        # Profile inline if profiling_data not provided
        if self.profiling_data is None:
            self.agent_logger.info("Profiling initial code inline...")
            try:
                annotated_ncu, init_ncu_log, _, cycles = await self.gather_perf_metrics(
                    self.code_to_optimize_fp
                )
                self.initial_cycles = cycles
                self.best_cycles = cycles
                self.last_ncu_log = init_ncu_log
                
                # Save initial files
                (self.folder / "0_init_annotated.cu").write_text(annotated_ncu)
                annotated_ncu_text = annotated_ncu
            except FeedbackError as e:
                self.agent_logger.warning(
                    f"Initial profiling failed; proceeding with fallback. Details: {e}"
                )
                # Fallback - no profiling data
                init_ncu_log = ""
                cycles = None
                annotated_ncu_text = self.code_to_optimize
                self.initial_cycles = None
                self.best_cycles = None
                self.last_ncu_log = ""
        else:
            # Use provided profiling data (backward compatibility)
            self.agent_logger.info("Using provided profiling data...")
            init_ncu_log = self.profiling_data.ncu_log
            annotated_ncu_text = self.profiling_data.annotated_ncu or self.code_to_optimize
            self.initial_cycles = self.profiling_data.cycles
            self.best_cycles = self.initial_cycles
            self.last_ncu_log = init_ncu_log
        
        # Simple prompt: code + profile + database knowledge -> optimized code
        # No state matching, no strategy selection - just pack everything into the prompt
        database_content = self.database.gpu_optimization_knowledge[:6000] if hasattr(self.database, 'gpu_optimization_knowledge') else ""
        
        self.init_user_prompt = f"""Optimize the following CUDA kernel code for better performance.

GPU Optimization Knowledge:
{database_content}

Original code:
```cpp
{self.code_to_optimize}
```

Profiling results:
{f"Initial cycles: {self.initial_cycles}" if self.initial_cycles is not None else "Profiling not available"}
{f"Annotated source with NCU metrics:\n```cpp\n{annotated_ncu_text}\n```" if annotated_ncu_text != self.code_to_optimize else ""}
{f"NCU log:\n{init_ncu_log}" if init_ncu_log else ""}

Please generate an optimized version of this code that:
1. Maintains correctness (same functionality)
2. Improves performance (reduces execution cycles)
3. Is complete and compilable (include all necessary headers, defines, etc.)
4. **CRITICAL: Keep the `launch_gpu_implementation()` function interface EXACTLY the same as the original code** - do not modify the function signature, parameter types, or parameter order unless you are explicitly fixing a bug in the signature itself.

Generate the complete optimized CUDA code in a single ```cpp code block."""
        
        self.agent_logger.info(f"Initialized with code + profile + database. Initial cycles: {self.initial_cycles}")
    
    async def _create_trajectory_dir(self, task_id: int) -> Path:
        """Create trajectory directory for a task (matching RL agent structure)."""
        async with self._trajectory_lock:
            self.total_trajectories += 1
            trajectory_index = self.total_trajectories
        
        # Use uuid suffix to avoid folder name collisions in concurrent runs
        _uid = uuid.uuid4().hex[:8]
        trajectory_dir = self.folder / f"trajectory_{trajectory_index}_{_uid}"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_dirs[task_id] = trajectory_dir
        return trajectory_dir
    
    async def __run(self, messages, attempt_id, task_id, remaining_fix_budget: int = MAX_FIX_BUDGET) -> tuple[Feedback, int]:
        """Override __run to save files in trajectory directories."""
        # Trajectory directory should already exist (created upfront)
        trajectory_dir = self.trajectory_dirs.get(task_id, self.folder)
        
        # Create logger and timer locally (not using self.task_loggers/self.timers)
        logger = self.agent_logger.bind(attempt_id=attempt_id, task_id=task_id)
        timer = NamedTimer()

        logger.info(f"[Task {task_id}] Starting LLM generation for attempt {attempt_id}")

        # Save prompt in trajectory directory
        prompt_path = trajectory_dir / f"attempt{attempt_id}_prompt.md"
        prompt_path.write_text(self._convert_messages_to_string(messages))

        timer.reset()
        timer.start("attempt")
        logger.info(f"[Task {task_id}] Generating response with {self.model}...")
        try:
            from .utils import generate_code_retry
            response = await generate_code_retry(messages, self.model, logger)
        except Exception as e:
            logger.error(f"Failed to generate code: {e}")
            return Feedback(), 0  # Return tuple: (Feedback, fixes_used)
        assert response.generations, "No generations found"
        logger.info(
            f"Response generation completed in {response.elapsed_time:0.2f} seconds"
        )

        generation = response.generations[0]
        # Update prompt with response
        prompt_path.write_text(
            self._convert_messages_to_string(
                messages, response=generation, usage=response.usage
            )
        )
        
        # Get feedback (this will save code file)
        new_feedback, fixes_used = await self.get_feedback(
            generation, attempt_id, task_id, logger, remaining_fix_budget=remaining_fix_budget
        )
        timer.stop("attempt")

        new_feedback.llm_calls.insert(0, response)
        new_feedback.durations = timer.elapsed.copy()
        return new_feedback, fixes_used
    
    def _convert_messages_to_string(self, messages, response=None, usage=None):
        """Helper to convert messages to string (imported from utils)."""
        from .utils import convert_messages_to_string
        return convert_messages_to_string(messages, response=response, usage=usage)
    
    async def run(self) -> Optional[Path]:
        """
        Override run() to match RL agent structure:
        - Create trajectory directories for each task
        - Save rl_iter_X_best.cu after each iteration
        - Save success_rl_optimization.cu at the end
        """
        existing_attempt = self.check_for_existing_run()
        if isinstance(existing_attempt, Path):
            return existing_attempt
        elif existing_attempt == "__failed__":
            return None

        successful_tasks = []
        chosen_task = None
        best_filename = None
        best_cycles = float('inf')
        start_time = asyncio.get_event_loop().time()
        import time

        from ..config import config
        custom_logger_id = self.agent_logger.add(
            self.folder / "run.log",
            level=config.LOG_LEVEL,
            backtrace=True,
            diagnose=True,
            format=config.CUSTOM_LOGGER_FORMAT,
            filter=lambda record: record["extra"].get("folder") == str(self.folder),
        )
        self.agent_logger.info(f"Running {self.folder}...")

        try:
            await self.initialize()
        except FeedbackError as e:
            self.agent_logger.error(
                f"Failed to initialize agent for {self.folder}: {e}"
            )
            return None

        from .utils.query import process_messages, trim_messages, TrimError
        from copy import deepcopy

        initial_messages = [
            {
                "role": "system",
                "content": self.system_prompt if self.system_prompt else "",
            },
            {
                "role": "user",
                "content": self.init_user_prompt if self.init_user_prompt else "",
            },
        ]
        threads = {
            i: {
                "messages": process_messages(initial_messages, self.model),
                "feedbacks": [],
                "running": True,
            }
            for i in range(self.num_pgen)
        }
        
        # Create all trajectory directories upfront (before any tasks start) for true concurrency
        self.agent_logger.info(f"Configuration: num_pgen={self.num_pgen}, max_attempts={self.max_attempts}")
        self.agent_logger.info(f"Creating {self.num_pgen} trajectory directories...")
        for i in range(self.num_pgen):
            if i not in self.trajectory_dirs:
                await self._create_trajectory_dir(i)
        self.agent_logger.info(f"Created {len(self.trajectory_dirs)} trajectory directories (expected {self.num_pgen})")
        
        # Restructure: Each trajectory runs independently for max_attempts iterations
        # Similar to RL agent's run_rollout pattern
        async def run_single_trajectory(task_id: int):
            """Run a single trajectory for max_attempts iterations."""
            trajectory_dir = self.trajectory_dirs.get(task_id, self.folder)
            thread = threads[task_id]
            trajectory_best_cycles = float('inf')
            trajectory_best_file = None
            consecutive_failures = 0
            remaining_fix_budget = MAX_FIX_BUDGET  # Shared fix budget across all iterations in this trajectory
            
            for attempt in range(self.max_attempts):
                if not thread["running"]:
                    break
                    
                try:
                    thread["messages"] = trim_messages(
                        thread["messages"],
                        logger=self.agent_logger,
                    )
                except TrimError as e:
                    self.agent_logger.warning(
                        f"Trajectory {task_id}: Failed to trim messages: {e}, terminating"
                    )
                    thread["running"] = False
                    break
                
                self.agent_logger.info(
                    f"Trajectory {task_id}: Starting iteration {attempt + 1}/{self.max_attempts} "
                    f"(remaining fix budget: {remaining_fix_budget})"
                )
                
                # Run this iteration with shared fix budget
                try:
                    feedback, fixes_used = await self.__run(
                        deepcopy(thread["messages"]),
                        attempt,
                        task_id,
                        remaining_fix_budget=remaining_fix_budget,
                    )
                    
                    # Update remaining fix budget
                    remaining_fix_budget -= fixes_used
                    if remaining_fix_budget < 0:
                        remaining_fix_budget = 0
                    
                    thread["feedbacks"].append(feedback)
                    for message in feedback.new_messages:
                        thread["messages"].append(message)
                    
                    # Check if this iteration succeeded (compiled and ran)
                    if feedback.success:
                        consecutive_failures = 0  # Reset failure counter
                        
                        # Extract cycles from feedback
                        cycles = None
                        if hasattr(feedback, 'durations') and 'cycles' in feedback.durations:
                            cycles = feedback.durations['cycles']
                        elif feedback.filename:
                            try:
                                filepath = Path(feedback.filename)
                                if filepath.exists():
                                    _, _, _, cycles = await self.gather_perf_metrics(filepath)
                            except Exception as e:
                                self.agent_logger.warning(f"Trajectory {task_id}: Failed to get cycles: {e}")
                        
                        if cycles is not None:
                            if cycles < trajectory_best_cycles:
                                trajectory_best_cycles = cycles
                                trajectory_best_file = Path(feedback.filename)
                            # Update global best (using nonlocal to modify outer scope)
                            nonlocal best_cycles, best_filename
                            if cycles < best_cycles:
                                best_cycles = cycles
                                best_filename = Path(feedback.filename)
                                successful_tasks.append(Path(feedback.filename))
                            
                            self.agent_logger.info(
                                f"Trajectory {task_id}, Iteration {attempt + 1}: Success with {cycles} cycles"
                            )
                    else:
                        # Iteration failed (after all retries)
                        consecutive_failures += 1
                        self.agent_logger.warning(
                            f"Trajectory {task_id}, Iteration {attempt + 1}: Failed after retries"
                        )
                        
                        # If too many consecutive failures, terminate trajectory early
                        if consecutive_failures >= self.max_attempts:
                            self.agent_logger.warning(
                                f"Trajectory {task_id}: Terminating early after {consecutive_failures} consecutive failures"
                            )
                            thread["running"] = False
                            break
                            
                except Exception as e:
                    self.agent_logger.error(f"Trajectory {task_id}, Iteration {attempt + 1}: Exception: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= self.max_attempts:
                        thread["running"] = False
                        break
            
            # Return best result from this trajectory
            return task_id, trajectory_best_file, trajectory_best_cycles
        
        # Run all trajectories in parallel
        trajectory_tasks = [
            asyncio.create_task(run_single_trajectory(i))
            for i in range(self.num_pgen)
        ]
        
        # Collect results as they complete
        iteration_best_files = {}  # Track best per iteration across all trajectories
        for coro in asyncio.as_completed(trajectory_tasks):
            task_id, traj_best_file, traj_best_cycles = await coro
            if traj_best_file and traj_best_file.exists():
                # Determine which iteration this file came from (by parsing filename)
                # Files are named attempt{attempt_id}.cu
                try:
                    attempt_num = int(traj_best_file.stem.replace("attempt", "").split("_")[0])
                    if attempt_num not in iteration_best_files or traj_best_cycles < iteration_best_files[attempt_num][1]:
                        iteration_best_files[attempt_num] = (traj_best_file, traj_best_cycles)
                except (ValueError, IndexError):
                    pass
        
        # Save rl_iter_X_best.cu after each iteration (matching RL agent)
        for attempt_num, (best_file, best_cycles) in iteration_best_files.items():
            best_iter_path = self.folder / f"rl_iter_{attempt_num}_best.cu"
            best_iter_path.write_text(
                best_file.read_text() + f"\n\n// Elapsed Cycles: {best_cycles}\n"
            )
            self.agent_logger.info(
                f"Saved best result from iteration {attempt_num}: {best_cycles} cycles"
            )
        
        # Save metrics
        from .utils import write_jsonl
        from .feedback import write_metrics
        write_metrics(self.folder / "metrics.jsonl", threads)

        (self.folder / ".finished").write_text("")
        duration = time.time() - start_time
        self.agent_logger.info(f"Agent completed in {duration:0.2f} seconds.")
        self.agent_logger.remove(custom_logger_id)
        
        # Ensure we have baseline cycles to judge improvement (matching RL agent)
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
        
        # Decide success vs failure based on improvement over baseline (matching RL agent)
        if best_filename and best_filename.exists():
            # We have a best result - check if it's an improvement
            if self.initial_cycles is not None and best_cycles < self.initial_cycles:
                # Success: improvement over baseline
                final_filename = self.folder / "success_rl_optimization.cu"
                final_filename.write_text(best_filename.read_text())
                self.agent_logger.info(
                    f"Saved success_rl_optimization.cu with {best_cycles} cycles "
                    f"(improvement: {((self.initial_cycles - best_cycles) / self.initial_cycles * 100):.2f}%)"
                )
                
                # Run final reprofiling (simplified - just get cycles, no detailed profiling)
                self.agent_logger.info("Running final profiling on optimized code...")
                try:
                    # Use our own gather_perf_metrics for consistency (profiles all kernels at once)
                    _, final_ncu_log, _, final_cycles = await self.gather_perf_metrics(final_filename)
                    self.agent_logger.info(
                        f"Final profiling complete: {final_cycles} cycles"
                    )
                    # Save final profiling results
                    final_profiling_dir = self.folder / "final_profiling"
                    final_profiling_dir.mkdir(exist_ok=True)
                    (final_profiling_dir / "ncu_log.txt").write_text(final_ncu_log)
                    (final_profiling_dir / "cycles.txt").write_text(str(final_cycles))
                except Exception as e:
                    self.agent_logger.warning(f"Failed to run final profiling: {e}")
                
                return final_filename
            else:
                # Failure: no improvement over baseline
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
                    "Optimization did not produce an improvement; wrote failure_rl_optimization.cu with baseline (if available)"
                )
                return failure_file
        else:
            # No successful trajectory - write failure file with baseline
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
                "All trajectories failed; wrote failure_rl_optimization.cu with baseline (if available)"
            )
            return failure_file
    
    async def gather_perf_metrics(self, filepath: Path) -> Tuple[str, str, str, int]:
        """
        Gather performance metrics using NCU profiling (same as RL agent).
        Returns: (annotated_ncu, ncu_log, stderr, cycles)
        """
        stdout_list, stderr_list, path, success = await compile_and_run_cu_file(
            self.test_code_fp,
            filepath,
            self.gpu,
            NamedTimer(),
            self.agent_logger,
            persistent_artifacts=False,  # Don't use persistent artifacts to avoid path issues
            timeout=self.timeout,
            num_runs=1,
            passed_keyword="passed",
        )
        
        if not success:
            FeedbackAgent.raise_numerics_verification_error(stdout_list, stderr_list)
        
        # Simplified profiling: profile all kernels at once (no -k flag profiles all kernels)
        kernel_names = await find_kernel_names_ncu(path, filepath, self.gpu, self.timeout)
        if not kernel_names:
            raise ValueError("No kernel names found for NCU profiling")
        
        # Profile all kernels at once (no -k flag)
        ncu_stdout, ncu_stderr = await run_gpu_executable(
            path, self.gpu, self.timeout,
            job_name=f"{filepath} (ncu all kernels)",
            prefix_command="NVIDIA_TF32_OVERRIDE=0 ncu --section SpeedOfLight",
        )
        
        if "No Kernels were profiled" in ncu_stdout:
            raise ValueError("NCU did not profile any kernels")
        
        # Extract total cycles from all kernels (sum all Elapsed Cycles values)
        # The log may contain multiple "Elapsed Cycles" entries, one per kernel
        import re
        cycles_pattern = r"Elapsed Cycles\s+\w+\s+(\d[\d,]*)"
        cycles_matches = re.findall(cycles_pattern, ncu_stdout, re.IGNORECASE | re.MULTILINE)
        
        total_cycles = 0
        for cycle_str in cycles_matches:
            try:
                total_cycles += int(cycle_str.replace(",", ""))
            except ValueError:
                continue
        
        if total_cycles == 0:
            # Fallback: try to get cycles using the utility function (gets first one)
            try:
                total_cycles = get_elapsed_cycles_ncu_log(ncu_stdout)
            except Exception as e:
                self.agent_logger.warning(f"Failed to extract cycles from NCU output: {e}")
                raise ValueError("Failed to extract cycles from NCU output")
        
        # Extract only the GPU Speed Of Light Throughput section for each kernel (with names)
        # The NCU output will have sections for each kernel, we'll extract them
        simplified_ncu = self._extract_speed_of_light_section(ncu_stdout, kernel_names)
        
        cycles = total_cycles
        
        # Use original code as annotated (we don't need full annotation for minimal agent)
        annotated_ncu = self.code_to_optimize
        stderr = "\n".join(stderr_list or [])
        combined_ncu = simplified_ncu
        
        if cycles == 0:
            raise ValueError("Failed to extract cycles from NCU output")
        
        return annotated_ncu, combined_ncu, stderr, cycles
    
    def _extract_speed_of_light_section(self, ncu_output: str, kernel_names: list) -> str:
        """
        Extract only the GPU Speed Of Light Throughput section from NCU log.
        Returns simplified log with kernel names and just the initial tables for each kernel.
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
            # NCU format: [timestamp] kernel_name@... or kernel_name(...) before sections
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
                
                # Always add kernel name header (we now match kernel names even without markers)
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
            # Last resort: return minimal info with cycles
            self.agent_logger.warning("Could not extract Speed Of Light sections, using minimal output")
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
            return f"Kernels: {', '.join(kernel_names)}\n(Full NCU log available but not shown for simplicity)"
        
        return "\n\n".join(sections)
    
    async def get_feedback(
        self, response, attempt_id, task_id, logger, remaining_fix_budget: int = 4
    ) -> tuple[Feedback, int]:
        """
        Generate optimized code and verify it compiles/runs.
        Implements retry logic similar to RL agent: retry failed code up to remaining_fix_budget times.
        
        Returns:
            (Feedback, int): The feedback object and the number of fix attempts used
        """
        logger.info(
            f"Extracting and verifying optimized code... (remaining fix budget: {remaining_fix_budget})"
        )
        
        # Get trajectory directory for this task
        trajectory_dir = self.trajectory_dirs.get(task_id, self.folder)
        
        # Extract code from LLM response
        code = extract_code_from_response(response)
        if code is None:
            raise FeedbackError(
                "Error: The code should be contained within ```cpp and ``` tags."
            )
        
        # Retry logic for failed code (similar to RL agent's apply_optimization)
        # Use the remaining fix budget for this iteration
        max_fix_attempts = min(remaining_fix_budget, MAX_FIX_ATTEMPTS_PER_ITERATION)  # Cap per iteration
        attempt_idx = 0
        optimized_code = code
        filepath = None
        
        while attempt_idx < max_fix_attempts:
            # Write the (potentially fixed) code to a unique file
            filepath = trajectory_dir / f"attempt{attempt_id}_fix{attempt_idx}.cu"
            write_code_to_file(optimized_code, filepath, logger)
            
            timer = NamedTimer()
            stdout_list = None
            stderr_list = None
            error_msg = None
            
            try:
                # Compile and run to verify
                stdout_list, stderr_list, compiled_path, success = await compile_and_run_cu_file(
                    self.test_code_fp,
                    filepath,
                    self.gpu,
                    timer,
                    logger,
                    persistent_artifacts=True,
                    timeout=self.timeout,
                    num_runs=1,
                    passed_keyword="passed",
                )
                
                if not success:
                    # Compilation/execution failed - will retry below
                    error_msg = f"Compilation/execution failed. stdout: {stdout_list}, stderr: {stderr_list}"
                    logger.warning(f"Attempt {attempt_idx} failed: {error_msg}")
                else:
                    # Success! Break out of retry loop
                    break
                    
            except Exception as e:
                # Compilation or runtime failed – capture error message
                error_msg = str(e)
                logger.warning(f"Attempt {attempt_idx} failed with exception: {error_msg}")
            
            attempt_idx += 1
            if attempt_idx >= max_fix_attempts:
                # Give up after all retries - return failed feedback
                final_error_msg = f"Failed after {attempt_idx} retry attempts. Last error: {error_msg}"
                logger.error(f"Optimized code failed verification after all retries: {final_error_msg}")
                return (
                    Feedback(
                        new_messages=[{
                            "role": "user",
                            "content": f"The optimized code failed to compile or execute after {attempt_idx} retry attempts. Please fix:\n{final_error_msg}"
                        }],
                        llm_calls=[],
                        success=False,
                        filename=str(filepath) if filepath else "",
                        contents=optimized_code,
                        feedback=final_error_msg,
                        durations={},  # No cycles for failed attempts
                    ),
                    attempt_idx,  # Return number of fixes used
                )
            
            # Build a fix prompt for the LLM using the error message
            # Extract compile errors from stdout/stderr if available
            compile_error = ""
            runtime_error = ""
            if stdout_list:
                compile_error = "\n".join(stdout_list) if isinstance(stdout_list, list) else str(stdout_list)
            if stderr_list:
                runtime_error = "\n".join(stderr_list) if isinstance(stderr_list, list) else str(stderr_list)
            
            # Combine errors for the prompt
            combined_error = error_msg if error_msg else "Unknown error"
            if compile_error or runtime_error:
                combined_error = ""
                if compile_error:
                    combined_error += f"COMPILATION OUTPUT:\n{compile_error}\n\n"
                if runtime_error:
                    combined_error += f"RUNTIME OUTPUT:\n{runtime_error}\n\n"
                if not combined_error:
                    combined_error = error_msg if error_msg else "Unknown error"
            
            from .utils import generate_code_retry
            fix_prompt = f"""The previously generated CUDA code failed to compile or run.

COMPILER / RUNTIME ERROR LOG:
```
{combined_error}
```

ORIGINAL CUDA CODE (for reference – please modify in place):
```cpp
{optimized_code}
```

Please provide a corrected, fully compilable version of the kernel. Return **complete CUDA code** in one ```cpp``` block.
Please keep the code structure otherwise unchanged; it is compiled together with separate test code, so do NOT add a main function.

Include ALL necessary components:
   - #include statements (cuda_fp16.h, cuda_runtime.h, etc.)
   - #define constants – DEFINE ALL CONSTANTS BEFORE USING THEM
   - Complete __global__ kernel function with proper signature
   - Complete launch_gpu_implementation() function
"""
            
            # Log the fix prompt to file (similar to main prompt logging)
            fix_prompt_path = trajectory_dir / f"attempt{attempt_id}_fix{attempt_idx}_prompt.md"
            fix_messages = [{"role": "user", "content": fix_prompt}]
            
            # Save initial prompt (matching main prompt format)
            fix_prompt_path.write_text(self._convert_messages_to_string(fix_messages))
            logger.info(f"Saved fix prompt to {fix_prompt_path}")
            
            # Ask the LLM to fix the code
            logger.info(f"Requesting fix from LLM (attempt {attempt_idx + 1}/{max_fix_attempts})...")
            try:
                fix_response = await generate_code_retry(
                    messages=fix_messages,
                    model=self.model,
                    logger=logger,
                    max_retries=MAX_FIX_RETRIES,
                )
                logger.info(
                    f"Fix response received in {fix_response.elapsed_time:0.2f} seconds"
                )
                
                # Log the fix response (update the prompt file with response, matching main prompt format)
                fix_prompt_path.write_text(
                    self._convert_messages_to_string(
                        fix_messages,
                        response=fix_response.generations[0],
                        usage=fix_response.usage if hasattr(fix_response, 'usage') else None
                    )
                )
                logger.info(f"Saved fix prompt and response to {fix_prompt_path}")
                
                # Extract new code for next iteration
                optimized_code = extract_code_from_response(fix_response.generations[0])
                if optimized_code is None:
                    logger.error("Failed to extract code from fix response")
                    break  # Give up if we can't extract code
                    
            except Exception as e:
                logger.error(f"Failed to get fix from LLM: {e}")
                break  # Give up if LLM call fails
        
        # If we get here, either we succeeded or gave up
        if attempt_idx >= max_fix_attempts:
            # Should have returned above, but just in case
            error_msg = f"Failed after {attempt_idx} retry attempts"
            return (
                Feedback(
                    new_messages=[{
                        "role": "user",
                        "content": f"The optimized code failed to compile or execute after {attempt_idx} retry attempts."
                    }],
                    llm_calls=[],
                    success=False,
                    filename=str(filepath) if filepath else "",
                    contents=optimized_code,
                    feedback=error_msg,
                    durations={},
                ),
                attempt_idx,  # Return number of fixes used
            )
        
        # Success! Profile the optimized code
        # Save the final successful code to attempt{attempt_id}.cu (without fix suffix)
        final_filepath = trajectory_dir / f"attempt{attempt_id}.cu"
        write_code_to_file(optimized_code, final_filepath, logger)
        
        try:
            # Profile the optimized code (get new perf results for next iteration)
            logger.info("Profiling optimized code...")
            annotated_ncu, ncu_log, _, cycles = await self.gather_perf_metrics(final_filepath)
            
            logger.info(f"Optimized cycles: {cycles}, Initial cycles: {self.initial_cycles}, Best so far: {self.best_cycles}")
            
            # Update best cycles if this is better
            if self.best_cycles is None or cycles < self.best_cycles:
                self.best_cycles = cycles
                self.last_ncu_log = ncu_log
                improvement = ((self.initial_cycles - cycles) / self.initial_cycles * 100) if self.initial_cycles and self.initial_cycles > 0 else 0
                logger.info(f"New best cycles: {cycles} (improvement: {improvement:.2f}% vs initial)")
            
            # Check if this is an improvement over initial
            is_faster = self.initial_cycles is not None and cycles < self.initial_cycles
            
            if is_faster:
                improvement = ((self.initial_cycles - cycles) / self.initial_cycles * 100)
                logger.info(f"Optimization successful! {improvement:.2f}% improvement vs initial ({self.initial_cycles} -> {cycles} cycles)")
                return (
                    Feedback(
                        new_messages=[],  # No feedback needed - optimization successful
                        llm_calls=[],
                        success=True,
                        filename=str(final_filepath),
                        contents=optimized_code,
                        feedback=f"Optimization successful! {improvement:.2f}% improvement",
                        durations={"cycles": cycles},  # Store cycles for best selection
                    ),
                    attempt_idx,  # Return number of fixes used (0 if succeeded on first try)
                )
            else:
                # Not an improvement - provide feedback with new profiling data for next iteration
                if self.initial_cycles is not None:
                    change_pct = ((cycles - self.initial_cycles) / self.initial_cycles * 100)
                    feedback_msg = f"Optimization did not improve over initial. {change_pct:.2f}% change ({self.initial_cycles} -> {cycles} cycles). Please try a different optimization approach."
                else:
                    feedback_msg = f"Optimized code cycles: {cycles}. Please try to improve further."
                
                logger.warning(feedback_msg)
                
                # For next iteration, include the new profiling data in the prompt
                database_content = self.database.gpu_optimization_knowledge[:6000] if hasattr(self.database, 'gpu_optimization_knowledge') else ""
                next_prompt = f"""The previous optimization attempt did not improve performance.

Current optimized code cycles: {cycles}
{f"Initial cycles: {self.initial_cycles}" if self.initial_cycles is not None else ""}

GPU Optimization Knowledge:
{database_content}

Current code:
```cpp
{optimized_code}
```

Profiling results from current code:
```cpp
{annotated_ncu}
```

NCU log:
{ncu_log}

Please generate a new optimized version that improves performance. Generate the complete optimized CUDA code in a single ```cpp code block.

**CRITICAL: Keep the `launch_gpu_implementation()` function interface EXACTLY the same as the original code** - do not modify the function signature, parameter types, or parameter order."""
                
                return (
                    Feedback(
                        new_messages=[{
                            "role": "user",
                            "content": next_prompt
                        }],
                        llm_calls=[],
                        success=True,  # Still success (compiled/ran), just not improved
                        filename=str(final_filepath),
                        contents=optimized_code,
                        feedback=feedback_msg,
                        durations={"cycles": cycles},  # Store cycles for best selection
                    ),
                    attempt_idx,  # Return number of fixes used (0 if succeeded on first try)
                )
            
        except Exception as e:
            error_msg = f"Error during verification: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return (
                Feedback(
                    new_messages=[{
                        "role": "user",
                        "content": f"Error during code verification: {error_msg}"
                    }],
                    llm_calls=[],
                    success=False,
                    filename=str(filepath) if filepath else "",
                    contents=code,
                    feedback=error_msg,
                    durations={},
                ),
                0,  # No fixes used if we hit an exception
            )
    
    def choose_best_task(self, successful_files: list[Path]) -> Path:
        """
        Choose the best task from successful files.
        For minimal agent, we prefer files with better cycles if available.
        """
        if not successful_files:
            return self.folder / "attempt0_task0.cu"
        
        # Try to find the one with best cycles by checking optimization summaries
        best_file = None
        best_cycles = float('inf')
        
        for file_path in successful_files:
            # Look for optimization summary in the same directory
            summary_file = file_path.parent / "optimization_summary.json"
            if summary_file.exists():
                try:
                    summary = json.loads(summary_file.read_text())
                    cycles = summary.get("optimized_cycles", float('inf'))
                    if cycles < best_cycles:
                        best_cycles = cycles
                        best_file = file_path
                except Exception:
                    pass
        
        return best_file if best_file else successful_files[0]
