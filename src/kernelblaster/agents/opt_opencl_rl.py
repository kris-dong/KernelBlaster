"""
Reinforcement Learning-based OpenCL Optimization Agent for Qualcomm Adreno GPUs.
Implements the LLM-based policy optimization via strategy-guided rollouts,
analogous to opt_ncu_rl.py but targeting OpenCL kernels instead of CUDA.
"""
from __future__ import annotations
from pathlib import Path
import re
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import json
import asyncio
import os
import math

from ..config import config
from .feedback import FeedbackAgent, Feedback, FeedbackConfig
from .database import OptimizationDatabase, OptimizationEntry, CompositeOptimization
from .rl_agents import (
    ReplayBuffer, Trajectory, TrajectoryStep,
    PolicyEvaluationAgent, PerfGapAnalysisAgent, ParameterUpdateAgent
)
from .utils import (
    FeedbackError,
    compile_and_run_opencl,
    NamedTimer,
    generate_code_retry,
    extract_code_from_response,
    write_code_to_file,
)
from .database import LLMInterface

# Strip a trailing "Kernel time" footer we add when persisting best kernels.
_KERNEL_TIME_FOOTER_RE = re.compile(
    r"\n*//\s*Kernel time:\s*[0-9]+(?:\.[0-9]+)?\s*ms\s*$", re.IGNORECASE | re.MULTILINE
)


def parse_opencl_profile(stdout: str) -> Dict[str, float]:
    """Parse kernel execution times from OpenCL event profiling output.

    Expected format per kernel:
        [PROFILE] kernel_name: 123.456 ms
    """
    timings: Dict[str, float] = {}
    for m in re.finditer(r"\[PROFILE\]\s+(\S+):\s+([0-9]+(?:\.[0-9]+)?)\s*ms", stdout):
        timings[m.group(1)] = float(m.group(2))
    return timings


def get_total_kernel_time_ms(stdout: str) -> float:
    """Sum all [PROFILE] kernel times reported in stdout."""
    timings = parse_opencl_profile(stdout)
    return sum(timings.values()) if timings else 0.0


def generate_opencl_strategy_prompt(
    optimization_entry: OptimizationEntry | CompositeOptimization,
    kernel_source: str,
    profile_output: str,
    database_content: str = "",
    override_description: str | None = None,
    original_code: str | None = None,
) -> str:
    """Generate a prompt that guides the LLM to optimise an OpenCL kernel for Adreno."""

    technique_descriptions = {
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

    source_display = kernel_source if kernel_source.strip() else (original_code or "// Source code not available")

    profile_section = ""
    if profile_output.strip():
        profile_section = f"""
OPENCL EVENT PROFILING OUTPUT:
```
{profile_output[:4000] if len(profile_output) > 4000 else profile_output}
```
"""

    if isinstance(optimization_entry, CompositeOptimization):
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

        return f"""You are an OpenCL kernel optimization expert targeting Qualcomm Adreno GPUs.

GPU OPTIMIZATION DATABASE:
```
{database_content[:6000] if database_content else "Database not available"}
```

COMPOSITE OPTIMIZATION STRATEGY:
{optimization_entry.get_composite_id()}
PREDICTED IMPROVEMENT: {optimization_entry.predicted_improvement}%

TECHNIQUES TO APPLY:
{composite_desc}

APPLICATION ORDER:
{order_desc}{params_desc}{side_effects_note}

CURRENT KERNEL:
```c
{source_display}
```
{profile_section}

OPTIMIZATION TASK:
Apply the optimization plan to the current OpenCL kernel. The kernel runs on a Qualcomm Adreno 650 GPU (3 compute units, 32KB local memory, OpenCL 2.0).

CRITICAL REQUIREMENTS:
1. Apply ALL specified techniques in the given order: {' -> '.join(techniques)}
2. Generate COMPLETE, COMPILABLE OpenCL C kernel code
3. The output must be a single .cl file with proper __kernel function(s)
4. Keep the same kernel function name and argument signature
5. Format ALL code in a single ```c code block
6. Adreno-specific: prefer float4 vectorisation, work-group sizes that are multiples of 32, and __local memory tiling with barrier(CLK_LOCAL_MEM_FENCE)

APPROACH:
1. Analyse the profiling data to understand current bottlenecks
2. Apply the composite strategy systematically
3. Generate optimised code addressing the identified performance issues"""

    else:
        technique_name = getattr(optimization_entry, "technique", str(optimization_entry))
        technique_desc = override_description if override_description else technique_descriptions.get(
            technique_name, f"Apply the {technique_name} optimization technique.")
        pred_impr = getattr(optimization_entry, "predicted_improvement", None)
        category = getattr(optimization_entry, "category", "general")
        pred_impr_str = f"{pred_impr}%" if pred_impr is not None else "N/A"

        return f"""OPTIMIZATION TASK:
You are an OpenCL kernel optimization expert targeting Qualcomm Adreno GPUs (Adreno 650, 3 CUs, 32KB local mem, OpenCL 2.0).

OPTIMIZATION STRATEGY: {technique_name}
PREDICTED IMPROVEMENT: {pred_impr_str}
CATEGORY: {category}

STRATEGY DESCRIPTION:
{technique_desc}

CURRENT KERNEL:
```c
{source_display}
```
{profile_section}

GPU OPTIMIZATION DATABASE:
```
{database_content}
```

CRITICAL REQUIREMENTS:
1. Generate COMPLETE, COMPILABLE OpenCL C kernel code
2. Keep the same kernel function name and argument signature
3. Format ALL code in a single ```c code block
4. Adreno-specific tips:
   - Prefer float4/int4 vector types for wider memory transactions
   - Work-group sizes should be multiples of 32 (Adreno wave size)
   - Use __local memory with barrier(CLK_LOCAL_MEM_FENCE) for tiling
   - Use mad() / fma() for fused multiply-add
   - Minimise barrier() calls — each is expensive on mobile GPUs
5. Summarise the optimisation applied and the reason before the code

APPROACH:
1. Apply the requested optimisation technique systematically
2. If applying a new technique, start with the most minimal correct example
3. Generate optimised code addressing the identified performance issues"""


@dataclass
class RLOpenCLFeedback(Feedback):
    kernel_time_ms: Optional[float] = None
    profile_output: Optional[str] = None
    optimization_technique: Optional[str] = None
    predicted_improvement: Optional[float] = None
    actual_improvement: Optional[float] = None
    state: Optional[str] = None


class RLOpenCLAgent(FeedbackAgent):
    """
    RL-based OpenCL optimization agent for Qualcomm Adreno GPUs.
    Mirrors RLNCUAgent but uses OpenCL event profiling instead of NCU.
    """

    def __init__(
        self,
        fb_config: FeedbackConfig,
        kernel_to_optimize_fp: Path,
        database_path: Path,
        max_rollout_steps: int = 5,
        replay_buffer_size: int = 1000,
        update_frequency: int = 10,
        database: Optional[OptimizationDatabase] = None,
    ):
        super().__init__(fb_config)

        self.test_code_fp = fb_config.test_code_fp  # driver.c
        self.test_code = fb_config.test_code_fp.read_text()
        self.kernel_to_optimize_fp = kernel_to_optimize_fp  # kernel.cl
        self.kernel_to_optimize = kernel_to_optimize_fp.read_text()

        gpu_report_path = Path(__file__).parent.parent.parent.parent.parent / "algo-sol-modeling/algo-space/gpu_optimization_report.md"
        llm_interface = LLMInterface(self.model, self.agent_logger)
        if database is not None:
            self.database = database
        else:
            self.database = OptimizationDatabase(database_path, gpu_report_path, llm_interface)
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
        self.max_rollout_steps = max_rollout_steps
        self.update_frequency = update_frequency

        self.policy_evaluation_agent = PolicyEvaluationAgent()
        self.perf_gap_analysis_agent = PerfGapAnalysisAgent()
        self.parameter_update_agent = ParameterUpdateAgent()

        self.iteration_count = 0
        self.total_trajectories = 0
        self.best_time_ms = float('inf')
        self.initial_time_ms = None

        self._trajectory_lock: asyncio.Lock = asyncio.Lock()
        self.current_trajectory = None
        self.num_rl_iterations = 50
        self.exec_timeout_s = int(os.getenv("KERNELBLASTER_OPENCL_TIMEOUT_S", "600"))

        # Fastest kernel seen across *any* successful profile run in this session (verifies
        # correctness; includes pre-step profiles, init, and fix-attempt passes — not only
        # the last accepted step in each trajectory).
        self._global_best_lock: asyncio.Lock = asyncio.Lock()
        self._global_best_time_ms: float = float("inf")
        self._global_best_code: Optional[str] = None
        self._global_best_source: Optional[Path] = None

    def _format_global_best_artifact(self, code: str, time_ms: float) -> str:
        body = _KERNEL_TIME_FOOTER_RE.sub("", code).rstrip()
        return f"{body}\n\n// Kernel time: {time_ms:.3f} ms\n"

    async def _reset_global_best_for_run(self) -> None:
        async with self._global_best_lock:
            self._global_best_time_ms = float("inf")
            self._global_best_code = None
            self._global_best_source = None

    async def _record_global_best_if_better(
        self, kernel_filepath: Path, time_ms: float
    ) -> None:
        if time_ms <= 0.0 or not math.isfinite(time_ms):
            return
        try:
            text = Path(kernel_filepath).read_text(encoding="utf-8")
        except OSError:
            return
        async with self._global_best_lock:
            if time_ms >= self._global_best_time_ms:
                return
            self._global_best_time_ms = time_ms
            self._global_best_code = _KERNEL_TIME_FOOTER_RE.sub("", text).rstrip()
            self._global_best_source = Path(kernel_filepath)
            self.best_time_ms = time_ms
            self.agent_logger.info(
                f"[Global best] {time_ms:.3f} ms ({kernel_filepath.name})"
            )

    # ------------------------------------------------------------------
    # File helpers — OpenCL uses .cl extension, not .cu
    # ------------------------------------------------------------------
    def get_intermediate_filepath(self, attempt_id, task_id) -> Path:
        return self.folder / f"attempt{attempt_id}_task{task_id}.cl"

    def get_code_from_response(
        self, response, attempt_id, task_id, logger
    ) -> tuple[str, Path]:
        code = extract_code_from_response(response, tag="c")
        if code is None:
            code = extract_code_from_response(response, tag="opencl")
        if code is None:
            raise FeedbackError(
                "Error: The code should be contained within ```c and ``` tags."
            )
        filepath = self.get_intermediate_filepath(attempt_id, task_id)
        write_code_to_file(code, filepath, logger)
        return code, filepath

    # ------------------------------------------------------------------
    # Profiling: compile + run with --profile, parse timings
    # ------------------------------------------------------------------
    async def gather_perf_metrics(self, kernel_filepath: Path) -> Tuple[str, str, float]:
        """Compile and execute on Adreno, return (profile_output, stderr, total_kernel_time_ms)."""
        extra_files = None
        if hasattr(self, "_reference_bin_path") and self._reference_bin_path and self._reference_bin_path.exists():
            extra_files = [str(self._reference_bin_path)]

        stdout_list, stderr_list, compiled_path, success = await compile_and_run_opencl(
            main_filepath=self.test_code_fp,
            kernel_filepath=kernel_filepath,
            gpu=self.gpu,
            timer=NamedTimer(),
            logger=self.agent_logger,
            timeout=self.exec_timeout_s,
            num_runs=1,
            passed_keyword="passed",
            profile=True,
            extra_files=extra_files,
        )

        if not success:
            FeedbackAgent.raise_numerics_verification_error(stdout_list, stderr_list)

        profile_output = "\n".join(stdout_list)
        stderr_text = "\n".join(stderr_list)
        total_ms = get_total_kernel_time_ms(profile_output)

        await self._record_global_best_if_better(kernel_filepath, total_ms)

        return profile_output, stderr_text, total_ms

    async def _generate_reference(self):
        """Run the driver with --generate-reference on the board to cache CPU reference output.

        Compiles and runs the driver directly via SSH (bypassing the GPU server) since
        we need to retrieve the output file. Subsequent --profile runs load this cached
        binary instead of recomputing the O(N^3) CPU reference each time.
        """
        self._reference_bin_path = self.folder / "reference_output.bin"

        if self._reference_bin_path.exists():
            self.agent_logger.info("Reference output already cached, skipping generation.")
            return

        self.agent_logger.info("Generating cached CPU reference (one-time, may take ~2 min)...")

        board_host = os.getenv("KERNELBLASTER_ADRENO_BOARD_HOST", "root@10.44.120.201")
        ssh_opts = "-o StrictHostKeyChecking=no -o ConnectTimeout=10"
        remote_dir = f"/tmp/kernelblaster_refgen_{os.getpid()}"

        driver_path = str(self.test_code_fp.resolve())
        kernel_path = str(self.kernel_to_optimize_fp.resolve())

        cmds = [
            f"ssh {ssh_opts} {board_host} 'mkdir -p {remote_dir}'",
            f"scp {ssh_opts} -q {driver_path} {board_host}:{remote_dir}/driver.c",
            f"scp {ssh_opts} -q {kernel_path} {board_host}:{remote_dir}/kernel.cl",
            (
                f"ssh {ssh_opts} {board_host} "
                f"'cd {remote_dir} && gcc -o main driver.c -I/usr/include -L/usr/lib -lOpenCL -lm "
                f"-DCL_TARGET_OPENCL_VERSION=200 && ./main --generate-reference'"
            ),
            f"scp {ssh_opts} -q {board_host}:{remote_dir}/reference_output.bin {self._reference_bin_path}",
            f"ssh {ssh_opts} {board_host} 'rm -rf {remote_dir}'",
        ]

        import subprocess
        for cmd in cmds:
            self.agent_logger.debug(f"Reference gen: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=1800)
            if result.returncode != 0:
                stderr_out = result.stderr.decode(errors="replace")
                self.agent_logger.warning(f"Reference gen step failed: {cmd}\n{stderr_out}")
                self._reference_bin_path = None
                return

        if self._reference_bin_path.exists():
            size_mb = self._reference_bin_path.stat().st_size / (1024 * 1024)
            self.agent_logger.info(f"Reference output cached: {self._reference_bin_path} ({size_mb:.1f} MB)")
        else:
            self.agent_logger.warning("Failed to download reference_output.bin")
            self._reference_bin_path = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    async def initialize(self):
        """Gather initial profiling data for the unoptimised kernel."""
        self.kernel_to_optimize_fp = self.folder / "init.cl"
        self.kernel_to_optimize_fp.write_text(self.kernel_to_optimize)

        # Generate CPU reference once (cached for all subsequent runs)
        await self._generate_reference()

        self.agent_logger.info("Gathering initial OpenCL profiling data...")
        try:
            profile_output, _, time_ms = await self.gather_perf_metrics(self.kernel_to_optimize_fp)
            self.initial_time_ms = time_ms
            self.best_time_ms = time_ms
            self.last_profile_output = profile_output

            initial_state = await self._get_state(profile_output, self.kernel_to_optimize, time_ms)
            self.agent_logger.info(f"Initial state: {initial_state}, time: {time_ms:.3f} ms")

            (self.folder / "0_init_kernel.cl").write_text(self.kernel_to_optimize)
        except FeedbackError as e:
            self.agent_logger.warning(
                f"Initial profiling failed; proceeding with fallback state. Details: {e}"
            )
            self.last_profile_output = ""

    async def _get_state(self, profile_output: str, code: str, time_ms: float) -> str:
        """Derive an optimisation state string from profiling output."""
        metrics = parse_opencl_profile(profile_output)
        metrics["total_kernel_time_ms"] = time_ms
        try:
            state = await self.database.get_state_from_ncu_report(
                profile_output, metrics, code, elapsed_cycles=int(time_ms * 1000)
            )
        except Exception:
            state = "opencl_unknown"
        return state

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------
    async def run(self) -> Path:
        """Run parallel RL iterations and return the best result.

        success_rl_optimization.cl and global_best_rl_optimization.cl use the fastest
        *verified* kernel from any successful profile in this run (not only the last
        step of a trajectory with the lowest in-trajectory time).
        """
        best_filename = None
        best_time = float("inf")

        await self._reset_global_best_for_run()

        if not hasattr(self, "last_profile_output") or not self.last_profile_output:
            await self.initialize()

        # `initialize()` may run on the host before `run()`; the reset above clears
        # session tracking, so re-establish the baseline from disk + recorded ms.
        if self.initial_time_ms and math.isfinite(self.initial_time_ms) and (self.folder / "init.cl").exists():
            await self._record_global_best_if_better(self.folder / "init.cl", self.initial_time_ms)

        initial_state = await self._get_state(
            self.last_profile_output, self.kernel_to_optimize, self.initial_time_ms or 0.0
        )

        async def _run_single_iteration(idx: int):
            self.agent_logger.info(f"[Async] RL Iteration {idx + 1}/{self.num_rl_iterations}")
            try:
                trajectory = await self.run_rollout(self.kernel_to_optimize, initial_state)
                return idx, trajectory
            except Exception as exc:
                self.agent_logger.error(f"RL iteration {idx + 1} failed: {exc}")
                return idx, None

        tasks = [asyncio.create_task(_run_single_iteration(i)) for i in range(self.num_rl_iterations)]

        for coro in asyncio.as_completed(tasks):
            idx, trajectory = await coro
            if trajectory is None:
                continue

            if trajectory.steps:
                best_step = min(trajectory.steps, key=lambda s: s.cycles)
                step_time = best_step.cycles / 1000.0  # cycles field stores microseconds
                if step_time < best_time:
                    best_time = step_time
                    fp = self.folder / f"rl_iter_{idx}_best.cl"
                    fp.write_text(best_step.code + f"\n\n// Kernel time: {step_time:.3f} ms\n")
                    best_filename = fp
                    self.agent_logger.info(f"[Async] New best from iter {idx}: {best_time:.3f} ms")

            if trajectory:
                self.replay_buffer.add_trajectory(trajectory)
                self.total_trajectories += 1

        # Persist database snapshot
        try:
            self.database._persist_database()
            persist_fp = self.database._persist_json_fp
            snapshots_dir = persist_fp.parent / "snapshots"
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            existing = sorted(snapshots_dir.glob("optimization_database_*.json"))
            snapshot_fp = snapshots_dir / f"optimization_database_{len(existing)}.json"
            snapshot_fp.write_text(persist_fp.read_text(encoding="utf-8"), encoding="utf-8")
            self.agent_logger.info(f"Saved database snapshot to {snapshot_fp}")
        except Exception as snap_exc:
            self.agent_logger.warning(f"Failed to write database snapshot: {snap_exc}")

        g_time = self._global_best_time_ms
        g_code = self._global_best_code
        if g_code is not None and math.isfinite(g_time):
            gp = self.folder / "global_best_rl_optimization.cl"
            gp.write_text(
                self._format_global_best_artifact(g_code, g_time),
                encoding="utf-8",
            )
            self.agent_logger.info(
                f"Global best (verified): {g_time:.3f} ms -> {gp.name} "
                f"(per-iter min: {best_time if best_time < float('inf') else 'n/a'})"
            )
        if g_code is not None and self.initial_time_ms is not None and g_time < self.initial_time_ms:
            final = self.folder / "success_rl_optimization.cl"
            final.write_text(self._format_global_best_artifact(g_code, g_time), encoding="utf-8")
            return final

        if best_filename is not None and self.initial_time_ms is not None and best_time < self.initial_time_ms:
            # Fallback: global tracking unavailable but iteration picked a true improvement.
            final = self.folder / "success_rl_optimization.cl"
            final.write_text(best_filename.read_text(encoding="utf-8"), encoding="utf-8")
            return final

        if best_filename is not None or g_code is not None:
            failure_file = self.folder / "failure_rl_optimization.cl"
            baseline_str = f"{self.initial_time_ms:.3f} ms" if self.initial_time_ms is not None else "N/A"
            failure_file.write_text(
                self.kernel_to_optimize + f"\n\n// Baseline: {baseline_str}\n"
            )
            self.agent_logger.error("RL did not improve over baseline; wrote failure file")
            return failure_file

        failure_file = self.folder / "failure_rl_optimization.cl"
        baseline_str = f"{self.initial_time_ms:.3f} ms" if self.initial_time_ms is not None else "N/A"
        failure_file.write_text(
            self.kernel_to_optimize + f"\n\n// Baseline: {baseline_str}\n"
        )
        self.agent_logger.error("All RL iterations failed; wrote failure file")
        return failure_file

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    async def run_rollout(self, initial_code: str, initial_state: str) -> Trajectory:
        """Run a single optimisation rollout trajectory."""
        import random, uuid as _uuid
        from dataclasses import asdict

        async with self._trajectory_lock:
            self.total_trajectories += 1
            trajectory_index = self.total_trajectories

        _uid = _uuid.uuid4().hex[:8]
        trajectory_dir = self.folder / f"trajectory_{trajectory_index}_{_uid}"
        trajectory_dir.mkdir(parents=True, exist_ok=True)

        trajectory = Trajectory()
        current_code = initial_code
        current_state = initial_state
        current_time_ms = self.initial_time_ms
        last_profile = getattr(self, "last_profile_output", "")

        self.agent_logger.info(f"Starting rollout from state: {current_state}")

        for step in range(self.max_rollout_steps):
            # 1) Analyse current state
            metrics = parse_opencl_profile(last_profile)
            metrics["total_kernel_time_ms"] = current_time_ms or 0.0
            try:
                profile = await self.database.analyze_performance_state(
                    last_profile, metrics, current_code,
                    elapsed_cycles=int((current_time_ms or 0) * 1000),
                )
                analysis_json = json.dumps(
                    {k: getattr(profile, k) for k in profile.__dataclass_fields__}, indent=2,
                )

                cur_iter = step + 1
                plan = await self.database.generate_optimization_plan(
                    analysis_json, current_code,
                    top_n=max(4, self.max_rollout_steps - cur_iter),
                )
            except Exception as exc:
                self.agent_logger.warning(f"Plan generation failed: {exc}")
                plan = []

            # 2) Pick technique
            optimization_entry = None
            strategy_description = ""
            if plan:
                def _safe_rel(x):
                    try:
                        return min(max(float(x), 0.0), 1.0)
                    except (TypeError, ValueError):
                        return 0.05

                force_top1 = os.getenv("KERNELAGENT_DB_FALLBACK_TOP1", "0") in ("1", "true", "True", "yes", "y", "on")
                if force_top1:
                    chosen = max(plan, key=lambda p: _safe_rel(p.get("relevance_score", 0.05)))
                else:
                    weights = [max(_safe_rel(p.get("relevance_score", 0.05)) ** 3, 0.001) for p in plan]
                    chosen = random.choices(plan, weights=weights, k=1)[0]
                technique_name = chosen.get("technique")
                optimization_entry = self._lookup_optim_entry_by_name(technique_name)
                strategy_description = chosen.get("description", "")
                self.agent_logger.info(
                    f"Selected technique: {technique_name} (relevance {chosen.get('relevance_score', 0):.2f})"
                )

            # 3) Fallback
            if optimization_entry is None:
                optimization_entry = self.database.select_best_optimization(current_state)
                if optimization_entry is None:
                    optimization_entry = self.database.select_best_optimization(current_state, exclude_used=True)
                if optimization_entry is None:
                    for state_name, sd in self.database.optimization_strategies.items():
                        if sd.get("optimizations"):
                            optimization_entry = self.database.select_best_optimization(state_name)
                            if optimization_entry:
                                break
                if optimization_entry is None:
                    if self._try_add_default_optimizations(current_state):
                        optimization_entry = self.database.select_best_optimization(current_state)
                if optimization_entry is None:
                    self.agent_logger.warning(f"No optimisation found for state {current_state}, stopping")
                    break

            if isinstance(optimization_entry, CompositeOptimization):
                technique_name = optimization_entry.get_composite_id()
            elif hasattr(optimization_entry, "technique"):
                technique_name = optimization_entry.technique
            else:
                technique_name = str(optimization_entry)

            _pred = getattr(optimization_entry, "predicted_improvement", None)
            self.agent_logger.info(
                f"Step {step}: Applying {technique_name}"
                + (f" (predicted: {_pred}%)" if _pred is not None else "")
            )

            try:
                optimized_code, new_time_ms, new_state, new_profile = await self.apply_optimization(
                    current_code, optimization_entry, step, trajectory_dir, strategy_description,
                )

                if current_time_ms and current_time_ms > 0:
                    actual_improvement = ((current_time_ms - new_time_ms) / current_time_ms) * 100
                else:
                    actual_improvement = 0.0

                reward = self.calculate_reward(
                    getattr(optimization_entry, "predicted_improvement", None),
                    actual_improvement,
                    current_time_ms is not None and new_time_ms < current_time_ms,
                )

                action_name = (
                    optimization_entry.get_composite_id()
                    if isinstance(optimization_entry, CompositeOptimization)
                    else getattr(optimization_entry, "technique", str(optimization_entry))
                )

                # Store time in microseconds in the cycles field for compatibility with Trajectory
                traj_step = TrajectoryStep(
                    state=current_state,
                    action=action_name,
                    code=optimized_code,
                    cycles=int(new_time_ms * 1000),
                    predicted_improvement=getattr(optimization_entry, "predicted_improvement", 0.0) or 0.0,
                    actual_improvement=actual_improvement,
                    reward=reward,
                )
                trajectory.add_step(traj_step)

                if isinstance(optimization_entry, CompositeOptimization):
                    self.database.update_composite_optimization_result(
                        current_state, technique_name, actual_improvement,
                    )
                else:
                    self.database.update_optimization_result(
                        current_state, technique_name, actual_improvement,
                        # OpenCL RL tracks percent improvement from measured ms; do
                        # not force CUDA baseline-file speedup parsing here.
                        current_file_path=None,
                    )

                self.agent_logger.info(
                    f"Step {step} result: {new_time_ms:.3f} ms "
                    f"({actual_improvement:.1f}% improvement, reward: {reward:.2f})"
                )

                current_code = optimized_code
                if new_state is not None:
                    current_state = new_state
                current_time_ms = new_time_ms
                last_profile = new_profile or last_profile

                if actual_improvement < -500:
                    self.agent_logger.warning(f"Stopping due to severe degradation: {actual_improvement:.1f}%")
                    break

            except Exception as e:
                import traceback
                self.agent_logger.error(f"Error in step {step}: {e}\n{traceback.format_exc()}")
                break

        return trajectory

    def _lookup_optim_entry_by_name(
        self, technique_name: str
    ) -> Optional[OptimizationEntry | CompositeOptimization]:
        for state_data in self.database.optimization_strategies.values():
            for opt in state_data.get("optimizations", []):
                if opt.technique == technique_name:
                    return opt
        for comps in self.database.composite_optimizations.values():
            for comp in comps:
                if comp.get_composite_id() == technique_name:
                    return comp
        return None

    # ------------------------------------------------------------------
    # Apply a single optimisation step
    # ------------------------------------------------------------------
    async def apply_optimization(
        self,
        code: str,
        optimization_entry: OptimizationEntry | CompositeOptimization,
        step: int,
        trajectory_dir: Path | None = None,
        strategy_description: str = "",
    ) -> Tuple[str, float, str | None, str]:
        """Apply an optimisation and return (optimised_code, time_ms, new_state, profile_output)."""

        def _save_log(label: str, prompt_text: str, response_text: str):
            if trajectory_dir is None:
                return
            log_fp = trajectory_dir / "agentic_steps_log.txt"
            with open(log_fp, "a", encoding="utf-8") as f:
                f.write(f"=== {label} ===\n--- PROMPT ---\n{prompt_text.rstrip()}\n--- RESPONSE ---\n{response_text.rstrip()}\n\n")

        if isinstance(optimization_entry, CompositeOptimization):
            technique_name = optimization_entry.get_composite_id()
        else:
            technique_name = getattr(optimization_entry, "technique", str(optimization_entry))
        base_label = f"step_{step}_{technique_name}"
        base_dir = trajectory_dir if trajectory_dir is not None else self.folder

        # Profile current code
        temp_file = base_dir / f"step_{step}_{technique_name}.cl"
        temp_file.write_text(code)
        try:
            profile_output, _, _ = await self.gather_perf_metrics(temp_file)
        except (FeedbackError, Exception) as e:
            self.agent_logger.warning(f"Profiling failed at step {step}: {e}; using empty profile")
            profile_output = ""

        # Build prompt
        try:
            database_content = self.database.get_database_md_text()
            if not database_content or not database_content.strip():
                database_content = self.database.get_database_footer_text()
            if not database_content or not database_content.strip():
                database_content = getattr(self.database, 'gpu_optimization_knowledge', '')[:6000] or ""
        except Exception:
            database_content = ""

        prompt = generate_opencl_strategy_prompt(
            optimization_entry, code, profile_output,
            database_content,
            override_description=strategy_description or None,
            original_code=code,
        )

        response = await generate_code_retry(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            logger=self.agent_logger,
            max_retries=3,
        )
        _save_log(f"{base_label}_initial", prompt, response.generations[0])

        optimized_code, filepath = self.get_code_from_response(
            response.generations[0], step, 0, self.agent_logger
        )
        try:
            target_fp = base_dir / f"{base_label}_initial.cl"
            if filepath != target_fp:
                try:
                    filepath.rename(target_fp)
                except Exception:
                    target_fp.write_text(optimized_code)
            filepath = target_fp
        except Exception:
            pass

        # Compile / run with fix attempts
        MAX_FIX_ATTEMPTS = 4
        attempt_idx = 0
        new_time_ms = 0.0
        new_profile = ""

        while attempt_idx < MAX_FIX_ATTEMPTS:
            filepath = base_dir / f"{base_label}_attempt{attempt_idx}.cl"
            filepath.write_text(optimized_code)

            try:
                new_profile, _, new_time_ms = await self.gather_perf_metrics(filepath)
                break
            except Exception as e:
                error_msg = str(e)
                if trajectory_dir:
                    with open(trajectory_dir / "agentic_steps_log.txt", "a") as f:
                        f.write(f"Compile/Run failed attempt {attempt_idx}: {error_msg}\n\n")

                attempt_idx += 1
                if attempt_idx >= MAX_FIX_ATTEMPTS:
                    raise

                fix_prompt = (
                    "The previously generated OpenCL kernel failed to compile or run.\n\n"
                    f"ERROR LOG:\n```\n{error_msg}\n```\n\n"
                    f"ORIGINAL KERNEL CODE:\n```c\n{optimized_code}\n```\n\n"
                    "Please provide a corrected, fully compilable version of the kernel. "
                    "Return complete OpenCL C code in one ```c``` block.\n"
                    "Keep the same kernel function name and argument signature.\n"
                )

                fix_response = await generate_code_retry(
                    messages=[{"role": "user", "content": fix_prompt}],
                    model=self.model,
                    logger=self.agent_logger,
                    max_retries=2,
                )
                _save_log(f"{base_label}_fix_{attempt_idx}", fix_prompt, fix_response.generations[0])
                optimized_code, _ = self.get_code_from_response(
                    fix_response.generations[0], step, attempt_idx, self.agent_logger
                )

        new_state = None
        return optimized_code, new_time_ms, new_state, new_profile

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def calculate_reward(self, predicted_improvement: Optional[float],
                         actual_improvement: float, is_faster: bool) -> float:
        base_reward = actual_improvement / 100.0
        try:
            safe_predicted = float(predicted_improvement) if predicted_improvement is not None else 0.0
        except (TypeError, ValueError):
            safe_predicted = 0.0

        if safe_predicted > 0.0:
            accuracy = min(actual_improvement / safe_predicted, 2.0)
            accuracy_bonus = 0.2 if 0.8 <= accuracy <= 1.2 else -0.1 * abs(accuracy - 1.0)
        else:
            accuracy_bonus = 0.0

        penalty = -0.5 if not is_faster else 0.0
        return base_reward + accuracy_bonus + penalty

    # ------------------------------------------------------------------
    # Default optimisations fallback
    # ------------------------------------------------------------------
    def _try_add_default_optimizations(self, current_state: str) -> bool:
        try:
            defaults = {
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

            for bottleneck, opts in defaults.items():
                if bottleneck in current_state:
                    for technique, improvement in opts:
                        self.database.add_new_optimization(current_state, technique, improvement)
                    self.agent_logger.info(f"Added {len(opts)} default optimisations for state: {current_state}")
                    return True
        except Exception as e:
            self.agent_logger.error(f"Error adding default optimisations: {e}")
        return False

    # ------------------------------------------------------------------
    # Feedback (for use via the base FeedbackAgent run loop)
    # ------------------------------------------------------------------
    async def get_feedback(self, response, attempt_id, task_id, logger) -> Feedback:
        if self.initial_time_ms is None:
            await self.initialize()

        logger.info(f"Starting RL optimisation trajectory for task {task_id}")
        code, filepath = self.get_code_from_response(response, attempt_id, task_id, logger)

        try:
            profile_output, _, time_ms = await self.gather_perf_metrics(filepath)
            initial_state = await self._get_state(profile_output, code, time_ms)
            trajectory = await self.run_rollout(code, initial_state)
            self.replay_buffer.add_trajectory(trajectory)
            self.total_trajectories += 1

            use_global = (
                self._global_best_code is not None
                and math.isfinite(self._global_best_time_ms)
            )
            if trajectory.steps or use_global:
                if use_global:
                    best_ms = self._global_best_time_ms
                    best_kernel = self._global_best_code
                    best_action = (
                        min(trajectory.steps, key=lambda s: s.cycles).action
                        if trajectory.steps
                        else "global_best_pool"
                    )
                    if trajectory.steps and self._global_best_code is not None:
                        in_traj = min(trajectory.steps, key=lambda s: s.cycles)
                        t_ms = in_traj.cycles / 1000.0
                        if abs(t_ms - best_ms) > 0.0005:
                            self.agent_logger.info(
                                f"Feedback uses global best {best_ms:.3f} ms (trajectory min was {t_ms:.3f} ms)"
                            )
                else:
                    best_step = min(trajectory.steps, key=lambda s: s.cycles)
                    best_ms = best_step.cycles / 1000.0
                    best_kernel = best_step.code
                    best_action = best_step.action

                if self.initial_time_ms and self.initial_time_ms > 0:
                    improvement_pct = ((self.initial_time_ms - best_ms) / self.initial_time_ms) * 100
                else:
                    improvement_pct = 0.0
                bstep = min(trajectory.steps, key=lambda s: s.cycles) if trajectory.steps else None

                feedback_msg = (
                    f"Optimisation trajectory completed with {len(trajectory.steps)} steps.\n\n"
                    f"BEST RESULT (fastest verified kernel this session):\n"
                    f"- Time: {best_ms:.3f} ms (vs initial: {self.initial_time_ms:.3f} ms)\n"
                    f"- Improvement: {improvement_pct:.1f}%\n"
                    f"- Best technique (in-trajectory min): {bstep.action if bstep else 'n/a'}\n"
                    f"- Total reward: {trajectory.total_reward:.2f}\n\n"
                    f"OPTIMISED KERNEL:\n```c\n{best_kernel}\n```\n"
                )

                best_file = self.folder / f"best_task_{task_id}.cl"
                best_file.write_text(
                    self._format_global_best_artifact(
                        _KERNEL_TIME_FOOTER_RE.sub("", best_kernel).rstrip(), best_ms
                    )
                )

                return RLOpenCLFeedback(
                    new_messages=[
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": feedback_msg},
                    ],
                    success=True,
                    filename=str(best_file),
                    contents=best_kernel,
                    kernel_time_ms=best_ms,
                    profile_output=profile_output,
                    optimization_technique=best_action,
                    predicted_improvement=bstep.predicted_improvement if bstep else 0.0,
                    actual_improvement=bstep.actual_improvement if bstep else 0.0,
                    state=initial_state,
                )
            else:
                return RLOpenCLFeedback(
                    new_messages=[
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": "No successful optimisation steps completed. Try a different approach."},
                    ],
                    success=False,
                    kernel_time_ms=time_ms,
                    profile_output=profile_output,
                    state=initial_state,
                )

        except FeedbackError as e:
            logger.error(f"Error in RL optimisation: {e}")
            return Feedback(
                new_messages=[
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": f"Optimisation failed: {e}. Please fix and try again."},
                ],
                success=False,
                feedback=e.feedback if hasattr(e, 'feedback') else str(e),
            )

    def get_performance_summary(self) -> Dict[str, Any]:
        return {
            'total_trajectories': self.total_trajectories,
            'iteration_count': self.iteration_count,
            'initial_time_ms': self.initial_time_ms,
            'best_time_ms': self.best_time_ms,
            'overall_improvement': (
                ((self.initial_time_ms - self.best_time_ms) / self.initial_time_ms * 100)
                if self.initial_time_ms else 0
            ),
            'buffer_stats': self.replay_buffer.get_statistics(),
            'database_stats': self.database.get_database_stats(),
        }
