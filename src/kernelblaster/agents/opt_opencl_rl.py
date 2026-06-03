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
from .opt_rl_base import RLAgentBase
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
    write_code_to_file,
)
from .database import LLMInterface

# Kernel-time footer regex moved to OpenCLBackend.format_result_artifact (Phase 4b).


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
    backend=None,
) -> str:
    """Generate a prompt that guides the LLM to optimise an OpenCL kernel for Adreno.

    The Adreno-tuned technique map is sourced from ``backend.technique_map``
    (single source of truth — see ``kernelblaster.backends.opencl``). The
    ``backend`` parameter is optional for back-compat; if omitted, the default
    OpenCL backend is constructed lazily. Phase 4b will fold this with the
    CUDA prompt function into a single backend-agnostic template.
    """
    if backend is None:
        from ..backends import get_backend
        backend = get_backend("opencl")
    technique_descriptions = backend.technique_map

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
    # ``metric`` is the primary backend metric (OpenCL kernel time in ms).
    # Renamed from ``kernel_time_ms`` in Phase 4d to harmonize with the
    # CUDA feedback shape.
    metric: Optional[float] = None
    profile_output: Optional[str] = None
    optimization_technique: Optional[str] = None
    predicted_improvement: Optional[float] = None
    actual_improvement: Optional[float] = None
    state: Optional[str] = None


class RLOpenCLAgent(RLAgentBase):
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
        # Phase 4f step 2: shared __init__ lives on RLAgentBase. The public
        # OpenCL constructor keeps ``kernel_to_optimize_fp`` for back-compat;
        # internally it's passed under the canonical kernel_source_fp name.
        # OpenCL-specific extras (exec_timeout_s, global-best verification-pool
        # fields) are set up via _init_backend_extras() below.
        super().__init__(
            fb_config=fb_config,
            kernel_source_fp=kernel_to_optimize_fp,
            database_path=database_path,
            max_rollout_steps=max_rollout_steps,
            replay_buffer_size=replay_buffer_size,
            update_frequency=update_frequency,
            database=database,
        )

    def _init_backend_extras(self) -> None:
        """OpenCL-specific instance attrs not present on the CUDA agent.

        - ``exec_timeout_s``: SSH-exec timeout (board can be slow). Env-var
          driven so operators can tune per-board.
        - ``_global_best_*``: verification-pool best-tracker. Captures the
          fastest correct kernel across all attempts (not just the final
          per-trajectory step). CUDA's libtorch-based reference verification
          doesn't use this pattern.
        """
        self.exec_timeout_s = int(os.getenv("KERNELBLASTER_OPENCL_TIMEOUT_S", "600"))
        self._global_best_lock: asyncio.Lock = asyncio.Lock()
        self._global_best_time_ms: float = float("inf")
        self._global_best_code: Optional[str] = None
        self._global_best_source: Optional[Path] = None

    def _format_global_best_artifact(self, code: str, time_ms: float) -> str:
        # Backend owns the footer format — see OpenCLBackend.format_result_artifact.
        return self.backend.format_result_artifact(code, time_ms)

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
            # Store body without the footer; _format_global_best_artifact
            # (-> backend.format_result_artifact) will re-strip and re-append
            # when this is serialized, so we just keep the raw body.
            self._global_best_code = text
            self._global_best_source = Path(kernel_filepath)
            self.best_metric = time_ms
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
        # Backend owns tag preferences (OpenCL: ```c -> ```opencl).
        code = self.backend.extract_code_from_response(response)
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

        # Phase 3c: persist a structured ProfileResult next to the kernel
        # so downstream tooling (analytics, future speedup-tracker) can read
        # JSON instead of regex-parsing driver stdout. raw_log is filtered to
        # just the [PROFILE] lines — keeping the full ~100KB driver dump per
        # step would balloon the run tree for no real gain.
        try:
            pr = self.backend.parse_profile(profile_output)
            pr.raw_log = "\n".join(
                ln for ln in profile_output.splitlines() if "[PROFILE]" in ln
            )
            profile_json_path = kernel_filepath.with_suffix(".profile.json")
            pr.write_json(profile_json_path)
        except Exception as e:
            # Persistence is best-effort — don't break the RL loop if the
            # filesystem hiccups or parse_profile chokes on weird output.
            self.agent_logger.warning(
                f"Failed to write profile.json for {kernel_filepath.name}: {e}"
            )

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

        # Backend already resolved the canonical board host in __init__ (Phase 2/6).
        board_host = self.backend.board_host
        ssh_opts = "-o StrictHostKeyChecking=no -o ConnectTimeout=10"
        remote_dir = f"/tmp/kernelblaster_refgen_{os.getpid()}"

        driver_path = str(self.test_code_fp.resolve())
        kernel_path = str(self.kernel_source_fp.resolve())

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
        self.kernel_source_fp = self.folder / "init.cl"
        self.kernel_source_fp.write_text(self.kernel_source)

        # Generate CPU reference once (cached for all subsequent runs)
        await self._generate_reference()

        self.agent_logger.info("Gathering initial OpenCL profiling data...")
        try:
            profile_output, _, time_ms = await self.gather_perf_metrics(self.kernel_source_fp)
            self.initial_metric = time_ms
            self.best_metric = time_ms
            self.last_profile_output = profile_output

            initial_state = await self._get_state(profile_output, self.kernel_source, time_ms)
            self.agent_logger.info(f"Initial state: {initial_state}, time: {time_ms:.3f} ms")

            (self.folder / "0_init_kernel.cl").write_text(self.kernel_source)
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
        if self.initial_metric and math.isfinite(self.initial_metric) and (self.folder / "init.cl").exists():
            await self._record_global_best_if_better(self.folder / "init.cl", self.initial_metric)

        initial_state = await self._get_state(
            self.last_profile_output, self.kernel_source, self.initial_metric or 0.0
        )

        async def _run_single_iteration(idx: int):
            self.agent_logger.info(f"[Async] RL Iteration {idx + 1}/{self.num_rl_iterations}")
            try:
                trajectory = await self.run_rollout(self.kernel_source, initial_state)
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
                    fp.write_text(self.backend.format_result_artifact(best_step.code, step_time))
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
        if g_code is not None and self.initial_metric is not None and g_time < self.initial_metric:
            final = self.folder / "success_rl_optimization.cl"
            final.write_text(self._format_global_best_artifact(g_code, g_time), encoding="utf-8")
            return final

        if best_filename is not None and self.initial_metric is not None and best_time < self.initial_metric:
            # Fallback: global tracking unavailable but iteration picked a true improvement.
            final = self.folder / "success_rl_optimization.cl"
            final.write_text(best_filename.read_text(encoding="utf-8"), encoding="utf-8")
            return final

        if best_filename is not None or g_code is not None:
            failure_file = self.folder / "failure_rl_optimization.cl"
            baseline_str = f"{self.initial_metric:.3f} ms" if self.initial_metric is not None else "N/A"
            failure_file.write_text(
                self.kernel_source + f"\n\n// Baseline: {baseline_str}\n"
            )
            self.agent_logger.error("RL did not improve over baseline; wrote failure file")
            return failure_file

        failure_file = self.folder / "failure_rl_optimization.cl"
        baseline_str = f"{self.initial_metric:.3f} ms" if self.initial_metric is not None else "N/A"
        failure_file.write_text(
            self.kernel_source + f"\n\n// Baseline: {baseline_str}\n"
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
        current_time_ms = self.initial_metric
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

    # _lookup_optim_entry_by_name lifted to RLAgentBase in Phase 4f.

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
            backend=self.backend,
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
    # calculate_reward and _try_add_default_optimizations lifted to
    # RLAgentBase in Phase 4f.

    # ------------------------------------------------------------------
    # Feedback (for use via the base FeedbackAgent run loop)
    # ------------------------------------------------------------------
    async def get_feedback(self, response, attempt_id, task_id, logger) -> Feedback:
        if self.initial_metric is None:
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

                if self.initial_metric and self.initial_metric > 0:
                    improvement_pct = ((self.initial_metric - best_ms) / self.initial_metric) * 100
                else:
                    improvement_pct = 0.0
                bstep = min(trajectory.steps, key=lambda s: s.cycles) if trajectory.steps else None

                feedback_msg = (
                    f"Optimisation trajectory completed with {len(trajectory.steps)} steps.\n\n"
                    f"BEST RESULT (fastest verified kernel this session):\n"
                    f"- Time: {best_ms:.3f} ms (vs initial: {self.initial_metric:.3f} ms)\n"
                    f"- Improvement: {improvement_pct:.1f}%\n"
                    f"- Best technique (in-trajectory min): {bstep.action if bstep else 'n/a'}\n"
                    f"- Total reward: {trajectory.total_reward:.2f}\n\n"
                    f"OPTIMISED KERNEL:\n```c\n{best_kernel}\n```\n"
                )

                best_file = self.folder / f"best_task_{task_id}.cl"
                # backend.format_result_artifact already strips any existing footer
                # before re-applying, so no need to pre-strip _KERNEL_TIME_FOOTER_RE here.
                best_file.write_text(
                    self.backend.format_result_artifact(best_kernel, best_ms)
                )

                return RLOpenCLFeedback(
                    new_messages=[
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": feedback_msg},
                    ],
                    success=True,
                    filename=str(best_file),
                    contents=best_kernel,
                    metric=best_ms,
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
                    metric=time_ms,
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

    # get_performance_summary skeleton lifted to RLAgentBase in Phase 4f;
    # this override only contributes the OpenCL-named metric keys.
    def _perf_summary_extras(self) -> Dict[str, Any]:
        return {
            "initial_time_ms": self.initial_metric,
            "best_time_ms": self.best_metric,
        }
