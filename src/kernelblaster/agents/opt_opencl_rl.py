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
from .rl_agents import ReplayBuffer, Trajectory, TrajectoryStep
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
    async def gather_profile_result(self, kernel_filepath: Path):
        """Phase 4f.3b: wrap the OpenCL tuple return into a ProfileResult.

        Delegates to ``backend.parse_profile`` (which already builds the
        right ProfileResult from a ``[PROFILE]``-bearing stdout) then
        stashes stderr in raw_metrics for parity with the CUDA wrapper.

        Phase 3c: persists ``<kernel>.profile.json`` via the shared
        ``_write_profile_json`` helper (matches CUDA's path in
        ``opt_ncu_rl.gather_profile_result``). ``raw_log`` is filtered to
        just the ``[PROFILE]`` lines — keeping the full ~100KB driver dump
        per step would balloon the run tree for no real gain.
        """
        profile_output, stderr, time_ms = await self.gather_perf_metrics(kernel_filepath)
        pr = self.backend.parse_profile(profile_output)
        pr.raw_metrics["stderr"] = stderr
        pr.raw_log = "\n".join(
            ln for ln in profile_output.splitlines() if "[PROFILE]" in ln
        )
        if pr.total_time_ms > 0:
            self._write_profile_json(kernel_filepath, pr)
        return pr

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

        # profile.json write moved up to ``gather_profile_result`` in Phase 3c
        # CUDA (this commit) so both backends persist through the same
        # ``RLAgentBase._write_profile_json`` helper.

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
    # initialize() lifted to RLAgentBase in Phase 4f.3c. This subclass
    # overrides _maybe_generate_reference (SSH-execs the reference-gen pass
    # on the board) and _write_init_artifact (writes raw kernel source as
    # the init artifact). _handle_init_failure inherits the no-op default.

    async def _maybe_generate_reference(self) -> None:
        """OpenCL needs a one-shot CPU-reference cache on the board so
        subsequent profile passes don't recompute it. Calls the existing
        SSH-based ``_generate_reference`` helper."""
        await self._generate_reference()

    def _write_init_artifact(self, profile_result) -> None:
        """Write ``0_init_kernel.cl`` — raw kernel source; no NCU-style
        annotation needed since OpenCL prompt context comes from
        ``[PROFILE]`` markers in ``profile_result.raw_log``."""
        (self.folder / "0_init_kernel.cl").write_text(self.kernel_source)


    # _get_state removed in Phase 4f.3d.f — replaced by the shared
    # ``RLAgentBase._derive_state`` (ProfileResult-based) for get_feedback and
    # ``_derive_shared_initial_state`` for run(). Both consult
    # ``backend.parse_state_metrics`` + ``state_cycles_from_metric``.

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------
    # run() lifted to RLAgentBase in Phase 4f.3d.f. OpenCL contributes two
    # hooks: ``_pre_run`` (reset + seed verification-pool global-best) and
    # ``_finalize_run`` (prefer global-best over per-iter min; write raw
    # ``// Baseline:`` failure file instead of format_result_artifact).
    async def _pre_run(self) -> None:
        await self._reset_global_best_for_run()

        if not hasattr(self, "last_profile_log") or not self.last_profile_log:
            await self.initialize()

        # ``initialize()`` may run on the host before ``run()``; the reset above
        # clears session tracking, so re-establish the baseline from disk +
        # recorded ms.
        init_fp = self.folder / f"init{self.backend.kernel_ext}"
        if (
            self.initial_metric
            and math.isfinite(self.initial_metric)
            and init_fp.exists()
        ):
            await self._record_global_best_if_better(init_fp, self.initial_metric)

    async def _finalize_run(self, best_filename, best_metric) -> Path:
        ext = self.backend.kernel_ext
        g_time = self._global_best_time_ms
        g_code = self._global_best_code

        if g_code is not None and math.isfinite(g_time):
            gp = self.folder / self.backend.best_filename()
            gp.write_text(
                self._format_global_best_artifact(g_code, g_time),
                encoding="utf-8",
            )
            self.agent_logger.info(
                f"Global best (verified): {g_time:.3f} ms -> {gp.name} "
                f"(per-iter min: {best_metric if best_metric < float('inf') else 'n/a'})"
            )
        if (
            g_code is not None
            and self.initial_metric is not None
            and g_time < self.initial_metric
        ):
            final = self.folder / f"success_rl_optimization{ext}"
            final.write_text(
                self._format_global_best_artifact(g_code, g_time), encoding="utf-8"
            )
            return final

        if (
            best_filename is not None
            and self.initial_metric is not None
            and best_metric < self.initial_metric
        ):
            # Global tracking unavailable but per-iter best is a real improvement.
            final = self.folder / f"success_rl_optimization{ext}"
            final.write_text(best_filename.read_text(encoding="utf-8"), encoding="utf-8")
            return final

        # Failure: preserved OpenCL-style body (raw source + "// Baseline:"
        # note, not format_result_artifact — a longstanding intentional
        # distinction between "measured baseline" vs "achieved kernel time").
        failure_file = self.folder / f"failure_rl_optimization{ext}"
        baseline_str = (
            f"{self.initial_metric:.3f} ms"
            if self.initial_metric is not None
            else "N/A"
        )
        failure_file.write_text(
            self.kernel_source + f"\n\n// Baseline: {baseline_str}\n"
        )
        if best_filename is not None or g_code is not None:
            self.agent_logger.error(
                "RL did not improve over baseline; wrote failure file"
            )
        else:
            self.agent_logger.error("All RL iterations failed; wrote failure file")
        return failure_file

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    # run_rollout lifted to RLAgentBase in Phase 4f.3d.e. Backend hooks
    # (parse_state_metrics, state_cycles_from_metric, metric_to_traj_cycles,
    # database_update_kwargs) supply the OpenCL-flavoured bits — no
    # override needed here.
    # ------------------------------------------------------------------
    # Apply a single optimisation step
    # ------------------------------------------------------------------
    # apply_optimization lifted to RLAgentBase in Phase 4f.3d. Backend
    # picks the prompt via ``build_strategy_prompt`` / ``build_fix_prompt``.
    # OpenCL's fix-response intermediate file is now relocated into the
    # trajectory dir (matching CUDA's behaviour) — a minor cleanup that
    # was previously leaking a stray ``attempt<N>_task<N>.cl`` into
    # ``self.folder`` per fix retry.

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    # calculate_reward and _try_add_default_optimizations lifted to
    # RLAgentBase in Phase 4f.

    # ------------------------------------------------------------------
    # Feedback (get_feedback lifted to RLAgentBase in Phase 4f.3d)
    # ------------------------------------------------------------------
    # OpenCL contributes two hooks: the verification-pool best-step
    # override, and the ms/profile-output-flavored feedback dataclass.
    def _get_verified_best(self, trajectory):
        """Prefer the verification-pool global best over per-trajectory min
        when it exists — captures the fastest correct kernel across all
        attempts, not just the current trajectory's last step.
        """
        use_global = (
            self._global_best_code is not None
            and math.isfinite(self._global_best_time_ms)
        )
        if use_global:
            bstep = (
                min(trajectory.steps, key=lambda s: s.cycles)
                if trajectory.steps
                else None
            )
            best_ms = self._global_best_time_ms
            best_kernel = self._global_best_code
            best_action = bstep.action if bstep else "global_best_pool"
            if bstep is not None:
                # Step 4: cycles is now the native metric (ms), no scaling.
                t_ms = bstep.cycles
                if abs(t_ms - best_ms) > 0.0005:
                    self.agent_logger.info(
                        f"Feedback uses global best {best_ms:.3f} ms "
                        f"(trajectory min was {t_ms:.3f} ms)"
                    )
            return best_kernel, best_ms, best_action, bstep
        if trajectory.steps:
            best_step = min(trajectory.steps, key=lambda s: s.cycles)
            return (
                best_step.code,
                best_step.cycles,  # already ms since Step 4
                best_step.action,
                best_step,
            )
        return None, float("inf"), "n/a", None

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
        profile_output = profile_result.raw_log
        best_kernel, best_ms, best_action, bstep = self._get_verified_best(trajectory)

        if best_kernel is not None:
            if self.initial_metric and self.initial_metric > 0:
                improvement_pct = ((self.initial_metric - best_ms) / self.initial_metric) * 100
            else:
                improvement_pct = 0.0

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

        return RLOpenCLFeedback(
            new_messages=[
                {"role": "assistant", "content": response},
                {
                    "role": "user",
                    "content": "No successful optimisation steps completed. Try a different approach.",
                },
            ],
            success=False,
            metric=initial_metric,
            profile_output=profile_output,
            state=initial_state,
        )

    # get_performance_summary skeleton lifted to RLAgentBase in Phase 4f;
    # this override only contributes the OpenCL-named metric keys.
    def _perf_summary_extras(self) -> Dict[str, Any]:
        return {
            "initial_time_ms": self.initial_metric,
            "best_time_ms": self.best_metric,
        }
