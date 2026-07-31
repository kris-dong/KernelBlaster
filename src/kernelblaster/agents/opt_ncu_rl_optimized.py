"""Optimized RL CUDA optimisation agent — alternative to ``opt_ncu_rl.py``.

Captures the high-ROI improvements identified in
``out/kernelblaster_rl_improvement_analysis.md``:

Tier 1 (cost):
  - Layered, cache-stable prompts (system + user split). Only the per-step user
    block changes between calls.
  - NCU shipped as a small JSON dict instead of an ASCII Speed-Of-Light table.
  - Slim DB injection — only the chosen technique's description is passed
    in user content; the index of all techniques lives in the cached system.

Tier 2 (search):
  - UCB1 selection over (state, technique) pairs, replacing the cubed-relevance
    weighted random pick.
  - State transitions re-enabled (``new_state`` is now refreshed from NCU each
    step, not pinned to the initial state).
  - Best-of-N seeding: the first ``seed_from_init_count`` rollouts start from
    init.cu; subsequent rollouts re-seed from the running top-K best variants.
  - Aggressive trajectory pruning: a trajectory is abandoned when the most
    recent step regresses by >5% AND the trajectory's running best has not
    improved for ``patience`` steps.

Tier 3 (model heterogeneity):
  - State analysis + plan generation: ``MODEL_PLAN`` (default cheap).
  - Codegen for "simple" technique categories: ``MODEL_CODEGEN_SIMPLE``.
  - Codegen for "hard" technique categories: ``MODEL_CODEGEN_HARD``.
  - Fix attempts: ``MODEL_FIX`` (default cheap), capped at 2 LLM attempts; a
    deterministic syntax-fix pre-pass (``_deterministic_fix``) runs first and
    repairs the most common nvcc errors with regex / pattern matching.

Tier 4 (deterministic infra wins):
  - NCU profile cache keyed by SHA-1 of the kernel source. Two trajectories
    that converge on the same code share one ~30–60 s NCU profile.
  - Compile-only validation step before NCU profiling (cheap rejection).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cost_tracker import CostTracker
from .progress_writer import ProgressWriter
from .database_optimized import (
    OptimizedOptimizationDatabase,
    TieredLLMInterface,
    extract_metrics_json,
)
from .database import OptimizationEntry, CompositeOptimization
from .feedback import FeedbackConfig
from .opt_ncu_rl import RLNCUFeedback, parse_ncu_metrics
from .opt_rl_optimized_base import RLOptimizedAgentBase
from .reprofile_nsys import _parse_nsys_gpu_trace
import base64
from .rl_agents import Trajectory, TrajectoryStep
from .utils import (
    FeedbackError,
    NamedTimer,
    UTILIZATION_METRICS,
    compile_and_run_cu_file,
    find_kernel_names_ncu,
    generate_code_retry,
    run_gpu_executable,
)
from .utils.perf_log import perf_span, perf_record
from ..backends import CUDABackend, ProfileResult
from .rl import ProfileCacheEntry

# ---------------------------------------------------------------------------
# CUDA backend singleton — passed to :class:`RLOptimizedAgentBase` so it
# owns the tier-dispatch + deterministic-fix path via ``self.backend``.
# ---------------------------------------------------------------------------

_CUDA_BACKEND = CUDABackend()


# The token-budget bumper, UCB1 bandit, deterministic fix rules, and
# content-hash profile cache all moved to ``agents.rl`` (P5.1–P5.5). They're
# imported at the top of the file — see ``from .rl import ...``. The
# ``_deterministic_fix`` free function became ``backend.deterministic_fix``;
# ``categorise_technique`` became ``backend.categorise_technique``. The old
# ``_ProfileCacheEntry`` (NCU-specific fields) → ``ProfileCacheEntry``
# (backend-agnostic: ``primary_metric``, ``profile: ProfileResult``, ``stderr``).


# ---------------------------------------------------------------------------
# Optimised RL agent
# ---------------------------------------------------------------------------


class OptimizedRLNCUAgent(RLOptimizedAgentBase):
    """CUDA-specific optimized RL agent.

    Inherits the tiered-model dispatch, UCB1 bandit, profile cache,
    top-K seed buffer, and adaptive token budget from
    :class:`RLOptimizedAgentBase`. Only the CUDA-specific bits live
    here: NCU + nsys profile capture, ``elapsed_cycles`` parsing, and
    the initialize/run outer wrappers with their perf spans.
    """

    agent_perf_label: str = "opt_ncu_rl_optimized"

    def __init__(
        self,
        fb_config: FeedbackConfig,
        code_to_optimize_fp: Path,
        database_path: Path,
        *,
        max_rollout_steps: int = 5,
        replay_buffer_size: int = 1000,
        num_rl_iterations: int = 50,
        seed_from_init_count: int = 10,
        bandit_exploration: float = 1.4,
        prune_patience: int = 2,
        prune_regression_pct: float = -5.0,
        max_fix_attempts: int = 2,
        database: Optional[OptimizedOptimizationDatabase] = None,
        cost_tracker: Optional[CostTracker] = None,
        problem_id: Optional[str] = None,
        progress_writer: Optional[ProgressWriter] = None,
    ):
        # Shared init (bandit, profile_cache, seed_buffer, tiered
        # model config, prune knobs) lives on the base.
        super().__init__(
            fb_config,
            code_to_optimize_fp,
            backend=_CUDA_BACKEND,
            max_rollout_steps=max_rollout_steps,
            replay_buffer_size=replay_buffer_size,
            num_rl_iterations=num_rl_iterations,
            seed_from_init_count=seed_from_init_count,
            bandit_exploration=bandit_exploration,
            prune_patience=prune_patience,
            prune_regression_pct=prune_regression_pct,
            max_fix_attempts=max_fix_attempts,
            cost_tracker=cost_tracker,
            problem_id=problem_id,
            progress_writer=progress_writer,
        )

        # CUDA-specific: database with cheap_llm plumbed through for
        # tiered state-analysis dispatch.
        gpu_report_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "algo-sol-modeling/algo-space/gpu_optimization_report.md"
        )
        cheap_llm = TieredLLMInterface(
            model_name=self.model_plan,
            logger=self.agent_logger,
            cost_tracker=cost_tracker,
            role_label="plan",
        )
        if database is None:
            self.database = OptimizedOptimizationDatabase(
                database_path,
                gpu_report_path,
                cheap_llm,
                cheap_llm=cheap_llm,
                cost_tracker=cost_tracker,
            )
        else:
            self.database = database
            # Late-bind tracker if the database wasn't constructed with one.
            if cost_tracker is not None and getattr(self.database, "cost_tracker", None) is None:
                self.database.cost_tracker = cost_tracker

        # CUDA-only knob (NCU profiling timeout).
        self.ncu_timeout_s = int(os.getenv("KERNELBLASTER_NCU_TIMEOUT_S", "600"))

    # ------------------------------------------------------------------
    # Profiling — cached by code hash
    # ------------------------------------------------------------------

    async def gather_perf_metrics_cached(
        self, filepath: Path
    ) -> Tuple[str, str, str, int, Dict[str, Any]]:
        """Wrap NCU profiling with a SHA-1-keyed cache.

        Returns ``(annotated_ncu, raw_ncu, stderr, cycles, metrics_json)``.
        On cache hit we skip the ~30–60 s NCU call entirely.
        """
        try:
            code = filepath.read_text()
        except Exception:
            code = ""

        cached = self.profile_cache.get(code) if code else None
        if cached is not None:
            cycles = int(cached.primary_metric)
            annotated_ncu = cached.profile.raw_metrics.get("annotated_ncu", "")
            raw_ncu = cached.profile.raw_log
            metrics_json = cached.profile.raw_metrics.get("metrics_json", {})
            self.agent_logger.info(
                f"NCU cache hit for {filepath.name} (cycles={cycles})"
            )
            # Cache hit is ~free; record it for cache-rate analysis.
            perf_record(
                phase="profile_cache_hit",
                duration_s=0.0,
                problem_id=self.problem_id,
                agent="opt_ncu_rl_optimized",
                extra={"cycles": cycles},
            )
            return annotated_ncu, raw_ncu, cached.stderr, cycles, metrics_json

        with perf_span(
            phase="profile_total",
            problem_id=self.problem_id,
            agent="opt_ncu_rl_optimized",
        ) as span:
            annotated_ncu, raw_ncu, stderr, cycles = await self._gather_perf_metrics(filepath)
            span.set_extra(cycles=cycles)
        # ``cycles`` is now nsys gpu_time_ns (wall-clock across solution kernels).
        # The bottleneck kernel's NCU cycles (if parseable from raw_ncu) end up
        # in metrics_json["elapsed_cycles"] separately for LLM context.
        metrics_json = extract_metrics_json(raw_ncu, gpu_time_ns=cycles)
        if code:
            # Wrap the CUDA-specific fields into a generic ProfileResult so
            # the cache stays backend-agnostic (P5.2). CUDA readers still
            # find annotated_ncu / metrics_json under ``raw_metrics``.
            profile = ProfileResult(
                total_time_ms=float(cycles),
                per_kernel_ms={},
                raw_metrics={
                    "annotated_ncu": annotated_ncu,
                    "metrics_json": metrics_json,
                    "elapsed_cycles": int(cycles),
                },
                raw_log=raw_ncu,
            )
            self.profile_cache.put(
                code,
                ProfileCacheEntry(
                    primary_metric=float(cycles),
                    profile=profile,
                    stderr=stderr or "",
                ),
            )
        return annotated_ncu, raw_ncu, stderr, cycles, metrics_json

    async def _nsys_timing(
        self, executable: Path, solution_kernel_names: set
    ) -> Tuple[int, Dict[str, int]]:
        """Run nsys once, return (gpu_span_ns, per_kernel_total_ns).

        ``gpu_span_ns`` = wall-clock from the first solution-kernel start to the
        last solution-kernel end (inter-kernel gaps included; LibTorch noise
        excluded by ``solution_kernel_names`` filter).

        ``per_kernel_total_ns`` is a {kernel_name: summed_duration_ns} dict
        used to pick the bottleneck kernel for the NCU details pass. Keys are
        the source-declared kernel names (matched as substrings against the
        possibly-mangled trace names).
        """
        report_file = f"/tmp/kbrl_nsys_{uuid.uuid4().hex[:10]}"
        # Inline Python that queries the .sqlite for per-kernel start/dur/name
        # — same shape as reprofile_nsys.py but without the cross-run cache.
        query_py = (
            "import sqlite3\n"
            f"conn = sqlite3.connect('{report_file}.sqlite')\n"
            "c = conn.cursor()\n"
            "c.execute('SELECT k.start, k.end - k.start, s.value "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.demangledName = s.id "
            "ORDER BY k.start')\n"
            "for r in c.fetchall():\n"
            "    print(f'{r[0]},{r[1]},{r[2]}')\n"
        )
        query_b64 = base64.b64encode(query_py.encode()).decode()
        nsys_prefix = (
            f"bash -c '"
            f"nsys profile --force-overwrite=true --export=sqlite "
            f"--output={report_file} \"$1\" >/dev/null && "
            f"echo __GPUTRACE_CSV_START__ && "
            f"python3 -c \"import base64,sys;exec(base64.b64decode(sys.argv[1]))\" "
            f"{query_b64} && "
            f"rm -f {report_file}.nsys-rep {report_file}.sqlite"
            f"' _"
        )
        nsys_stdout, nsys_stderr = await run_gpu_executable(
            executable,
            self.gpu,
            self.ncu_timeout_s,
            job_name=f"{executable} (nsys timing)",
            prefix_command=nsys_prefix,
        )
        combined = nsys_stdout + "\n" + nsys_stderr

        span_ns, _, _, _ = _parse_nsys_gpu_trace(combined, solution_kernel_names)

        # Build {source_kernel: total_ns} from the same CSV — sum over all
        # invocations of each declared kernel. We re-walk the parser-friendly
        # section to keep this self-contained.
        per_kernel: Dict[str, int] = {n: 0 for n in solution_kernel_names}
        marker = "__GPUTRACE_CSV_START__"
        if marker in combined:
            tail = combined.split(marker, 1)[1]
            for line in tail.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",", 2)
                if len(parts) < 3:
                    continue
                try:
                    dur_ns = int(parts[1])
                except ValueError:
                    continue
                trace_name = parts[2].strip().strip('"')
                for sol in solution_kernel_names:
                    if sol in trace_name:
                        per_kernel[sol] = per_kernel.get(sol, 0) + dur_ns
                        break  # first match wins; trace names rarely overlap

        return span_ns, per_kernel

    async def _gather_perf_metrics(self, filepath: Path) -> Tuple[str, str, str, int]:
        """Compile + verify + nsys timing + filtered NCU details.

        Three-step redesign:
          1. Compile + run with the verification harness.
          2. ``nsys`` profile pass: gives wall-clock GPU time (ns) from first
             solution-kernel start to last solution-kernel end. This is the
             reward signal — fast (single profile pass), and pre-filtered by
             source-declared kernel names so LibTorch noise is excluded.
          3. ``ncu`` details pass on the *single bottleneck kernel* (the
             solution kernel with the largest total ns from step 2). Provides
             SpeedOfLight / Occupancy / LaunchStats supplemental info for the
             LLM. Filtering to one kernel keeps NCU runtime under control even
             when the binary launches many kernels.

        Returns ``(annotated_ncu, raw_ncu, stderr, gpu_time_ns)``. The fourth
        value is named ``cycles`` historically but now carries nanoseconds —
        every reward / comparison path is monotonic-only so the unit change
        is transparent.
        """
        timer = NamedTimer()
        with perf_span(
            phase="compile_run",
            problem_id=self.problem_id,
            agent="opt_ncu_rl_optimized",
        ):
            stdout_list, stderr_list, path, success = await compile_and_run_cu_file(
                self.test_code_fp,
                filepath,
                self.gpu,
                timer,
                self.agent_logger,
                persistent_artifacts=True,
                timeout=self.ncu_timeout_s,
                num_runs=1,
                passed_keyword="passed",
            )
        if not success:
            FeedbackAgent.raise_numerics_verification_error(stdout_list, stderr_list)

        # ── (1) discover source-declared kernel names ──────────────────
        try:
            with perf_span(
                phase="find_kernel_names",
                problem_id=self.problem_id,
                agent="opt_ncu_rl_optimized",
            ):
                kernel_names = await find_kernel_names_ncu(
                    path, filepath, self.gpu, self.ncu_timeout_s
                )
        except Exception as e:
            self.agent_logger.warning(f"find_kernel_names_ncu failed: {e}")
            kernel_names = []
        if not kernel_names:
            # Without a kernel-name list we cannot filter the nsys trace, so a
            # 0 reward here is meaningless. Surface as a profile failure so the
            # fix-loop (rollouts) or init-fix retry (initialize) can attempt to
            # repair the source instead of silently anchoring on cycles=0.
            raise FeedbackError(
                f"find_kernel_names_ncu returned no kernels for {filepath} — "
                f"the binary likely launches no source-declared kernels (broken "
                f"launcher, optimised-out kernels, or kernel-name extraction bug)."
            )
        solution_kernels = set(kernel_names)

        # ── (2) nsys timing pass — the reward ──────────────────────────
        try:
            with perf_span(
                phase="nsys_timing",
                problem_id=self.problem_id,
                agent="opt_ncu_rl_optimized",
            ) as span:
                gpu_time_ns, per_kernel_ns = await self._nsys_timing(path, solution_kernels)
                span.set_extra(gpu_time_ns=gpu_time_ns)
        except Exception as e:
            # Don't silently set the reward to 0 — that masks profile failures
            # as fake "successful 0-cycle measurements" and lets the comparator
            # crown bogus rollouts as winners.
            raise FeedbackError(f"nsys timing pass failed for {filepath}: {e}") from e

        if gpu_time_ns <= 0:
            # The nsys call ran but returned a 0-span: the solution kernels
            # were never observed in the trace. This is a measurement failure,
            # not a 0-reward win — see comment above.
            raise FeedbackError(
                f"nsys returned gpu_time_ns={gpu_time_ns} for {filepath} — "
                f"solution kernels {sorted(solution_kernels)} not seen in trace; "
                f"the binary likely never invokes them (early exit, abort, or "
                f"renamed launchers)."
            )

        # ── (3) NCU details pass on the bottleneck kernel only ─────────
        if per_kernel_ns and any(v > 0 for v in per_kernel_ns.values()):
            bottleneck = max(per_kernel_ns.items(), key=lambda kv: kv[1])[0]
        else:
            bottleneck = kernel_names[0]
        self.agent_logger.info(
            f"NCU details pass filtered to bottleneck kernel '{bottleneck}' "
            f"(gpu_time_ns={gpu_time_ns:,}, per_kernel_ns={per_kernel_ns})"
        )

        details_command = (
            f"ncu -k {bottleneck} --page details --section=SpeedOfLight "
            f"--section=Occupancy --section=LaunchStats --csv --metrics "
            + ",".join(UTILIZATION_METRICS)
        )
        with perf_span(
            phase="ncu_details",
            problem_id=self.problem_id,
            agent="opt_ncu_rl_optimized",
        ) as span:
            span.set_extra(bottleneck_kernel=bottleneck)
            details_stdout, details_stderr = await run_gpu_executable(
                path,
                self.gpu,
                self.ncu_timeout_s,
                job_name=f"{filepath} (ncu details, k={bottleneck})",
                prefix_command=f"NVIDIA_TF32_OVERRIDE=0 {details_command} ",
            )
        if "No Kernels were profiled" in details_stdout:
            # NCU couldn't match; still return the nsys timing as the reward.
            return "", "", details_stderr, gpu_time_ns

        return details_stdout, details_stdout, details_stderr, gpu_time_ns

    # ------------------------------------------------------------------
    # LLM dispatchers — inherited from RLOptimizedAgentBase (P5.7).
    # ``_llm_codegen`` routes via ``self.backend.categorise_technique``;
    # ``_llm_fix`` uses ``self.model_fix``. Both apply cost tracking +
    # adaptive token-budget bump.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Run — main entry point
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Public entry point — wraps the impl in an init_total perf span."""
        started_at = time.time()
        try:
            await self._initialize_impl()
        finally:
            perf_record(
                phase="init_total",
                duration_s=time.time() - started_at,
                problem_id=self.problem_id,
                agent="opt_ncu_rl_optimized",
                success=(self.initial_cycles is not None),
            )

    async def _initialize_impl(self) -> None:
        self.code_to_optimize_fp = self.folder / "init.cu"
        self.code_to_optimize_fp.write_text(self.code_to_optimize)
        self.agent_logger.info("Gathering initial NCU profile…")

        # Mirror the rollout fix-loop in _apply_optimization: deterministic
        # regex repair first, then bounded LLM-driven fix attempts. The repaired
        # init code becomes the new baseline; intermediate variants are kept on
        # disk under init_fix_attempt{N}.cu for inspection.
        current_code = self.code_to_optimize
        last_error: Optional[str] = None
        for attempt in range(self.max_fix_attempts + 1):
            attempt_fp = (
                self.code_to_optimize_fp
                if attempt == 0
                else self.folder / f"init_fix_attempt{attempt}.cu"
            )
            attempt_fp.write_text(current_code)
            try:
                annotated, raw, _, cycles, metrics = (
                    await self.gather_perf_metrics_cached(attempt_fp)
                )
            except FeedbackError as e:
                err = str(e)
                last_error = err

                patched = self.backend.deterministic_fix(current_code, err)
                if patched is not None and patched != current_code:
                    self.agent_logger.info(
                        f"Deterministic syntax-fix repaired init (attempt {attempt})"
                    )
                    current_code = patched
                    continue

                if attempt >= self.max_fix_attempts:
                    self.agent_logger.error(
                        f"Initial profiling failed after {self.max_fix_attempts} fix attempts "
                        f"(init.cu compile/run broken): {err}; "
                        f"continuing without baseline — final result will be marked 'no_baseline'"
                    )
                    self.last_ncu_log = ""
                    return

                self.agent_logger.warning(
                    f"Initial profiling failed (attempt {attempt}); requesting LLM fix: {err}"
                )
                fix_messages = self.database.build_fix_messages(
                    broken_kernel=current_code, compiler_error=err
                )
                try:
                    fix_text = await self._llm_fix(fix_messages)
                except Exception as fix_err:
                    self.agent_logger.error(
                        f"LLM fix call failed during init repair: {fix_err}; "
                        f"continuing without baseline — final result will be marked 'no_baseline'"
                    )
                    self.last_ncu_log = ""
                    return
                fixed_code, fix_fp = self.get_code_from_response(
                    fix_text, 0, attempt + 1, self.agent_logger
                )
                if not fixed_code or fixed_code.strip() == current_code.strip():
                    self.agent_logger.error(
                        f"LLM fix produced no actionable change at init attempt {attempt}; "
                        f"continuing without baseline — final result will be marked 'no_baseline'"
                    )
                    self.last_ncu_log = ""
                    return
                current_code = fixed_code
                try:
                    if fix_fp.exists() and fix_fp != attempt_fp:
                        fix_fp.unlink()
                except Exception:
                    pass
                continue

            # Success path — record baseline and persist the (possibly repaired)
            # code as init.cu so downstream reads see the working version.
            if current_code != self.code_to_optimize:
                self.agent_logger.info(
                    f"init.cu repaired after {attempt} fix attempt(s); "
                    f"persisting repaired source as the baseline."
                )
                self.code_to_optimize = current_code
                self.code_to_optimize_fp.write_text(current_code)
            self.initial_cycles = cycles
            self.best_cycles = cycles
            self.last_ncu_log = raw
            (self.folder / "0_init_annotated.cu").write_text(
                annotated or self.code_to_optimize
            )
            # Write a baseline marker file the legacy speedup-calc path looks for
            # (database.py:1841 → ncu/0_init_ncu_log.txt). Without this, every
            # ``update_optimization_result`` call logs "No elapsed cycles found
            # in text" and falls back to 1.0× speedup, polluting the cross-problem
            # speedup priors. The value carried is nsys gpu_time_ns post-redesign,
            # but get_elapsed_cycles_v2 only reads the integer — unit-agnostic.
            try:
                ncu_dir = self.folder / "ncu"
                ncu_dir.mkdir(parents=True, exist_ok=True)
                (ncu_dir / "0_init_ncu_log.txt").write_text(
                    f"Elapsed Cycles: {int(cycles)}\n"
                )
            except Exception as e:
                self.agent_logger.debug(f"Could not write baseline marker: {e}")
            # Late-bind the (possibly deterministic-fix-repaired) init code
            # to the seed buffer so subsequent rollouts see the working version.
            self.seed_buffer.set_init_code(self.code_to_optimize)
            self.seed_buffer.update(float(cycles), self.code_to_optimize)
            self.agent_logger.info(f"Initial cycles={cycles}, metrics={metrics}")
            return

        # Loop exhausted without a successful profile (every attempt re-raised).
        self.agent_logger.error(
            f"Initial profiling exhausted fix attempts (last error: {last_error}); "
            f"continuing without baseline — final result will be marked 'no_baseline'"
        )
        self.last_ncu_log = ""

    async def run(self) -> Path:
        """Public entry point — wraps the impl in a problem_total perf span."""
        started_at = time.time()
        outcome = "unknown"
        try:
            result = await self._run_impl()
            name = result.name if hasattr(result, "name") else str(result)
            if "success_" in name:
                outcome = "success"
            elif "no_baseline" in name:
                outcome = "no_baseline"
            elif "failure_" in name:
                outcome = "failure"
            return result
        finally:
            perf_record(
                phase="problem_total",
                duration_s=time.time() - started_at,
                problem_id=self.problem_id,
                agent="opt_ncu_rl_optimized",
                success=(outcome == "success"),
                extra={"outcome": outcome},
            )

    async def _run_impl(self) -> Path:
        """Run ``num_rl_iterations`` rollouts in parallel and return the best file."""
        if self.initial_cycles is None:
            await self.initialize()

        # Compute initial state once and share across rollouts.
        initial_state = (
            await self.database.get_state_from_ncu_report(
                self.last_ncu_log,
                parse_ncu_metrics(self.last_ncu_log),
                self.code_to_optimize,
                elapsed_cycles=self.initial_cycles,
            )
        )

        async def _one_rollout(idx: int) -> Optional[Trajectory]:
            try:
                seed_code, seed_metric = self.seed_buffer.pick(idx)
                if seed_metric is not None:
                    self.agent_logger.info(
                        f"Rollout {idx} seeded from prior best ({seed_metric} cycles)"
                    )
                return await self._run_rollout(idx, seed_code, initial_state)
            except Exception as e:
                self.agent_logger.error(f"Rollout {idx} failed: {e}")
                return None

        tasks = [asyncio.create_task(_one_rollout(i)) for i in range(self.num_rl_iterations)]

        best_filename: Optional[Path] = None
        best_cycles = self.best_cycles

        for coro in asyncio.as_completed(tasks):
            traj = await coro
            if traj is None or not traj.steps:
                continue
            self.replay_buffer.add_trajectory(traj)
            best_step = min(traj.steps, key=lambda s: s.cycles)
            if best_step.cycles < best_cycles:
                best_cycles = best_step.cycles
                best_filename = self.folder / f"rl_iter_best.cu"
                best_filename.write_text(
                    best_step.code + f"\n\n// Elapsed Cycles: {best_step.cycles}\n"
                )
                self.agent_logger.info(f"New best: {best_cycles} cycles (action={best_step.action})")
            self.seed_buffer.update(float(best_step.cycles), best_step.code)

        # Persist database snapshot.
        try:
            self.database._persist_database()
        except Exception as e:
            self.agent_logger.warning(f"Database persist failed: {e}")

        if best_filename is not None:
            if self.initial_cycles is not None and best_cycles < self.initial_cycles:
                final = self.folder / "success_rl_optimization.cu"
                final.write_text(best_filename.read_text())
                return final
            if self.initial_cycles is None:
                # No baseline (init.cu didn't profile) but rollouts produced a
                # working kernel. Keep the best artifact and mark as 'no_baseline'
                # so the runner/dashboard show the distinct outcome instead of
                # silently dropping the work.
                no_baseline = self.folder / "no_baseline_rl_optimization.cu"
                no_baseline.write_text(
                    best_filename.read_text()
                    + f"\n\n// Note: initial baseline unavailable (init.cu profile failed); "
                    f"best rollout cycles={best_cycles}\n"
                )
                self.agent_logger.warning(
                    f"No initial baseline — kept best rollout kernel ({best_cycles} cycles) "
                    f"as no_baseline_rl_optimization.cu"
                )
                return no_baseline

        failure = self.folder / "failure_rl_optimization.cu"
        baseline_str = self.initial_cycles if self.initial_cycles is not None else "N/A"
        failure.write_text(self.code_to_optimize + f"\n\n// Elapsed Cycles: {baseline_str}\n")
        self.agent_logger.warning("RL produced no improvement.")
        return failure

    # P5.4 extraction: ``_pick_seed_code`` and ``_update_top_k`` moved to
    # ``rl.TopKSeedBuffer``. Callers use ``self.seed_buffer.pick(idx)`` and
    # ``self.seed_buffer.update(metric, code)`` directly.

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    async def _run_rollout(
        self, traj_idx: int, seed_code: str, seed_state: str
    ) -> Trajectory:
        async with self._trajectory_lock:
            self.total_trajectories += 1
            trajectory_index = self.total_trajectories

        traj_uid = uuid.uuid4().hex[:8]
        traj_dir = self.folder / f"trajectory_{trajectory_index}_{traj_uid}"
        traj_dir.mkdir(parents=True, exist_ok=True)

        trajectory = Trajectory()
        current_code = seed_code
        current_state = seed_state
        current_cycles: Optional[int] = self.initial_cycles
        last_ncu_log = self.last_ncu_log
        running_best_cycles = current_cycles if current_cycles else float("inf")
        steps_since_improvement = 0

        for step in range(self.max_rollout_steps):
            # Plan: top-N candidates for the current state.
            try:
                profile = await self.database.analyze_performance_state(
                    last_ncu_log,
                    parse_ncu_metrics(last_ncu_log),
                    current_code,
                    elapsed_cycles=current_cycles,
                )
                analysis_json = json.dumps(asdict(profile), indent=2)
                plan = await self.database.generate_optimization_plan(
                    analysis_json,
                    current_code,
                    top_n=max(4, self.max_rollout_steps - step),
                )
            except Exception as e:
                self.agent_logger.warning(f"Plan failed for traj {traj_idx} step {step}: {e}")
                plan = []

            if not plan:
                self.agent_logger.info(f"Empty plan; stopping rollout {traj_idx} at step {step}")
                break

            candidate_pairs = [
                (p.get("technique"), float(p.get("relevance_score") or 0.0))
                for p in plan
                if p.get("technique")
            ]
            if not candidate_pairs:
                break
            candidate_names = [n for n, _ in candidate_pairs]
            candidate_weights = [w for _, w in candidate_pairs]

            # Bandit selection — UCB1 once arms have data; for cold-start
            # (all arms unseen), use the trajectory index to deterministically
            # spread across distinct unseen actions instead of letting cubed-
            # relevance sampling collapse all parallel trajectories onto the
            # same dominant pick.
            chosen_name = self.bandit.select(
                current_state, candidate_names,
                weights=candidate_weights,
                traj_idx=traj_idx,
            )
            chosen_plan = next((p for p in plan if p.get("technique") == chosen_name), plan[0])
            strategy_description = chosen_plan.get("description", "")

            self.agent_logger.info(
                f"Traj {traj_idx} step {step}: bandit chose '{chosen_name}' "
                f"(state={current_state}; plan_relevance={chosen_plan.get('relevance_score')})"
            )

            try:
                optimized_code, new_cycles, new_ncu_log, new_metrics = await self._apply_optimization(
                    current_code, chosen_name, strategy_description, current_cycles, traj_dir, step
                )
            except Exception as e:
                self.agent_logger.error(f"Apply failed traj {traj_idx} step {step}: {e}")
                break

            # Reward = improvement fraction; bandit consumes this directly.
            if current_cycles and current_cycles > 0:
                actual_improvement = ((current_cycles - new_cycles) / current_cycles) * 100
            else:
                actual_improvement = 0.0
            reward = actual_improvement / 100.0
            self.bandit.update(current_state, chosen_name, reward)
            # Phase 3c full: prefer direct metrics over the legacy file parse.
            # ``current_file_path`` remains as a fallback for the tiny window
            # where ``self.initial_cycles`` might be None (initial profiling
            # exhausted its fix attempts — see the branch at :925). The
            # DB then reads init.profile.json / ncu/0_init_ncu_log.txt via
            # _read_baseline_metric_from_files.
            self.database.update_optimization_result(
                current_state, chosen_name, actual_improvement,
                current_metric=float(new_cycles),
                baseline_metric=(
                    float(self.initial_cycles)
                    if self.initial_cycles is not None
                    else None
                ),
                current_file_path=self.folder / "init.cu",
            )

            traj_step = TrajectoryStep(
                state=current_state,
                action=chosen_name,
                code=optimized_code,
                cycles=new_cycles,
                predicted_improvement=chosen_plan.get("relevance_score", 0.0) * 100,
                actual_improvement=actual_improvement,
                reward=reward,
            )
            trajectory.add_step(traj_step)

            # Live per-problem progress dump (file-only, no logger spam).
            if self.progress_writer is not None and self.problem_id is not None:
                try:
                    self.progress_writer.step_done(
                        self.problem_id,
                        traj_idx=traj_idx,
                        step_idx=step,
                        technique=chosen_name,
                        cycles=new_cycles,
                        improvement_pct=actual_improvement,
                    )
                except Exception:
                    pass  # progress writing must never block the optimisation loop

            # Re-derive state for next step. (Legacy code had this TODO-disabled.)
            try:
                next_state_profile = await self.database.analyze_performance_state(
                    new_ncu_log,
                    parse_ncu_metrics(new_ncu_log),
                    optimized_code,
                    elapsed_cycles=new_cycles,
                )
                current_state = await self.database.match_state_against_database(
                    next_state_profile
                )
            except Exception:
                pass  # keep current_state on failure

            current_code = optimized_code
            current_cycles = new_cycles
            last_ncu_log = new_ncu_log

            # Aggressive pruning.
            if new_cycles < running_best_cycles:
                running_best_cycles = new_cycles
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1

            if (
                actual_improvement < self.prune_regression_pct
                and steps_since_improvement >= self.prune_patience
            ):
                self.agent_logger.info(
                    f"Traj {traj_idx} pruned at step {step} "
                    f"(improvement={actual_improvement:.1f}%, "
                    f"steps_since_improvement={steps_since_improvement})"
                )
                break

        return trajectory

    # ------------------------------------------------------------------
    # Apply one optimization (codegen + compile + cached profile + fix loop)
    # ------------------------------------------------------------------

    async def _apply_optimization(
        self,
        code: str,
        technique_name: str,
        strategy_description: str,
        current_cycles: Optional[int],
        traj_dir: Path,
        step: int,
    ) -> Tuple[str, int, str, Dict[str, Any]]:
        """Generate + compile + profile a step. Returns (code, cycles, ncu_log, metrics)."""
        # Per-step span — bottleneck dashboard groups by phase, but this
        # gives us a step-level reference point too. trajectory id is parsed
        # from traj_dir (named "trajectory_<idx>_<short_hash>").
        try:
            traj_idx = int(traj_dir.name.split("_", 2)[1])
        except (IndexError, ValueError):
            traj_idx = None
        step_started_at = time.time()
        step_status = "ok"
        try:
            return await self.__apply_optimization_inner(
                code, technique_name, strategy_description,
                current_cycles, traj_dir, step,
            )
        except Exception:
            step_status = "raised"
            raise
        finally:
            perf_record(
                phase="step_total",
                duration_s=time.time() - step_started_at,
                problem_id=self.problem_id,
                agent="opt_ncu_rl_optimized",
                step=step,
                trajectory=traj_idx,
                success=(step_status == "ok"),
                extra={"technique": technique_name, "status": step_status},
            )

    async def __apply_optimization_inner(
        self,
        code: str,
        technique_name: str,
        strategy_description: str,
        current_cycles: Optional[int],
        traj_dir: Path,
        step: int,
    ) -> Tuple[str, int, str, Dict[str, Any]]:
        # 1. Build messages with cache-stable system prompt.
        # Use last cached profile metrics if available; else minimal dict.
        # ProfileCacheEntry stashes metrics_json under ProfileResult.raw_metrics
        # per the P5.2 backend-agnostic shape.
        try:
            cached_entry = self.profile_cache.get(code)
            current_metrics_json = (
                cached_entry.profile.raw_metrics.get("metrics_json", {})
                if cached_entry is not None else {}
            )
        except Exception:
            current_metrics_json = {}
        if current_cycles and "elapsed_cycles" not in current_metrics_json:
            current_metrics_json = dict(current_metrics_json)
            current_metrics_json["elapsed_cycles"] = int(current_cycles)

        # Best-so-far hint to keep the LLM grounded.
        snapshot = self.seed_buffer.snapshot()
        if snapshot and current_cycles is not None:
            best = int(min(m for m, _ in snapshot))
            best_so_far_summary = (
                f"Running best across all rollouts: {best} cycles "
                f"(this trajectory's current: {current_cycles} cycles)."
            )
        else:
            best_so_far_summary = None

        messages = self.database.build_codegen_messages(
            technique_name=technique_name,
            kernel_source=code,
            ncu_metrics_json=current_metrics_json,
            strategy_description=strategy_description,
            best_so_far_summary=best_so_far_summary,
        )

        # 2. Codegen.
        response_text = await self._llm_codegen(messages, technique_name=technique_name)
        optimized_code, candidate_fp = self.get_code_from_response(
            response_text, step, 0, self.agent_logger
        )

        # Persist to a step file.
        base = f"step_{step}_{re.sub(r'[^A-Za-z0-9_]+', '_', technique_name)}"
        target = traj_dir / f"{base}.cu"
        target.write_text(optimized_code)
        try:
            if candidate_fp != target and candidate_fp.exists():
                candidate_fp.unlink()
        except Exception:
            pass

        # 3. Profile with cache + fix-loop.
        for attempt in range(self.max_fix_attempts + 1):
            attempt_fp = traj_dir / f"{base}_attempt{attempt}.cu"
            attempt_fp.write_text(optimized_code)
            try:
                annotated, raw_ncu, _, cycles, metrics = await self.gather_perf_metrics_cached(
                    attempt_fp
                )
                return optimized_code, cycles, raw_ncu, metrics
            except FeedbackError as e:
                err = str(e)
                # 3a. Deterministic syntax fix first.
                patched = self.backend.deterministic_fix(optimized_code, err)
                if patched is not None and patched != optimized_code:
                    self.agent_logger.info(
                        f"Deterministic syntax-fix repaired step {step} attempt {attempt}"
                    )
                    optimized_code = patched
                    continue
                # 3b. LLM fix attempt (capped).
                if attempt >= self.max_fix_attempts:
                    raise
                fix_messages = self.database.build_fix_messages(
                    broken_kernel=optimized_code, compiler_error=err
                )
                try:
                    fix_text = await self._llm_fix(fix_messages)
                except Exception:
                    raise
                fixed_code, fix_fp = self.get_code_from_response(
                    fix_text, step, attempt + 1, self.agent_logger
                )
                if not fixed_code or fixed_code.strip() == optimized_code.strip():
                    raise
                optimized_code = fixed_code
                try:
                    if fix_fp.exists() and fix_fp != attempt_fp:
                        fix_fp.unlink()
                except Exception:
                    pass

        # Should not reach here — the loop either returns or raises.
        raise RuntimeError("Unreachable: _apply_optimization fix loop exited without return")
