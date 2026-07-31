# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Optimized RL agent for RISC-V + Zephyr + spike/FireSim (P5.11).

Inherits :class:`~kernelblaster.agents.opt_rl_optimized_base.RLOptimizedAgentBase`.
The base owns the shared bandit / profile cache / seed buffer / tier-
dispatched LLM calls (P5.1-P5.7); this subclass only owns the RISC-V-
specific bits:

* Profile capture: shells out through the framework's compile server
  (``ZephyrCompileStrategy`` → ``west build``) and exec server
  (``SpikeExecStrategy`` → ``spike_runner``); parses the ``MODELBLASTER_*``
  markers via :meth:`RiscvZephyrBackend.parse_profile`.
* Baseline + fix loop with ``init.c`` naming and ``self.backend.deterministic_fix``
  (gcc/picolibc-flavoured, populated in :class:`RiscvZephyrBackend`).
* RL loop uses ``self.backend.parse_state_metrics(...)`` where the CUDA
  variant uses NCU-specific ``parse_ncu_metrics(...)`` — this is the one
  place the two agents diverge in structure.

**Scope note:** the CUDA agent (``opt_ncu_rl_optimized.py``) has richer
per-step observability (nsys per-kernel span, NCU details cache, cost-
tracker roles). This agent ships with the minimum viable equivalent —
enough to run a real RL loop end-to-end against spike or FireSim; the
richer bits can be lifted from CUDA on demand.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..backends import ProfileResult, RiscvZephyrBackend
from ..config import GPUType
from .database import GPUOptimizationDatabase, OptimizationEntry
from .database_optimized import TieredLLMInterface
from .cost_tracker import CostTracker
from .feedback import FeedbackConfig
from .opt_rl_optimized_base import RLOptimizedAgentBase
from .progress_writer import ProgressWriter
from .rl import ProfileCacheEntry
from .rl_agents import Trajectory, TrajectoryStep
from .utils import FeedbackError, NamedTimer
from .utils.commands import compile_and_run_riscv, compile_and_run_riscv_batched
from .utils.exec_batch_client import ExecBatchClient, get_exec_batch_client


class OptimizedRLRiscvAgent(RLOptimizedAgentBase):
    """RISC-V (Zephyr + spike) optimized RL agent.

    Structural mirror of :class:`OptimizedRLNCUAgent`. The tiered-model
    dispatch, UCB1 bandit, profile cache, top-K seed buffer, and adaptive
    token budget all come from :class:`RLOptimizedAgentBase`. What's
    RISC-V-specific:

      - ``_gather_perf_metrics`` calls :func:`compile_and_run_riscv`
        (through the framework's compile + spike-exec servers) and hands
        the spike stdout to :meth:`RiscvZephyrBackend.parse_profile`.
      - ``initialize`` uses ``init.c`` (not ``init.cu``); the baseline
        marker is a ``profile.json`` under the trajectory dir rather
        than the CUDA ``ncu/0_init_ncu_log.txt``.
      - State analysis feeds through the backend's
        :meth:`parse_state_metrics` (which the RISC-V backend's
        implementation reads WALL_CYCLES + per-op mcycle out of the
        modelblaster CSV block).

    Constructor kwargs and knobs are otherwise identical to
    :class:`OptimizedRLNCUAgent`.
    """

    agent_perf_label: str = "opt_riscv_rl_optimized"

    def __init__(
        self,
        fb_config: FeedbackConfig,
        code_to_optimize_fp: Path,
        database_path: Path,
        *,
        gpu: GPUType = GPUType.RISCV_SPIKE,
        max_rollout_steps: int = 5,
        replay_buffer_size: int = 1000,
        num_rl_iterations: int = 50,
        seed_from_init_count: int = 10,
        bandit_exploration: float = 1.4,
        prune_patience: int = 2,
        prune_regression_pct: float = -5.0,
        max_fix_attempts: int = 2,
        database: Optional[GPUOptimizationDatabase] = None,
        cost_tracker: Optional[CostTracker] = None,
        problem_id: Optional[str] = None,
        progress_writer: Optional[ProgressWriter] = None,
        io_npz_path: Optional[Path] = None,
        spike_args_str: str = "",
        use_exec_batching: bool = True,
    ):
        backend = RiscvZephyrBackend(gpu=gpu)

        super().__init__(
            fb_config,
            code_to_optimize_fp,
            backend=backend,
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

        self.gpu = gpu
        self.io_npz_path = io_npz_path
        self.spike_args_str = spike_args_str

        # Database — RISC-V uses the base :class:`GPUOptimizationDatabase`
        # with the RISC-V footer markdown (author-tuned states like
        # ``memory_bound_stencil`` / ``compute_bound_gemm_small``).
        # ``cheap_llm`` is plumbed through so state-analysis dispatches
        # to the tiered path (same infrastructure win the CUDA agent gets).
        report_placeholder = Path("/dev/null")
        cheap_llm = TieredLLMInterface(
            model_name=self.model_plan,
            logger=self.agent_logger,
            cost_tracker=cost_tracker,
            role_label="plan",
        )
        if database is None:
            self.database = GPUOptimizationDatabase(
                database_path,
                report_placeholder,
                cheap_llm,
                backend=backend,
                cheap_llm=cheap_llm,
                cost_tracker=cost_tracker,
            )
        else:
            self.database = database
            if cost_tracker is not None and getattr(self.database, "cost_tracker", None) is None:
                self.database.cost_tracker = cost_tracker

        # Spike-execution timeout (long — spike is CPU-bound and can take
        # minutes on stock-dim KernelBench problems).
        self.spike_timeout_s = int(os.getenv("KERNELBLASTER_SPIKE_TIMEOUT_S", "1800"))

        # Batching: when enabled (default) the RL exec calls route
        # through a client-side :class:`ExecBatchClient` that coalesces
        # concurrent rollout submits into ``/gpu/batch`` calls. Lazily
        # created on first use (needs an event loop). Users can disable
        # per-run to A/B against single-item exec.
        self._use_exec_batching = use_exec_batching
        self._batch_client: Optional[ExecBatchClient] = None

        # Legacy field the RL loop still references; RISC-V uses cycles
        # directly. ``last_ncu_log`` is repurposed to store the raw
        # spike stdout so the state analyzer has full context.

    # ------------------------------------------------------------------
    # Profile capture — RISC-V specific
    # ------------------------------------------------------------------

    async def _gather_perf_metrics(
        self, filepath: Path
    ) -> Tuple[str, str, str, int, Dict[str, Any]]:
        """Compile ``filepath`` + run on spike; return the CUDA-shaped
        5-tuple ``(annotated, raw_log, stderr, cycles, metrics_json)``.

        ``annotated`` is the same as ``raw_log`` for RISC-V — spike
        output isn't post-annotated the way NCU logs are. ``metrics_json``
        carries per-op cycles + wall mtime ticks under keys the state-
        analysis prompt can consume verbatim.
        """
        timer = NamedTimer()

        # Driver = fb_config.test_code_fp (the KernelBench driver.c).
        driver_fp = self.test_code_fp

        # Route through the batch client when enabled — concurrent
        # rollouts naturally arrive here nearly-simultaneously, and the
        # coordinator coalesces them into one ``/gpu/batch`` HTTP call.
        # When batching is off (or the client can't be created), fall
        # back to the single-item path with identical semantics.
        if self._use_exec_batching:
            if self._batch_client is None:
                self._batch_client = await get_exec_batch_client(self.gpu)
            stdout_list, stderr_list, _elf_path, _success = await compile_and_run_riscv_batched(
                driver_fp,
                filepath,
                self.gpu,
                timer,
                self.agent_logger,
                self._batch_client,
                timeout=self.spike_timeout_s,
                num_runs=1,
                io_npz_path=self.io_npz_path,
                spike_args_str=self.spike_args_str,
                passed_keyword=None,
            )
        else:
            stdout_list, stderr_list, _elf_path, _success = await compile_and_run_riscv(
                driver_fp,
                filepath,
                self.gpu,
                timer,
                self.agent_logger,
                timeout=self.spike_timeout_s,
                num_runs=1,
                io_npz_path=self.io_npz_path,
                spike_args_str=self.spike_args_str,
                passed_keyword=None,
            )
        stdout = "\n".join(stdout_list) if isinstance(stdout_list, list) else str(stdout_list)
        stderr = "\n".join(stderr_list) if isinstance(stderr_list, list) else str(stderr_list)

        profile = self.backend.parse_profile(stdout)
        cycles = int(self.backend.extract_primary_metric(profile))

        metrics_json = {
            "total_cycles": profile.raw_metrics.get("total_cycles", cycles),
            "wall_mtime_ticks": profile.raw_metrics.get("wall_mtime_ticks", 0),
            "per_op_cycles": dict(profile.per_kernel_ms),
        }
        return stdout, stdout, stderr, cycles, metrics_json

    async def gather_perf_metrics_cached(
        self, filepath: Path
    ) -> Tuple[str, str, str, int, Dict[str, Any]]:
        """SHA-1-keyed wrapper. On a hit, we skip the compile + spike
        run entirely — same behavior as CUDA's cache but keyed on the
        RISC-V ``.c`` source instead of the ``.cu``."""
        try:
            code = filepath.read_text()
        except Exception:
            code = ""

        cached = self.profile_cache.get(code) if code else None
        if cached is not None:
            cycles = int(cached.primary_metric)
            raw_log = cached.profile.raw_log
            metrics_json = cached.profile.raw_metrics.get("metrics_json", {})
            self.agent_logger.info(
                f"Spike cache hit for {filepath.name} (cycles={cycles})"
            )
            return raw_log, raw_log, cached.stderr, cycles, metrics_json

        annotated, raw_log, stderr, cycles, metrics_json = (
            await self._gather_perf_metrics(filepath)
        )
        if code:
            profile = ProfileResult(
                total_time_ms=float(cycles),
                per_kernel_ms=metrics_json.get("per_op_cycles", {}),
                raw_metrics={
                    "elapsed_cycles": int(cycles),
                    "metrics_json": metrics_json,
                    "wall_mtime_ticks": metrics_json.get("wall_mtime_ticks", 0),
                },
                raw_log=raw_log,
            )
            self.profile_cache.put(
                code,
                ProfileCacheEntry(
                    primary_metric=float(cycles),
                    profile=profile,
                    stderr=stderr or "",
                ),
            )
        return annotated, raw_log, stderr, cycles, metrics_json

    # ------------------------------------------------------------------
    # Outer entry points (initialize + run) — analogues of CUDA's
    # ``initialize`` / ``run`` with ``.c`` filenames + spike-log baseline.
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Establish the RISC-V baseline. Fix-loop pattern lifted from
        the CUDA agent — deterministic pre-pass then bounded LLM fix.
        """
        self.code_to_optimize_fp = self.folder / f"init{self.backend.kernel_ext}"
        self.code_to_optimize_fp.write_text(self.code_to_optimize)
        self.agent_logger.info("Gathering initial spike profile…")

        current_code = self.code_to_optimize
        last_error: Optional[str] = None
        for attempt in range(self.max_fix_attempts + 1):
            attempt_fp = (
                self.code_to_optimize_fp
                if attempt == 0
                else self.folder / f"init_fix_attempt{attempt}{self.backend.kernel_ext}"
            )
            attempt_fp.write_text(current_code)
            try:
                _, raw_log, _, cycles, metrics = (
                    await self.gather_perf_metrics_cached(attempt_fp)
                )
            except FeedbackError as e:
                err = str(e)
                last_error = err

                # Deterministic pre-pass (gcc-flavored). Cheap; runs
                # before we burn an LLM call.
                patched = self.backend.deterministic_fix(current_code, err)
                if patched is not None and patched != current_code:
                    self.agent_logger.info(
                        f"Deterministic gcc-fix repaired init (attempt {attempt})"
                    )
                    current_code = patched
                    continue

                if attempt >= self.max_fix_attempts:
                    self.agent_logger.error(
                        f"Initial profiling failed after {self.max_fix_attempts} fix attempts "
                        f"(init.c compile/run broken): {err}; "
                        f"continuing without baseline"
                    )
                    self.last_ncu_log = ""
                    return

                self.agent_logger.warning(
                    f"Initial profiling failed (attempt {attempt}); requesting LLM fix: {err}"
                )
                fix_messages = self._build_riscv_fix_messages(
                    broken_kernel=current_code, compiler_error=err,
                )
                try:
                    fix_text = await self._llm_fix(fix_messages)
                except Exception as fix_err:
                    self.agent_logger.error(
                        f"LLM fix call failed during init repair: {fix_err}; "
                        f"continuing without baseline"
                    )
                    self.last_ncu_log = ""
                    return
                fixed_code = self.backend.extract_code_from_response(fix_text)
                if not fixed_code or fixed_code.strip() == current_code.strip():
                    self.agent_logger.error(
                        f"LLM fix produced no actionable change at init attempt {attempt}; "
                        f"continuing without baseline"
                    )
                    self.last_ncu_log = ""
                    return
                current_code = fixed_code
                continue

            # Success path.
            if current_code != self.code_to_optimize:
                self.agent_logger.info(
                    f"init{self.backend.kernel_ext} repaired after {attempt} fix attempt(s); "
                    f"persisting repaired source as the baseline."
                )
                self.code_to_optimize = current_code
                self.code_to_optimize_fp.write_text(current_code)
                self.seed_buffer.set_init_code(current_code)
            self.initial_cycles = cycles
            self.best_cycles = float(cycles)
            self.last_ncu_log = raw_log
            (self.folder / f"0_init_annotated{self.backend.kernel_ext}").write_text(
                self.code_to_optimize
            )
            # Baseline marker (analogue of CUDA's ncu/0_init_ncu_log.txt).
            try:
                (self.folder / "init.profile.json").write_text(
                    json.dumps({"total_cycles": int(cycles), **metrics}, indent=2)
                )
            except Exception as e:
                self.agent_logger.debug(f"Could not write baseline marker: {e}")
            self.seed_buffer.update(float(cycles), self.code_to_optimize)
            self.agent_logger.info(f"Initial cycles={cycles}, metrics={metrics}")
            return

        self.agent_logger.error(
            f"Initial profiling exhausted fix attempts (last error: {last_error}); "
        )
        self.last_ncu_log = ""

    async def run(self) -> Path:
        """Spawn ``num_rl_iterations`` concurrent rollouts, keep the
        running best. Returns the path to the best-so-far kernel.
        """
        if self.initial_cycles is None:
            no_baseline = self.folder / f"no_baseline_rl_optimization{self.backend.kernel_ext}"
            no_baseline.write_text(self.code_to_optimize)
            self.agent_logger.warning(
                "No RISC-V baseline established; RL loop skipped."
            )
            return no_baseline

        try:
            initial_state_profile = await self.database.analyze_performance_state(
                self.last_ncu_log,
                self.backend.parse_state_metrics(self.last_ncu_log, self.initial_cycles),
                self.code_to_optimize,
                elapsed_cycles=self.backend.state_cycles_from_metric(self.initial_cycles),
            )
            initial_state = await self.database.match_state_against_database(
                initial_state_profile
            )
        except Exception as e:
            self.agent_logger.warning(f"Initial state derivation failed: {e}")
            initial_state = "hybrid_bound"

        async def _one_rollout(idx: int) -> Optional[Trajectory]:
            try:
                seed_code, seed_metric = self.seed_buffer.pick(idx)
                if seed_metric is not None:
                    self.agent_logger.info(
                        f"Rollout {idx} seeded from prior best ({int(seed_metric)} cycles)"
                    )
                return await self._run_rollout(idx, seed_code, initial_state)
            except Exception as e:
                self.agent_logger.error(f"Rollout {idx} failed: {e}")
                return None

        tasks = [
            asyncio.create_task(_one_rollout(i))
            for i in range(self.num_rl_iterations)
        ]

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
                best_filename = self.folder / self.backend.best_filename()
                best_filename.write_text(
                    self.backend.format_result_artifact(
                        best_step.code, int(best_step.cycles)
                    )
                )
                self.agent_logger.info(
                    f"New best: {int(best_cycles)} cycles (action={best_step.action})"
                )
            self.seed_buffer.update(float(best_step.cycles), best_step.code)

        try:
            self.database._persist_database()
        except Exception as e:
            self.agent_logger.warning(f"Database persist failed: {e}")

        if best_filename is not None:
            self.best_cycles = best_cycles
            return best_filename

        failure = self.folder / f"failure_rl_optimization{self.backend.kernel_ext}"
        failure.write_text(
            self.backend.format_result_artifact(
                self.code_to_optimize, int(self.initial_cycles or 0)
            )
        )
        self.agent_logger.warning("RL produced no improvement.")
        return failure

    # ------------------------------------------------------------------
    # Rollout — backend-agnostic body via ``self.backend.parse_state_metrics``
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
        last_raw_log = self.last_ncu_log
        running_best_cycles = current_cycles if current_cycles else float("inf")
        steps_since_improvement = 0

        for step in range(self.max_rollout_steps):
            try:
                profile = await self.database.analyze_performance_state(
                    last_raw_log,
                    self.backend.parse_state_metrics(last_raw_log, current_cycles),
                    current_code,
                    elapsed_cycles=self.backend.state_cycles_from_metric(current_cycles),
                )
                analysis_json = json.dumps(asdict(profile), indent=2)
                plan = await self.database.generate_optimization_plan(
                    analysis_json,
                    current_code,
                    top_n=max(4, self.max_rollout_steps - step),
                )
            except Exception as e:
                self.agent_logger.warning(
                    f"Plan failed for traj {traj_idx} step {step}: {e}"
                )
                plan = []

            if not plan:
                self.agent_logger.info(
                    f"Empty plan; stopping rollout {traj_idx} at step {step}"
                )
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

            chosen_name = self.bandit.select(
                current_state,
                candidate_names,
                weights=candidate_weights,
                traj_idx=traj_idx,
            )
            chosen_plan = next(
                (p for p in plan if p.get("technique") == chosen_name), plan[0]
            )
            strategy_description = chosen_plan.get("description", "")

            self.agent_logger.info(
                f"Traj {traj_idx} step {step}: bandit chose '{chosen_name}' "
                f"(state={current_state}; plan_relevance={chosen_plan.get('relevance_score')})"
            )

            try:
                (
                    optimized_code,
                    new_cycles,
                    new_raw_log,
                    new_metrics,
                ) = await self._apply_optimization(
                    current_code,
                    chosen_name,
                    strategy_description,
                    current_cycles,
                    traj_dir,
                    step,
                )
            except Exception as e:
                self.agent_logger.error(
                    f"Apply failed traj {traj_idx} step {step}: {e}"
                )
                break

            if current_cycles and current_cycles > 0:
                actual_improvement = (
                    (current_cycles - new_cycles) / current_cycles
                ) * 100
            else:
                actual_improvement = 0.0
            reward = actual_improvement / 100.0
            self.bandit.update(current_state, chosen_name, reward)

            self.database.update_optimization_result(
                current_state,
                chosen_name,
                actual_improvement,
                current_metric=float(new_cycles),
                baseline_metric=(
                    float(self.initial_cycles)
                    if self.initial_cycles is not None
                    else None
                ),
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
                    pass

            # Re-derive state for next step.
            try:
                next_profile = await self.database.analyze_performance_state(
                    new_raw_log,
                    self.backend.parse_state_metrics(new_raw_log, new_cycles),
                    optimized_code,
                    elapsed_cycles=self.backend.state_cycles_from_metric(new_cycles),
                )
                current_state = await self.database.match_state_against_database(
                    next_profile
                )
            except Exception:
                pass

            current_code = optimized_code
            current_cycles = new_cycles
            last_raw_log = new_raw_log

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
    # RISC-V-flavored prompt builders (P5.15)
    # ------------------------------------------------------------------
    #
    # These replace the CUDA-tuned :meth:`GPUOptimizationDatabase.build_codegen_messages`
    # / :meth:`build_fix_messages` when the backend is RISC-V. The system
    # block is the cache-stable one — the same across every call in a
    # run — so a prefix-cache-aware LLM only charges input price for
    # the delta in the user block.

    _RISCV_SYSTEM_PROMPT_TMPL = (
        "You are an expert RISC-V + Zephyr kernel optimisation engineer. "
        "You receive an optimisation technique to apply, a C kernel "
        "source, and a JSON of RISC-V profile metrics. Apply the "
        "technique to the kernel and emit the COMPLETE rewritten C "
        "file in a single ```c code block.\n\n"
        "Hard rules for the RISC-V + Zephyr target:\n"
        "- Output ONLY the rewritten C file in one ```c fenced block. "
        "No prose before or after.\n"
        "- The kernel compiles with gcc against Zephyr's minimal libc + "
        "picolibc. Do NOT rely on <math.h> transcendentals (sinf, "
        "expf, tanhf, sqrtf) unless you range-reduce or approximate "
        "yourself — no libm linkage is guaranteed. Basic <stdint.h>, "
        "<stddef.h>, <string.h> (for memcpy/memset) are safe.\n"
        "- No <malloc.h>, no dynamic allocation — Zephyr's default has "
        "no heap.\n"
        "- Preserve every function's declared signature exactly. Do not "
        "rename the entry-point kernel functions. modelblaster's "
        "generated dispatcher calls them by name.\n"
        "- The target is a RISC-V in-order scalar core (Rocket) or "
        "dual-issue with the V extension (Saturn, rv64gcv_zicntr). "
        "Assume 16 KB L1D, 16 KB L1I, VLEN=128 bits when RVV is "
        "available. Two-cycle load-use bubble; no OoO scheduling.\n"
        "- Available techniques (this run's catalog):\n"
        "  {technique_names}\n\n"
        "Style preferences:\n"
        "- Use `static inline` for helpers so they don't multiply-define "
        "at link.\n"
        "- Prefer `__builtin_expect` and predicated arithmetic over "
        "conditional branches in hot loops (Rocket has static predict-"
        "not-taken).\n"
        "- Loop-unrolling depth 4-8 is usually right for hiding the "
        "load-use bubble.\n"
        "- If the technique cannot be cleanly applied, return the input "
        "unchanged inside the same ```c block.\n\n"
        "RVV idiom cheatsheet (use these — they are ONE vector op each, "
        "not the mask+merge equivalents):\n"
        "- `vfmax_vf_f32m8(v, 0.0f, vl)` for ReLU / max-with-scalar. "
        "Do NOT emit `vmfgt_vf_*` + `vmerge_vvm_*` for this — that's "
        "three vector ops where one suffices.\n"
        "- `vfmin_vf_f32m8(v, cap, vl)` for clamp-max / ReLU6 upper "
        "bound; chain with vfmax_vf for two-sided clamp.\n"
        "- `vfabs_v_f32m8(v, vl)` for absolute value.\n"
        "- `vfsgnj_vv` / `vfsgnjn_vv` / `vfsgnjx_vv` for sign "
        "manipulation.\n"
        "- `vfmacc_vv_f32m8(acc, a, b, vl)` (a*b + acc) for reductions / "
        "dot products.\n"
        "- `vfredosum_vs` / `vfredmax_vs` for in-vector reductions to a "
        "scalar (one instruction, LMUL-wide).\n"
        "- Use LMUL=m8 for pure-elementwise ops (widens the memory "
        "transaction). Drop to LMUL=m2 or m4 when register pressure "
        "matters (reductions, register tiling).\n\n"
        "Memory-vs-compute bottleneck triage (READ THIS BEFORE PICKING "
        "AN OPTIMISATION SHAPE):\n"
        "- Compute intensity ~= FLOPs / (input_bytes + output_bytes). "
        "For pure elementwise on fp32 (relu, add, mul, sigmoid): "
        "~0.25 FLOP/byte. That is DEEPLY memory-bandwidth bound on "
        "this SoC (~1-2 GB/s effective DDR to a Rocket + Saturn). "
        "Compute optimisations (unroll, pipeline, more vec lanes) "
        "CANNOT help — the vector unit is already idle waiting for "
        "DDR.\n"
        "- When you see per_op_cycles roughly matching "
        "(elements * 20-40) at LMUL=8, you are in the memory-bound "
        "regime. Reach for: prefetch (`__builtin_prefetch(&x[i+64], "
        "0, 0)`), streaming stores (avoid cache pollution when there "
        "is no output reuse), or restructuring to fuse producers/"
        "consumers so intermediate memory traffic is eliminated.\n"
        "- Compute-bound signal: matmul, conv2d, softmax, layernorm "
        "with weight reuse. Here vectorise + tile + pipeline give "
        "3-5x. Elementwise ops rarely go past 1.2x on this SoC — "
        "consider fusion into the surrounding op instead of "
        "optimising the elementwise in isolation.\n"
    )

    def _build_riscv_codegen_messages(
        self,
        *,
        technique_name: str,
        kernel_source: str,
        metrics_json: Dict[str, Any],
        strategy_description: str = "",
        best_so_far_summary: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        technique_names = ", ".join(sorted(self.backend.technique_map.keys()))
        system = self._RISCV_SYSTEM_PROMPT_TMPL.format(
            technique_names=technique_names or "(catalog empty — LLM discretion)"
        )

        # Technique description — prefer the backend's technique_map
        # entry (rich prose). Fall back to whatever the DB has if the
        # LLM discovered a novel state.
        tech_desc = self.backend.technique_map.get(technique_name)
        if tech_desc is None:
            try:
                tech_desc = self.database.get_technique_description(technique_name)
            except Exception:
                tech_desc = technique_name

        user_blocks: list[str] = []
        user_blocks.append(f"SELECTED TECHNIQUE: {technique_name}\n{tech_desc}")
        if strategy_description:
            user_blocks.append(f"STRATEGY NOTE FROM PLAN:\n{strategy_description}")
        if best_so_far_summary:
            user_blocks.append(f"PRIOR-STEP CONTEXT:\n{best_so_far_summary}")
        # Metrics JSON — RISC-V shape (total_cycles, wall_mtime_ticks,
        # per_op_cycles) rather than CUDA's SoL fields.
        user_blocks.append(
            "RISC-V PROFILE METRICS (JSON):\n" + json.dumps(metrics_json, sort_keys=True, indent=2)
        )
        user_blocks.append(
            "CURRENT KERNEL SOURCE:\n```c\n" + kernel_source + "\n```"
        )
        user_blocks.append(
            "Apply the SELECTED TECHNIQUE to the source. Return only the "
            "rewritten C file in a single ```c code block. Keep the "
            "existing function signatures intact."
        )

        return [
            {"role": "system", "content": "<!-- cache_control: ephemeral -->\n" + system},
            {"role": "user", "content": "\n\n".join(user_blocks)},
        ]

    _RISCV_FIX_SYSTEM_PROMPT = (
        "You are a RISC-V + Zephyr compile-error fixer. Given a C "
        "kernel that failed gcc compilation or a runtime check, return "
        "the corrected COMPLETE kernel inside a single ```c code block. "
        "Preserve the entry-point function signature. Do not change the "
        "intent — only fix the error. Avoid <math.h> transcendentals "
        "and dynamic allocation."
    )

    def _build_riscv_fix_messages(
        self, *, broken_kernel: str, compiler_error: str,
    ) -> List[Dict[str, str]]:
        err = compiler_error[:2000]
        kernel = broken_kernel
        if len(kernel) > 12000:
            kernel = kernel[:6000] + "\n// ... [trimmed] ...\n" + kernel[-6000:]
        user_msg = (
            "COMPILER / RUNTIME ERROR:\n```\n"
            f"{err}\n```\n\n"
            "BROKEN KERNEL SOURCE:\n```c\n"
            f"{kernel}\n```\n\n"
            "Return the fixed complete kernel in a single ```c block."
        )
        return [
            {"role": "system", "content": self._RISCV_FIX_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    # ------------------------------------------------------------------
    # Apply — codegen + compile + spike + fix loop
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
        # Build cache-stable messages via the database — same prompt
        # shape both CUDA and RISC-V paths use.
        cached_entry = self.profile_cache.get(code)
        current_metrics_json = (
            cached_entry.profile.raw_metrics.get("metrics_json", {})
            if cached_entry is not None else {}
        )
        if current_cycles and "total_cycles" not in current_metrics_json:
            current_metrics_json = dict(current_metrics_json)
            current_metrics_json["total_cycles"] = int(current_cycles)

        snapshot = self.seed_buffer.snapshot()
        if snapshot and current_cycles is not None:
            best = int(min(m for m, _ in snapshot))
            best_so_far = (
                f"Running best across all rollouts: {best} cycles "
                f"(this trajectory's current: {current_cycles} cycles)."
            )
        else:
            best_so_far = None

        # P5.15: RISC-V-specific codegen prompt. The base DB's builder is
        # CUDA-vocab (mentions warps, cuda_fp16.h, cudaMemcpy, ```cpp
        # fence) which is actively misleading here. This override keeps
        # the cache-stable system-prompt shape (backend/OS/hard-rules
        # block is stable across a run; only the technique + kernel
        # rotate in the user block).
        messages = self._build_riscv_codegen_messages(
            technique_name=technique_name,
            kernel_source=code,
            metrics_json=current_metrics_json,
            strategy_description=strategy_description,
            best_so_far_summary=best_so_far,
        )

        # Tiered codegen call (routes via backend.categorise_technique).
        text = await self._llm_codegen(messages, technique_name=technique_name)
        candidate = self.backend.extract_code_from_response(text)
        if not candidate:
            raise FeedbackError(
                f"No code block in LLM response for technique {technique_name!r}"
            )

        # Persist per-step artifact.
        step_fp = traj_dir / self.backend.step_filename(0, step, technique_name)
        step_fp.write_text(candidate)

        current_code = candidate
        last_error: Optional[str] = None

        for attempt in range(self.max_fix_attempts + 1):
            attempt_fp = (
                step_fp
                if attempt == 0
                else traj_dir / f"step_{step}_{technique_name}_fix{attempt}{self.backend.kernel_ext}"
            )
            attempt_fp.write_text(current_code)
            try:
                _, raw_log, _, cycles, metrics = (
                    await self.gather_perf_metrics_cached(attempt_fp)
                )
                return current_code, cycles, raw_log, metrics
            except FeedbackError as e:
                err = str(e)
                last_error = err

                patched = self.backend.deterministic_fix(current_code, err)
                if patched is not None and patched != current_code:
                    self.agent_logger.info(
                        f"Deterministic gcc-fix applied at step {step} attempt {attempt}"
                    )
                    current_code = patched
                    continue

                if attempt >= self.max_fix_attempts:
                    break

                fix_messages = self._build_riscv_fix_messages(
                    broken_kernel=current_code, compiler_error=err,
                )
                try:
                    fix_text = await self._llm_fix(fix_messages)
                except Exception as fix_err:
                    self.agent_logger.error(
                        f"LLM fix call failed at step {step} attempt {attempt}: {fix_err}"
                    )
                    break
                fixed = self.backend.extract_code_from_response(fix_text)
                if not fixed or fixed.strip() == current_code.strip():
                    break
                current_code = fixed

        raise FeedbackError(
            f"Step {step} exhausted fix attempts (last error: {last_error})"
        )
