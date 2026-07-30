# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Shared base class for the per-backend RL optimization agents (Phase 4f).

After Phases 4a–4e established the Backend abstraction (technique_map,
parse_profile, extract_code_from_response, format_result_artifact,
get_default_optimizations, metric_name / format_metric / extract_primary_metric,
RLNodeConfig), the two RL agents share enough method bodies to extract
a base class. This module owns the genuinely-shared methods; per-backend
subclasses (``RLNCUAgent``, ``RLOpenCLAgent``) override what's truly
hardware-specific:

  - NCU-flavoured profile gathering / Speed-Of-Light section extraction (CUDA)
  - OpenCL event-time profile gathering / on-board reference generation (OpenCL)
  - Verification-pool global-best tracking (OpenCL-only)
  - Database policy-update lifecycle hook (CUDA-only)

Methods that live here:

  - ``calculate_reward`` — byte-identical between agents.
  - ``_lookup_optim_entry_by_name`` — byte-identical lookup.
  - ``_try_add_default_optimizations`` — uses ``backend.get_default_optimizations``.
  - ``get_performance_summary`` — backend-named JSON keys produced by the
    ``_perf_summary_extras`` hook so existing JSON consumers see the same
    schema as before this refactor.

Phase 4f-future work that can lift more methods to this base once the
per-backend kernel-source attribute names (``code_to_optimize_fp`` /
``kernel_to_optimize_fp``) are unified onto a single canonical attribute:
``__init__`` skeleton, ``initialize``, ``run``, ``run_rollout``,
``apply_optimization``, ``get_feedback`` skeleton.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .feedback import FeedbackAgent, FeedbackConfig
from .database import (
    OptimizationDatabase,
    OptimizationEntry,
    CompositeOptimization,
    LLMInterface,
)
from .rl_agents import ReplayBuffer
from ..backends import ProfileResult


class RLAgentBase(FeedbackAgent):
    """Shared scaffolding for ``RLNCUAgent`` and ``RLOpenCLAgent``.

    The shared ``__init__`` sets up everything common to both backends:
    the Backend object (``self.backend`` via ``self.gpu.backend()``),
    test-code references, kernel-source references under canonical names
    (``self.kernel_source_fp`` / ``self.kernel_source``), the
    ``OptimizationDatabase``, ``ReplayBuffer``, RL sub-agents, state
    counters, and concurrency locks. Subclasses override
    ``_init_backend_extras`` to add backend-specific attributes (e.g.
    OpenCL's global-best verification-pool fields and SSH-exec timeout).

    Subclasses MUST pass their per-backend kernel-source path under the
    canonical name ``kernel_source_fp`` to ``super().__init__``; they may
    keep their original public parameter name (``code_to_optimize_fp`` /
    ``kernel_to_optimize_fp``) in their own ``__init__`` signature for
    back-compat with existing callers.
    """

    def __init__(
        self,
        fb_config: FeedbackConfig,
        kernel_source_fp: Path,
        database_path: Path,
        max_rollout_steps: int = 5,
        replay_buffer_size: int = 1000,
        update_frequency: int = 10,
        database: Optional[OptimizationDatabase] = None,
    ):
        super().__init__(fb_config)

        # Phase 2 Backend abstraction: single source of truth for technique
        # map, file-naming, profile parsing, board_host (OpenCL), etc.
        # Routed via gpu.backend() so this picks the right backend for the
        # GPU type carried in fb_config.
        self.backend = self.gpu.backend()

        # Test driver + kernel source (canonical names — Phase 4f rename).
        self.test_code_fp = fb_config.test_code_fp
        self.test_code = fb_config.test_code_fp.read_text()
        self.kernel_source_fp = kernel_source_fp
        self.kernel_source = kernel_source_fp.read_text()

        # Database: use the shared instance if one was passed in (graph
        # nodes do this when running multiple problems in one process);
        # otherwise construct a fresh one rooted at ``database_path``.
        gpu_report_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "algo-sol-modeling/algo-space/gpu_optimization_report.md"
        )
        llm_interface = LLMInterface(self.model, self.agent_logger)
        if database is not None:
            self.database = database
        else:
            self.database = OptimizationDatabase(
                database_path, gpu_report_path, llm_interface, backend=self.backend
            )

        # RL components.
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
        self.max_rollout_steps = max_rollout_steps
        self.update_frequency = update_frequency

        # State tracking — unified naming since Phase 4d.
        self.iteration_count = 0
        self.total_trajectories = 0
        self.best_metric = float("inf")
        self.initial_metric = None

        # Concurrency.
        self._trajectory_lock: asyncio.Lock = asyncio.Lock()
        self.current_trajectory = None

        # Default RL iteration count (can be overridden by the workflow).
        self.num_rl_iterations = 50

        # Backend-specific extras (OpenCL global-best fields, etc.) go in
        # the subclass hook so they always run AFTER all shared state is set.
        self._init_backend_extras()

    def _init_backend_extras(self) -> None:
        """Subclass hook for backend-specific instance attributes.

        Default: no-op (CUDA agent has no extras beyond what the base sets).
        OpenCL agent overrides to set up SSH timeout + global-best fields.
        """
        return None

    # ------------------------------------------------------------------
    # Shared state-derivation glue (Phase 4f.3a)
    # ------------------------------------------------------------------
    async def gather_profile_result(self, kernel_fp):
        """Backend-agnostic wrapper that returns a ProfileResult.

        Subclasses override to convert their existing ``gather_perf_metrics``
        tuple return into a ``ProfileResult``. Shared methods in this base
        call ``gather_profile_result`` rather than the per-backend
        ``gather_perf_metrics`` so they don't have to know the tuple shape.
        """
        raise NotImplementedError(
            "Subclasses must implement gather_profile_result"
        )

    def _write_profile_json(self, kernel_filepath: Path, pr: ProfileResult) -> None:
        """Best-effort persist a ``ProfileResult`` next to the kernel file.

        Phase 3c: downstream tooling (analytics, future speedup-tracker)
        should read JSON instead of regex-parsing driver stdout. The file
        is written to ``<kernel_filepath>.with_suffix('.profile.json')``.

        Never raises — a filesystem hiccup or JSON-encode issue should not
        break the RL loop. Failures are logged at WARNING.
        """
        try:
            profile_json_path = Path(kernel_filepath).with_suffix(".profile.json")
            pr.write_json(profile_json_path)
        except Exception as e:
            self.agent_logger.warning(
                f"Failed to write profile.json for {kernel_filepath.name}: {e}"
            )

    async def _derive_state(self, profile_result, code: str) -> str:
        """Compute the optimisation-state string for the current kernel.

        Backend-agnostic — the per-backend bits (metrics extraction, cycles
        arg conversion) come from ``backend.derive_metrics_for_state`` and
        ``backend.state_cycles_arg``. On any database exception, returns a
        ``"<backend>_unknown"`` sentinel so callers can proceed.
        """
        metrics = self.backend.derive_metrics_for_state(profile_result)
        cycles_arg = self.backend.state_cycles_arg(profile_result)
        try:
            return await self.database.get_state_from_ncu_report(
                profile_result.raw_log, metrics, code, elapsed_cycles=cycles_arg
            )
        except Exception:
            return f"{self.backend.name}_unknown"

    # ------------------------------------------------------------------
    # initialize() lift (Phase 4f.3c)
    # ------------------------------------------------------------------
    async def _maybe_generate_reference(self) -> None:
        """Subclass hook: backend-specific pre-profile setup.

        CUDA: no-op (default). OpenCL: SSH-execs ``--generate-reference``
        on the board to cache ``reference_output.bin``.
        """
        return None

    def _write_init_artifact(self, profile_result) -> None:
        """Subclass hook: write the per-backend ``0_init_*`` artifact.

        CUDA writes the NCU-annotated source; OpenCL writes the raw kernel
        source (the [PROFILE] markers are already in raw_log, no annotation
        needed for prompts).
        """
        raise NotImplementedError(
            "Subclasses must implement _write_init_artifact"
        )

    def _handle_init_failure(self) -> None:
        """Subclass hook: optional cleanup/fallback when initial profiling
        raised ``FeedbackError``. Default: no-op. CUDA overrides to populate
        a fallback state via the database and write a placeholder annotated
        artifact so downstream steps have something to read.
        """
        return None

    async def initialize(self):
        """Gather initial profiling data for the unoptimised kernel.

        Backend-agnostic shape (Phase 4f.3c) — the per-backend bits go
        through three hooks: ``_maybe_generate_reference`` (pre-profile
        setup), ``_write_init_artifact`` (post-profile artifact), and
        ``_handle_init_failure`` (fallback when the first profile pass
        raises). Profile extraction goes through ``gather_profile_result``
        (Phase 4f.3b) so the return shape is ``ProfileResult`` regardless
        of backend.
        """
        from .utils import FeedbackError

        # Anchor the kernel source under the per-backend init filename.
        self.kernel_source_fp = self.folder / f"init{self.backend.kernel_ext}"
        self.kernel_source_fp.write_text(self.kernel_source)

        # OpenCL-specific: pre-cache CPU reference output on the board.
        # CUDA's libtorch driver computes its reference in-process — no-op.
        await self._maybe_generate_reference()

        self.agent_logger.info(
            f"Gathering initial {self.backend.name.upper()} profiling data..."
        )
        try:
            pr = await self.gather_profile_result(self.kernel_source_fp)
            metric = self.backend.extract_primary_metric(pr)
            self.initial_metric = metric
            self.best_metric = metric
            self.last_profile_log = pr.raw_log

            initial_state = await self._derive_state(pr, self.kernel_source)
            self.agent_logger.info(
                f"Initial state: {initial_state}, "
                f"{self.backend.format_metric(metric)}"
            )

            self._write_init_artifact(pr)
        except FeedbackError as e:
            self.agent_logger.warning(
                f"Initial profiling failed; proceeding with fallback state. Details: {e}"
            )
            self.last_profile_log = ""
            self._handle_init_failure()

    # ------------------------------------------------------------------
    # apply_optimization() lift (Phase 4f.3d)
    # ------------------------------------------------------------------
    def _load_database_content(self) -> str:
        """Resolve the database content payload used in strategy prompts.

        Byte-identical to the historical CUDA fallback chain (which was the
        more verbose of the two agents): ``get_database_md_text`` → footer →
        ``gpu_optimization_knowledge[:6000]``. Emitted as a helper so both
        the lifted ``apply_optimization`` and its subclass overrides (should
        any survive) share the exact same policy.
        """
        try:
            database_content = self.database.get_database_md_text()
            if not database_content or database_content.strip() == "":
                self.agent_logger.warning("Database markdown is empty, trying footer")
                database_content = self.database.get_database_footer_text()
                if not database_content or database_content.strip() == "":
                    self.agent_logger.warning(
                        "Database footer is also empty, using GPU optimization knowledge"
                    )
                    database_content = (
                        getattr(self.database, "gpu_optimization_knowledge", "")[:6000] or ""
                    )
        except Exception as e:
            self.agent_logger.warning(f"Failed to load database content: {e}")
            try:
                database_content = self.database.get_database_footer_text()
            except Exception:
                database_content = (
                    getattr(self.database, "gpu_optimization_knowledge", "")[:6000] or ""
                )
        if database_content:
            self.agent_logger.debug(
                f"Using database content: {len(database_content)} characters"
            )
        else:
            self.agent_logger.warning("No database content available for prompt")
        return database_content

    def _load_database_footer(self) -> str:
        """Read the optimization-database footer for fix-prompt scaffolding.

        Silent on all errors — the fix prompt is best-effort and each
        backend's ``build_fix_prompt`` handles an empty footer gracefully
        (OpenCL ignores it entirely; CUDA omits the reference-snippets block).
        """
        try:
            footer_path = getattr(self.database, "optimization_db_footer_path", None)
            if footer_path is not None and footer_path.exists():
                return footer_path.read_text(encoding="utf-8")
        except Exception:
            pass
        return ""

    @staticmethod
    def _append_agentic_log(
        trajectory_dir: Optional[Path], label: str, prompt_text: str, response_text: str
    ) -> None:
        if trajectory_dir is None:
            return
        log_fp = trajectory_dir / "agentic_steps_log.txt"
        with open(log_fp, "a", encoding="utf-8") as f:
            f.write(
                f"=== {label} ===\n"
                f"--- PROMPT ---\n{prompt_text.rstrip()}\n"
                f"--- RESPONSE ---\n{response_text.rstrip()}\n\n"
            )

    async def apply_optimization(
        self,
        code: str,
        optimization_entry: OptimizationEntry | CompositeOptimization,
        step: int,
        trajectory_dir: Optional[Path] = None,
        strategy_description: str = "",
    ) -> Tuple[str, float, None, str]:
        """Backend-agnostic per-step optimisation loop.

        Returns ``(optimized_code, primary_metric, new_state, new_raw_log)``.
        ``primary_metric`` is a ``float`` — CUDA callers cast to ``int`` when
        assigning to ``TrajectoryStep.cycles`` (which the type hint still
        pins to ``int``). ``new_state`` is always ``None`` here; state
        recomputation is done by the trajectory loop (or dead in the
        current codepath).

        Per-backend bits go through ``backend.build_strategy_prompt`` /
        ``backend.build_fix_prompt`` (Phase 4f.3d.a) and through
        ``gather_profile_result`` (Phase 4f.3b) so this body doesn't
        reference NCU or OpenCL profiling directly.
        """
        from .utils import FeedbackError, generate_code_retry

        ext = self.backend.kernel_ext
        if isinstance(optimization_entry, CompositeOptimization):
            technique_name = optimization_entry.get_composite_id()
        else:
            technique_name = getattr(optimization_entry, "technique", str(optimization_entry))
        base_label = f"step_{step}_{technique_name}"
        base_dir = trajectory_dir if trajectory_dir is not None else self.folder

        # First profile pass of the incoming code — tolerate verification /
        # profiling failures. An empty ProfileResult keeps the prompt shape
        # stable (backends' build_strategy_prompt handles empty raw_log).
        temp_file = base_dir / f"{base_label}{ext}"
        temp_file.write_text(code)
        try:
            current_pr = await self.gather_profile_result(temp_file)
        except FeedbackError as prof_err:
            self.agent_logger.warning(
                f"Profiling failed at step {step} with FeedbackError; "
                f"using empty profile. Details: {prof_err}"
            )
            current_pr = ProfileResult(
                total_time_ms=0.0, per_kernel_ms={}, raw_metrics={}, raw_log=""
            )
        except Exception as prof_other:
            self.agent_logger.warning(
                f"Unexpected profiling error at step {step}: {prof_other}; "
                f"continuing with empty profile."
            )
            current_pr = ProfileResult(
                total_time_ms=0.0, per_kernel_ms={}, raw_metrics={}, raw_log=""
            )

        database_content = self._load_database_content()
        prompt = self.backend.build_strategy_prompt(
            optimization_entry,
            code,
            current_pr,
            database_content,
            strategy_description or "",
        )

        response = await generate_code_retry(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            logger=self.agent_logger,
            max_retries=3,
        )
        self._append_agentic_log(
            trajectory_dir, f"{base_label}_initial", prompt, response.generations[0]
        )

        optimized_code, filepath = self.get_code_from_response(
            response.generations[0], step, 0, self.agent_logger
        )
        # Relocate the intermediate file into the trajectory folder so
        # concurrent trajectories don't collide on shared filenames in
        # ``self.folder`` (CUDA already did this; OpenCL didn't for the fix
        # branch — now unified).
        try:
            target_fp = base_dir / f"{base_label}_initial{ext}"
            if filepath != target_fp:
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
            pass

        MAX_FIX_ATTEMPTS = 4
        attempt_idx = 0
        new_metric: float = 0.0
        new_pr: Optional[ProfileResult] = None

        while attempt_idx < MAX_FIX_ATTEMPTS:
            filepath = base_dir / f"{base_label}_attempt{attempt_idx}{ext}"
            filepath.write_text(optimized_code)
            try:
                new_pr = await self.gather_profile_result(filepath)
                new_metric = self.backend.extract_primary_metric(new_pr)
                if trajectory_dir is not None:
                    log_fp = trajectory_dir / "agentic_steps_log.txt"
                    with open(log_fp, "a", encoding="utf-8") as f:
                        f.write(
                            "Compile success: True\n"
                            "Run success    : True\n"
                            f"{self.backend.metric_name}: "
                            f"{self.backend.format_metric(new_metric, with_unit=False)}\n\n"
                        )
                break
            except Exception as e:
                error_msg = str(e)
                if trajectory_dir is not None:
                    log_fp = trajectory_dir / "agentic_steps_log.txt"
                    with open(log_fp, "a", encoding="utf-8") as f:
                        f.write(
                            f"Compile/Run failed on attempt {attempt_idx}: {error_msg}\n\n"
                        )
                attempt_idx += 1
                if attempt_idx >= MAX_FIX_ATTEMPTS:
                    raise

                db_footer = self._load_database_footer()
                fix_prompt = self.backend.build_fix_prompt(
                    optimized_code, error_msg, db_footer
                )
                fix_response = await generate_code_retry(
                    messages=[{"role": "user", "content": fix_prompt}],
                    model=self.model,
                    logger=self.agent_logger,
                    max_retries=2,
                )
                self._append_agentic_log(
                    trajectory_dir,
                    f"{base_label}_fix_attempt_{attempt_idx}",
                    fix_prompt,
                    fix_response.generations[0],
                )
                optimized_code, fix_fp = self.get_code_from_response(
                    fix_response.generations[0], step, attempt_idx, self.agent_logger
                )
                # Relocate the fix intermediate into the trajectory dir.
                # CUDA did this; OpenCL used to leak these into self.folder —
                # unifying keeps per-trajectory artifacts self-contained.
                try:
                    fix_target_fp = base_dir / f"{base_label}_attempt{attempt_idx}_llm{ext}"
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

        new_raw_log = new_pr.raw_log if new_pr is not None else ""
        return optimized_code, new_metric, None, new_raw_log

    # ------------------------------------------------------------------
    # run() lift (Phase 4f.3d.f)
    # ------------------------------------------------------------------
    async def _pre_run(self) -> None:
        """Subclass hook: pre-run bookkeeping.

        Default: no-op (CUDA). OpenCL overrides to reset its
        verification-pool global-best fields at the start of each run and
        seed the pool from the initial kernel if it exists on disk.
        """
        return None

    async def _derive_shared_initial_state(self) -> str:
        """Compute the trajectory-shared initial state once per ``run()``.

        Backend-agnostic (Phase 4f.3d.f) — uses the same hooks as
        ``run_rollout``. On database exceptions, returns
        ``"<backend>_unknown"`` — historically CUDA raised; OpenCL fell
        back to a sentinel. Unifying on the sentinel is a small robustness
        improvement (an early state-analysis error no longer crashes the
        whole parallel-iteration fan-out).
        """
        last_log = getattr(self, "last_profile_log", "") or ""
        metrics = self.backend.parse_state_metrics(last_log, self.initial_metric or 0.0)
        try:
            return await self.database.get_state_from_ncu_report(
                last_log,
                metrics,
                self.kernel_source,
                elapsed_cycles=self.backend.state_cycles_from_metric(self.initial_metric),
            )
        except Exception:
            return f"{self.backend.name}_unknown"

    async def _finalize_run(self, best_filename, best_metric):
        """Subclass hook: assemble the final ``success_*`` / ``failure_*`` file.

        Called after all trajectories complete and the DB snapshot has been
        written. Subclass returns the ``Path`` that ``run()`` hands back to
        its caller. Divergence intentionally stays here — OpenCL runs a
        verification-pool preference on top of the per-iter best; CUDA
        does a baseline-recompute fallback.
        """
        raise NotImplementedError("Subclasses must implement _finalize_run")

    async def run(self):
        """Parallel RL iterations; return the best resulting artifact.

        Backend-agnostic shape (Phase 4f.3d.f). Per-backend bits: pre-run
        hook (``_pre_run``), initial-state derivation (shared),
        best-per-iter tracking (via ``backend.metric_from_traj_cycles`` +
        ``format_result_artifact``), and final artifact assembly
        (``_finalize_run``).
        """
        import asyncio as _asyncio

        best_filename: Optional[Path] = None
        best_metric: float = float("inf")

        await self._pre_run()

        if not hasattr(self, "last_profile_log") or not self.last_profile_log:
            await self.initialize()

        initial_state = await self._derive_shared_initial_state()

        async def _run_single_iteration(idx: int):
            self.agent_logger.info(
                f"[Async] RL Iteration {idx + 1}/{self.num_rl_iterations}"
            )
            try:
                trajectory = await self.run_rollout(self.kernel_source, initial_state)
                return idx, trajectory
            except Exception as exc:
                self.agent_logger.error(f"RL iteration {idx + 1} failed: {exc}")
                return idx, None

        tasks = [
            _asyncio.create_task(_run_single_iteration(i))
            for i in range(self.num_rl_iterations)
        ]

        ext = self.backend.kernel_ext
        for coro in _asyncio.as_completed(tasks):
            idx, trajectory = await coro
            if trajectory is None:
                continue

            if trajectory.steps:
                best_step = min(trajectory.steps, key=lambda s: s.cycles)
                # Step 4: ``TrajectoryStep.cycles`` is the backend's native
                # metric (float, no shape conversion needed).
                step_metric = best_step.cycles
                if step_metric < best_metric:
                    best_metric = step_metric
                    fp = self.folder / f"rl_iter_{idx}_best{ext}"
                    fp.write_text(
                        self.backend.format_result_artifact(best_step.code, step_metric)
                    )
                    best_filename = fp
                    self.agent_logger.info(
                        f"[Async] New best from iter {idx}: "
                        f"{self.backend.format_metric(best_metric)}"
                    )

            if trajectory:
                self.replay_buffer.add_trajectory(trajectory)
                self.total_trajectories += 1

        # Persist a numbered snapshot of the optimisation database JSON.
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

        return await self._finalize_run(best_filename, best_metric)

    # ------------------------------------------------------------------
    # run_rollout() lift (Phase 4f.3d.e)
    # ------------------------------------------------------------------
    async def run_rollout(self, initial_code: str, initial_state: str):
        """Execute one RL trajectory: analyse -> plan -> apply -> record.

        Backend-agnostic (Phase 4f.3d.e). The per-backend bits are
        already in place on ``Backend``:

          - ``parse_state_metrics(raw_log, current_metric)`` — dict shape
            expected by ``database.analyze_performance_state``.
          - ``state_cycles_from_metric(current_metric)`` — int passed as
            ``elapsed_cycles=`` to the analyse call.
          - ``metric_to_traj_cycles(metric)`` — how the primary metric
            lands in ``TrajectoryStep.cycles`` (typed int).
          - ``database_update_kwargs()`` — CUDA {} / OpenCL {current_file_path: None}.

        ``apply_optimization`` (lifted in 4f.3d.c) returns ``float`` for
        the new metric; we hand it back into the loop as ``current_metric``
        for the next iteration.
        """
        import json as _json
        import random
        import uuid as _uuid
        from dataclasses import asdict
        from .rl_agents import Trajectory, TrajectoryStep

        # Per-trajectory index + directory (uuid suffix guards against
        # folder collisions when multiple trajectories run concurrently).
        async with self._trajectory_lock:
            self.total_trajectories += 1
            trajectory_index = self.total_trajectories

        _uid = _uuid.uuid4().hex[:8]
        trajectory_dir = self.folder / f"trajectory_{trajectory_index}_{_uid}"
        trajectory_dir.mkdir(parents=True, exist_ok=True)

        trajectory = Trajectory()
        current_code: str = initial_code
        current_state: str = initial_state
        current_metric = self.initial_metric
        last_raw_log: str = getattr(self, "last_profile_log", "")

        self.agent_logger.info(f"Starting rollout from state: {current_state}")

        for step in range(self.max_rollout_steps):
            # 1) Analyse current performance state via the LLM helper.
            metrics = self.backend.parse_state_metrics(last_raw_log, current_metric)
            try:
                profile = await self.database.analyze_performance_state(
                    last_raw_log,
                    metrics,
                    current_code,
                    elapsed_cycles=self.backend.state_cycles_from_metric(current_metric),
                )
                analysis_json_str = _json.dumps(asdict(profile), indent=2)

                # 2) Generate a ranked plan. top_n shrinks as we advance.
                cur_iter = step + 1
                plan = await self.database.generate_optimization_plan(
                    analysis_json_str,
                    current_code,
                    top_n=max(4, self.max_rollout_steps - cur_iter),
                )
            except Exception as exc:
                self.agent_logger.warning(f"Plan generation failed, falling back: {exc}")
                plan = []

            # 3) Pick a technique — weighted by (relevance ** 3) unless the
            #    KERNELAGENT_DB_FALLBACK_TOP1 env var forces deterministic top-1.
            optimization_entry = None
            strategy_description = ""
            if plan:
                def _safe_rel(x):
                    try:
                        r = float(x)
                    except (TypeError, ValueError):
                        r = 0.05
                    return min(max(r, 0.0), 1.0)

                import os as _os
                force_top1 = _os.getenv("KERNELAGENT_DB_FALLBACK_TOP1", "0") in (
                    "1", "true", "True", "yes", "YES", "y", "on", "ON",
                )
                if force_top1:
                    chosen_plan = max(plan, key=lambda p: _safe_rel(p.get("relevance_score", 0.05)))
                    self.agent_logger.info(
                        f"KERNELAGENT_DB_FALLBACK_TOP1 is set; deterministically selecting top-1: "
                        f"{chosen_plan.get('technique')} "
                        f"(relevance {chosen_plan.get('relevance_score', 0.0)})"
                    )
                else:
                    weights = [
                        max(_safe_rel(p.get("relevance_score", 0.05)) ** 3, 0.001)
                        for p in plan
                    ]
                    chosen_plan = random.choices(plan, weights=weights, k=1)[0]

                technique_name = chosen_plan.get("technique")
                optimization_entry = self._lookup_optim_entry_by_name(technique_name)
                strategy_description = chosen_plan.get("description", "")
                self.agent_logger.info(
                    f"Selected technique from optimisation plan: {technique_name} "
                    f"(relevance {chosen_plan.get('relevance_score', 0.0):.2f})"
                )

            # 4) Legacy fallback chain if no plan / lookup failed.
            if optimization_entry is None:
                optimization_entry = self.database.select_best_optimization(current_state)
                if optimization_entry is None:
                    optimization_entry = self.database.select_best_optimization(
                        current_state, exclude_used=True
                    )
                if optimization_entry is None:
                    for state_name, sd in self.database.optimization_strategies.items():
                        if sd.get("optimizations"):
                            optimization_entry = self.database.select_best_optimization(state_name)
                            if optimization_entry:
                                self.agent_logger.info(
                                    f"Using fallback optimization from state: {state_name}"
                                )
                                break
                if optimization_entry is None:
                    if self._try_add_default_optimizations(current_state):
                        optimization_entry = self.database.select_best_optimization(current_state)
                        if optimization_entry is not None:
                            self.agent_logger.info(
                                f"Using default optimization for new state: {current_state}"
                            )
                if optimization_entry is None:
                    self.agent_logger.warning(
                        f"No optimization found for state: {current_state}, "
                        f"stopping rollout at step {step}"
                    )
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
                + f" | entry_type={type(optimization_entry).__name__}"
            )

            try:
                # 5) Apply the optimisation (lifted in 4f.3d.c).
                optimized_code, new_metric, new_state, new_raw_log = await self.apply_optimization(
                    current_code,
                    optimization_entry,
                    step,
                    trajectory_dir,
                    strategy_description,
                )

                # 6) Reward + trajectory step.
                if current_metric is not None and current_metric > 0:
                    actual_improvement = ((current_metric - new_metric) / current_metric) * 100
                else:
                    actual_improvement = 0.0
                reward = self.calculate_reward(
                    getattr(optimization_entry, "predicted_improvement", None),
                    actual_improvement,
                    (current_metric is not None and new_metric < current_metric),
                )

                action_name = (
                    optimization_entry.get_composite_id()
                    if isinstance(optimization_entry, CompositeOptimization)
                    else getattr(optimization_entry, "technique", str(optimization_entry))
                )
                traj_step = TrajectoryStep(
                    state=current_state,
                    action=action_name,
                    code=optimized_code,
                    # Step 4: cycles field now holds the backend's native
                    # metric verbatim (float).
                    cycles=new_metric,
                    predicted_improvement=(
                        getattr(optimization_entry, "predicted_improvement", 0.0) or 0.0
                    ),
                    actual_improvement=actual_improvement,
                    reward=reward,
                )
                self.agent_logger.info(f"Adding trajectory step: {traj_step}")
                trajectory.add_step(traj_step)

                # 7) Record actuals into the database.
                self.agent_logger.info(
                    f"Updating database with actual results for {technique_name} in state "
                    f"{current_state} with actual improvement {actual_improvement}"
                )
                if isinstance(optimization_entry, CompositeOptimization):
                    self.database.update_composite_optimization_result(
                        current_state, technique_name, actual_improvement,
                    )
                else:
                    # Phase 3c full: pass primary metrics directly. The DB's
                    # speedup calc prefers this over any legacy file-based
                    # parsing. ``current_metric`` and ``self.initial_metric``
                    # are in the backend's primary unit (CUDA cycles / OpenCL
                    # ms) — direction-consistent (lower is faster). This
                    # supersedes the ``database_update_kwargs`` hook that used
                    # to suppress the CUDA-flavored file parse for OpenCL.
                    self.database.update_optimization_result(
                        current_state,
                        technique_name,
                        actual_improvement,
                        current_metric=float(new_metric),
                        baseline_metric=(
                            float(self.initial_metric)
                            if self.initial_metric is not None
                            else None
                        ),
                    )

                self.agent_logger.info(
                    f"Step {step} result: "
                    f"{self.backend.format_metric(new_metric)} "
                    f"({actual_improvement:.1f}% improvement, reward: {reward:.2f})"
                )

                # 8) Advance loop state.
                current_code = optimized_code
                if new_state is not None:
                    current_state = new_state
                current_metric = new_metric
                last_raw_log = new_raw_log or last_raw_log

                if actual_improvement < -500:
                    self.agent_logger.warning(
                        f"Stopping rollout due to severe degradation: {actual_improvement:.1f}%"
                    )
                    break

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                try:
                    self.agent_logger.error(
                        f"Error in step {step}: {e}\n"
                        f"Technique: {technique_name} | Entry type: {type(optimization_entry).__name__}\n"
                        f"Raw optimization entry: {optimization_entry}\n"
                        f"Traceback:\n{tb}"
                    )
                except Exception:
                    print(f"Error in step {step}: {e}\n{tb}")
                break

        return trajectory

    # ------------------------------------------------------------------
    # get_feedback() lift (Phase 4f.3d.d)
    # ------------------------------------------------------------------
    def _get_verified_best(self, trajectory):
        """Return ``(best_code, best_metric, best_action, best_step)``.

        Default: trajectory minimum (CUDA). OpenCL overrides to prefer its
        verification-pool global-best when set — the fastest CORRECT kernel
        across all attempts, not only the current trajectory's last step.
        ``best_metric`` is a ``float`` in the backend's primary unit
        (CUDA: cycles-as-float, OpenCL: ms).

        Returns ``(None, inf, "n/a", None)`` when nothing usable exists.
        """
        if not trajectory.steps:
            return None, float("inf"), "n/a", None
        best_step = min(trajectory.steps, key=lambda s: s.cycles)
        return best_step.code, float(best_step.cycles), best_step.action, best_step

    def _build_feedback(
        self,
        *,
        response,
        task_id,
        code: str,
        initial_metric: float,
        profile_result: "ProfileResult",
        initial_state: str,
        trajectory,
    ):
        """Subclass hook: assemble the backend-specific ``Feedback`` subclass.

        Called on the happy path (whether or not the trajectory produced
        steps). Subclass returns ``RLNCUFeedback`` / ``RLOpenCLFeedback``
        with the trajectory + best-step framing they need.
        """
        raise NotImplementedError("Subclasses must implement _build_feedback")

    async def get_feedback(self, response, attempt_id, task_id, logger):
        """Main feedback loop for the RL optimisation agent.

        Backend-agnostic shape (Phase 4f.3d.d): profile the incoming code,
        derive an initial state (via ``_derive_state`` — Phase 4f.3a),
        run a full ``run_rollout``, then hand off to the per-backend
        ``_build_feedback`` hook for message + dataclass shape. The
        ``FeedbackError`` fallback is byte-identical between backends and
        is handled here.
        """
        from .utils import FeedbackError
        from .feedback import Feedback

        if self.initial_metric is None:
            await self.initialize()
        logger.info(f"Starting RL optimisation trajectory for task {task_id}")

        code, filepath = self.get_code_from_response(response, attempt_id, task_id, logger)
        try:
            pr = await self.gather_profile_result(filepath)
            initial_metric = self.backend.extract_primary_metric(pr)
            initial_state = await self._derive_state(pr, code)

            trajectory = await self.run_rollout(code, initial_state)
            self.replay_buffer.add_trajectory(trajectory)
            self.total_trajectories += 1

            return self._build_feedback(
                response=response,
                task_id=task_id,
                code=code,
                initial_metric=initial_metric,
                profile_result=pr,
                initial_state=initial_state,
                trajectory=trajectory,
            )
        except FeedbackError as e:
            logger.error(f"Error in RL optimisation: {e}")
            return Feedback(
                new_messages=[
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": f"Optimisation failed: {e}. Please fix and try again.",
                    },
                ],
                success=False,
                feedback=e.feedback if hasattr(e, "feedback") else str(e),
            )

    # ------------------------------------------------------------------
    # Reward calculation — backend-independent.
    # Byte-identical in both agents pre-Phase-4f.
    # ------------------------------------------------------------------
    def calculate_reward(
        self,
        predicted_improvement: Optional[float],
        actual_improvement: float,
        is_faster: bool,
    ) -> float:
        """Calculate reward based on prediction accuracy and actual performance.

        Safely handles None/zero predicted_improvement by skipping accuracy bonus.
        """
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
    # Database lookup — backend-independent.
    # Byte-identical in both agents pre-Phase-4f.
    # ------------------------------------------------------------------
    def _lookup_optim_entry_by_name(
        self, technique_name: str
    ) -> Optional[OptimizationEntry | CompositeOptimization]:
        """Search the database for an OptimizationEntry or CompositeOptimization by name."""
        # Search individual techniques.
        for state_data in self.database.optimization_strategies.values():
            for opt in state_data.get("optimizations", []):
                if opt.technique == technique_name:
                    return opt
        # Search composite optimisations.
        for comps in self.database.composite_optimizations.values():
            for comp in comps:
                if comp.get_composite_id() == technique_name:
                    return comp
        return None

    # ------------------------------------------------------------------
    # Default optimisations fallback — backend-aware via Phase 4c.
    # ------------------------------------------------------------------
    def _try_add_default_optimizations(self, current_state: str) -> bool:
        """Fallback when no optimizations are recorded for a discovered state.

        The catalog (bottleneck -> [(technique, predicted_pct), ...]) lives on
        the backend — see ``Backend.get_default_optimizations``.
        """
        try:
            defaults = self.backend.get_default_optimizations()
            primary_bottleneck = next(
                (b for b in defaults if b in current_state), None
            )
            if primary_bottleneck is not None:
                for technique, improvement in defaults[primary_bottleneck]:
                    self.database.add_new_optimization(current_state, technique, improvement)
                self.agent_logger.info(
                    f"Added {len(defaults[primary_bottleneck])} default optimizations for state: {current_state}"
                )
                return True
        except Exception as e:
            self.agent_logger.error(f"Error adding default optimizations: {e}")
        return False

    # ------------------------------------------------------------------
    # Performance summary — shared skeleton with backend-named JSON keys.
    # ------------------------------------------------------------------
    def get_performance_summary(self) -> Dict[str, Any]:
        """Comprehensive performance summary as a JSON-serializable dict.

        Backend-named metric keys (``initial_cycles`` / ``best_cycles`` for CUDA,
        ``initial_time_ms`` / ``best_time_ms`` for OpenCL) come from the
        ``_perf_summary_extras`` hook so the persisted JSON schema is
        preserved across the Phase 4f refactor (downstream analytics tools
        keep working).
        """
        overall_improvement = (
            ((self.initial_metric - self.best_metric) / self.initial_metric * 100)
            if self.initial_metric
            else 0
        )
        return {
            "total_trajectories": self.total_trajectories,
            "iteration_count": self.iteration_count,
            "overall_improvement": overall_improvement,
            "buffer_stats": self.replay_buffer.get_statistics(),
            "database_stats": self.database.get_database_stats(),
            **self._perf_summary_extras(),
        }

    def _perf_summary_extras(self) -> Dict[str, Any]:
        """Subclass hook for backend-named metric keys in the summary dict.

        Default: empty (no extra keys). Subclasses return e.g.
        ``{"initial_cycles": self.initial_metric, "best_cycles": self.best_metric}``.
        """
        return {}
