# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Unified RL-optimization graph node (Phase 4e).

Replaces the two near-duplicate node bodies in ``optimization_rl_ncu.py``
and ``optimization_rl_opencl.py`` with one function parameterized on a
``Backend``. Per-backend specifics (curated paths, state-dict keys, agent
class wiring, final filename, global-best preference) come from
``backend.rl_node_config()`` — see ``kernelblaster.backends.base.RLNodeConfig``.

The two public ``optimization_rl_ncu`` / ``optimization_rl_opencl``
function names remain in their original files as thin shims that
construct the right backend and call ``_run_rl_optimization_node``.
"""
from __future__ import annotations

from pathlib import Path

from ...agents import FeedbackConfig
from ..state import GraphState, save_state_to_json


async def _run_rl_optimization_node(state: GraphState, backend) -> dict:
    """Unified RL-optimization graph node body.

    Resolves curated artifacts (kernel source + driver) then instantiates
    the per-backend RL agent. Two resolution mechanisms, tried in order:

      1. **State override**. If ``state[cfg.state_kernel_fp_input]`` /
         ``state[cfg.state_test_code_fp_key]`` are set (typically by an
         upstream kgen node in the same graph), use them verbatim.
      2. **Problem-driven** (Item 2, Phase 5). If ``state["problem"]``
         is set — a :class:`data.sources.Problem` — read
         ``problem.curated_artifacts["kernel"]`` and ``["driver"]``.

    The pre-Step-1c "derive-from-run-folder-parent-name" fallback was
    deleted once every in-tree caller was migrated: run_RL.py always
    sets ``state["problem"]``; the HTTP API (serve_api.py) sets it too
    since Step 1b; the intermediate kgen node populates the state
    override keys.

    Returns a dict with the backend-specific state-output key so
    existing downstream nodes don't need to change.
    """
    cfg = backend.rl_node_config()

    base_folder = Path(state["folder"])
    base_folder.mkdir(parents=True, exist_ok=True)

    problem = state.get("problem")
    problem_name = None
    kernel_fp = state.get(cfg.state_kernel_fp_input)
    test_code_fp = state.get(cfg.state_test_code_fp_key)

    if problem is not None:
        # Problem-driven resolution: role-keyed artifacts, no derivation.
        problem_name = problem.problem_name
        if kernel_fp is None:
            kernel_fp = problem.curated_artifacts.get("kernel")
            if kernel_fp is not None:
                state["logger"].info(
                    f"Using Problem-provided kernel artifact as "
                    f"{cfg.state_kernel_fp_input}: {kernel_fp}"
                )
        if test_code_fp is None:
            test_code_fp = problem.curated_artifacts.get("driver")
            if test_code_fp is not None:
                state["logger"].info(
                    f"Using Problem-provided driver artifact as "
                    f"{cfg.state_test_code_fp_key}: {test_code_fp}"
                )

    if kernel_fp is None:
        state["logger"].error(
            f"No {cfg.state_kernel_fp_input} available and no Problem-"
            f"provided kernel artifact. Set state[{cfg.state_kernel_fp_input!r}] "
            f"upstream (e.g. via kgen) or provide state['problem'] with a "
            f"'kernel' curated artifact."
        )
        return {cfg.state_perf_fp_output: None}
    if test_code_fp is None:
        state["logger"].error(
            f"No {cfg.state_test_code_fp_key} available and no Problem-"
            f"provided driver artifact. Set state[{cfg.state_test_code_fp_key!r}] "
            f"upstream (e.g. via kgen) or provide state['problem'] with a "
            f"'driver' curated artifact."
        )
        return {cfg.state_perf_fp_output: None}

    kernel_fp = Path(kernel_fp)
    test_code_fp = Path(test_code_fp)
    problem_name = problem_name or base_folder.name

    save_state_to_json(state, base_folder / "state.json")

    # ---------- Agent construction ----------
    fb_config = FeedbackConfig(
        agent_name=cfg.fb_config_agent_name,
        base_folder=base_folder,
        logger=state["logger"],
        init_user_prompt="",
        model=state["model"],
        gpu=state["gpu"],
        test_code_fp=test_code_fp,
        retry_failed=state["retry_failed"],
        num_pgen=cfg.num_pgen,
    )

    database_path = base_folder.parent.parent.parent / "optimization_database.md"
    max_rollout_steps = state.get("rl_rollout_steps", 5)
    replay_buffer_size = state.get("rl_buffer_size", 100)
    update_frequency = state.get("rl_update_frequency", 3)
    rl_iterations = state.get("rl_iterations", 10)

    agent_kwargs = {
        cfg.agent_kernel_fp_kwarg: kernel_fp,
        "fb_config": fb_config,
        "database_path": database_path,
        "max_rollout_steps": max_rollout_steps,
        "replay_buffer_size": replay_buffer_size,
        "update_frequency": update_frequency,
        "database": state.get("shared_optimization_database"),
    }
    agent = cfg.agent_class(**agent_kwargs)

    # ---------- Run ----------
    await agent.initialize()
    state["logger"].info(
        f"Starting {backend.name.upper()} RL optimization with {rl_iterations} iterations"
    )
    agent.num_rl_iterations = rl_iterations

    best_filename = await agent.run()

    if best_filename is not None:
        state["logger"].info(
            f"{backend.name.upper()} RL optimization completed successfully: {best_filename}"
        )
    else:
        state["logger"].error(
            f"{backend.name.upper()} RL optimization failed to produce any valid results"
        )

    # ---------- Save final artifact ----------
    # OpenCL prefers global_best_rl_optimization.cl if it exists (the
    # verification-pool best across all attempts); CUDA uses the agent's
    # returned best_filename verbatim.
    if cfg.use_global_best_preference:
        global_best = base_folder / backend.best_filename()
        src = global_best if global_best.exists() else best_filename
    else:
        src = best_filename

    if src is not None:
        final_file = base_folder / cfg.final_filename
        final_file.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        suffix = f" (from {src.name})" if cfg.use_global_best_preference and src != best_filename else ""
        state["logger"].info(
            f"{backend.name.upper()} RL best kernel written to {final_file}{suffix}"
        )

    save_state_to_json(
        {**state, cfg.state_perf_fp_output: best_filename},
        base_folder / "state.json",
    )
    return {cfg.state_perf_fp_output: best_filename}
