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

    Resolves curated artifacts (kernel source + driver) via the backend's
    ``RLNodeConfig``, instantiates the per-backend RL agent, runs it, and
    saves the final artifact. Returns a dict with the backend-specific
    state-output key so existing downstream nodes don't need to change.
    """
    cfg = backend.rl_node_config()

    base_folder = Path(state["folder"])
    base_folder.mkdir(parents=True, exist_ok=True)

    # ---------- Curated artifact resolution ----------
    # Repo root is the project root (e.g. /path/to/KernelBlaster); this file
    # lives at .../src/kernelblaster/graph/nodes/_rl_node.py — parents[4].
    repo_root = Path(__file__).resolve().parents[4]
    curated_root = Path(
        state.get(cfg.curated_root_state_key, cfg.curated_root_default)
    )
    if not curated_root.exists():
        # Fallback: try the standard alt path under data/.
        alt = repo_root / "data" / f"kernelbench-{backend.name}"
        if alt.exists():
            state["logger"].warning(
                f"Curated root missing at {curated_root}; falling back to {alt}"
            )
            curated_root = alt

    # Tier resolution: CUDA uses parent name as-is (level1/...); OpenCL maps
    # the run-output parent name to a benchmark directory (sol-level2 -> L2).
    parent_name = base_folder.parent.name
    tier_dir = cfg.tier_resolver(parent_name) if cfg.tier_resolver else parent_name
    problem_name = base_folder.name
    curated_dir = curated_root / tier_dir / problem_name

    curated_driver = curated_dir / backend.driver_filename
    curated_kernel = curated_dir / cfg.kernel_filename

    run_driver = base_folder / backend.driver_filename
    run_kernel = base_folder / cfg.kernel_filename

    # Resolve kernel source path: explicit state -> curated -> run-folder.
    kernel_fp = state.get(cfg.state_kernel_fp_input)
    if kernel_fp is None:
        if curated_kernel.exists():
            kernel_fp = curated_kernel
            state["logger"].info(
                f"Using curated {cfg.kernel_filename} from {curated_dir} "
                f"as {cfg.state_kernel_fp_input}: {kernel_fp}"
            )
        elif run_kernel.exists():
            kernel_fp = run_kernel
            state["logger"].info(
                f"Using run-folder {cfg.kernel_filename} as "
                f"{cfg.state_kernel_fp_input}: {kernel_fp}"
            )
        else:
            state["logger"].error(
                f"No {cfg.state_kernel_fp_input} available. Required files not found:\n"
                f"  - Curated: {curated_kernel}\n"
                f"  - Run folder: {run_kernel}\n"
                f"Skipping problem {problem_name} - curated {backend.name} files are required."
            )
            return {cfg.state_perf_fp_output: None}

    kernel_fp = Path(kernel_fp)

    # Resolve driver path: explicit state -> curated -> run-folder.
    test_code_fp = state.get(cfg.state_test_code_fp_key)
    if test_code_fp is None:
        if curated_driver.exists():
            test_code_fp = curated_driver
            state["logger"].info(
                f"Using curated {backend.driver_filename} from {curated_dir} "
                f"as {cfg.state_test_code_fp_key}: {test_code_fp}"
            )
        elif run_driver.exists():
            test_code_fp = run_driver
            state["logger"].info(
                f"Using run-folder {backend.driver_filename} as "
                f"{cfg.state_test_code_fp_key}: {test_code_fp}"
            )
        else:
            state["logger"].error(
                f"No {cfg.state_test_code_fp_key} available. Required files not found:\n"
                f"  - Curated: {curated_driver}\n"
                f"  - Run folder: {run_driver}\n"
                f"Skipping problem {problem_name} - curated {backend.driver_filename} is required."
            )
            return {cfg.state_perf_fp_output: None}

    test_code_fp = Path(test_code_fp)

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
