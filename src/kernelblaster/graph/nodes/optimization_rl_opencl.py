from pathlib import Path

from data.kernelbench_opencl import (
    default_benchmark_opencl_root,
    run_output_parent_to_benchmark_dir,
)

from ...agents import FeedbackConfig
from ...agents.opt_opencl_rl import RLOpenCLAgent
from ..state import GraphState, save_state_to_json


async def optimization_rl_opencl(state: GraphState):
    """
    RL-based OpenCL optimization node for Qualcomm Adreno GPUs.
    Takes the OpenCL kernel from the curated benchmark and applies
    RL-based optimization using the RLOpenCLAgent.
    """
    base_folder = Path(state["folder"])
    base_folder.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[4]
    curated_root = Path(
        state.get("kernelbench_opencl_root", default_benchmark_opencl_root())
    )
    if not curated_root.exists():
        alt = repo_root / "data" / "kernelbench-opencl"
        if alt.exists():
            state["logger"].warning(
                f"Benchmark OpenCL root missing at {curated_root}; falling back to {alt}"
            )
            curated_root = alt

    parent_name = base_folder.parent.name
    bench_tier = run_output_parent_to_benchmark_dir(parent_name)
    problem_name = base_folder.name
    curated_dir = curated_root / bench_tier / problem_name

    curated_driver_c = curated_dir / "driver.c"
    curated_kernel_cl = curated_dir / "kernel.cl"

    run_driver_c = base_folder / "driver.c"
    run_kernel_cl = base_folder / "kernel.cl"

    kernel_cl_fp = state.get("kernel_cl_fp")
    if kernel_cl_fp is None:
        if curated_kernel_cl.exists():
            kernel_cl_fp = curated_kernel_cl
            state["logger"].info(
                f"Using curated kernel.cl from {curated_dir} as kernel_cl_fp: {kernel_cl_fp}"
            )
        elif run_kernel_cl.exists():
            kernel_cl_fp = run_kernel_cl
            state["logger"].info(f"Using run-folder kernel.cl as kernel_cl_fp: {kernel_cl_fp}")
        else:
            state["logger"].error(
                f"No kernel_cl_fp available. Required files not found:\n"
                f"  - Curated: {curated_kernel_cl}\n"
                f"  - Run folder: {run_kernel_cl}\n"
                f"Skipping problem {problem_name} - curated OpenCL files are required."
            )
            return {"rl_opencl_perf_fp": None}

    kernel_cl_fp = Path(kernel_cl_fp)

    test_code_fp = state.get("test_code_fp")
    if test_code_fp is None:
        if curated_driver_c.exists():
            test_code_fp = curated_driver_c
            state["logger"].info(
                f"Using curated driver.c from {curated_dir} as test_code_fp: {test_code_fp}"
            )
        elif run_driver_c.exists():
            test_code_fp = run_driver_c
            state["logger"].info(
                f"Using run-folder driver.c as test_code_fp: {test_code_fp}"
            )
        else:
            state["logger"].error(
                f"No test_code_fp (driver.c) for this task. Expected alongside OpenCL under:\n"
                f"  {curated_dir}/driver.c\n"
                f"or run-folder:\n"
                f"  {run_driver_c}"
            )
            return {"rl_opencl_perf_fp": None}

    test_code_fp = Path(test_code_fp)

    save_state_to_json(state, base_folder / "state.json")

    fb_config = FeedbackConfig(
        agent_name="rl_opencl",
        base_folder=base_folder,
        logger=state["logger"],
        init_user_prompt="",
        model=state["model"],
        gpu=state["gpu"],
        test_code_fp=test_code_fp,
        retry_failed=state["retry_failed"],
        num_pgen=1,
    )

    database_path = base_folder.parent.parent.parent / "optimization_database.md"
    max_rollout_steps = state.get("rl_rollout_steps", 5)
    replay_buffer_size = state.get("rl_buffer_size", 100)
    update_frequency = state.get("rl_update_frequency", 3)
    rl_iterations = state.get("rl_iterations", 10)

    agent_rl_opencl = RLOpenCLAgent(
        fb_config=fb_config,
        kernel_to_optimize_fp=kernel_cl_fp,
        database_path=database_path,
        max_rollout_steps=max_rollout_steps,
        replay_buffer_size=replay_buffer_size,
        update_frequency=update_frequency,
        database=state.get("shared_optimization_database"),
    )

    await agent_rl_opencl.initialize()

    state["logger"].info(f"Starting OpenCL RL optimization with {rl_iterations} iterations")
    agent_rl_opencl.num_rl_iterations = rl_iterations

    best_filename = await agent_rl_opencl.run()

    if best_filename is not None:
        state["logger"].info(f"OpenCL RL optimization completed successfully: {best_filename}")
    else:
        state["logger"].error("OpenCL RL optimization failed to produce any valid results")

    # Prefer the fastest *verified* kernel (any successful profiled run), not only `success_*.cl`.
    global_best = base_folder / "global_best_rl_optimization.cl"
    src = global_best if global_best.exists() else best_filename
    if src is not None:
        final_file = base_folder / "final_rl_opencl_perf.cl"
        final_file.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        state["logger"].info(
            f"OpenCL RL best verified kernel written to {final_file} (from {src.name})"
        )
    else:
        state["logger"].error("OpenCL RL optimization failed to produce any valid results")

    save_state_to_json({**state, "rl_opencl_perf_fp": best_filename}, base_folder / "state.json")

    return {"rl_opencl_perf_fp": best_filename}
