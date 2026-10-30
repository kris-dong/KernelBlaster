from pathlib import Path

# Item 2 cleanup Phase 3: migrated off the ``data.kernelbench_opencl``
# back-compat shim onto the canonical source class. The two functions
# used to be free re-exports at the shim's module level — they're now
# static methods on :class:`KernelBenchOpenCLSource`.
from data.sources.kernelbench_opencl_source import (
    KernelBenchOpenCLSource,
    _SUBSET_TO_BENCHMARK_DIR,
)

from ...integration.kgen_opencl import run_kgen_opencl_pipeline
from ..state import GraphState, save_state_to_json


def default_benchmark_opencl_root() -> Path:
    return KernelBenchOpenCLSource._default_benchmark_root()


def run_output_parent_to_benchmark_dir(parent: str) -> str:
    return _SUBSET_TO_BENCHMARK_DIR.get(parent, parent)


async def kgen_opencl(state: GraphState):
    """
    OpenCL kgen node: translate PyTorch reference.py into driver.c + kernel.cl
    via LLM.  Skips if curated or existing files are already present.
    """
    if not state.get("run_kgen_opencl"):
        return {}

    reference = state.get("reference_code")
    base_folder = Path(state["folder"])
    logger = state["logger"]

    if not reference:
        logger.info(
            "run_kgen_opencl is set but reference_code is empty; skipping kgen."
        )
        return {}

    # Check if curated files already exist in benchmark-opencl
    repo_root = Path(__file__).resolve().parents[4]
    curated_root = Path(
        state.get("kernelbench_opencl_root", default_benchmark_opencl_root())
    )
    if not curated_root.exists():
        alt = repo_root / "data" / "kernelbench-opencl"
        if alt.exists():
            curated_root = alt

    parent_name = base_folder.parent.name
    bench_tier = run_output_parent_to_benchmark_dir(parent_name)
    problem_name = base_folder.name
    curated_dir = curated_root / bench_tier / problem_name

    curated_driver = curated_dir / "driver.c"
    curated_kernel = curated_dir / "kernel.cl"

    if curated_driver.exists() and curated_kernel.exists():
        logger.info(
            f"Curated OpenCL files already exist at {curated_dir}; skipping kgen."
        )
        return {
            "test_code_fp": curated_driver,
            "kernel_cl_fp": curated_kernel,
        }

    # Check if run folder already has files
    run_driver = base_folder / "driver.c"
    run_kernel = base_folder / "kernel.cl"

    if run_driver.exists() and run_kernel.exists():
        logger.info(
            f"Run folder already has driver.c + kernel.cl; skipping kgen."
        )
        return {
            "test_code_fp": run_driver,
            "kernel_cl_fp": run_kernel,
        }

    save_state_to_json(state, base_folder / "state.json")

    try:
        ok = await run_kgen_opencl_pipeline(
            folder=base_folder,
            reference_code=reference,
            logger=logger,
            model=state["model"],
            gpu=state["gpu"],
            max_attempts=int(state.get("kgen_max_attempts", 8)),
            num_pgen=int(state.get("kgen_num_pgen", 1)),
            retry_failed=state.get("retry_failed", False),
            precision=state.get("precision", "fp16"),
        )
    except Exception:
        logger.exception(
            "kgen_opencl failed with an unexpected error; "
            "downstream RL will require curated files."
        )
        ok = False

    save_state_to_json(state, base_folder / "state.json")

    if not ok:
        return {}

    return {
        "test_code_fp": base_folder / "driver.c",
        "kernel_cl_fp": base_folder / "kernel.cl",
    }
