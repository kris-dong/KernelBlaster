"""
OpenCL kgen integration: translate a PyTorch reference.py into
driver.c + kernel.cl for Qualcomm Adreno GPUs.

Single-stage generation — both files are produced in one LLM call and
validated with a two-step process (dummy kernel + real kernel).
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ..config import GPUType, config
from ..agents.feedback import FeedbackConfig
from ..agents.kgen_opencl import OpenCLKgenAgent


async def run_kgen_opencl_pipeline(
    *,
    folder: Path,
    reference_code: str,
    logger,
    model: str,
    gpu: GPUType,
    max_attempts: int = 8,
    num_pgen: int = 1,
    retry_failed: bool = False,
    precision: str = "fp16",
) -> bool:
    """
    Run the OpenCL kgen agent to produce driver.c + kernel.cl.

    Returns True if valid files were produced; False on failure.
    """
    folder.mkdir(parents=True, exist_ok=True)

    fb_config = FeedbackConfig(
        agent_name="kgen_opencl",
        base_folder=folder,
        logger=logger,
        init_user_prompt="",
        model=model,
        gpu=gpu,
        retry_failed=retry_failed,
        num_pgen=num_pgen,
        max_attempts=max_attempts,
    )

    agent = OpenCLKgenAgent(
        fb_config=fb_config,
        reference_code=reference_code,
        precision=precision,
    )

    best = await agent.run()

    if best is None:
        logger.warning("kgen_opencl failed: no valid driver.c + kernel.cl produced.")
        return False

    best = Path(best)
    if not best.exists():
        logger.warning(f"kgen_opencl: success file missing: {best}")
        return False

    # The agent saves success_attempt*_task*_driver.c and the matching _kernel.cl.
    # Copy them to canonical names in the folder.
    kernel_fp = Path(str(best).replace("_driver.c", "_kernel.cl"))
    if not kernel_fp.exists():
        logger.warning(f"kgen_opencl: matching kernel file missing: {kernel_fp}")
        return False

    final_driver = folder / "driver.c"
    final_kernel = folder / "kernel.cl"
    shutil.copy2(best, final_driver)
    shutil.copy2(kernel_fp, final_kernel)

    logger.info(f"kgen_opencl succeeded: {final_driver}, {final_kernel}")
    return True
