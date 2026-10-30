from pathlib import Path

from ...integration.kgen_cudacoder import run_kgen_pipeline
from ..state import GraphState, save_state_to_json


async def kgen_cudacoder(state: GraphState):
    """
    Optional first stage: CudaCoder LLM agents generate ``driver.cpp`` and
    ``final_cuda.cu`` in the run folder. On failure, downstream RL optimization
    falls back to curated ``data/benchmark/<L1|L2|L3>/...`` CUDA (and optional driver) artifacts.
    """
    if not state.get("run_kgen"):
        return {}

    reference = state.get("reference_code")
    base_folder = Path(state["folder"])
    logger = state["logger"]

    if not reference:
        logger.info(
            "run_kgen is set but reference_code is empty; skipping kgen "
            "(RL will use curated or run-folder CUDA if present)."
        )
        return {}

    save_state_to_json(state, base_folder / "state.json")

    cudacoder_src = None
    if state.get("cudacoder_root"):
        cr = Path(state["cudacoder_root"])
        cudacoder_src = cr / "src" if (cr / "src").is_dir() else cr

    try:
        ok = await run_kgen_pipeline(
            folder=base_folder,
            reference_code=reference,
            logger=logger,
            model=state["model"],
            gpu=state["gpu"],
            max_attempts=int(state.get("kgen_max_attempts", 8)),
            num_coders=int(state.get("kgen_num_coders", 4)),
            llm_client_name=str(state.get("kgen_llm_client", "openai")),
            stream=bool(state.get("kgen_stream", False)),
            retry_on_llm_error=bool(state.get("kgen_retry_on_llm_error", False)),
            cudacoder_src=cudacoder_src,
        )
    except ImportError as err:
        logger.warning(
            f"Kgen unavailable (install cudacoder or set CUDACODER_SRC): {err}"
        )
        ok = False
    except Exception:
        logger.exception(
            "Kgen failed with an unexpected error; continuing with curated CUDA."
        )
        ok = False

    save_state_to_json(state, base_folder / "state.json")

    if not ok:
        return {}

    final_cu = base_folder / "final_cuda.cu"
    driver = base_folder / "driver.cpp"
    return {"cuda_fp": final_cu, "test_code_fp": driver}
