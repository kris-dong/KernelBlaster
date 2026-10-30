"""
CudaCoder \"kgen\" integration: test-driver generation + CUDA kernel generation.

Mirrors ``cudacoder.graph.workflow`` nodes ``generation_test_driver`` and
``generation_cuda``, producing ``driver.cpp`` and ``final_cuda.cu`` under the
run folder. Used as the initial CUDA snapshot before KernelBlaster RL NCU
optimization.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

from ..config import GPUType


def _kb_repo_root() -> Path:
    # .../<repo-root>/src/kernelblaster/integration/kgen_cudacoder.py
    return Path(__file__).resolve().parents[3]


def default_cudacoder_parent_on_syspath() -> Path | None:
    """Directory to prepend to ``sys.path`` so ``import cudacoder`` resolves.

    Resolution order:

    1. ``CUDACODER_SRC`` — parent directory of the ``cudacoder`` package
       (e.g. ``.../cudacoder-main/src`` or ``.../<repo-root>/third_party``).
    2. Vendored copy: ``<repo>/third_party`` (contains ``cudacoder/``).
    3. Sibling checkout: ``../cudacoder-main/src``.
    """
    env = os.environ.get("CUDACODER_SRC", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            return p
    kb_repo = _kb_repo_root()
    bundled = kb_repo / "third_party"
    if (bundled / "cudacoder").is_dir() and (bundled / "cudacoder" / "__init__.py").is_file():
        return bundled
    sibling = kb_repo.parent / "cudacoder-main" / "src"
    if sibling.is_dir():
        return sibling
    return None


def ensure_cudacoder_on_path(cudacoder_parent: Path | None) -> Path:
    """Insert *parent of the cudacoder package* on sys.path (not the package dir itself)."""
    root = cudacoder_parent or default_cudacoder_parent_on_syspath()
    if root is None or not root.is_dir():
        raise ImportError(
            "CudaCoder is not available. Expected a vendored copy under "
            "<repo>/third_party/cudacoder, or set CUDACODER_SRC to the directory "
            "that contains the cudacoder package (e.g. cudacoder-main/src), or "
            "pass --cudacoder-root to a full cudacoder repo / third_party path."
        )
    pkg = root / "cudacoder"
    if not pkg.is_dir():
        raise ImportError(
            f"Directory {root} does not contain a cudacoder package (missing {pkg})."
        )
    s = str(root.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def llm_client_from_string(name: str):
    from cudacoder.types import LLMClientType

    n = (name or "openai").strip().lower()
    for m in LLMClientType:
        if m.value == n:
            return m
    return LLMClientType.OPENAI


def _make_cudacoder_workflow_config(
    *,
    model: str,
    gpu: GPUType,
    max_attempts: int,
    num_coders: int,
    llm_client,
    stream: bool,
    retry_on_llm_error: bool,
):
    from cudacoder.types import WorkflowConfig as CCWorkflowConfig, WorkflowType

    args = SimpleNamespace(
        model=model,
        gpu=gpu.value,
        workflow=WorkflowType.CUDA.value,
        num_coders=num_coders,
        num_coders_ncu_opt=8,
        max_attempts=max_attempts,
        ncu_opt_required_improvements=3,
        stream=stream,
        llm_client_type=llm_client,
        retry_on_llm_error=retry_on_llm_error,
    )
    return CCWorkflowConfig(args)


async def run_kgen_pipeline(
    *,
    folder: Path,
    reference_code: str,
    logger,
    model: str,
    gpu: GPUType,
    max_attempts: int = 8,
    num_coders: int = 4,
    llm_client_name: str = "openai",
    stream: bool = False,
    retry_on_llm_error: bool = False,
    cudacoder_src: Path | None = None,
) -> bool:
    """
    Run CudaCoder test-driver generation then CUDA kernel generation.

    Returns True if ``final_cuda.cu`` and ``driver.cpp`` were produced; False on
    failure (caller should keep curated / existing CUDA files).
    """
    import os

    llm_client = llm_client_from_string(llm_client_name)
    if llm_client.value == "openai" and not os.environ.get("OPENAI_API_KEY"):
        logger.warning(
            "Kgen skipped: OPENAI_API_KEY is not set (export it or use --openai-api-key)."
        )
        return False

    # ``cudacoder_src`` may be ``.../src`` (full repo) or ``.../third_party`` (vendored layout).
    ensure_cudacoder_on_path(cudacoder_src)

    from cudacoder.graph.nodes.cuda import generation_cuda
    from cudacoder.graph.nodes.test_driver import generation_test_driver

    folder.mkdir(parents=True, exist_ok=True)
    wf_config = _make_cudacoder_workflow_config(
        model=model,
        gpu=gpu,
        max_attempts=max_attempts,
        num_coders=num_coders,
        llm_client=llm_client,
        stream=stream,
        retry_on_llm_error=retry_on_llm_error,
    )

    state = {
        "reference_code": reference_code,
        "folder": folder,
        "logger": logger,
        "config": wf_config,
        "test_driver_fp": None,
        "cuda_fp": None,
    }

    logger.info("Kgen: running CudaCoder test driver generation…")
    delta_td = await generation_test_driver(state)
    state.update(delta_td)
    if state.get("test_driver_fp") is None:
        logger.warning("Kgen failed: no test driver generated.")
        return False

    logger.info("Kgen: running CudaCoder CUDA kernel generation…")
    delta_cu = await generation_cuda(state)
    state.update(delta_cu)
    if state.get("cuda_fp") is None:
        logger.warning("Kgen failed: no CUDA kernel generated.")
        return False

    final_cu = folder / "final_cuda.cu"
    driver = folder / "driver.cpp"
    if not final_cu.is_file() or not driver.is_file():
        logger.warning(
            f"Kgen reported success but missing outputs: {final_cu=} {driver=}"
        )
        return False

    logger.info(f"Kgen succeeded: {driver} , {final_cu}")
    return True
