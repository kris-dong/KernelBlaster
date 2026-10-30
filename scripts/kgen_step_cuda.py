"""Per-step CUDA kgen helper, meant to be invoked inside a serve container via
`docker exec`. Same contract as ``scripts/kgen_step.py`` (the OpenCL version)
but produces / validates ``driver.cpp`` + ``final_cuda.cu`` for NVIDIA GPUs.

Subcommands
-----------
  validate      Static rules + launcher-signature check; writes dummy_final_cuda.cu.
  run-dummy     Compile+run driver.cpp against the no-op kernel; expect "failed".
  run-real      Compile+run driver.cpp against final_cuda.cu; expect "passed".
  full          validate -> run-dummy -> run-real; the canonical flow.
  prompt        Print system+user prompts for a given reference.py (no run).

Exit 0 on success, 1 on kgen-domain failure (with feedback text the caller
should show the LLM), 2 on infrastructure errors.

The kgen flow assumes ``compile`` and ``gpu`` servers are reachable via
``COMPILE_SERVER_URL`` and ``GPU_SERVER_URL_<GPU>`` environment variables,
typically set by the serve-mode docker container on localhost:2001/2002.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path("/kernelblaster") if Path("/kernelblaster").exists() else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Default the servers to localhost if unset, so `kgen_step_cuda.py` Just Works
# inside the serve container without extra env plumbing.
os.environ.setdefault("COMPILE_SERVER_URL", "http://localhost:2001")
# Pick a sensible default GPU URL env; users targeting a different GPU should
# set GPU_SERVER_URL_<GPU> themselves before invoking.
os.environ.setdefault("GPU_SERVER_URL_L40S", "http://localhost:2002")

from loguru import logger

from src.kernelblaster.config import GPUType, config
from src.kernelblaster.agents.utils import (
    FeedbackError,
    NamedTimer,
    compile_and_run_cu_file,
)
from src.kernelblaster.agents.prompt.kgen_cuda import (
    build_system_prompt,
    build_user_prompt,
)


# ---------------------------------------------------------------------------
# Static rules for driver.cpp
# ---------------------------------------------------------------------------

def _strip_c_like_comments_and_strings(text: str) -> str:
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    text = re.sub(r"'(?:\\.|[^'\\])*'", "''", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    return text


def _rule_driver_has_torch_header(code: str) -> None:
    if "torch/torch.h" not in code:
        raise FeedbackError(
            "driver.cpp must #include <torch/torch.h>. Add the missing header."
        )


def _rule_driver_has_main(code: str) -> None:
    if re.search(r"\bint\s+main\s*\(", code) is None:
        raise FeedbackError(
            "driver.cpp must define int main(...). Add the missing entry point."
        )


def _rule_driver_has_launcher_decl(code: str) -> None:
    stripped = _strip_c_like_comments_and_strings(code)
    if re.search(
        r"\bvoid\s+launch_gpu_implementation\s*\(", stripped
    ) is None:
        raise FeedbackError(
            "driver.cpp must forward-declare "
            "`void launch_gpu_implementation(...)` and call it from main. "
            "The declaration must exactly match the definition in final_cuda.cu."
        )


def _rule_driver_has_allclose(code: str) -> None:
    if "torch::allclose" not in code:
        raise FeedbackError(
            "driver.cpp must verify GPU output with `torch::allclose(...)` "
            "before printing passed/failed."
        )


def _rule_driver_has_passed_failed(code: str) -> None:
    if '"passed' not in code or '"failed' not in code:
        raise FeedbackError(
            'driver.cpp must print "passed" or "failed" for verification. '
            "Add the missing verification output."
        )


def _rule_driver_has_zeros_like(code: str) -> None:
    # The driver must deterministically initialize gpu_output before calling
    # the kernel, so a no-op kernel is detectable. Two patterns satisfy this:
    #   - `torch::zeros_like(ref_output)` (the original idiom — relies on the
    #     reference being non-zero almost everywhere)
    #   - `fill_(<sentinel>)` (a stronger pattern: any unwritten element keeps
    #     a NaN / 1e30f / etc. canary, which `allclose` rejects regardless of
    #     reference content)
    # Either is acceptable.
    stripped = _strip_c_like_comments_and_strings(code)
    if "zeros_like" in stripped:
        return
    if re.search(r"\.fill_\s*\(", stripped):
        return
    raise FeedbackError(
        "driver.cpp must pre-initialize gpu_output so a no-op kernel is "
        "detectable. Use either `torch::zeros_like(ref_output)` or pre-fill "
        "with a canary via `gpu_output.fill_(NAN)` (or `1e30f`)."
    )


DRIVER_RULES = [
    _rule_driver_has_torch_header,
    _rule_driver_has_main,
    _rule_driver_has_launcher_decl,
    _rule_driver_has_allclose,
    _rule_driver_has_passed_failed,
    _rule_driver_has_zeros_like,
]


# ---------------------------------------------------------------------------
# Launcher contract: driver's decl vs .cu's definition must agree on arg count
# ---------------------------------------------------------------------------


_LAUNCHER_SIG_RE = re.compile(
    r"\bvoid\s+launch_gpu_implementation\s*\(([^)]*)\)",
    re.DOTALL,
)


def _launcher_params(code: str) -> list[str] | None:
    stripped = _strip_c_like_comments_and_strings(code)
    m = _LAUNCHER_SIG_RE.search(stripped)
    if not m:
        return None
    params = m.group(1).strip()
    if not params:
        return []
    return [p.strip() for p in params.split(",") if p.strip()]


def _validate_contract(driver_code: str, cuda_code: str) -> None:
    dparams = _launcher_params(driver_code)
    cparams = _launcher_params(cuda_code)
    if dparams is None:
        raise FeedbackError(
            "Cannot find `void launch_gpu_implementation(...)` declaration in driver.cpp."
        )
    if cparams is None:
        raise FeedbackError(
            "Cannot find `void launch_gpu_implementation(...)` definition in final_cuda.cu. "
            "The .cu file must define the launcher as the single host entry point."
        )
    if len(dparams) != len(cparams):
        raise FeedbackError(
            f"Launcher signature mismatch: driver.cpp declares "
            f"{len(dparams)} parameter(s) but final_cuda.cu defines {len(cparams)}. "
            "They must match exactly.\n\n"
            f"  driver.cpp: launch_gpu_implementation({', '.join(dparams) or 'void'})\n"
            f"  final_cuda.cu: launch_gpu_implementation({', '.join(cparams) or 'void'})"
        )


def _generate_dummy_cuda(driver_code: str, cuda_code: str) -> str:
    """Generate a no-op `launch_gpu_implementation` with a matching signature.

    The driver's gpu_output is pre-initialized to zeros, so a no-op launcher
    leaves zeros in the buffer; `torch::allclose(zeros, ref)` is almost
    certainly false for random inputs, so the driver prints "failed" — which
    is exactly what the dummy step wants.
    """
    params = _launcher_params(cuda_code)
    if params is None:
        raise FeedbackError(
            "Cannot extract launcher signature from final_cuda.cu for dummy generation."
        )
    sig_params = ", ".join(params) if params else "void"
    return (
        "// dummy no-op CUDA launcher for driver verification\n"
        "#include <cuda_runtime.h>\n\n"
        f"void launch_gpu_implementation({sig_params}) {{\n"
        "    // intentionally empty — driver must print 'failed' for this\n"
        "}\n"
    )


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _fail(msg: str, code: int = 1) -> "NoReturn":  # type: ignore[name-defined]
    print(msg, file=sys.stderr)
    sys.exit(code)


def _driver_cuda(folder: Path) -> tuple[Path, Path]:
    d = folder / "driver.cpp"
    c = folder / "final_cuda.cu"
    if not d.exists():
        _fail(f"driver.cpp not found at {d}")
    if not c.exists():
        _fail(f"final_cuda.cu not found at {c}")
    return d, c


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_validate(args) -> None:
    folder = Path(args.folder).resolve()
    d, c = _driver_cuda(folder)
    driver_code = d.read_text()
    cuda_code = c.read_text()

    for rule in DRIVER_RULES:
        try:
            rule(driver_code)
        except FeedbackError as e:
            _fail(str(e))

    try:
        _validate_contract(driver_code, cuda_code)
    except FeedbackError as e:
        _fail(str(e))

    try:
        dummy_code = _generate_dummy_cuda(driver_code, cuda_code)
    except FeedbackError as e:
        _fail(str(e))

    dummy_fp = folder / "dummy_final_cuda.cu"
    dummy_fp.write_text(dummy_code)
    print(f"validate ok — launcher matches, dummy written to {dummy_fp}")


async def _run_once(
    folder: Path, cuda_filename: str, gpu: GPUType, timeout: int
) -> tuple[str, str, bool]:
    """Compile driver.cpp + given .cu and run once; return (stdout, stderr, passed)."""
    d, _ = _driver_cuda(folder)
    cu_fp = folder / cuda_filename
    if not cu_fp.exists():
        _fail(f"{cuda_filename} not found at {cu_fp}")

    timer = NamedTimer()
    try:
        stdout_list, stderr_list, _, success = await compile_and_run_cu_file(
            main_filepath=d,
            cuda_filepath=cu_fp,
            gpu=gpu,
            timer=timer,
            logger=logger,
            timeout=timeout,
            num_runs=1,
            passed_keyword="passed",
            persistent_artifacts=True,
        )
    except FeedbackError as e:
        return "", str(e), False

    stdout = stdout_list[0] if stdout_list else ""
    stderr = stderr_list[0] if stderr_list else ""
    return stdout, stderr, success


def _check_dummy_result(stdout: str, stderr: str, success: bool) -> None:
    if success and "passed" in stdout.lower():
        _fail(
            "DRIVER VERIFICATION BUG: driver printed 'passed' for a no-op kernel.\n"
            "The verifier (torch::allclose or the reference computation) is broken — "
            "it's not actually comparing GPU output against the LibTorch reference, "
            "or the tolerance is far too loose.\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    low = (stdout + stderr).lower()
    if not success and "failed" in low:
        print("run-dummy ok — driver correctly rejected the no-op kernel.")
        return
    _fail(
        "Running driver.cpp with a dummy kernel crashed or failed to produce 'failed'.\n"
        "The driver should print 'failed' cleanly for a no-op kernel. Fix the driver "
        "(pre-init gpu_output to zeros, run torch::allclose, print passed/failed).\n"
        f"stdout:\n{stdout}\nstderr:\n{stderr}"
    )


def _check_real_result(stdout: str, stderr: str, success: bool) -> None:
    if success and "passed" in stdout.lower():
        print("run-real ok — final_cuda.cu verified against LibTorch reference.")
        print(stdout)
        return
    _fail(
        "Real kernel failed verification (driver did not print 'passed').\n"
        f"stdout:\n{stdout}\nstderr:\n{stderr}\n\n"
        "Fix the kernel so its output matches the torch reference within tolerance."
    )


async def _async_run_dummy(folder: Path, gpu: GPUType, timeout: int) -> None:
    dummy_fp = folder / "dummy_final_cuda.cu"
    if not dummy_fp.exists():
        _fail(f"dummy_final_cuda.cu not found — run `validate` first. ({dummy_fp})")
    stdout, stderr, success = await _run_once(folder, "dummy_final_cuda.cu", gpu, timeout)
    _check_dummy_result(stdout, stderr, success)


async def _async_run_real(folder: Path, gpu: GPUType, timeout: int) -> None:
    stdout, stderr, success = await _run_once(folder, "final_cuda.cu", gpu, timeout)
    _check_real_result(stdout, stderr, success)


def cmd_run_dummy(args) -> None:
    folder = Path(args.folder).resolve()
    asyncio.run(_async_run_dummy(folder, GPUType(args.gpu), args.timeout))


def cmd_run_real(args) -> None:
    folder = Path(args.folder).resolve()
    asyncio.run(_async_run_real(folder, GPUType(args.gpu), args.timeout))


async def _async_full(folder: Path, gpu: GPUType, timeout: int) -> None:
    # Single event loop — the shared aiohttp session cached by TCPClient
    # breaks if we straddle multiple asyncio.run() calls.
    await _async_run_dummy(folder, gpu, timeout)
    await _async_run_real(folder, gpu, timeout)


def cmd_full(args) -> None:
    cmd_validate(args)
    folder = Path(args.folder).resolve()
    asyncio.run(_async_full(folder, GPUType(args.gpu), args.timeout))
    print("full ok — both dummy and real verification steps passed.")


def cmd_prompt(args) -> None:
    ref_path = Path(args.reference).resolve()
    if not ref_path.exists():
        _fail(f"reference file not found: {ref_path}")
    ref_code = ref_path.read_text()
    sys_prompt = build_system_prompt(args.precision, args.problem_class)
    user_prompt = build_user_prompt(ref_code, args.precision, args.problem_class)
    print("=" * 60)
    print("SYSTEM PROMPT")
    print("=" * 60)
    print(sys_prompt)
    print("=" * 60)
    print("USER PROMPT")
    print("=" * 60)
    print(user_prompt)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def _common(sp):
        sp.add_argument("--folder", required=True, help="Folder with driver.cpp + final_cuda.cu")
        sp.add_argument("--gpu", default="l40s",
                        help="GPUType value (e.g. l40s, h100, a100). Default: l40s.")
        sp.add_argument("--timeout", type=int, default=600)

    sp = sub.add_parser("validate", help="Static rules + contract check")
    sp.add_argument("--folder", required=True)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("run-dummy", help="Run driver.cpp with a no-op kernel (expect 'failed')")
    _common(sp)
    sp.set_defaults(func=cmd_run_dummy)

    sp = sub.add_parser("run-real", help="Run driver.cpp with final_cuda.cu (expect 'passed')")
    _common(sp)
    sp.set_defaults(func=cmd_run_real)

    sp = sub.add_parser("full", help="validate -> run-dummy -> run-real")
    _common(sp)
    sp.set_defaults(func=cmd_full)

    sp = sub.add_parser("prompt", help="Print system+user prompts for a reference.py")
    sp.add_argument("--reference", required=True)
    sp.add_argument("--precision", default="fp16", choices=["fp16", "fp32", "bf16"])
    sp.add_argument(
        "--problem-class",
        default="l1",
        choices=["l1", "l2", "deep"],
        help=(
            "Tolerance tier. l1: single op (default). l2: composite forward "
            "(<=5 chained matmuls). deep: backward / sol-level2 / heavy chain."
        ),
    )
    sp.set_defaults(func=cmd_prompt)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
