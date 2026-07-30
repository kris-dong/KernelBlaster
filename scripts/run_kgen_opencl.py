"""
Standalone script to run OpenCL kgen (translate PyTorch reference.py → driver.c + kernel.cl)
without the RL optimization step.

Usage:
    python scripts/run_kgen_opencl.py --subset L1 --problem-numbers 1-5
    python scripts/run_kgen_opencl.py --subset L1 --problem-numbers 1 --model anthropic.claude-opus-4-6-v1
    python scripts/run_kgen_opencl.py --single-file-path data/KernelBench/KernelBench/level1/23_Softmax.py
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import signal
from pathlib import Path
from loguru import logger

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

# SSL cert bootstrap (same as run_RL.py)
if not os.environ.get("SSL_CERT_FILE") and not os.environ.get("REQUESTS_CA_BUNDLE"):
    try:
        import certifi
        _cert_file = certifi.where()
    except Exception:
        _cert_file = None
    if _cert_file and Path(_cert_file).exists():
        os.environ["SSL_CERT_FILE"] = _cert_file
if os.environ.get("SSL_CERT_FILE") and not os.environ.get("REQUESTS_CA_BUNDLE"):
    os.environ["REQUESTS_CA_BUNDLE"] = os.environ["SSL_CERT_FILE"]
elif not os.environ.get("SSL_CERT_FILE") and Path("/etc/ssl/certs/ca-certificates.crt").exists():
    os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"
    os.environ["REQUESTS_CA_BUNDLE"] = os.environ["SSL_CERT_FILE"]

from src.kernelblaster.config import config, GPUType
from src.kernelblaster.resources import OpenCLCompileServer, AdrenoGPUServer
from src.kernelblaster.integration.kgen_opencl import run_kgen_opencl_pipeline

OPENCL_COMPILE_SERVER = None
ADRENO_GPU_SERVER = None
CLEANUP_IN_PROGRESS = False


def cleanup_servers():
    global CLEANUP_IN_PROGRESS
    if CLEANUP_IN_PROGRESS:
        return
    CLEANUP_IN_PROGRESS = True
    try:
        import threading
        for name, server in [
            ("OpenCL compiler", OPENCL_COMPILE_SERVER),
            ("Adreno GPU", ADRENO_GPU_SERVER),
        ]:
            if server is not None:
                logger.info(f"Cleaning up {name} server...")
                t = threading.Thread(target=server.cleanup)
                t.daemon = True
                t.start()
                t.join(timeout=5.0)
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
    finally:
        CLEANUP_IN_PROGRESS = False


def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, cleaning up...")
    cleanup_servers()
    sys.exit(0)


# Precision-injection helper — Item 2 cleanup Phase 2b: dropped the
# ``_inject_precision`` local alias in favour of the canonical helper
# in ``data.sources._precision``. Callers use ``inject_precision`` here.
from data.sources._precision import inject_precision


def collect_problems(args) -> list["Problem"]:
    """Composite walker for kgen's two input sources.

    Yields ``data.sources.Problem`` objects with ``reference_code``
    inlined (precision-injected). Two disk trees, tried in order:

      1. ``data/benchmark-opencl/<L*>/<prob>/reference.py`` — the
         curated OpenCL bench (also the base for
         ``KernelBenchOpenCLSource``). Problems here have hand-authored
         driver.c + kernel.cl artifacts alongside the reference.
      2. ``data/KernelBench/KernelBench/<level*>/<NNN>_*.py`` — the
         legacy KernelBench pytorch corpus, used as a fallback for
         problems not yet ported into ``benchmark-opencl``.

    Deduped by problem number: benchmark-opencl wins when both trees
    contain a given problem. ``--single-file-path`` short-circuits
    everything via :func:`data.sources.iter_problems_for_args` and
    returns a single ``source="custom"`` Problem.

    Migrated from the pre-cleanup dict-yielding walker in Item 2
    cleanup Phase 2b — the composite-fallback semantics are kept
    intact locally rather than pushed into a `CompositeSource`,
    because this two-tree fallback is script-specific (kgen ingests
    torch references; the fallback tree isn't a general concern for
    other consumers).
    """
    from data.sources import Problem, iter_problems_for_args, parse_problem_numbers

    if args.single_file_path:
        # Reuse the canonical single-file-path handling — same
        # source="custom" Problem shape as run_RL.py.
        args_local = argparse.Namespace(
            **{**vars(args), "dataset": "kernelbench"},  # dataset unused but required
        )
        return iter_problems_for_args(args_local)

    _nums = parse_problem_numbers(args.problem_numbers)
    problem_numbers = set(_nums) if _nums is not None else None

    subset = args.subset or "L1"
    level_map = {
        "L1": "level1", "L2": "level2", "L3": "level3",
        "level1": "level1", "level2": "level2", "level3": "level3",
    }
    kb_level = level_map.get(subset)
    if not kb_level:
        logger.error(f"Unsupported subset for kgen: {subset}")
        return []

    # Normalize to L1/L2/L3 for output paths — the OpenCL bench uses
    # the L* spelling; problem IDs downstream match this.
    out_level = {"level1": "L1", "level2": "L2", "level3": "L3"}.get(kb_level, subset)

    opencl_bench = ROOT_DIR / "data" / "benchmark-opencl" / out_level
    kb_dir = ROOT_DIR / "data" / "KernelBench" / "KernelBench" / kb_level

    def _in_range(num: int) -> bool:
        if problem_numbers is not None and num not in problem_numbers:
            return False
        if args.start and num < args.start:
            return False
        if args.end and num > args.end:
            return False
        return True

    problems: list[Problem] = []
    seen: set[int] = set()

    # Tier 1: benchmark-opencl reference.py.
    if opencl_bench.is_dir():
        for prob_dir in sorted(opencl_bench.iterdir()):
            if not prob_dir.is_dir():
                continue
            ref_py = prob_dir / "reference.py"
            if not ref_py.exists():
                continue
            try:
                num = int(prob_dir.name.split("_", 1)[0])
            except ValueError:
                continue
            if not _in_range(num):
                continue
            problems.append(Problem(
                id=f"kernelbench-opencl:{out_level}/{prob_dir.name}",
                source="kernelbench-opencl",
                tier=out_level,
                problem_num=num,
                problem_name=prob_dir.name,
                curated_artifacts={"reference_py": ref_py},
                reference_code=inject_precision(ref_py.read_text(), args.precision),
                metadata={"precision": args.precision},
                backends_supported=frozenset({"opencl"}),
            ))
            seen.add(num)

    # Tier 2: KernelBench torch files not covered by Tier 1.
    if kb_dir.is_dir():
        for py_file in sorted(kb_dir.glob("*.py")):
            try:
                num = int(py_file.name.split("_", 1)[0])
            except ValueError:
                continue
            if num in seen or not _in_range(num):
                continue
            problem_name = py_file.stem
            problems.append(Problem(
                id=f"kernelbench:{out_level}/{problem_name}",
                source="kernelbench",
                tier=out_level,
                problem_num=num,
                problem_name=problem_name,
                curated_artifacts={"reference_py": py_file},
                reference_code=inject_precision(py_file.read_text(), args.precision),
                metadata={"precision": args.precision},
                backends_supported=frozenset({"cuda", "opencl"}),
            ))

    problems.sort(key=lambda p: p.id)
    return problems


async def async_main():
    parser = argparse.ArgumentParser(
        description="Run OpenCL kgen: translate PyTorch → driver.c + kernel.cl"
    )
    parser.add_argument("--model", type=str, default=config.MODEL)
    parser.add_argument("--gpu", type=str, default="adreno650",
                        choices=[g.value for g in GPUType])
    parser.add_argument("--subset", type=str, default="L1",
                        choices=["L1", "L2", "L3", "level1", "level2", "level3"])
    parser.add_argument("--problem-numbers", type=str, default=None,
                        help="e.g. '1,2,3' or '1-10'")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--single-file-path", type=Path, default=None)
    parser.add_argument("--precision", type=str, default="fp16",
                        choices=["fp16", "fp32"],
                        help="Data precision for generated driver/kernel (default: fp16)")
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--experiment-name", type=str, default="kgen_opencl")
    parser.add_argument("--board-host", type=str, default=None)
    parser.add_argument("--compiler-port", type=int, default=None)
    parser.add_argument("--gpu-port", type=int, default=None)
    parser.add_argument("--gpu-server-url", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=240,
                        help="Timeout per problem in minutes")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--retry", action="store_true")
    args = parser.parse_args()

    gpu_type = GPUType(args.gpu)

    # Update config model
    if args.model != config.MODEL:
        config.MODEL = args.model

    model_name = (
        config.MODEL.replace("llmgateway/", "")
        .replace("eos/", "")
        .replace("chipnemo/", "")
        .replace("azure/", "")
        .replace("/", "-")
        .lower()
    )
    OUT_DIR = ROOT_DIR / "out" / "kernelbench-opencl" / args.experiment_name / model_name

    log_file = OUT_DIR / "run.log"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.configure(
        handlers=[
            dict(sink=sys.stderr, format=config.CUSTOM_LOGGER_FORMAT,
                 level=config.LOG_LEVEL, colorize=True, backtrace=True, diagnose=True),
            dict(sink=log_file, format=config.CUSTOM_LOGGER_FORMAT,
                 level=config.LOG_LEVEL, colorize=False, backtrace=True, diagnose=True),
        ],
        extra=dict(agent_name="kgen_opencl", attempt_id=None, task_id=None),
    )
    logger.info(f"Logging to {log_file}")

    # Start servers
    global OPENCL_COMPILE_SERVER, ADRENO_GPU_SERVER
    board_host = args.board_host

    if args.gpu_server_url:
        logger.info(f"Using existing Adreno GPU server at {args.gpu_server_url}")
        os.environ["KERNELBLASTER_ADRENO_GPU_SERVER_URL"] = args.gpu_server_url
    else:
        ADRENO_GPU_SERVER = AdrenoGPUServer(
            logger, OUT_DIR, board_host=board_host, port=args.gpu_port
        )
        ADRENO_GPU_SERVER.wait_for_connection()
        os.environ["KERNELBLASTER_ADRENO_GPU_SERVER_URL"] = ADRENO_GPU_SERVER.url
        logger.info(f"Adreno GPU server started at {ADRENO_GPU_SERVER.url}")

    OPENCL_COMPILE_SERVER = OpenCLCompileServer(
        logger, OUT_DIR, board_host=board_host, port=args.compiler_port
    )
    OPENCL_COMPILE_SERVER.wait_for_connection()
    os.environ["KERNELBLASTER_OPENCL_COMPILE_SERVER_URL"] = OPENCL_COMPILE_SERVER.url
    logger.info(f"OpenCL compile server started at {OPENCL_COMPILE_SERVER.url}")

    # Collect problems
    problems = collect_problems(args)
    if not problems:
        logger.error("No problems found. Check --subset, --problem-numbers, or --single-file-path.")
        return

    logger.info(f"Processing {len(problems)} problems")
    config.print_config(logger)

    semaphore = asyncio.Semaphore(args.concurrency)
    succeeded = 0
    failed = 0

    async def process_one(problem):
        nonlocal succeeded, failed
        # Item 2 cleanup Phase 2b: ``problem`` is a Problem now.
        # ``filesystem_id`` keeps the pre-migration folder layout
        # (drops the ``<source>:`` prefix from ``.id``).
        problem_id = problem.id
        folder = OUT_DIR / problem.filesystem_id

        job_logger = logger.bind(problem_id=problem_id)
        job_logger_id = job_logger.add(
            folder / "run.log",
            level=config.LOG_LEVEL,
            backtrace=True,
            diagnose=True,
            format=config.CUSTOM_LOGGER_FORMAT,
            filter=lambda record, pid=problem_id: record["extra"].get("problem_id") == pid,
        )

        async with semaphore:
            try:
                ok = await asyncio.wait_for(
                    run_kgen_opencl_pipeline(
                        folder=folder,
                        reference_code=problem.reference_code,
                        logger=job_logger,
                        model=config.MODEL,
                        gpu=gpu_type,
                        max_attempts=args.max_attempts,
                        retry_failed=args.retry,
                        precision=args.precision,
                    ),
                    timeout=args.timeout * 60,
                )
                if ok:
                    job_logger.info(f"kgen succeeded for {problem_id}")
                    succeeded += 1
                else:
                    job_logger.error(f"kgen failed for {problem_id}")
                    failed += 1
            except asyncio.TimeoutError:
                job_logger.error(f"kgen timed out for {problem_id}")
                failed += 1
            except Exception as e:
                job_logger.exception(f"kgen error for {problem_id}: {e}")
                failed += 1

        job_logger.remove(job_logger_id)

    tasks = [asyncio.create_task(process_one(entry)) for entry in problems]
    await asyncio.gather(*tasks)

    logger.info(f"Done: {succeeded} succeeded, {failed} failed out of {len(problems)}")


def main():
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.error("KeyboardInterrupt, cleaning up...")
    finally:
        cleanup_servers()


if __name__ == "__main__":
    main()
