#!/usr/bin/env python3
"""
Wrapper script to run the NsysReProfileAgent for profiling existing
success_rl_optimization.cu files with Nsight Systems.

Unlike the NCU-based reprofiler (run_reprofile.py), this measures wall-clock
GPU time from first kernel start to last kernel end, capturing inter-kernel
overhead such as memory copies, host code, and kernel launch latency.

Usage:
    python run_reprofile_nsys.py --base-dir <directory> [options]

Example:
    python run_reprofile_nsys.py \
        --base-dir out/kernelbench-cuda/fp16/sol-level1/kernelblaster-test/anthropic.claude-opus-4-6-v1/sol-level1 \
        --gpu L40S --num-runs 5

Environment Variables:
    COMPILE_SERVER_URL: URL of compilation server (if not set, will auto-start)
    GPU_SERVER_URL_L40S: URL of GPU server for L40S (if not set, will auto-start)
    GPU_SERVER_URL_A6000: URL of GPU server for A6000 (if not set, will auto-start)
    GPU_SERVER_URL_H100: URL of GPU server for H100 (if not set, will auto-start)
    GPU_SERVER_URL_A100: URL of GPU server for A100 (if not set, will auto-start)
"""
import argparse
import asyncio
import sys
import os
from pathlib import Path
from loguru import logger
from contextlib import contextmanager

# Add src to path so that `import kernelblaster` resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kernelblaster.agents.reprofile_nsys import NsysReProfileAgent
from kernelblaster.config import GPUType, config
from kernelblaster.servers.management import (
    initialize_compiler_server,
    initialize_gpu_server,
    test_server_connection,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile success_rl_optimization.cu files with Nsight Systems "
                    "(wall-clock GPU span including inter-kernel overhead)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory to search for success_rl_optimization.cu files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for profiling results (default: alongside each file)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="L40S",
        choices=["L40S", "A6000", "H100", "A100"],
        help="GPU type to use for profiling (default: L40S)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of profiling runs per file; best (shortest) span is kept (default: 5)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel profiling jobs (default: 1, sequential)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout per file in seconds (default: 3600)",
    )
    parser.add_argument(
        "--problem-numbers",
        type=str,
        default=None,
        help="Comma-separated list of problem numbers to profile (e.g., '8,10,25'). "
             "If not specified, all problems will be profiled.",
    )
    parser.add_argument(
        "--profile-init",
        action="store_true",
        help="Profile init.cu (initial code) instead of success_rl_optimization.cu",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Force re-profiling all files, ignoring cached results from previous runs",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def get_gpu_type(gpu_str: str) -> GPUType:
    gpu_map = {
        "L40S": GPUType.L40S,
        "A6000": GPUType.A6000,
        "H100": GPUType.H100,
        "A100": GPUType.A100,
    }
    return gpu_map[gpu_str.upper()]


@contextmanager
def setup_servers(gpu: GPUType, log_dir: Path):
    """Context manager to set up and tear down servers."""
    compile_server_process = None
    gpu_server_process = None
    log_file = None

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(log_dir / "server.log", "a")

        compile_server_url = os.getenv("COMPILE_SERVER_URL")
        gpu_server_url = os.getenv(f"GPU_SERVER_URL_{gpu.value.upper()}")

        # Compilation server
        if compile_server_url is None:
            logger.info("Starting compilation server...")
            compile_server_process, compile_server_url = initialize_compiler_server(
                log_file=log_file,
                compile_server_url=None,
                artifacts_dir=Path("/tmp/kernelblaster/artifacts"),
                port=None,
            )
            if compile_server_process:
                if not test_server_connection(compile_server_process, compile_server_url, timeout=30):
                    raise RuntimeError(f"Compilation server failed to start at {compile_server_url}")
                config.set_compile_server_url(compile_server_url)
                logger.info(f"Compilation server started at {compile_server_url}")
        else:
            logger.info(f"Using existing compilation server at {compile_server_url}")

        # GPU server
        if gpu_server_url is None:
            logger.info(f"Starting GPU server for {gpu.value}...")
            gpu_server_process, gpu_server_url = initialize_gpu_server(
                log_file=log_file,
                gpu=gpu,
                port=None,
            )
            if gpu_server_process:
                if not test_server_connection(gpu_server_process, gpu_server_url, timeout=30):
                    raise RuntimeError(f"GPU server failed to start at {gpu_server_url}")
                logger.info(f"GPU server started at {gpu_server_url}")
        else:
            logger.info(f"Using existing GPU server at {gpu_server_url}")

        yield

    finally:
        for name, proc in [("compilation", compile_server_process), ("GPU", gpu_server_process)]:
            if proc:
                logger.info(f"Terminating {name} server...")
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        if log_file:
            log_file.close()


async def main():
    args = parse_args()

    logger.remove()
    logger.add(
        sys.stderr,
        level=args.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        logger.error(f"Base directory does not exist: {base_dir}")
        sys.exit(1)

    try:
        gpu = get_gpu_type(args.gpu)
    except KeyError:
        logger.error(f"Invalid GPU type: {args.gpu}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    problem_numbers = None
    if args.problem_numbers:
        problem_numbers = [pn.strip() for pn in args.problem_numbers.split(",")]

    log_dir = base_dir / "nsys_reprofile_logs"

    logger.info("=" * 60)
    logger.info("NsysReProfileAgent - Wall-clock GPU span profiling")
    logger.info("=" * 60)
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Output directory: {output_dir or 'alongside each file'}")
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Num runs (best-of): {args.num_runs}")
    logger.info(f"Profile init.cu: {args.profile_init}")
    logger.info(f"Fresh (ignore cache): {args.fresh}")
    logger.info(f"Max workers: {args.max_workers}")
    logger.info(f"Timeout: {args.timeout}s")
    if problem_numbers:
        logger.info(f"Problem numbers filter: {problem_numbers}")
    logger.info("=" * 60)

    with setup_servers(gpu, log_dir):
        agent = NsysReProfileAgent(
            base_folder=base_dir,
            gpu=gpu,
            logger=logger,
            timeout=args.timeout,
            num_runs=args.num_runs,
            profile_init=args.profile_init,
        )

        try:
            results = await agent.profile_all(
                base_directory=base_dir,
                output_base=output_dir,
                max_workers=args.max_workers,
                problem_numbers=problem_numbers,
                use_cached=not args.fresh,
            )

            logger.info("=" * 60)
            logger.info("Profiling Summary")
            logger.info("=" * 60)

            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]

            logger.info(f"Total files processed: {len(results)}")
            logger.info(f"Successful: {len(successful)}")
            logger.info(f"Failed: {len(failed)}")

            if failed:
                logger.warning("Failed files:")
                for r in failed:
                    logger.warning(f"  - {r.success_file}: {r.error}")

            if successful:
                times = [r.gpu_time_ns for r in successful if r.gpu_time_ns > 0]
                if times:
                    logger.info(f"GPU span range: {min(times):,} - {max(times):,} ns")
                    logger.info(f"Average GPU span: {sum(times) / len(times):,.0f} ns")

                logger.info("")
                logger.info(f"{'Problem':<55} {'GPU Span (ns)':>15} {'Kernels':>8}")
                logger.info("-" * 80)
                for r in sorted(successful, key=lambda x: x.success_file):
                    pname = Path(r.success_file).parent.parent.name
                    if pname == "rl_ncu":
                        pname = Path(r.success_file).parent.parent.parent.name
                    logger.info(f"{pname:<55} {r.gpu_time_ns:>15,} {r.kernel_count:>8}")

            logger.info("=" * 60)

            if failed:
                sys.exit(1)

        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
            sys.exit(130)
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
