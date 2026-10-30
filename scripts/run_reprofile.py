# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python3
"""
Wrapper script to run the ReProfileAgent for profiling existing success_rl_optimization.cu files.

This script automatically starts compilation and GPU servers if they're not already configured
via environment variables.

Usage:
    python run_reprofile.py --base-dir <directory> [options]

Example:
    python run_reprofile.py --base-dir out/kernelbench/fp16/rl_baseline_debug_computelab_l40s_level2_best --gpu L40S

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
import signal
import subprocess
import time
from pathlib import Path
from loguru import logger
from contextlib import contextmanager

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from kernelagent.agents.reprofile import ReProfileAgent
from kernelagent.config import GPUType, config
from kernelagent.servers.management import (
    initialize_compiler_server,
    initialize_gpu_server,
    test_server_connection,
    find_free_port,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile existing success_rl_optimization.cu files"
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
        "--cycles-only",
        action="store_true",
        help="Only collect cycles, skip detailed profiling (faster)",
    )
    parser.add_argument(
        "--no-detailed",
        action="store_true",
        help="Skip detailed profiling (source annotation, etc.)",
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
        "--no-skip-existing",
        action="store_true",
        help="Re-profile files even if they already have profiling results",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
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
    
    return parser.parse_args()


def get_gpu_type(gpu_str: str) -> GPUType:
    """Convert string to GPUType enum."""
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
        # Create log directory
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(log_dir / "server.log", "a")
        
        # Check if servers are already configured via environment
        compile_server_url = os.getenv("COMPILE_SERVER_URL")
        gpu_server_url = os.getenv(f"GPU_SERVER_URL_{gpu.value.upper()}")
        
        # Set up compilation server
        if compile_server_url is None:
            logger.info("Starting compilation server...")
            compile_server_process, compile_server_url = initialize_compiler_server(
                log_file=log_file,
                compile_server_url=None,
                cudnn_frontend_includes_dir=Path(
                    os.getenv("CUDNN_FRONTEND_INCLUDES_DIR", "/workspace/cudnn-frontend/include")
                ),
                cudnn_backend_includes_dir=Path(
                    os.getenv("CUDNN_BACKEND_INCLUDES_DIR", "/usr/include")
                ),
                cudnn_backend_lib_dir=Path(
                    os.getenv("CUDNN_BACKEND_LIB_DIR", "/usr/lib/x86_64-linux-gnu")
                ),
                cutlass_dir=Path(os.getenv("CUTLASS_DIR", "/usr/include/cutlass")),
                artifacts_dir=Path("/tmp/kernelagent/artifacts"),
                port=None,  # Auto-assign port
            )
            if compile_server_process:
                # Wait for server to be ready
                if not test_server_connection(compile_server_process, compile_server_url, timeout=30):
                    raise RuntimeError(f"Compilation server failed to start at {compile_server_url}")
                config.set_compile_server_url(compile_server_url)
                logger.info(f"✅ Compilation server started at {compile_server_url}")
        else:
            logger.info(f"Using existing compilation server at {compile_server_url}")
        
        # Set up GPU server
        if gpu_server_url is None:
            logger.info(f"Starting GPU server for {gpu.value}...")
            gpu_server_process, gpu_server_url = initialize_gpu_server(
                log_file=log_file,
                gpu=gpu,
                port=None,  # Auto-assign port
            )
            if gpu_server_process:
                # Wait for server to be ready
                if not test_server_connection(gpu_server_process, gpu_server_url, timeout=30):
                    raise RuntimeError(f"GPU server failed to start at {gpu_server_url}")
                logger.info(f"✅ GPU server started at {gpu_server_url}")
        else:
            logger.info(f"Using existing GPU server at {gpu_server_url}")
        
        yield
        
    finally:
        # Cleanup: terminate servers we started
        if compile_server_process:
            logger.info("Terminating compilation server...")
            try:
                compile_server_process.terminate()
                compile_server_process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error terminating compilation server: {e}")
                try:
                    compile_server_process.kill()
                except Exception:
                    pass
        
        if gpu_server_process:
            logger.info("Terminating GPU server...")
            try:
                gpu_server_process.terminate()
                gpu_server_process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error terminating GPU server: {e}")
                try:
                    gpu_server_process.kill()
                except Exception:
                    pass
        
        if log_file:
            log_file.close()


async def main():
    args = parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level=args.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )
    
    # Validate base directory
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        logger.error(f"Base directory does not exist: {base_dir}")
        sys.exit(1)
    
    # Get GPU type
    try:
        gpu = get_gpu_type(args.gpu)
    except KeyError:
        logger.error(f"Invalid GPU type: {args.gpu}")
        sys.exit(1)
    
    # Create output directory if specified
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up log directory for servers
    log_dir = base_dir / "reprofile_logs"
    
    logger.info("=" * 60)
    logger.info("ReProfileAgent - Profiling success_rl_optimization.cu files")
    logger.info("=" * 60)
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Output directory: {output_dir or 'alongside each file'}")
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Cycles only: {args.cycles_only}")
    logger.info(f"Detailed profiling: {not args.no_detailed and not args.cycles_only}")
    logger.info(f"Profile init.cu: {args.profile_init}")
    logger.info(f"Max workers: {args.max_workers}")
    logger.info(f"Timeout: {args.timeout}s")
    logger.info(f"Skip existing: {not args.no_skip_existing}")
    
    # Parse problem numbers if provided
    problem_numbers = None
    if args.problem_numbers:
        problem_numbers = [pn.strip() for pn in args.problem_numbers.split(',')]
        logger.info(f"Problem numbers filter: {problem_numbers}")
    else:
        logger.info("Problem numbers filter: None (all problems)")
    
    logger.info("=" * 60)
    
    # Set up servers (will use existing if env vars are set, otherwise start new ones)
    with setup_servers(gpu, log_dir):
        # Create agent
        agent = ReProfileAgent(
            base_folder=base_dir,
            gpu=gpu,
            logger=logger,
            timeout=args.timeout,
            cycles_only=args.cycles_only,
            detailed_profiling=not args.no_detailed and not args.cycles_only,
            profile_init=args.profile_init,
        )
        
        # Run profiling
        try:
            results = await agent.profile_all(
                base_directory=base_dir,
                output_base=output_dir,
                max_workers=args.max_workers,
                skip_existing=not args.no_skip_existing,
                problem_numbers=problem_numbers,
            )
            
            # Print summary
            logger.info("=" * 60)
            logger.info("Profiling Summary")
            logger.info("=" * 60)
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            logger.info(f"Total files processed: {len(results)}")
            logger.info(f"Successful: {successful}")
            logger.info(f"Failed: {failed}")
            
            if failed > 0:
                logger.warning("Failed files:")
                for result in results:
                    if not result.success:
                        logger.warning(f"  - {result.success_file}: {result.error}")
            
            # Print cycles summary
            if successful > 0:
                cycles_list = [r.cycles for r in results if r.success and r.cycles > 0]
                if cycles_list:
                    logger.info(f"Cycles range: {min(cycles_list)} - {max(cycles_list)}")
                    logger.info(f"Average cycles: {sum(cycles_list) / len(cycles_list):.0f}")
            
            logger.info("=" * 60)
            
            # Exit with error code if any failed
            if failed > 0:
                sys.exit(1)
            
        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
            sys.exit(130)
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
