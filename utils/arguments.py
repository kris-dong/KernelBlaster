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
import argparse
from pathlib import Path

from src.kernelblaster.config import WorkflowConfig, config, GPUType

__all__ = [
    "add_common_arguments",
    "validate_common_arguments",
    "create_workflow_config",
]


def add_common_arguments(parser: argparse.ArgumentParser):
    parser = parser
    parser.add_argument(
        "--timeout",
        type=int,
        default=240,
        help="Timeout per problem in minutes (default: 240 minutes = 4 hours)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use for generation",
        default=config.MODEL,
    )
    parser.add_argument("--cuda", action="store_true", help="Run CUDA generation")
    parser.add_argument("--cuda-bench", action="store_true", help="Run CUDA benchmark")
    parser.add_argument(
        "--cuda-perf",
        action="store_true",
        help="Run the NCU agent to optimize the CUDA code",
    )
    parser.add_argument(
        "--kgen-opencl",
        action="store_true",
        help="Run the OpenCL kgen agent to translate PyTorch reference into driver.c + kernel.cl",
    )
    parser.add_argument(
        "--opencl-perf",
        action="store_true",
        help="Run the OpenCL RL agent to optimize kernels for Adreno GPUs",
    )
    parser.add_argument(
        "--board-host",
        type=str,
        default=None,
        help="SSH target for Adreno board (e.g., root@192.0.2.201)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark agent to augment generated code with benchmarking harness.",
    )
    parser.add_argument("--retry", action="store_true", help="Retry failed agents")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="experiment",
        help="Stores output in ./out/<dataset>/<exp_name>/<model>",
    )
    parser.add_argument(
        # Item 2, Phase 8: ``--source`` is the new preferred spelling
        # (aligns with the ``data.sources.get_source(name)`` factory).
        # ``--dataset`` retains as an alias for one release so existing
        # scripts + docs keep working without churn. Value lands in
        # ``args.dataset`` (unchanged).
        "--dataset", "--source",
        type=str,
        default="kernelbench-cuda",
        dest="dataset",
        choices=[
            "kernelbench",
            "kernelbench-cuda",
            "kernelbench-opencl",
        ],
    )
    parser.add_argument(
        "--problem-numbers",
        type=str,
        default=None,
        help="Problem numbers to run. Supports comma-separated numbers (e.g., '8,9,10') and ranges (e.g., '8-60' or '8-20,25,30-35')",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Starting problem number (inclusive)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending problem number (inclusive)",
    )
    parser.add_argument(
        "--single-file-path",
        type=Path,
        default=None,
        help="Path to a single problem. Should be a .py file containing the pytorch grparserh.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["fp32", "fp16", "bf16"],
        help="Run on a subset of the dataset",
    )
    parser.add_argument(
        # Item 2, Phase 8: ``--source-tier`` is the new preferred spelling.
        # ``--subset`` retains as an alias. Value lands in ``args.subset``.
        "--subset", "--source-tier",
        type=str,
        default=None,
        dest="subset",
        choices=[
            "cub",
            "thrust",
            "cuda",
            "L1",
            "L2",
            "L3",
            "level1",
            "level2",
            "level3",
            "sol-level1",
            "sol-level2",
            "A",
            "B",
            "H",
        ],
        help="Run on a subset of the dataset",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default=None,
        choices=["train", "test"],
        help="Run on a subset of the dataset",
    )
    try:
        _default_gpu = GPUType.current().value
    except (ValueError, Exception):
        _default_gpu = None
    parser.add_argument(
        "--gpu",
        type=str,
        default=_default_gpu,
        choices=[gpu.value for gpu in GPUType],
        help="GPU to use for generation.",
    )
    
    # RL optimization arguments
    parser.add_argument(
        "--use-rl",
        action="store_true",
        help="Use RL-based optimization instead of regular NCU optimization",
    )
    parser.add_argument(
        "--rl-iterations",
        type=int,
        default=10,
        help="Number of RL iterations to run (default: 10)",
    )
    parser.add_argument(
        "--rl-rollout-steps",
        type=int,
        default=5,
        help="Number of rollout steps per RL iteration (default: 5)",
    )
    parser.add_argument(
        "--rl-buffer-size",
        type=int,
        default=100,
        help="Size of RL replay buffer (default: 100)",
    )
    parser.add_argument(
        "--rl-update-frequency",
        type=int,
        default=3,
        help="Frequency of RL database updates (default: 3)",
    )
    parser.add_argument(
        "--compile-server-port",
        type=int,
        default=None,
        help="Port for the compile server (default: auto-assign starting from 2001)",
    )
    parser.add_argument(
        "--gpu-server-port",
        type=int,
        default=None,
        help="Port for the GPU server (default: auto-assign starting from 2002)",
    )
    parser.add_argument(
        "--no-baseline-optimization",
        action="store_true",
        help="Disable baseline optimization mode (allows running without --cuda-perf)",
    )


def validate_common_arguments(parser, args):
    if args.cuda_perf and not args.cuda:
        parser.error("--cuda-perf cannot be specified without --cuda")
    if args.dataset == "kernelbench" and not args.precision:
        parser.error("--precision is required for --dataset kernelbench")
    if args.benchmark and not args.cuda:
        parser.error(
            "--benchmark cannot be specified without --cuda"
        )
    if getattr(args, "opencl_perf", False) and args.dataset != "kernelbench-opencl":
        parser.error("--opencl-perf requires --dataset kernelbench-opencl")


def create_workflow_config(args: argparse.Namespace) -> WorkflowConfig:
    if args.gpu is None:
        try:
            args.gpu = GPUType.current().value
        except (ValueError, Exception):
            raise RuntimeError(
                "No GPU detected and --gpu not specified. "
                "Please pass --gpu (e.g., --gpu adreno650 or --gpu l40s)."
            )

    # Update config.MODEL if model argument is provided (so config print shows correct model)
    if args.model != config.MODEL:
        config.MODEL = args.model

    # Enable RL optimization if requested (CUDA or OpenCL)
    if (hasattr(args, 'use_rl') and args.use_rl) or getattr(args, 'opencl_perf', False):
        import os
        os.environ["KERNELBLASTER_OPT_RL_NCU"] = "1"
        from src.kernelblaster.config.config import ExperimentalFeatures
        ExperimentalFeatures.OPT_RL_NCU = True

    run_kgen = bool(getattr(args, "kgen", False))
    if getattr(args, "use_rl", False) and not getattr(args, "no_kgen", False):
        run_kgen = True
    if getattr(args, "no_kgen", False):
        run_kgen = False

    workflow_config = WorkflowConfig(
        model=args.model,
        run_cuda=args.cuda,
        run_cuda_perf=args.cuda_perf,
        run_cuda_bench=args.cuda and args.benchmark,
        run_cuda_perf_bench=args.cuda_perf and args.benchmark,
        retry_failed=args.retry,
        gpu=GPUType(args.gpu),
        run_kgen_opencl=getattr(args, "kgen_opencl", False),
        run_opencl_perf=getattr(args, "opencl_perf", False),
        use_baseline_optimization=not args.no_baseline_optimization,
        run_kgen=run_kgen,
        kgen_max_attempts=getattr(args, "kgen_max_attempts", 8),
        kgen_num_coders=getattr(args, "kgen_num_coders", 4),
        kgen_llm_client=getattr(args, "kgen_llm_client", "openai"),
        kgen_stream=getattr(args, "kgen_stream", False),
        kgen_retry_on_llm_error=getattr(args, "kgen_retry_on_llm_error", False),
        cudacoder_root=getattr(args, "cudacoder_root", None),
    )

    # Add RL parameters to workflow config
    if (hasattr(args, 'use_rl') and args.use_rl) or getattr(args, 'opencl_perf', False):
        workflow_config.rl_iterations = args.rl_iterations
        workflow_config.rl_rollout_steps = args.rl_rollout_steps
        workflow_config.rl_buffer_size = args.rl_buffer_size
        workflow_config.rl_update_frequency = args.rl_update_frequency

    workflow_config.validate()
    return workflow_config
