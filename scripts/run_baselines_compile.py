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
import argparse
import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple, Optional

import subprocess
import csv
import tempfile
import torch


@dataclass
class ProblemSpec:
    path: str
    module_name: str


def find_problem_files(root_dir: str) -> List[str]:
    problem_files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "problem.py":
                problem_files.append(os.path.join(dirpath, filename))
    # If no problem.py files found, look for .py files directly in root_dir
    if not problem_files:
        root_path = os.path.abspath(root_dir)
        for filename in os.listdir(root_path):
            if filename.endswith(".py") and os.path.isfile(os.path.join(root_path, filename)):
                problem_files.append(os.path.join(root_path, filename))
    problem_files.sort()
    return problem_files


def extract_problem_name(problem_path: str) -> str:
    """
    Extract problem name from a problem file path.
    - If it's a problem.py file in a subdirectory, use the directory name
    - If it's a direct Python file, use the filename (without extension) with zero-padded number
    """
    filename = os.path.basename(problem_path)
    if filename == "problem.py":
        # Traditional structure: use directory name
        return os.path.basename(os.path.dirname(problem_path))
    else:
        # Direct Python file: use filename without extension, zero-pad the number
        name_without_ext = os.path.splitext(filename)[0]
        # Check if it starts with a number followed by underscore
        parts = name_without_ext.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            # Zero-pad the number to 3 digits
            number = int(parts[0])
            return f"{number:03d}_{parts[1]}"
        else:
            # No number prefix, return as-is
            return name_without_ext


def load_module_from_path(path: str, unique_suffix: str) -> Any:
    module_name = f"problem_module_{unique_suffix}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec for module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def move_inputs_to_device(inputs: Sequence[Any], device: torch.device, target_float_dtype: torch.dtype) -> Tuple[Tuple[Any, ...], List[torch.Tensor]]:
    moved: List[Any] = []
    tensors_to_keep_for_later_sync: List[torch.Tensor] = []
    for item in inputs:
        if isinstance(item, torch.Tensor):
            if torch.is_floating_point(item):
                moved_tensor = item.to(device=device, dtype=target_float_dtype, non_blocking=True)
            else:
                moved_tensor = item.to(device=device, non_blocking=True)
            moved.append(moved_tensor)
            tensors_to_keep_for_later_sync.append(moved_tensor)
        elif isinstance(item, (list, tuple)):
            nested_tuple, nested_tensors = move_inputs_to_device(item, device, target_float_dtype)
            moved.append(list(nested_tuple))
            tensors_to_keep_for_later_sync.extend(nested_tensors)
        else:
            moved.append(item)
    return tuple(moved), tensors_to_keep_for_later_sync


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * pct
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return d0 + d1


def get_sm_clock_hz(override_mhz: float) -> Optional[float]:
    if override_mhz and override_mhz > 0:
        return override_mhz * 1e6
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        line = out.stdout.strip().splitlines()[0]
        mhz = float(line)
        return mhz * 1e6
    except Exception:
        return None


def convert_times(times_ms: List[float], unit: str, sm_clock_hz: Optional[float]) -> List[float]:
    if unit == "ms":
        return times_ms
    if unit == "us":
        return [t * 1000.0 for t in times_ms]
    if unit == "ns":
        return [t * 1_000_000.0 for t in times_ms]
    if unit == "cycles_per_sm":
        # cycles_per_sm = time_s * sm_clock_hz
        if sm_clock_hz is None:
            return times_ms
        return [int((t / 1000.0) * sm_clock_hz) for t in times_ms]
    return times_ms


def time_model_forward_gpu(
    model: torch.nn.Module,
    inputs: Tuple[Any, ...],
    iters: int,
    warmup: int,
    use_nvtx: bool,
    nvtx_tag: str,
    use_cuda_profiler_guard: bool = False,
) -> List[float]:
    times_ms: List[float] = []
    stream = torch.cuda.current_stream()
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs)
    stream.synchronize()
    # Timed runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    if use_cuda_profiler_guard:
        try:
            torch.cuda.profiler.start()
        except Exception:
            pass
    with torch.no_grad():
        for i in range(iters):
            if use_nvtx:
                torch.cuda.nvtx.range_push(f"{nvtx_tag}/iter_{i}")
            start_event.record(stream)
            _ = model(*inputs)
            end_event.record(stream)
            end_event.synchronize()
            iter_ms = start_event.elapsed_time(end_event)
            times_ms.append(iter_ms)
            if use_nvtx:
                torch.cuda.nvtx.range_pop()
    stream.synchronize()
    if use_cuda_profiler_guard:
        try:
            torch.cuda.profiler.stop()
        except Exception:
            pass
    return times_ms


def time_model_forward_cpu(
    model: torch.nn.Module,
    inputs: Tuple[Any, ...],
    iters: int,
    warmup: int,
) -> List[float]:
    times_ms: List[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs)
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = model(*inputs)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    return times_ms


def maybe_profile(
    enabled: bool,
    model: torch.nn.Module,
    inputs: Tuple[Any, ...],
    on_cuda: bool,
    activities: Tuple[torch.profiler.ProfilerActivity, ...],
    profile_iters: int,
    profile_output_dir: str,
    tag: str,
) -> None:
    if not enabled:
        return
    os.makedirs(profile_output_dir, exist_ok=True)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
        with_flops=False,
    ) as prof:
        with torch.no_grad():
            for _ in range(profile_iters):
                _ = model(*inputs)
                if on_cuda:
                    torch.cuda.synchronize()
                prof.step()
    trace_path = os.path.join(profile_output_dir, f"{tag.replace(os.sep, '_')}.json")
    try:
        prof.export_chrome_trace(trace_path)
    except Exception:
        pass


def run_one_problem(
    problem_path: str,
    device_str: str,
    iters: int,
    warmup: int,
    use_nvtx: bool,
    use_profiler: bool,
    profile_iters: int,
    profile_dir: str,
    fail_fast: bool,
    use_cuda_profiler_guard: bool = False,
) -> Tuple[str, List[float]]:
    problem_name = extract_problem_name(problem_path)
    rel_tag = os.path.relpath(problem_path, start=os.getcwd())
    unique_suffix = str(abs(hash(problem_path)))
    try:
        module = load_module_from_path(problem_path, unique_suffix)
        # Build model and inputs
        Model = getattr(module, "Model")
        get_inputs = getattr(module, "get_inputs")
        get_init_inputs = getattr(module, "get_init_inputs")

        init_args = get_init_inputs()
        if not isinstance(init_args, (list, tuple)):
            init_args = [init_args]
        model = Model(*init_args)
        model.eval()

        device = torch.device(device_str)
        on_cuda = device.type == "cuda"

        # Handle dtype/device policy
        target_float_dtype = torch.float16 if on_cuda else torch.float32

        # Generate inputs once
        # Some problem files set default dtype to float16; we will cast appropriately below
        raw_inputs_list = get_inputs()
        if not isinstance(raw_inputs_list, (list, tuple)):
            raw_inputs_list = [raw_inputs_list]
        inputs, _ = move_inputs_to_device(tuple(raw_inputs_list), device, target_float_dtype)

        # Move model after input creation to ensure parameter dtypes
        model = model.to(device=device, dtype=target_float_dtype if any(
            isinstance(x, torch.Tensor) and torch.is_floating_point(x) for x in inputs
        ) else None)

        # Compile model to use PyTorch compile instead of native eager
        try:
            model = torch.compile(
                model,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": True},
            )
        except Exception as compile_exc:
            # Fall back to eager if compile is unavailable or fails
            print(f"[WARN] torch.compile failed; falling back to eager. err={compile_exc}", file=sys.stderr)

        # Optional profiler
        activities: Tuple[torch.profiler.ProfilerActivity, ...] = (
            (torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU)
            if on_cuda else (torch.profiler.ProfilerActivity.CPU,)
        )
        maybe_profile(
            enabled=use_profiler,
            model=model,
            inputs=inputs,
            on_cuda=on_cuda,
            activities=activities,
            profile_iters=profile_iters,
            profile_output_dir=profile_dir,
            tag=rel_tag,
        )

        # Timing
        if on_cuda:
            times_ms = time_model_forward_gpu(
                model=model,
                inputs=inputs,
                iters=iters,
                warmup=warmup,
                use_nvtx=use_nvtx,
                nvtx_tag=rel_tag,
                use_cuda_profiler_guard=use_cuda_profiler_guard,
            )
        else:
            times_ms = time_model_forward_cpu(
                model=model,
                inputs=inputs,
                iters=iters,
                warmup=warmup,
            )
        return problem_name, times_ms
    except Exception as e:
        if fail_fast:
            raise
        print(f"[ERROR] {rel_tag}: {e}", file=sys.stderr)
        return problem_name, []


def run_ncu_and_sum_metric(
    problem_path: str,
    metric: str,
    nvtx_include: str,
    ncu_set: str,
    profile_from_start: str,
    debug: bool,
    rep_dir: Optional[str],
    problem_name: str,
    agg: str,
    child_iters: int,
    child_warmup: int,
) -> Optional[float]:
    script_path = os.path.abspath(__file__)
    # Determine report path: user-specified dir or a temporary file
    if rep_dir:
        os.makedirs(rep_dir, exist_ok=True)
        safe_name = problem_name.replace(os.sep, "_")
        rep_path = os.path.join(rep_dir, f"{safe_name}.ncu-rep")
        cleanup = False
    else:
        tmp_rep = tempfile.NamedTemporaryFile(suffix=".ncu-rep", delete=False)
        rep_path = tmp_rep.name
        tmp_rep.close()
        cleanup = True

    cmd = [
        "ncu",
        "--target-processes", "all",
        "--export", rep_path,
        "--force-overwrite",
    ]
    if nvtx_include:
        cmd.extend(["--nvtx", "--nvtx-include", nvtx_include])
    cmd.extend(["--profile-from-start", profile_from_start])
    if ncu_set:
        cmd.extend(["--set", ncu_set])
    cmd.extend([
        sys.executable, script_path,
        "--single-problem", problem_path,
        "--device", "cuda",
        "--iters", str(child_iters),
        "--warmup", str(child_warmup),
        "--nvtx",
        "--quiet",
        "--cuda-profiler-guard",
    ])
    proc1 = subprocess.run(cmd, capture_output=True, text=True)

    # Now import and extract the metric as CSV raw
    cmd2 = [
        "ncu",
        "--import", rep_path,
        "--csv",
        "--page", "raw",
        "--metrics", metric,
    ]
    proc2 = subprocess.run(cmd2, capture_output=True, text=True)
    stdout = proc2.stdout
    stderr = proc2.stderr

    if cleanup and not debug:
        try:
            os.remove(rep_path)
        except OSError:
            pass

    # Parse CSV in a version-robust way: support both long and wide formats
    lines = [ln for ln in stdout.splitlines() if ln and not ln.startswith('#')]
    if not lines:
        if debug:
            print("[NCU PARSE] No CSV lines to parse", file=sys.stderr)
        return None

    reader = csv.reader(lines)
    header = next(reader, None)
    if not header:
        return None

    total = 0.0
    count = 0

    # Case 1: wide format where the metric is a header column
    if metric in header and "Metric Name" not in header:
        metric_idx = header.index(metric)
        # There may be a units row immediately after header; skip any non-numeric rows
        for row in reader:
            if len(row) <= metric_idx:
                continue
            cell = row[metric_idx].strip()
            if not cell:
                continue
            try:
                total += float(cell)
                count += 1
            except ValueError:
                # Likely a units row like "us"; skip
                continue
        value = (total / count) if (agg == "mean" and count > 0) else total
        if count > 0 or proc2.returncode == 0:
            return value

    # Case 2: long format with Metric Name / Metric Value columns
    if "Metric Name" in header and "Metric Value" in header:
        name_idx = header.index("Metric Name")
        value_idx = header.index("Metric Value")
        for row in reader:
            if len(row) <= max(name_idx, value_idx):
                continue
            if row[name_idx] == metric:
                try:
                    total += float(row[value_idx])
                    count += 1
                except ValueError:
                    continue
        value = (total / count) if (agg == "mean" and count > 0) else total
        if count > 0 or proc2.returncode == 0:
            return value

    if debug:
        # Export step diagnostics
        print(f"[NCU CMD1] {' '.join(cmd)}", file=sys.stderr)
        print(f"[NCU RC1] {proc1.returncode}", file=sys.stderr)
        if proc1.stdout:
            print(f"[NCU STDOUT1]\n{proc1.stdout}", file=sys.stderr)
        if proc1.stderr:
            print(f"[NCU STDERR1]\n{proc1.stderr}", file=sys.stderr)
        # Import step diagnostics
        print(f"[NCU CMD2] {' '.join(cmd2)}", file=sys.stderr)
        print(f"[NCU RC2] {proc2.returncode}", file=sys.stderr)
        if stdout:
            print(f"[NCU STDOUT2]\n{stdout}", file=sys.stderr)
        if stderr:
            print(f"[NCU STDERR2]\n{stderr}", file=sys.stderr)
        # Report path info
        exists = os.path.exists(rep_path)
        print(f"[NCU REPORT] path={rep_path} exists={exists}", file=sys.stderr)
    return None


def run_ncu_details_elapsed_cycles(
    problem_path: str,
    nvtx_include: str,
    ncu_set: str,
    profile_from_start: str,
    debug: bool,
    rep_dir: Optional[str],
    problem_name: str,
    agg: str,
    child_iters: int,
    child_warmup: int,
) -> Optional[float]:
    """Use Nsight Compute details page to sum 'Elapsed Cycles' across kernels (matches opt_ncu_rl.py)."""
    script_path = os.path.abspath(__file__)
    # Report path
    if rep_dir:
        os.makedirs(rep_dir, exist_ok=True)
        safe_name = problem_name.replace(os.sep, "_")
        rep_path = os.path.join(rep_dir, f"{safe_name}.ncu-rep")
        cleanup = False
    else:
        tmp_rep = tempfile.NamedTemporaryFile(suffix=".ncu-rep", delete=False)
        rep_path = tmp_rep.name
        tmp_rep.close()
        cleanup = True

    # Export report
    cmd1 = [
        "ncu",
        "--target-processes", "all",
        "--export", rep_path,
        "--force-overwrite",
    ]
    if nvtx_include:
        cmd1.extend(["--nvtx", "--nvtx-include", nvtx_include])
    cmd1.extend(["--profile-from-start", profile_from_start])
    if ncu_set:
        cmd1.extend(["--set", ncu_set])
    cmd1.extend([
        sys.executable, script_path,
        "--single-problem", problem_path,
        "--device", "cuda",
        "--iters", str(child_iters),
        "--warmup", str(child_warmup),
        "--nvtx",
        "--quiet",
        "--cuda-profiler-guard",
    ])
    p1 = subprocess.run(cmd1, capture_output=True, text=True)

    # Import details and parse
    cmd2 = [
        "ncu",
        "--import", rep_path,
        "--csv",
        "--page", "details",
        "--section", "SpeedOfLight",
    ]
    p2 = subprocess.run(cmd2, capture_output=True, text=True)
    out = p2.stdout
    err = p2.stderr

    if cleanup and not debug:
        try:
            os.remove(rep_path)
        except OSError:
            pass

    # Parse streaming CSV: handle repeated headers across kernels
    lines = [ln for ln in out.splitlines() if ln and not ln.startswith('#')]
    if not lines:
        if debug:
            print("[NCU PARSE] No CSV lines to parse (details)", file=sys.stderr)
        return None

    total = 0.0
    count = 0
    name_idx = -1
    value_idx = -1
    for row in csv.reader(lines):
        if ("Metric Name" in row) and ("Metric Value" in row):
            # New header block
            try:
                name_idx = row.index("Metric Name")
                value_idx = row.index("Metric Value")
            except ValueError:
                name_idx = -1
                value_idx = -1
            continue
        if name_idx != -1 and value_idx != -1:
            if len(row) <= max(name_idx, value_idx):
                continue
            metric_name = row[name_idx]
            metric_value = row[value_idx]
            if metric_name.strip() == "Elapsed Cycles":
                try:
                    total += float(metric_value.replace(",", ""))
                    count += 1
                except ValueError:
                    continue
    value = (total / count) if (agg == "mean" and count > 0) else total
    if count > 0 or p2.returncode == 0:
        return value

    if debug:
        print(f"[NCU CMD1] {' '.join(cmd1)}", file=sys.stderr)
        print(f"[NCU RC1] {p1.returncode}", file=sys.stderr)
        if p1.stdout:
            print(f"[NCU STDOUT1]\n{p1.stdout}", file=sys.stderr)
        if p1.stderr:
            print(f"[NCU STDERR1]\n{p1.stderr}", file=sys.stderr)
        print(f"[NCU CMD2] {' '.join(cmd2)}", file=sys.stderr)
        print(f"[NCU RC2] {p2.returncode}", file=sys.stderr)
        if out:
            print(f"[NCU STDOUT2]\n{out}", file=sys.stderr)
        if err:
            print(f"[NCU STDERR2]\n{err}", file=sys.stderr)
        exists = os.path.exists(rep_path)
        print(f"[NCU REPORT] path={rep_path} exists={exists}", file=sys.stderr)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PyTorch latency baselines over problem.py files.")
    parser.add_argument("--root", type=str, default=os.getcwd(), help="Root directory to search for problem.py files.")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), choices=["cuda", "cpu"], help="Device for execution.")
    parser.add_argument("--iters", type=int, default=100, help="Number of timed iterations per problem.")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations per problem.")
    parser.add_argument("--unit", type=str, default="ms", choices=["ms", "us", "ns", "cycles_per_sm"], help="Output unit for latency metrics. cycles_per_sm is approximate and uses current SM clock.")
    parser.add_argument("--sm-clock-mhz", type=float, default=0.0, help="Override SM clock in MHz when using cycles_per_sm. If 0, tries to read via nvidia-smi.")
    parser.add_argument("--nvtx", action="store_true", help="Emit NVTX ranges around each iteration for NCU correlation.")
    parser.add_argument("--profile", action="store_true", help="Enable lightweight PyTorch profiler for a few iterations.")
    parser.add_argument("--profile-iters", type=int, default=5, help="Number of iterations to run under profiler when enabled.")
    parser.add_argument("--profile-dir", type=str, default="./profiler_traces", help="Directory to write profiler traces.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first error.")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of problems to run (0 means no limit).")
    parser.add_argument("--single-problem", type=str, default="", help="Run only the specified problem.py path.")
    parser.add_argument("--quiet", action="store_true", help="Suppress normal CSV output (useful for NCU child runs).")
    parser.add_argument("--ncu", action="store_true", help="Use Nsight Compute to collect a metric instead of internal timing.")
    parser.add_argument("--ncu-measure", type=str, default="details_elapsed_cycles", choices=["details_elapsed_cycles", "raw_metric"], help="How to measure in NCU mode. details_elapsed_cycles sums 'Elapsed Cycles' from details page (default). raw_metric sums a raw metric across kernels.")
    parser.add_argument("--ncu-metric", type=str, default="sm__cycles_elapsed.sum", help="Raw metric name for --ncu-measure=raw_metric.")
    parser.add_argument("--ncu-agg", type=str, default="mean", choices=["sum", "mean"], help="Aggregate across kernel launches (default: mean).")
    parser.add_argument("--ncu-include", type=str, default="", help="NVTX include pattern for Nsight Compute to select kernels. Empty disables NVTX filtering.")
    parser.add_argument("--ncu-set", type=str, default="", help="Nsight Compute preset set (e.g., 'full', 'base'). Empty string to skip.")
    parser.add_argument("--ncu-profile-from-start", type=str, default="on", choices=["on", "off"], help="Whether to profile from start or honor CUDA profiler start/stop.")
    parser.add_argument("--ncu-debug", action="store_true", help="Print NCU stdout/stderr and commands on failure or for debugging.")
    parser.add_argument("--ncu-rep-dir", type=str, default="", help="Directory to save .ncu-rep files (kept). If empty, a temp report is used and deleted.")
    parser.add_argument("--cuda-profiler-guard", action="store_true", help="Call torch.cuda.profiler.start/stop() around the measured GPU iterations.")
    parser.add_argument("--skip", type=int, default=0, help="Skip the first N discovered problem files (start from index N).")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root)

    if args.single_problem:
        problem_files = [os.path.abspath(args.single_problem)]
    else:
        problem_files = find_problem_files(root_dir)
        if args.skip > 0:
            problem_files = problem_files[args.skip:]
        if args.limit > 0:
            problem_files = problem_files[: args.limit]

    if args.ncu:
        if args.device != "cuda":
            print("[ERROR] NCU mode requires --device cuda", file=sys.stderr)
            sys.exit(2)
        # Header based on measure type and aggregator
        if args.ncu_measure == "details_elapsed_cycles":
            col = "avg" if args.ncu_agg == "mean" else "total"
            print(f"problem,{col}_Elapsed_Cycles")
        else:
            col = "avg" if args.ncu_agg == "mean" else "total"
            print(f"problem,{col}_{args.ncu_metric}")
        for path in problem_files:
            problem_name = extract_problem_name(path)
            # Aggregation: sum across kernels within a run; mean across repeated runs only if requested
            values: List[float] = []
            if args.ncu_agg == "mean" and args.iters > 1:
                # Run NCU per-iteration: one launch per child, average across runs
                repeats = args.iters
                for _ in range(repeats):
                    if args.ncu_measure == "details_elapsed_cycles":
                        v = run_ncu_details_elapsed_cycles(
                            problem_path=path,
                            nvtx_include=args.ncu_include,
                            ncu_set=args.ncu_set,
                            profile_from_start=args.ncu_profile_from_start,
                            debug=args.ncu_debug,
                            rep_dir=(args.ncu_rep_dir or None),
                            problem_name=problem_name,
                            agg="sum",
                            child_iters=1,
                            child_warmup=0,
                        )
                    else:
                        v = run_ncu_and_sum_metric(
                            problem_path=path,
                            metric=args.ncu_metric,
                            nvtx_include=args.ncu_include,
                            ncu_set=args.ncu_set,
                            profile_from_start=args.ncu_profile_from_start,
                            debug=args.ncu_debug,
                            rep_dir=(args.ncu_rep_dir or None),
                            problem_name=problem_name,
                            agg="sum",
                            child_iters=1,
                            child_warmup=0,
                        )
                    if v is not None:
                        values.append(v)
                result = (sum(values) / len(values)) if values else None
            else:
                # Single run with requested iters/warmup, sum across kernels
                if args.ncu_measure == "details_elapsed_cycles":
                    result = run_ncu_details_elapsed_cycles(
                        problem_path=path,
                        nvtx_include=args.ncu_include,
                        ncu_set=args.ncu_set,
                        profile_from_start=args.ncu_profile_from_start,
                        debug=args.ncu_debug,
                        rep_dir=(args.ncu_rep_dir or None),
                        problem_name=problem_name,
                        agg="sum" if args.ncu_agg == "sum" else "sum",
                        child_iters=args.iters,
                        child_warmup=args.warmup,
                    )
                else:
                    result = run_ncu_and_sum_metric(
                        problem_path=path,
                        metric=args.ncu_metric,
                        nvtx_include=args.ncu_include,
                        ncu_set=args.ncu_set,
                        profile_from_start=args.ncu_profile_from_start,
                        debug=args.ncu_debug,
                        rep_dir=(args.ncu_rep_dir or None),
                        problem_name=problem_name,
                        agg="sum" if args.ncu_agg == "sum" else "sum",
                        child_iters=args.iters,
                        child_warmup=args.warmup,
                    )
            if result is None:
                print(f"{problem_name},nan")
            else:
                if abs(result - round(result)) < 1e-6:
                    print(f"{problem_name},{int(round(result))}")
                else:
                    print(f"{problem_name},{result}")
        return

    if args.quiet:
        for path in problem_files:
            run_one_problem(
                problem_path=path,
                device_str=args.device,
                iters=args.iters,
                warmup=args.warmup,
                use_nvtx=args.nvtx,
                use_profiler=args.profile,
                profile_iters=args.profile_iters,
                profile_dir=args.profile_dir,
                fail_fast=args.fail_fast,
                use_cuda_profiler_guard=args.cuda_profiler_guard,
            )
        return

    header_unit = args.unit
    print(f"problem,mean_{header_unit},p50_{header_unit},p90_{header_unit},p99_{header_unit},min_{header_unit},max_{header_unit},iters")

    sm_clock_hz: Optional[float] = None
    if args.unit == "cycles_per_sm":
        if args.device != "cuda":
            print("[WARN] cycles_per_sm requested on CPU; falling back to ms.", file=sys.stderr)
            args.unit = "ms"
        else:
            sm_clock_hz = get_sm_clock_hz(args.sm_clock_mhz)
            if sm_clock_hz is None:
                print("[WARN] Could not determine SM clock; reporting ms instead of cycles.", file=sys.stderr)
                args.unit = "ms"
                header_unit = "ms"

    for idx, path in enumerate(problem_files):
        tag, times_ms = run_one_problem(
            problem_path=path,
            device_str=args.device,
            iters=args.iters,
            warmup=args.warmup,
            use_nvtx=args.nvtx,
            use_profiler=args.profile,
            profile_iters=args.profile_iters,
            profile_dir=args.profile_dir,
            fail_fast=args.fail_fast,
            use_cuda_profiler_guard=args.cuda_profiler_guard,
        )
        if times_ms:
            converted = convert_times(times_ms, args.unit, sm_clock_hz)
            mean_v = sum(converted) / len(converted)
            p50 = percentile(converted, 0.5)
            p90 = percentile(converted, 0.9)
            p99 = percentile(converted, 0.99)
            min_v = min(converted)
            max_v = max(converted)
            print(f"{tag},{mean_v:.4f},{p50:.4f},{p90:.4f},{p99:.4f},{min_v:.4f},{max_v:.4f},{len(converted)}")
        else:
            print(f"{tag},nan,nan,nan,nan,nan,nan,0")
        sys.stdout.flush()


if __name__ == "__main__":
    main() 