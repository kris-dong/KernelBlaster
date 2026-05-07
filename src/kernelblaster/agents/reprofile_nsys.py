"""
Nsight Systems Re-Profiling Agent for existing success_rl_optimization.cu files.

Unlike the NCU-based reprofile agent which measures kernel-only elapsed cycles,
this agent uses nsys to capture wall-clock GPU time from first kernel launch to
last kernel completion. This includes host/device synchronization, memory copies,
and kernel launch overhead between kernels -- giving a more realistic end-to-end
measurement.

Outputs a CSV with columns: problem, nsys_gpu_time_ns
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import asyncio
import re
import csv
import base64
import loguru
from dataclasses import dataclass, asdict

from ..config import GPUType
from .utils import (
    compile_and_run_cu_file,
    run_gpu_executable,
    find_kernel_names,
    NamedTimer,
    FeedbackError,
)


NSYS_STATS_CUDA_API_RE = re.compile(
    r"^\s*[\d.]+\s+[\d.]+\s+(\d[\d,]*)\s+(\d[\d,]*)\s+(\d[\d,]*)\s+(\d[\d,]*)\s+\d+\s+(\S+)",
    re.MULTILINE,
)


@dataclass
class NsysProfilingResult:
    """Results from nsys profiling a single file."""
    success_file: str
    test_code_file: str
    gpu_time_ns: int
    first_kernel_start_ns: int
    last_kernel_end_ns: int
    kernel_count: int
    nsys_output: str
    output_path: str
    success: bool
    error: Optional[str] = None


def _kernel_matches(kernel_name_in_trace: str, solution_kernel_names: set) -> bool:
    """Check if a kernel name from nsys trace matches any solution kernel name.

    nsys trace names are often mangled/templated (e.g. 'conv2d_implicit_gemm_kernel(...)'),
    so we check if any solution kernel name is a substring of the trace name.
    """
    for sol_name in solution_kernel_names:
        if sol_name in kernel_name_in_trace:
            return True
    return False


def _parse_nsys_gpu_trace(
    nsys_output: str,
    solution_kernel_names: Optional[set] = None,
) -> Tuple[int, int, int, int]:
    """Parse nsys output to find first solution-kernel start and last solution-kernel end.

    Only accepts per-invocation timestamp data (sqlite3 CSV or gputrace CSV) that
    provides absolute start/end times for each kernel. Does NOT fall back to summary
    tables which cannot measure the wall-clock span between kernels.

    When solution_kernel_names is provided, only kernels whose names match those in
    the solution .cu file are considered. The span is from the first matching kernel's
    start to the last matching kernel's end — capturing inter-kernel overhead between
    solution kernels but excluding harness setup/teardown.

    Returns:
        (gpu_span_ns, first_start_ns, last_end_ns, kernel_count)
        Returns (0, 0, 0, 0) on failure — caller should treat this as an error.
    """
    first_start = None
    last_end = None
    kernel_count = 0

    lines = nsys_output.split("\n")

    # Strategy 1: sqlite3 CSV output after __GPUTRACE_CSV_START__ marker
    marker_idx = None
    for i, line in enumerate(lines):
        if "__GPUTRACE_CSV_START__" in line:
            marker_idx = i
            break

    if marker_idx is not None:
        for line in lines[marker_idx + 1:]:
            line = line.strip()
            if not line:
                continue
            # sqlite3 CSV: start,duration,demangledName
            # Name may contain commas (e.g. template args), so split only first 2
            parts = line.split(",", 2)
            if len(parts) < 3:
                continue
            try:
                start_ns = int(parts[0])
                dur_ns = int(parts[1])
            except (ValueError, IndexError):
                continue
            trace_name = parts[2].strip().strip('"')

            if solution_kernel_names and not _kernel_matches(trace_name, solution_kernel_names):
                continue

            end_ns = start_ns + dur_ns
            kernel_count += 1
            if first_start is None or start_ns < first_start:
                first_start = start_ns
            if last_end is None or end_ns > last_end:
                last_end = end_ns

    if kernel_count > 0 and first_start is not None and last_end is not None:
        return last_end - first_start, first_start, last_end, kernel_count

    # Strategy 2: CSV from `nsys stats --report gputrace --format csv`
    header_idx = None
    start_col = None
    dur_col = None
    name_col = None
    for i, line in enumerate(lines):
        if "Start (ns)" in line and "Duration (ns)" in line:
            cols = [c.strip().strip('"') for c in line.split(",")]
            for j, c in enumerate(cols):
                if c == "Start (ns)":
                    start_col = j
                elif c == "Duration (ns)":
                    dur_col = j
                elif c == "Name":
                    name_col = j
            if start_col is not None and dur_col is not None:
                header_idx = i
                break

    if header_idx is not None and start_col is not None and dur_col is not None:
        for line in lines[header_idx + 1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) <= max(start_col, dur_col):
                continue
            try:
                start_ns = int(parts[start_col].strip().strip('"').replace(",", ""))
                dur_ns = int(parts[dur_col].strip().strip('"').replace(",", ""))
            except (ValueError, IndexError):
                continue

            if solution_kernel_names and name_col is not None and len(parts) > name_col:
                trace_name = parts[name_col].strip().strip('"')
                if not _kernel_matches(trace_name, solution_kernel_names):
                    continue

            end_ns = start_ns + dur_ns
            kernel_count += 1
            if first_start is None or start_ns < first_start:
                first_start = start_ns
            if last_end is None or end_ns > last_end:
                last_end = end_ns

    if kernel_count > 0 and first_start is not None and last_end is not None:
        return last_end - first_start, first_start, last_end, kernel_count

    # No per-invocation timestamp data found — do not fall back to summary tables.
    return 0, 0, 0, 0


class NsysReProfileAgent:
    """
    Agent to re-profile existing success_rl_optimization.cu files using Nsight Systems.

    Measures wall-clock GPU time from first kernel start to last kernel end,
    capturing all inter-kernel overhead (memory copies, host code, launch latency).
    """

    def __init__(
        self,
        base_folder: Path,
        gpu: GPUType,
        logger: loguru.Logger,
        timeout: int = 3600,
        num_warmup: int = 1,
        num_runs: int = 5,
        profile_init: bool = False,
    ):
        self.base_folder = Path(base_folder)
        self.gpu = gpu
        self.logger = logger
        self.timeout = timeout
        self.num_warmup = num_warmup
        self.num_runs = num_runs
        self.profile_init = profile_init

    # ---- discovery helpers (reused from reprofile.py) ----

    def discover_success_files(
        self,
        base_directory: Optional[Path] = None,
        problem_numbers: Optional[List[str]] = None,
    ) -> List[Tuple[Path, Optional[Path]]]:
        pattern = "init.cu" if self.profile_init else "success_rl_optimization.cu"
        if base_directory is None:
            base_directory = self.base_folder

        normalized = None
        if problem_numbers:
            normalized = [pn.lstrip("0") or "0" for pn in problem_numbers]
            self.logger.info(f"Filtering to problem numbers: {problem_numbers}")

        pairs: List[Tuple[Path, Optional[Path]]] = []
        for sf in sorted(base_directory.rglob(pattern)):
            if problem_numbers is not None:
                num = self._get_problem_number(sf)
                if num is None:
                    continue
                if (num.lstrip("0") or "0") not in normalized:
                    continue
            tc = self._find_test_code(sf)
            pairs.append((sf, tc))
            if tc is None:
                self.logger.warning(f"No test code for {sf}, will be skipped")

        self.logger.info(f"Discovered {len(pairs)} files")
        return pairs

    def _find_test_code(self, success_file: Path) -> Optional[Path]:
        for d in [success_file.parent, success_file.parent.parent]:
            candidate = d / "driver.cpp"
            if candidate.exists():
                return candidate

        for check_dir in [success_file.parent, success_file.parent.parent, success_file.parent.parent.parent]:
            state_json = check_dir / "state.json"
            if state_json.exists():
                try:
                    state = json.load(open(state_json))
                    tc = state.get("test_code_fp")
                    if tc:
                        p = Path(tc)
                        if p.is_file():
                            return p
                except Exception:
                    pass

        # Fallback: curated data directory  data/kernelbench-cuda/<tier>/<problem>/driver.cpp
        problem_name = self._get_problem_name(success_file)
        tier = self._get_tier(success_file)
        if tier and problem_name:
            repo_root = self._find_repo_root(success_file)
            if repo_root:
                curated = repo_root / "data" / "kernelbench-cuda" / tier / problem_name / "driver.cpp"
                if curated.is_file():
                    return curated

        for d in [success_file.parent, success_file.parent.parent]:
            for pat in ["*driver*", "*test*.cpp", "*test*.cu"]:
                hits = list(d.glob(pat))
                if hits:
                    return hits[0]
        return None

    def _get_problem_name(self, path: Path) -> str:
        tier_markers = ("L1", "L2", "L3", "level1", "level2", "level3",
                        "sol-level1", "sol-level2", "sol-level3")
        parts = path.parts
        # Use the *last* tier marker (paths may contain the marker more than once)
        last_idx = None
        for i, part in enumerate(parts):
            if part in tier_markers and i + 1 < len(parts):
                last_idx = i
        if last_idx is not None:
            return parts[last_idx + 1]
        return path.parent.parent.name if path.parent.name == "rl_ncu" else path.parent.name

    def _get_tier(self, path: Path) -> Optional[str]:
        """Extract the tier directory name (e.g. 'sol-level1') from the path."""
        tier_markers = ("sol-level1", "sol-level2", "sol-level3",
                        "L1", "L2", "L3", "level1", "level2", "level3")
        parts = path.parts
        last_idx = None
        for i, part in enumerate(parts):
            if part in tier_markers:
                last_idx = i
        if last_idx is not None:
            return parts[last_idx]
        return None

    @staticmethod
    def _find_repo_root(path: Path) -> Optional[Path]:
        """Walk up from path to find the repo root (contains 'data' and 'src' dirs)."""
        current = path if path.is_dir() else path.parent
        for _ in range(20):
            if (current / "data" / "kernelbench-cuda").is_dir():
                return current
            parent = current.parent
            if parent == current:
                break
            current = parent
        return None

    def _get_problem_number(self, path: Path) -> Optional[str]:
        name = self._get_problem_name(path)
        m = re.match(r"^(\d+)_", name)
        return m.group(1) if m else None

    # ---- profiling ----

    async def profile_file(
        self,
        success_file: Path,
        test_code_fp: Path,
        output_dir: Optional[Path] = None,
    ) -> NsysProfilingResult:
        problem_name = self._get_problem_name(success_file)
        if output_dir is None:
            output_dir = success_file.parent.parent / "nsys_results" / problem_name
        else:
            output_dir = Path(output_dir) / problem_name
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"[nsys] Profiling {success_file}")

        try:
            # Extract solution kernel names from the .cu source for filtering
            solution_kernel_names = None
            try:
                sol_kernels = find_kernel_names(success_file)
                if sol_kernels:
                    solution_kernel_names = set(sol_kernels)
                    self.logger.info(
                        f"[nsys] Solution kernels for filtering: {solution_kernel_names}"
                    )
            except Exception as e:
                self.logger.warning(
                    f"[nsys] Could not extract kernel names from {success_file}: {e}. "
                    f"Will include all kernels in span."
                )

            # Step 1: compile
            timer = NamedTimer()
            stdout_list, stderr_list, compiled_path, ok = await compile_and_run_cu_file(
                test_code_fp, success_file, self.gpu, timer, self.logger,
                persistent_artifacts=True, timeout=self.timeout,
                num_runs=1, passed_keyword="passed",
            )
            if not ok:
                err = f"Compilation/execution failed: {stdout_list}"
                self.logger.error(err)
                return NsysProfilingResult(
                    success_file=str(success_file), test_code_file=str(test_code_fp),
                    gpu_time_ns=0, first_kernel_start_ns=0, last_kernel_end_ns=0,
                    kernel_count=0, nsys_output="", output_path=str(output_dir),
                    success=False, error=err,
                )

            # Step 2: nsys profile + gputrace extraction in a single command.
            # We use bash -c so that "$1" receives the binary path appended by
            # the GPU server, and nsys stats runs on the .nsys-rep afterwards.
            # The .nsys-rep is copied to a persistent location for later analysis.
            nsys_report_dir = f"/tmp/nsys_reports/{problem_name}"

            best_span = None
            best_output = ""
            best_first = 0
            best_last = 0
            best_count = 0

            for run_idx in range(self.num_runs):
                report_file = f"/tmp/nsys_reprofile_{problem_name}_{run_idx}"
                # Profile, then query the sqlite DB for per-kernel start/end timestamps
                # using Python's built-in sqlite3 module (sqlite3 CLI may not be installed).
                # We base64-encode a Python query script to avoid nested quoting issues.
                query_py = (
                    f"import sqlite3\n"
                    f"conn = sqlite3.connect('{report_file}.sqlite')\n"
                    f"c = conn.cursor()\n"
                    f"c.execute('SELECT k.start, k.end - k.start, s.value "
                    f"FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                    f"JOIN StringIds s ON k.demangledName = s.id "
                    f"ORDER BY k.start')\n"
                    f"for r in c.fetchall():\n"
                    f"    print(f'{{r[0]}},{{r[1]}},{{r[2]}}')\n"
                )
                query_b64 = base64.b64encode(query_py.encode()).decode()
                nsys_prefix = (
                    f"bash -c '"
                    f"nsys profile --force-overwrite=true --export=sqlite --output={report_file} \"$1\" && "
                    f"mkdir -p {nsys_report_dir} && "
                    f"cp {report_file}.nsys-rep {nsys_report_dir}/run_{run_idx}.nsys-rep && "
                    f"echo __GPUTRACE_CSV_START__ && "
                    f"python3 -c \"import base64,sys;exec(base64.b64decode(sys.argv[1]))\" {query_b64}"
                    f"' _"
                )
                nsys_stdout, nsys_stderr = await run_gpu_executable(
                    Path(compiled_path), self.gpu, self.timeout,
                    job_name=f"{success_file} (nsys run {run_idx})",
                    prefix_command=nsys_prefix,
                )

                combined = nsys_stdout + "\n" + nsys_stderr
                (logs_dir / f"nsys_run_{run_idx}_stdout.txt").write_text(nsys_stdout)
                (logs_dir / f"nsys_run_{run_idx}_stderr.txt").write_text(nsys_stderr)

                span, first, last, count = _parse_nsys_gpu_trace(
                    combined, solution_kernel_names
                )

                if span > 0 and (best_span is None or span < best_span):
                    best_span = span
                    best_output = combined
                    best_first = first
                    best_last = last
                    best_count = count

            if best_span is None or best_span == 0:
                err = "Failed to extract GPU kernel timing from nsys output"
                self.logger.error(f"{err} for {success_file}")
                (output_dir / "nsys_output.txt").write_text(best_output or combined)
                return NsysProfilingResult(
                    success_file=str(success_file), test_code_file=str(test_code_fp),
                    gpu_time_ns=0, first_kernel_start_ns=0, last_kernel_end_ns=0,
                    kernel_count=0, nsys_output=best_output or combined,
                    output_path=str(output_dir), success=False, error=err,
                )

            # Save results
            (output_dir / "nsys_output.txt").write_text(best_output)
            summary = {
                "problem_name": problem_name,
                "success_file": str(success_file),
                "test_code_file": str(test_code_fp),
                "compiled_binary": str(compiled_path),
                "nsys_reports_dir": nsys_report_dir,
                "gpu_time_ns": best_span,
                "first_kernel_start_ns": best_first,
                "last_kernel_end_ns": best_last,
                "kernel_count": best_count,
                "num_runs": self.num_runs,
                "solution_kernel_names": sorted(solution_kernel_names) if solution_kernel_names else None,
            }
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

            self.logger.info(
                f"[nsys] {problem_name}: {best_span:,} ns span, "
                f"{best_count} kernels, best of {self.num_runs} runs"
            )

            return NsysProfilingResult(
                success_file=str(success_file), test_code_file=str(test_code_fp),
                gpu_time_ns=best_span, first_kernel_start_ns=best_first,
                last_kernel_end_ns=best_last, kernel_count=best_count,
                nsys_output=best_output, output_path=str(output_dir),
                success=True,
            )

        except Exception as e:
            import traceback
            err = f"Error profiling {success_file}: {e}"
            self.logger.error(err, exc_info=True)
            try:
                (output_dir / "error.json").write_text(json.dumps({
                    "error": err, "traceback": traceback.format_exc(),
                }, indent=2))
            except Exception:
                pass
            return NsysProfilingResult(
                success_file=str(success_file), test_code_file=str(test_code_fp),
                gpu_time_ns=0, first_kernel_start_ns=0, last_kernel_end_ns=0,
                kernel_count=0, nsys_output="", output_path=str(output_dir),
                success=False, error=err,
            )

    def _load_cached_result(
        self, success_file: Path, output_base: Optional[Path]
    ) -> Optional[NsysProfilingResult]:
        """Try to load a previously-computed profiling result from summary.json."""
        problem_name = self._get_problem_name(success_file)
        if output_base:
            output_dir = Path(output_base) / problem_name
        else:
            output_dir = success_file.parent.parent / "nsys_results" / problem_name
        summary_path = output_dir / "summary.json"
        if not summary_path.exists():
            return None
        try:
            data = json.loads(summary_path.read_text())
            gpu_time = data.get("gpu_time_ns", 0)
            if gpu_time <= 0:
                return None
            return NsysProfilingResult(
                success_file=str(success_file),
                test_code_file=data.get("test_code_file", ""),
                gpu_time_ns=gpu_time,
                first_kernel_start_ns=data.get("first_kernel_start_ns", 0),
                last_kernel_end_ns=data.get("last_kernel_end_ns", 0),
                kernel_count=data.get("kernel_count", 0),
                nsys_output="",
                output_path=str(output_dir),
                success=True,
            )
        except Exception:
            return None

    async def profile_all(
        self,
        base_directory: Optional[Path] = None,
        output_base: Optional[Path] = None,
        max_workers: int = 1,
        problem_numbers: Optional[List[str]] = None,
        use_cached: bool = True,
    ) -> List[NsysProfilingResult]:
        if base_directory is None:
            base_directory = self.base_folder

        pairs = self.discover_success_files(
            base_directory=base_directory, problem_numbers=problem_numbers,
        )
        to_profile = [(sf, tc) for sf, tc in pairs if tc is not None]

        self.logger.info(f"Discovered {len(to_profile)} files (of {len(pairs)} total)")
        if not to_profile:
            return []

        results: List[NsysProfilingResult] = []
        need_profiling: List[tuple] = []

        if use_cached:
            for sf, tc in to_profile:
                cached = self._load_cached_result(sf, output_base)
                if cached is not None:
                    self.logger.info(
                        f"[nsys] Using cached result for {self._get_problem_name(sf)}: "
                        f"{cached.gpu_time_ns:,} ns"
                    )
                    results.append(cached)
                else:
                    need_profiling.append((sf, tc))
            self.logger.info(
                f"Cached: {len(results)}, need profiling: {len(need_profiling)}"
            )
        else:
            need_profiling = to_profile

        if need_profiling:
            if max_workers == 1:
                for sf, tc in need_profiling:
                    results.append(await self.profile_file(sf, tc, output_dir=output_base))
            else:
                sem = asyncio.Semaphore(max_workers)

                async def _run(sf, tc):
                    async with sem:
                        return await self.profile_file(sf, tc, output_dir=output_base)

                new_results = await asyncio.gather(
                    *[_run(sf, tc) for sf, tc in need_profiling]
                )
                results.extend(new_results)

        # Write CSV summary
        csv_path = (output_base or base_directory) / "nsys_profiling_results.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        ok_results = [r for r in results if r.success and r.gpu_time_ns > 0]
        ok_results.sort(key=lambda r: self._get_problem_name(Path(r.success_file)))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "problem", "nsys_gpu_time_ns", "kernel_count",
            ])
            w.writeheader()
            for r in ok_results:
                w.writerow({
                    "problem": self._get_problem_name(Path(r.success_file)),
                    "nsys_gpu_time_ns": r.gpu_time_ns,
                    "kernel_count": r.kernel_count,
                })
        self.logger.info(
            f"Profiling complete: {len(ok_results)}/{len(results)} successful. "
            f"CSV: {csv_path}"
        )

        # Also write JSONL
        jsonl_path = csv_path.with_suffix(".jsonl")
        with open(jsonl_path, "w") as f:
            for r in results:
                f.write(json.dumps(asdict(r)) + "\n")

        return list(results)
