#!/usr/bin/env python3
"""
Copy OpenCL L1 artifacts (kernel.cl + driver.c) into benchmark-opencl/L1.

Supports both:
1) Legacy kgen layout with success_* files under kgen_opencl/
2) Batch layout with kernel.cl + driver.c directly in each problem directory

This script only fills in missing files in the destination and never overwrites
existing files.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


DEFAULT_SRC = Path(
    # "out/kernelbench-opencl/kgen_opencl/gpt-5-mini-2025-08-07/L1"
    "out/kgen_claude_batch/L1"
)
DEFAULT_DST = Path("data/benchmark-opencl/L1")


def _attempt_num(stem: str) -> int:
    match = re.search(r"attempt(\d+)", stem)
    if not match:
        return -1
    return int(match.group(1))


def pick_best_pair(problem_dir: Path) -> tuple[Path, Path] | None:
    """Pick (kernel, driver) from supported source layouts."""

    # New batch layout: files are directly under problem_dir.
    direct_kernel = problem_dir / "kernel.cl"
    direct_driver = problem_dir / "driver.c"
    if direct_kernel.exists() and direct_driver.exists():
        return direct_kernel, direct_driver

    # Legacy layout: pick from succeeded kgen artifacts.
    kgen_dir = problem_dir / "kgen_opencl"
    if not kgen_dir.exists():
        return None

    kernels_by_stem: dict[str, Path] = {}
    drivers_by_stem: dict[str, Path] = {}

    for kernel_path in kgen_dir.glob("*_kernel.cl"):
        if kernel_path.name.endswith("_dummy_kernel.cl"):
            continue
        stem = kernel_path.name[: -len("_kernel.cl")]
        if not stem.startswith("success_"):
            continue
        kernels_by_stem[stem] = kernel_path

    for driver_path in kgen_dir.glob("*_driver.c"):
        stem = driver_path.name[: -len("_driver.c")]
        if not stem.startswith("success_"):
            continue
        drivers_by_stem[stem] = driver_path

    paired_stems = sorted(set(kernels_by_stem) & set(drivers_by_stem))
    if not paired_stems:
        return None

    # Pick the latest successful attempt.
    def rank(stem: str) -> tuple[int, float]:
        attempt_rank = _attempt_num(stem)
        mtime_rank = kernels_by_stem[stem].stat().st_mtime
        return (attempt_rank, mtime_rank)

    best_stem = max(paired_stems, key=rank)
    return kernels_by_stem[best_stem], drivers_by_stem[best_stem]


def copy_problem(problem_src: Path, problem_dst: Path, overwrite: bool) -> str:
    # Keep parameter for CLI compatibility, but destination files are never
    # overwritten by design.
    _ = overwrite

    pair = pick_best_pair(problem_src)
    if pair is None:
        return "no-source-pair"

    src_kernel, src_driver = pair
    dst_kernel = problem_dst / "kernel.cl"
    dst_driver = problem_dst / "driver.c"
    problem_dst.mkdir(parents=True, exist_ok=True)

    copied_any = False
    if not dst_kernel.exists():
        shutil.copy2(src_kernel, dst_kernel)
        copied_any = True

    if not dst_driver.exists():
        shutil.copy2(src_driver, dst_driver)
        copied_any = True

    if copied_any:
        return "copied"
    return "already-present"


def plan_problem_action(problem_src: Path, problem_dst: Path, overwrite: bool) -> str:
    """Return action without performing file writes."""
    # Keep parameter for CLI compatibility, but destination files are never
    # overwritten by design.
    _ = overwrite

    pair = pick_best_pair(problem_src)
    if pair is None:
        return "no-source-pair"

    dst_kernel = problem_dst / "kernel.cl"
    dst_driver = problem_dst / "driver.c"

    if dst_kernel.exists() and dst_driver.exists():
        return "already-present"
    return "would-copy"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy OpenCL L1 kernel.cl/driver.c pairs into data/benchmark-opencl/L1."
        )
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deprecated/no-op: destination files are never overwritten.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without copying files.",
    )
    args = parser.parse_args()

    if not args.src.exists():
        raise SystemExit(f"Source directory not found: {args.src}")
    if not args.src.is_dir():
        raise SystemExit(f"Source path is not a directory: {args.src}")
    if args.overwrite:
        print("Warning: --overwrite is ignored. Existing destination files are never overwritten.")

    stats = {
        "copied": 0,
        "already-present": 0,
        "no-source-pair": 0,
    }

    for problem_src in sorted(p for p in args.src.iterdir() if p.is_dir()):
        problem_dst = args.dst / problem_src.name
        if args.dry_run:
            action = plan_problem_action(problem_src, problem_dst, overwrite=args.overwrite)
            print(f"{problem_src.name}: {action}")
            continue

        action = copy_problem(problem_src, problem_dst, overwrite=args.overwrite)
        stats[action] += 1
        print(f"{problem_src.name}: {action}")

    if args.dry_run:
        return

    print(
        "Done. copied={copied}, already-present={already}, no-source-pair={missing}".format(
            copied=stats["copied"],
            already=stats["already-present"],
            missing=stats["no-source-pair"],
        )
    )


if __name__ == "__main__":
    main()
