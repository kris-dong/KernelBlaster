#!/usr/bin/env python3
"""
Copy CUDA artifacts (final_cuda.cu + driver.cpp) into kernelbench-cuda layout.

Supports both:
1) Legacy kgen layout with success_* files under kgen_cuda/
2) Batch layout with final_cuda.cu + driver.cpp directly in each problem directory
   (this is what scripts/kgen_step_cuda.py / the Claude Code CUDA flow produces)

This script only fills in missing files in the destination and never overwrites
existing files.

Default source: out/kgen_claude_batch/sol-level2 (the CUDA Claude-Code batch).
Default destination: data/kernelbench-cuda/sol-level2.

Override with --src / --dst for L1, sol-level1, etc.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


DEFAULT_SRC = Path("out/kgen_claude_batch/sol-level2")
DEFAULT_DST = Path("data/kernelbench-cuda/sol-level2")


def _attempt_num(stem: str) -> int:
    match = re.search(r"attempt(\d+)", stem)
    if not match:
        return -1
    return int(match.group(1))


def pick_best_pair(problem_dir: Path) -> tuple[Path, Path] | None:
    """Pick (cuda, driver) from supported source layouts.

    Returns (cuda_path, driver_path) or None if no usable pair is found.
    """

    # New batch layout: files are directly under problem_dir.
    direct_cuda = problem_dir / "final_cuda.cu"
    direct_driver = problem_dir / "driver.cpp"
    if direct_cuda.exists() and direct_driver.exists():
        return direct_cuda, direct_driver

    # Legacy layout: pick from succeeded kgen artifacts under kgen_cuda/.
    kgen_dir = problem_dir / "kgen_cuda"
    if not kgen_dir.exists():
        return None

    cudas_by_stem: dict[str, Path] = {}
    drivers_by_stem: dict[str, Path] = {}

    for cuda_path in kgen_dir.glob("*_final_cuda.cu"):
        if cuda_path.name.endswith("_dummy_final_cuda.cu"):
            continue
        stem = cuda_path.name[: -len("_final_cuda.cu")]
        if not stem.startswith("success_"):
            continue
        cudas_by_stem[stem] = cuda_path

    for driver_path in kgen_dir.glob("*_driver.cpp"):
        stem = driver_path.name[: -len("_driver.cpp")]
        if not stem.startswith("success_"):
            continue
        drivers_by_stem[stem] = driver_path

    paired_stems = sorted(set(cudas_by_stem) & set(drivers_by_stem))
    if not paired_stems:
        return None

    # Pick the latest successful attempt.
    def rank(stem: str) -> tuple[int, float]:
        attempt_rank = _attempt_num(stem)
        mtime_rank = cudas_by_stem[stem].stat().st_mtime
        return (attempt_rank, mtime_rank)

    best_stem = max(paired_stems, key=rank)
    return cudas_by_stem[best_stem], drivers_by_stem[best_stem]


def copy_problem(problem_src: Path, problem_dst: Path, overwrite: bool) -> str:
    # Keep parameter for CLI compatibility, but destination files are never
    # overwritten by design.
    _ = overwrite

    pair = pick_best_pair(problem_src)
    if pair is None:
        return "no-source-pair"

    src_cuda, src_driver = pair
    dst_cuda = problem_dst / "final_cuda.cu"
    dst_driver = problem_dst / "driver.cpp"
    problem_dst.mkdir(parents=True, exist_ok=True)

    copied_any = False
    if not dst_cuda.exists():
        shutil.copy2(src_cuda, dst_cuda)
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

    dst_cuda = problem_dst / "final_cuda.cu"
    dst_driver = problem_dst / "driver.cpp"

    if dst_cuda.exists() and dst_driver.exists():
        return "already-present"
    return "would-copy"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy CUDA final_cuda.cu/driver.cpp pairs into data/kernelbench-cuda/<level>."
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
