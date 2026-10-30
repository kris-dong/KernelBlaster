#!/usr/bin/env python3
"""
Copy generated CUDA artifacts (driver.cpp + final_cuda.cu) into KernelBlaster's
`data/kernelbench-cuda/sol-level1` layout.

Example:
  python scripts/copy_kernelbench_cuda_artifacts.py \
    --source-root /path/to/upstream/run/sol-level1 \
    --dest-root data/kernelbench-cuda/sol-level1
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Root containing per-problem directories from tachyon output.",
    )
    p.add_argument(
        "--dest-root",
        type=Path,
        default=Path("data/kernelbench-cuda/sol-level1"),
        help="Destination root with KernelBlaster sol-level1 folder structure.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination files.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without copying files.",
    )
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of problem directory names to copy.",
    )
    return p.parse_args()


def iter_problem_dirs(source_root: Path):
    for d in sorted(source_root.iterdir()):
        if d.is_dir():
            yield d


def copy_problem(problem_dir: Path, dest_root: Path, overwrite: bool, dry_run: bool) -> str:
    src_driver = problem_dir / "driver.cpp"
    src_cuda = problem_dir / "final_cuda.cu"

    if not src_driver.is_file() or not src_cuda.is_file():
        return f"SKIP {problem_dir.name}: missing driver.cpp or final_cuda.cu"

    out_dir = dest_root / problem_dir.name
    dst_driver = out_dir / "driver.cpp"
    dst_cuda = out_dir / "final_cuda.cu"

    if (dst_driver.exists() or dst_cuda.exists()) and not overwrite:
        return f"SKIP {problem_dir.name}: destination exists (use --overwrite)"

    if dry_run:
        return (
            f"DRY-RUN {problem_dir.name}: "
            f"{src_driver} -> {dst_driver}, {src_cuda} -> {dst_cuda}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_driver, dst_driver)
    shutil.copy2(src_cuda, dst_cuda)
    return f"COPIED {problem_dir.name}"


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    dest_root = args.dest_root.resolve()

    if not source_root.is_dir():
        print(f"ERROR: source root not found: {source_root}")
        return 2

    selected = set(args.only) if args.only else None
    total = 0
    copied = 0
    skipped = 0

    for problem_dir in iter_problem_dirs(source_root):
        if selected is not None and problem_dir.name not in selected:
            continue
        total += 1
        msg = copy_problem(problem_dir, dest_root, args.overwrite, args.dry_run)
        print(msg)
        if msg.startswith("COPIED"):
            copied += 1
        else:
            skipped += 1

    print(
        f"\nDone. scanned={total} copied={copied} skipped={skipped} "
        f"dest_root={dest_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

