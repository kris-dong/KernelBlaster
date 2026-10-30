#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot backfill: write ``profile.json`` next to every raw profiling log
in an existing run tree.

Phase 3b infrastructure. Existing run trees under ``out/kernelbench-*/``
have raw profiling logs (CUDA: ``ncu/*_ncu_log.txt``; OpenCL: stdout
captures containing ``[PROFILE] name: X ms`` markers) but no structured
``ProfileResult`` JSON. This script walks a tree, calls
``backend.parse_profile()`` on each log it can recognize, and writes a
``profile.json`` next to the log so future tooling can read the
structured form instead of re-parsing text.

Auto-detection of backend per file is based on log content:
  - ``Elapsed Cycles`` token  -> CUDA  -> CUDABackend.parse_profile
  - ``[PROFILE]`` marker      -> OpenCL -> OpenCLBackend.parse_profile
  - neither                   -> skip (warn)

Pass ``--backend cuda`` or ``--backend opencl`` to force a backend (skips
auto-detection).

Idempotent: by default skips files that already have a sibling
``profile.json``. Pass ``--force`` to overwrite.

Usage::

    # Walk every NCU log under an opt_ncu_rl_optimized run tree
    python scripts/backfill_profile_json.py \\
        /scratch/.../out/kernelbench-cuda/opt_ncu_rl_optimized/ \\
        --glob '**/ncu/*.txt'

    # OpenCL run tree (logs captured into run.log)
    python scripts/backfill_profile_json.py \\
        /scratch/.../out/kernelbench-opencl/opencl_rl/ \\
        --glob '**/run.log' --backend opencl

    # Dry run
    python scripts/backfill_profile_json.py <root> --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when running this script from a checkout.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.kernelblaster.backends import get_backend  # noqa: E402


PROFILE_JSON_NAME = "profile.json"


def detect_backend(raw_log: str) -> str | None:
    """Return ``"cuda"`` / ``"opencl"`` / ``None`` based on log content."""
    if "Elapsed Cycles" in raw_log:
        return "cuda"
    if "[PROFILE]" in raw_log:
        return "opencl"
    return None


def backfill_one(
    log_path: Path,
    *,
    force_backend: str | None,
    force: bool,
    dry_run: bool,
) -> str:
    """Process one log file. Returns a one-word status: written / skipped / ...
    """
    out_path = log_path.parent / PROFILE_JSON_NAME
    if out_path.exists() and not force:
        return "skipped-exists"

    try:
        raw_log = log_path.read_text(errors="replace")
    except OSError as e:
        print(f"  read-error: {log_path}: {e}", file=sys.stderr)
        return "read-error"

    backend_name = force_backend or detect_backend(raw_log)
    if backend_name is None:
        return "skipped-unrecognized"

    backend = get_backend(backend_name)
    try:
        pr = backend.parse_profile(raw_log)
    except Exception as e:
        # Some logs (e.g. CUDA with no Elapsed Cycles in this run) will fail.
        # That's fine — record as parse-error and move on.
        print(f"  parse-error: {log_path}: {e}", file=sys.stderr)
        return "parse-error"

    if dry_run:
        return f"would-write({backend_name})"

    try:
        pr.write_json(out_path)
    except OSError as e:
        print(f"  write-error: {out_path}: {e}", file=sys.stderr)
        return "write-error"
    return f"written({backend_name})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="Run-tree root to walk")
    parser.add_argument(
        "--glob",
        default="**/ncu/*.txt",
        help="Glob (relative to root) of raw log files. Default: %(default)s",
    )
    parser.add_argument(
        "--backend",
        choices=("cuda", "opencl"),
        default=None,
        help="Force a backend instead of auto-detecting from log content.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing profile.json files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be written; don't actually write.",
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"ERROR: {args.root} is not a directory", file=sys.stderr)
        return 2

    print(f"Walking {args.root} for '{args.glob}' (backend={args.backend or 'auto'})")

    counts: dict[str, int] = {}
    for log_path in args.root.glob(args.glob):
        if not log_path.is_file():
            continue
        status = backfill_one(
            log_path,
            force_backend=args.backend,
            force=args.force,
            dry_run=args.dry_run,
        )
        counts[status] = counts.get(status, 0) + 1

    print()
    print("Summary:")
    for status in sorted(counts):
        print(f"  {status:25s} {counts[status]}")
    print(f"  total                     {sum(counts.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
