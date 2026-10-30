#!/usr/bin/env python3
"""Plot OpenCL / Adreno RL speedup vs initial kernel (KernelBlaster rl_opencl runs).

Per-kernel vertical-bar layout matching plot_speedup.py: kernels on the X
axis (no labels), log-scale Y, sort by speedup descending, with categorical
failure colors and a transition vline annotated with the fraction of
problems that ran faster.

Data sources per problem (under each ``rl_opencl/`` dir):
- Baseline: median of the **first** verified ``[PROFILE]`` ms in each
  ``trajectory_*/agentic_steps_log.txt`` (falls back to
  ``// Baseline:`` in ``failure_rl_optimization.cl``).
- Best: minimum verified ``[PROFILE]`` ms across all trajectory logs plus
  ``// Kernel time:`` footers in ``global_best_rl_optimization.cl``,
  ``success_rl_optimization.cl``, and ``rl_iter_*_best.cl``.

Example::

    python scripts/plot_speedup_opencl_adreno.py \\
        --base out/kernelbench-opencl/opencl_rl/gpt-5-mini-2025-08-07/L1
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]

PROFILE_LINE_RE = re.compile(
    r"\[PROFILE\]\s+\S+:\s+([0-9]+(?:\.[0-9]+)?)\s*ms", re.MULTILINE
)
KERNEL_TIME_FOOTER_RE = re.compile(
    r"//\s*Kernel time:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE
)
FAILURE_BASELINE_RE = re.compile(
    r"//\s*Baseline:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE
)


def iter_verified_profile_ms(text: str) -> list[float]:
    """Kernel times (ms) where the driver reported ``passed`` right after the profile line."""
    out: list[float] = []
    for m in PROFILE_LINE_RE.finditer(text):
        tail = text[m.end() : m.end() + 400].lstrip()
        if tail.lower().startswith("passed"):
            out.append(float(m.group(1)))
    return out


def first_verified_profile_ms(text: str) -> float | None:
    times = iter_verified_profile_ms(text)
    return times[0] if times else None


def artifact_kernel_times_ms(rl_dir: Path) -> list[float]:
    times: list[float] = []
    for name in (
        "global_best_rl_optimization.cl",
        "success_rl_optimization.cl",
    ):
        p = rl_dir / name
        if p.is_file():
            times.extend(
                float(x)
                for x in KERNEL_TIME_FOOTER_RE.findall(
                    p.read_text(encoding="utf-8", errors="ignore")
                )
            )
    for p in sorted(rl_dir.glob("rl_iter_*_best.cl")):
        if p.is_file():
            times.extend(
                float(x)
                for x in KERNEL_TIME_FOOTER_RE.findall(
                    p.read_text(encoding="utf-8", errors="ignore")
                )
            )
    return times


def baseline_ms_for_problem(rl_dir: Path) -> float | None:
    firsts: list[float] = []
    for log in sorted(rl_dir.glob("trajectory_*/agentic_steps_log.txt")):
        t = first_verified_profile_ms(log.read_text(encoding="utf-8", errors="ignore"))
        if t is not None:
            firsts.append(t)
    if firsts:
        return float(np.median(firsts))
    fb = rl_dir / "failure_rl_optimization.cl"
    if fb.is_file():
        m = FAILURE_BASELINE_RE.search(fb.read_text(encoding="utf-8", errors="ignore"))
        if m:
            return float(m.group(1))
    return None


def global_best_correct_ms(rl_dir: Path) -> float | None:
    """Minimum verified kernel time from logs + artifact footers."""
    candidates: list[float] = []
    for log in rl_dir.glob("trajectory_*/agentic_steps_log.txt"):
        candidates.extend(
            iter_verified_profile_ms(log.read_text(encoding="utf-8", errors="ignore"))
        )
    candidates.extend(artifact_kernel_times_ms(rl_dir))
    return min(candidates) if candidates else None


def find_rl_opencl_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("rl_opencl") if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=REPO / "out/kernelbench-opencl/opencl_rl",
        help="Root to search recursively for rl_opencl/ directories",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "opencl_adreno_speedup.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV path for numeric summary (per problem)",
    )
    parser.add_argument(
        "--xlabel",
        type=str,
        default="Kernel Bench L1 Problem",
        help="X-axis label",
    )
    args = parser.parse_args()
    base = args.base.resolve()
    if not base.exists():
        print(f"Base directory not found: {base}", file=sys.stderr)
        sys.exit(1)

    rl_dirs = find_rl_opencl_dirs(base)
    if not rl_dirs:
        print(f"No rl_opencl/ directories under {base}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for rl in rl_dirs:
        name = rl.parent.name
        rel = str(rl.relative_to(base)) if rl.is_relative_to(base) else str(rl)
        b = baseline_ms_for_problem(rl)
        g = global_best_correct_ms(rl)
        b_failed = b is None or b <= 0
        g_failed = g is None or g <= 0
        if not b_failed and not g_failed:
            cat, height = "valid", b / g
        elif b_failed and g_failed:
            cat, height = "both_failed", 1.0
        elif b_failed:
            cat, height = "init_failed", 1000.0
        else:
            cat, height = "best_failed", 1.0
        rows.append(
            {
                "name": name, "cat": cat, "height": height, "color": None,
                "baseline_ms": b, "best_ms": g, "speedup": (b / g) if cat == "valid" else None,
                "rel_path": rel,
            }
        )

    # Drop both-failed problems entirely so they don't pad the X axis.
    both_failed_dropped = [r for r in rows if r["cat"] == "both_failed"]
    rows = [r for r in rows if r["cat"] != "both_failed"]

    # Sort: init_failed first (leftmost, green spike), valid speedups desc,
    # best_failed (red) at the end.
    priority = {"init_failed": 0, "valid": 1, "best_failed": 2}
    rows.sort(key=lambda r: (priority[r["cat"]], -r["height"] if r["cat"] == "valid" else 0))

    for r in rows:
        if r["cat"] == "valid":
            r["color"] = "steelblue" if r["height"] >= 1.0 else "lightsteelblue"
        elif r["cat"] == "init_failed":
            r["color"] = "green"
        elif r["cat"] == "best_failed":
            r["color"] = "red"
        else:
            r["color"] = "orange"

    n_total = len(rows)
    x = np.arange(n_total)
    heights = [r["height"] for r in rows]
    colors = [r["color"] for r in rows]

    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(x, heights, color=colors, alpha=0.8, edgecolor="black", linewidth=0.4)
    ax.set_yscale("log")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No Speedup (1×)")

    valid_speedups = [r["height"] for r in rows if r["cat"] == "valid"]
    geomean_val = float(np.exp(np.mean(np.log(valid_speedups)))) if valid_speedups else None
    median_val = float(np.median(valid_speedups)) if valid_speedups else None
    if geomean_val is not None:
        ax.axhline(y=geomean_val, color="#1f77b4", linestyle=":", linewidth=1.5, alpha=0.85)
        ax.text(len(rows) - 0.5, geomean_val, f" geomean {geomean_val:.2f}×",
                va="center", ha="left", fontsize=9, fontweight="bold",
                color="#1f77b4",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#1f77b4", alpha=0.9))
    if median_val is not None:
        ax.axhline(y=median_val, color="#1f77b4", linestyle=":", linewidth=1.5, alpha=0.85)
        ax.text(len(rows) - 0.5, median_val, f" median {median_val:.2f}×",
                va="center", ha="left", fontsize=9, fontweight="bold",
                color="#1f77b4",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#1f77b4", alpha=0.9))

    init_failed_count = sum(1 for r in rows if r["cat"] == "init_failed")
    best_failed_count = sum(1 for r in rows if r["cat"] == "best_failed")
    both_failed_count = len(both_failed_dropped)
    n_meaningful = len(valid_speedups) + init_failed_count

    if n_meaningful > 0:
        ge1 = init_failed_count + sum(1 for s in valid_speedups if s >= 1.0)
        transition = next(
            (i for i, r in enumerate(rows) if r["cat"] == "valid" and r["height"] < 1.0),
            n_total,
        )
        pct = ge1 / n_meaningful * 100
        if 0 < transition < n_total:
            ax.axvline(x=transition - 0.5, color="darkblue", linestyle=":",
                       alpha=0.8, linewidth=2)
            ax.text(transition - 0.5, 3.0,
                    f"{pct:.1f}% of problems are faster",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                    rotation=90)

    geomean = geomean_val if geomean_val is not None else float("nan")
    median = median_val if median_val is not None else float("nan")

    ax.set_xlabel(args.xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel("Speedup (init median ms / global best verified ms) [Log Scale]",
                  fontsize=12, fontweight="bold")
    ax.set_title(
        f"OpenCL / Adreno RL — verified global best  |  "
        f"N={len(valid_speedups)}, geomean={geomean:.2f}×, median={median:.2f}×",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks([])
    ax.set_xlim(-0.6, n_total - 0.4)
    ax.grid(axis="y", alpha=0.3)

    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.8, label="RL Faster (>1×)"),
        Patch(facecolor="lightsteelblue", alpha=0.8, label="RL Slower (<1×)"),
    ]
    present = {r["color"] for r in rows}
    if "green" in present:
        legend_elements.append(Patch(facecolor="green", alpha=0.8,
                                     label="Only Best Found (Init Failed)"))
    if "red" in present:
        legend_elements.append(Patch(facecolor="red", alpha=0.8,
                                     label="Only Init Measured (No Verified Best)"))
    legend_elements.append(plt.Line2D([0], [0], color="gray", linestyle="--",
                                      alpha=0.5, label="No Speedup (1×)"))
    if geomean_val is not None:
        legend_elements.append(plt.Line2D([0], [0], color="#1f77b4", linestyle=":",
                                          linewidth=1.5,
                                          label=f"Geomean ({geomean_val:.2f}×)"))
    if median_val is not None:
        legend_elements.append(plt.Line2D([0], [0], color="#1f77b4", linestyle=":",
                                          linewidth=1.5,
                                          label=f"Median ({median_val:.2f}×)"))
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {args.out}")
    print(f"  Total: {n_total}, valid: {len(valid_speedups)}, "
          f"init-failed: {init_failed_count}, best-failed: {best_failed_count}, "
          f"both-failed: {both_failed_count}")
    if valid_speedups:
        print(f"  geomean={geomean:.2f}×, median={median:.2f}×, "
              f"min={min(valid_speedups):.2f}×, max={max(valid_speedups):.2f}×")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["problem", "status", "baseline_ms_median",
                        "best_verified_ms", "speedup", "rl_opencl_path"])
            for r in sorted(rows, key=lambda r: r["name"]):
                w.writerow([
                    r["name"], r["cat"],
                    f"{r['baseline_ms']:.6f}" if r["baseline_ms"] is not None else "",
                    f"{r['best_ms']:.6f}" if r["best_ms"] is not None else "",
                    f"{r['speedup']:.6f}" if r["speedup"] is not None else "",
                    r["rel_path"],
                ])
        print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
