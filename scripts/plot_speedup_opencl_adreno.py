#!/usr/bin/env python3
"""Plot OpenCL / Adreno RL speedup vs initial kernel (KernelBlaster rl_opencl runs).

Uses the **fastest verified** kernel time per problem, not only ``success_rl_optimization.cl``:

1. All ``[PROFILE] ... ms`` lines immediately followed by ``passed`` in
   ``trajectory_*/agentic_steps_log.txt`` (works for older runs without
   ``global_best_rl_optimization.cl``).
2. ``// Kernel time: ... ms`` footers in ``global_best_rl_optimization.cl``,
   ``success_rl_optimization.cl``, and ``rl_iter_*_best.cl`` when present.

Baseline (init) is the **median** of the **first** verified profile per trajectory
(first ``[PROFILE]`` + ``passed`` in each ``agentic_steps_log.txt``), which matches
the shared starting kernel across parallel rollouts. If no trajectories exist,
falls back to ``// Baseline: ... ms`` in ``failure_rl_optimization.cl`` when present.

Example::

    python scripts/plot_speedup_opencl_adreno.py \\
        --base out/kernelbench-opencl/opencl_rl/gpt-5-mini-2025-08-07/L1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
# Updated 2026-04-30: ``_fixed`` file replaces the legacy single-field schema
# with separate ``avg_ms_wallclock`` (end-to-end QNN run() time, includes
# framework overhead) and ``avg_ms_gpu_kernel`` (per-op GPU event time
# matching our cl_event methodology). We pick min() of the two below: the
# GPU-kernel value is what's directly comparable to our [PROFILE] number,
# but on some problems wallclock is actually smaller (their reported
# kernel time appears to count multi-pass replays for instruction metrics
# whereas wallclock is a single end-to-end run). Taking the minimum gives
# the most charitable QNN baseline — defensible against "you cherry-picked
# the metric that flatters our flow".
DEFAULT_QNN_JSON = REPO / "level1_qnn_qualcomm_results_fixed.json"

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
            times.extend(float(x) for x in KERNEL_TIME_FOOTER_RE.findall(p.read_text(encoding="utf-8", errors="ignore")))
    for p in sorted(rl_dir.glob("rl_iter_*_best.cl")):
        if p.is_file():
            times.extend(float(x) for x in KERNEL_TIME_FOOTER_RE.findall(p.read_text(encoding="utf-8", errors="ignore")))
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
        candidates.extend(iter_verified_profile_ms(log.read_text(encoding="utf-8", errors="ignore")))
    candidates.extend(artifact_kernel_times_ms(rl_dir))
    return min(candidates) if candidates else None


def find_rl_opencl_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("rl_opencl") if p.is_dir())


def make_label(name: str, max_desc: int = 36) -> str:
    m = re.match(r"(\d+)_(.*)", name)
    if m:
        num = m.group(1)
        desc = m.group(2).replace("_", " ")
        if len(desc) > max_desc:
            desc = desc[: max_desc - 3] + "..."
        return f"{num}: {desc}"
    return name


def problem_id(name: str) -> str | None:
    m = re.match(r"^\s*(\d+)_", name)
    return m.group(1) if m else None


def load_qnn_baseline_ms(path: Path) -> dict[str, float]:
    """Map problem id -> QNN baseline ms from the JSON results file.

    Supports two schemas:

    1. Legacy ``level1_qnn_qualcomm_results.json``: a single
       ``avg_ms_adreno_fp16`` field per entry (wallclock-flavoured; includes
       QNN framework overhead).

    2. ``level1_qnn_qualcomm_results_fixed.json`` (post 2026-04-30): two
       fields per entry — ``avg_ms_wallclock`` (end-to-end ``run()``) and
       ``avg_ms_gpu_kernel`` (QNN per-op GPU event timing). We take the
       ``min(wallclock, gpu_kernel)`` of whichever pair is present, which
       gives QNN the most favourable possible baseline. Rationale:
         - ``gpu_kernel`` is the "right" comparable quantity (matches our
           ``cl_event``-based ``[PROFILE]`` numbers).
         - But on a few problems QNN's reported kernel time is *larger* than
           its wallclock — likely a multi-pass profiler artefact (the GPU
           kernel time counter accumulates across instruction-replay passes
           while wallclock is a single dispatch). Falling back to wallclock
           in that case keeps the comparison honest in QNN's favour.
       Either way, we only ever pick ONE timing per problem; we never sum.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for _, rec in raw.items():
        if not isinstance(rec, dict):
            continue
        if rec.get("status") != "ok":
            continue
        name = rec.get("name")
        if not isinstance(name, str):
            continue
        pid = problem_id(name)
        if pid is None:
            continue

        candidates: list[float] = []
        # New schema fields first.
        for field in ("avg_ms_gpu_kernel", "avg_ms_wallclock"):
            v = rec.get(field)
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v > 0:
                candidates.append(v)
        # Legacy schema fallback.
        if not candidates:
            legacy = rec.get("avg_ms_adreno_fp16")
            try:
                legacy = float(legacy) if legacy is not None else None
            except (TypeError, ValueError):
                legacy = None
            if legacy is not None and legacy > 0:
                candidates.append(legacy)

        if candidates:
            out[pid] = min(candidates)
    return out


def plot_bars(ax, labels: list[str], speedups: list[float], xlabel: str, title: str) -> None:
    if not speedups:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    colors = ["#2ecc71" if s >= 1.0 else "#e74c3c" for s in speedups]
    bars = ax.barh(range(len(labels)), speedups, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    xmax = max(max(speedups) * 2.0, 1.05)
    ax.set_xlim(0.1, xmax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, s in zip(bars, speedups):
        x_pos = bar.get_width() + 0.02 * max(speedups)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{s:.2f}x", va="center", fontsize=7, fontweight="bold")


def collect_rows(base: Path) -> list[tuple[str, float, float, float, str]]:
    """Rows: (problem_name, baseline_ms, best_ms, speedup, rel_path)."""
    rows: list[tuple[str, float, float, float, str]] = []
    for rl in find_rl_opencl_dirs(base):
        problem = rl.parent.name
        rel = str(rl.relative_to(base)) if rl.is_relative_to(base) else str(rl)
        b = baseline_ms_for_problem(rl)
        g = global_best_correct_ms(rl)
        if b is None or g is None or b <= 0 or g <= 0:
            continue
        speedup = b / g
        rows.append((problem, b, g, speedup, rel))
    return rows


def load_qnn_components(path: Path) -> dict[str, dict[str, float]]:
    """Like ``load_qnn_baseline_ms`` but returns the raw components per pid:
    ``{pid: {"wallclock": float|None, "gpu_kernel": float|None,
              "legacy": float|None, "chosen": float}}``.

    Used for the printed table so the reader can see what we picked and why.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, float]] = {}
    for _, rec in raw.items():
        if not isinstance(rec, dict) or rec.get("status") != "ok":
            continue
        name = rec.get("name")
        if not isinstance(name, str):
            continue
        pid = problem_id(name)
        if pid is None:
            continue
        comps: dict[str, float] = {}
        for field, key in (
            ("avg_ms_gpu_kernel", "gpu_kernel"),
            ("avg_ms_wallclock", "wallclock"),
            ("avg_ms_adreno_fp16", "legacy"),
        ):
            v = rec.get(field)
            try:
                v = float(v) if v is not None else None
            except (TypeError, ValueError):
                v = None
            if v is not None and v > 0:
                comps[key] = v
        if comps:
            comps["chosen"] = min(comps.values())
            out[pid] = comps
    return out


def collect_qnn_rows(
    rows: list[tuple[str, float, float, float, str]], qnn_ms_by_id: dict[str, float]
) -> list[tuple[str, float, float, float]]:
    """Rows: (problem_name, qnn_ms, best_ms, qnn_vs_rl_speedup)."""
    out: list[tuple[str, float, float, float]] = []
    for name, _, best_ms, _, _ in rows:
        pid = problem_id(name)
        if pid is None:
            continue
        qnn_ms = qnn_ms_by_id.get(pid)
        if qnn_ms is None or qnn_ms <= 0 or best_ms <= 0:
            continue
        out.append((name, qnn_ms, best_ms, qnn_ms / best_ms))
    return out


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
        help="Optional CSV path for numeric summary",
    )
    parser.add_argument(
        "--qnn-json",
        type=Path,
        default=DEFAULT_QNN_JSON,
        help="QNN baseline JSON path (problem id keyed by name prefix)",
    )
    parser.add_argument(
        "--out-qnn",
        type=Path,
        default=None,
        help="Optional output PNG path for QNN-vs-RL plot "
        "(defaults to <out_stem>_vs_qnn<out_suffix>)",
    )
    args = parser.parse_args()
    base = args.base.resolve()
    if not base.exists():
        print(f"Base directory not found: {base}", file=sys.stderr)
        sys.exit(1)

    rows = collect_rows(base)
    if not rows:
        print(f"No rl_opencl problems with baseline+best under {base}", file=sys.stderr)
        sys.exit(1)

    rows.sort(key=lambda r: r[3], reverse=True)
    labels = [make_label(r[0]) for r in rows]
    speedups = [r[3] for r in rows]

    fig_h = max(6.0, len(labels) * 0.42)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    geomean = float(np.exp(np.mean(np.log(speedups))))
    plot_bars(
        ax,
        labels,
        speedups,
        xlabel="Speedup (init median ms / global best verified ms)",
        title=f"OpenCL / Adreno RL — verified global best\n"
        f"Geo-mean: {geomean:.2f}x  |  N={len(rows)}  |  scan: {base}",
    )
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.out}")

    print(f"\n{'Problem':<48} {'Init_ms':>10} {'Best_ms':>10} {'Speedup':>10}  path")
    print("-" * 120)
    for name, b, g, sp, rel in rows:
        print(f"{make_label(name):<48} {b:>10.3f} {g:>10.3f} {sp:>10.2f}x  {rel}")
    print(
        f"\nGeo-mean speedup: {geomean:.2f}x  |  mean: {float(np.mean(speedups)):.2f}x  |  "
        f"median: {float(np.median(speedups)):.2f}x"
    )

    if args.csv:
        import csv

        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["problem", "baseline_ms_median", "best_verified_ms", "speedup", "rl_opencl_path"])
            for name, b, g, sp, rel in rows:
                w.writerow([name, f"{b:.6f}", f"{g:.6f}", f"{sp:.6f}", rel])
        print(f"Wrote CSV: {args.csv}")

    qnn_json = args.qnn_json.resolve()
    if not qnn_json.is_file():
        print(f"QNN JSON not found; skipping QNN comparison: {qnn_json}")
        return

    qnn_by_id = load_qnn_baseline_ms(qnn_json)
    qnn_components = load_qnn_components(qnn_json)
    qnn_rows = collect_qnn_rows(rows, qnn_by_id)
    if not qnn_rows:
        print(f"No overlapping successful QNN + RL problems found in {qnn_json}")
        return

    qnn_rows.sort(key=lambda r: r[3], reverse=True)
    qnn_labels = [make_label(r[0]) for r in qnn_rows]
    qnn_speedups = [r[3] for r in qnn_rows]
    qnn_geomean = float(np.exp(np.mean(np.log(qnn_speedups))))

    out_qnn = args.out_qnn
    if out_qnn is None:
        out_qnn = args.out.with_name(f"{args.out.stem}_vs_qnn{args.out.suffix}")

    fig_h2 = max(6.0, len(qnn_labels) * 0.42)
    fig2, ax2 = plt.subplots(figsize=(14, fig_h2))
    plot_bars(
        ax2,
        qnn_labels,
        qnn_speedups,
        xlabel="Speedup (min(QNN wallclock, QNN gpu_kernel) / RL best verified ms)",
        title=f"OpenCL / Adreno RL vs QNN baseline (charitable: min wallclock/gpu)\n"
        f"Geo-mean: {qnn_geomean:.2f}x  |  N={len(qnn_rows)}  |  qnn: {qnn_json.name}",
    )
    plt.tight_layout()
    out_qnn.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(out_qnn, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_qnn}")

    print(
        f"\n{'Problem':<48} {'QNN_wall':>10} {'QNN_gpu':>10} "
        f"{'QNN_chosen':>10} {'RL_best':>10} {'QNN/RL':>9}  src"
    )
    print("-" * 110)
    for name, qnn_ms, best_ms, sp in qnn_rows:
        pid = problem_id(name)
        comps = qnn_components.get(pid or "", {})
        wall = comps.get("wallclock")
        gpu = comps.get("gpu_kernel")
        # Annotate which field "won" min(): w=wall, g=gpu, l=legacy
        chosen_src = (
            "g"
            if gpu is not None and gpu == qnn_ms
            else "w"
            if wall is not None and wall == qnn_ms
            else "l"
        )
        wall_s = f"{wall:>10.3f}" if wall is not None else f"{'—':>10}"
        gpu_s = f"{gpu:>10.3f}" if gpu is not None else f"{'—':>10}"
        print(
            f"{make_label(name):<48} {wall_s} {gpu_s} {qnn_ms:>10.3f} "
            f"{best_ms:>10.3f} {sp:>8.2f}x  {chosen_src}"
        )
    print(
        f"\nGeo-mean QNN/RL speedup: {qnn_geomean:.2f}x  |  mean: "
        f"{float(np.mean(qnn_speedups)):.2f}x  |  median: {float(np.median(qnn_speedups)):.2f}x"
    )
    print("(QNN_chosen = min(QNN_wall, QNN_gpu); src column: g=gpu_kernel, w=wallclock, l=legacy)")

    if args.csv:
        import csv

        qnn_csv = args.csv.with_name(f"{args.csv.stem}_vs_qnn{args.csv.suffix}")
        with open(qnn_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "problem",
                    "qnn_wallclock_ms",
                    "qnn_gpu_kernel_ms",
                    "qnn_chosen_ms",
                    "qnn_chosen_source",
                    "rl_best_verified_ms",
                    "qnn_vs_rl_speedup",
                ]
            )
            for name, qnn_ms, best_ms, sp in qnn_rows:
                pid = problem_id(name)
                comps = qnn_components.get(pid or "", {})
                wall = comps.get("wallclock")
                gpu = comps.get("gpu_kernel")
                chosen_src = (
                    "gpu_kernel"
                    if gpu is not None and gpu == qnn_ms
                    else "wallclock"
                    if wall is not None and wall == qnn_ms
                    else "legacy"
                )
                w.writerow(
                    [
                        name,
                        f"{wall:.6f}" if wall is not None else "",
                        f"{gpu:.6f}" if gpu is not None else "",
                        f"{qnn_ms:.6f}",
                        chosen_src,
                        f"{best_ms:.6f}",
                        f"{sp:.6f}",
                    ]
                )
        print(f"Wrote QNN CSV: {qnn_csv}")


if __name__ == "__main__":
    main()
