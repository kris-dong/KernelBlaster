#!/usr/bin/env python3
"""Compare OpenCL/Adreno baselines on KernelBench L1:
   - QNN (Qualcomm, vendor, Adreno fp16)              [existing]
   - TVM (optimization-based compiler, Adreno OpenCL)  [NEW]
   - KernelBlaster (LLM-generated OpenCL, RL best)      [our method]

Produces a unified per-problem CSV, summary stats (with the compute-bound outlier
filter from adreno_opencl_profiling_methodology.md), and comparison plots.
"""
import os, re, json, csv, math
from collections import Counter
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tvm_paths import DATA as D, PLOTS as P, RESULTS, QNN, KB_SUMMARY
TVM, KB = RESULTS, KB_SUMMARY

def num_of(s):
    m = re.match(r'0*(\d+)_', s); return int(m.group(1)) if m else None

# --- load ---
tvm = json.load(open(TVM)) if os.path.exists(TVM) else {}
qnn = json.load(open(QNN))
kb = {}
with open(KB) as f:
    for row in csv.DictReader(f):
        n = num_of(row["problem"])
        if n is not None:
            kb[n] = {"name": row["problem"],
                     "best_ms": float(row["best_verified_ms"]),
                     "baseline_ms": float(row["baseline_ms_median"])}

# tvm keyed by str(num); qnn keyed by str(num)
tvm_by = {int(k): v for k, v in tvm.items()}
qnn_by = {int(k): v for k, v in qnn.items()}

allnums = sorted(set(tvm_by) | set(qnn_by) | set(kb))
rows = []
for n in allnums:
    t = tvm_by.get(n, {}); q = qnn_by.get(n, {}); k = kb.get(n, {})
    name = (t.get("name") or q.get("name") or k.get("name") or str(n))
    tvm_path = t.get("path")
    tvm_ms = t.get("avg_ms_gpu_kernel") if t.get("status") == "ok" else None
    qnn_ms = q.get("avg_ms_gpu_kernel") if q.get("status") == "ok" else None
    kb_ms  = k.get("best_ms")
    rows.append({
        "num": n, "name": re.sub(r'^0*\d+_', '', name),
        "qnn_ms": qnn_ms, "tvm_ms": tvm_ms, "kb_ms": kb_ms,
        "tvm_path": tvm_path,
        "tvm_optimized": (tvm_path == "adreno/float16"),
        "tvm_status": t.get("status", "-"), "tvm_stage": t.get("stage", "-"),
        "kb_over_tvm": (tvm_ms / kb_ms) if (tvm_ms and kb_ms) else None,
        "kb_over_qnn": (qnn_ms / kb_ms) if (qnn_ms and kb_ms) else None,
        "tvm_over_qnn": (qnn_ms / tvm_ms) if (qnn_ms and tvm_ms) else None,
    })

# --- CSV ---
os.makedirs(D, exist_ok=True); os.makedirs(P, exist_ok=True)
cols = ["num", "name", "qnn_ms", "tvm_ms", "kb_ms", "kb_over_tvm", "kb_over_qnn",
        "tvm_over_qnn", "tvm_path", "tvm_optimized", "tvm_status", "tvm_stage"]
with open(os.path.join(D, "opencl_baseline_comparison.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
    for r in rows: w.writerow(r)

def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return float(np.exp(np.mean(np.log(xs)))) if xs else None

# compute-bound filter (per methodology doc): both sides must clear the floor
def filt(r, key_a, key_b, a_min, b_min):
    return r[key_a] and r[key_b] and r[key_a] >= a_min and r[key_b] >= b_min

tvm_ok = [r for r in rows if r["tvm_status"] == "ok"]
summary = {
    "n_problems_total": len(rows),
    "tvm_ok": len(tvm_ok),
    "tvm_ok_optimized": sum(1 for r in tvm_ok if r["tvm_optimized"]),
    "tvm_ok_fallback": sum(1 for r in tvm_ok if not r["tvm_optimized"]),
    "tvm_fallback_paths": dict(Counter(r["tvm_path"] for r in tvm_ok if not r["tvm_optimized"])),
    "tvm_fail": sum(1 for r in rows if r["tvm_status"] not in ("ok", "-")),
    "qnn_ok": sum(1 for r in rows if r["qnn_ms"]),
    "kb_covered": sum(1 for r in rows if r["kb_ms"]),
    "tvm_fail_stages": {},
    # all-problem geomeans (indicative)
    "geomean_kb_over_tvm_all": geomean([r["kb_over_tvm"] for r in rows]),
    "geomean_tvm_over_qnn_all": geomean([r["tvm_over_qnn"] for r in rows]),
    "geomean_kb_over_qnn_all": geomean([r["kb_over_qnn"] for r in rows]),
    # compute-bound subset (TVM_ms>=10 and KB_ms>=1 : both clearly above launch floor)
    "geomean_kb_over_tvm_computebound": geomean(
        [r["kb_over_tvm"] for r in rows if r["tvm_ms"] and r["kb_ms"] and r["tvm_ms"] >= 10 and r["kb_ms"] >= 1]),
    "n_computebound_kb_vs_tvm": sum(
        1 for r in rows if r["tvm_ms"] and r["kb_ms"] and r["tvm_ms"] >= 10 and r["kb_ms"] >= 1),
    "geomean_tvm_over_qnn_computebound": geomean(
        [r["tvm_over_qnn"] for r in rows if r["tvm_ms"] and r["qnn_ms"] and r["tvm_ms"] >= 10 and r["qnn_ms"] >= 200]),
}
from collections import Counter
summary["tvm_fail_stages"] = dict(Counter(
    r["tvm_stage"] for r in rows if r["tvm_status"] not in ("ok", "-")))
json.dump(summary, open(os.path.join(D, "opencl_baseline_summary.json"), "w"), indent=2)

# ================= PLOTS =================
plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True, "grid.alpha": .3, "axes.axisbelow": True})

# 1. grouped bars: QNN vs TVM vs KB, problems where all three exist
tri = [r for r in rows if r["qnn_ms"] and r["tvm_ms"] and r["kb_ms"]]
tri.sort(key=lambda r: r["tvm_ms"], reverse=True)
if tri:
    fig, ax = plt.subplots(figsize=(max(10, .5*len(tri)), 5))
    x = np.arange(len(tri)); w = .27
    ax.bar(x-w, [r["qnn_ms"] for r in tri], w, label="QNN (vendor)", color="#7570b3", edgecolor="k")
    ax.bar(x,   [r["tvm_ms"] for r in tri], w, label="TVM (compiler, NEW)", color="#d95f0e", edgecolor="k")
    ax.bar(x+w, [r["kb_ms"] for r in tri], w, label="KernelBlaster (LLM)", color="#1b9e77", edgecolor="k")
    ax.set_yscale("log"); ax.set_ylabel("GPU op time (ms, log)")
    ax.set_xticks(x); ax.set_xticklabels([r["name"][:22] for r in tri], rotation=75, ha="right", fontsize=7)
    ax.set_title("Adreno OpenCL baselines vs KernelBlaster (problems measured by all three)")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(P, "opencl_baselines_grouped.png")); plt.close(fig)

# 2. KernelBlaster speedup over TVM (per problem, where both exist)
pair = [r for r in rows if r["kb_over_tvm"]]
pair.sort(key=lambda r: r["kb_over_tvm"], reverse=True)
if pair:
    fig, ax = plt.subplots(figsize=(max(8, .4*len(pair)), 4.8))
    vals = [r["kb_over_tvm"] for r in pair]
    ax.bar(range(len(pair)), vals, color=["#1b9e77" if v > 1 else "#bdbdbd" for v in vals], edgecolor="k")
    ax.axhline(1, color="r", ls="--"); ax.set_yscale("log")
    gm = geomean(vals)
    if gm: ax.axhline(gm, color="g", lw=1.4, label=f"geomean={gm:.2f}x")
    ax.set_xticks(range(len(pair))); ax.set_xticklabels([r["name"][:22] for r in pair], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("KernelBlaster speedup over TVM (x, log)")
    ax.set_title("LLM-generated OpenCL vs TVM-compiled OpenCL (Adreno) — higher = LLM faster")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(P, "opencl_kb_vs_tvm.png")); plt.close(fig)

# 3. coverage / status
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
labels = ["TVM ok", "TVM fail", "QNN ok", "KB covered"]
vals = [summary["tvm_ok"], summary["tvm_fail"], summary["qnn_ok"], summary["kb_covered"]]
ax[0].bar(labels, vals, color=["#d95f0e", "#bdbdbd", "#7570b3", "#1b9e77"], edgecolor="k")
ax[0].set_ylabel("# problems"); ax[0].set_title(f"Baseline coverage (of {summary['n_problems_total']} L1 problems)")
for i, v in enumerate(vals): ax[0].text(i, v+0.3, str(v), ha="center")
fs = summary["tvm_fail_stages"]
if fs:
    ax[1].bar(list(fs.keys()), list(fs.values()), color="#de2d26", edgecolor="k")
    ax[1].set_title("TVM failure stage breakdown"); ax[1].set_ylabel("# problems")
    plt.setp(ax[1].get_xticklabels(), rotation=20, ha="right")
fig.tight_layout(); fig.savefig(os.path.join(P, "opencl_baseline_coverage.png")); plt.close(fig)

print("=== baseline comparison summary ===")
print(json.dumps(summary, indent=2))
print(f"\nwrote data/opencl_baseline_comparison.csv, data/opencl_baseline_summary.json")
print("wrote plots: opencl_baselines_grouped.png, opencl_kb_vs_tvm.png, opencl_baseline_coverage.png")
