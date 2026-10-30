#!/usr/bin/env python3
"""KernelBlaster speedup over BOTH baselines (QNN vendor + TVM compiler) in one plot.
Bars where TVM used a fallback compile path (not the optimized Adreno pipeline) are
hatched, so optimized vs fallback TVM comparisons stay visually distinct."""
import csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
from tvm_paths import DATA, PLOTS
CSV = os.path.join(DATA, "opencl_baseline_comparison.csv")
OUT = os.path.join(PLOTS, "kb_speedup_vs_qnn_and_tvm.png")
os.makedirs(PLOTS, exist_ok=True)

def f(x):
    try: return float(x)
    except: return None

rows = list(csv.DictReader(open(CSV)))
data = []
for r in rows:
    kq, kt = f(r["kb_over_qnn"]), f(r["kb_over_tvm"])
    if kq or kt:
        data.append({"name": r["name"], "kq": kq, "kt": kt,
                     # only a *fallback bar* if TVM actually produced a number (kt) via a non-optimized path
                     "tvm_fallback": (kt is not None) and (r.get("tvm_optimized", "True") == "False")})
def key(d):
    vs = [v for v in (d["kq"], d["kt"]) if v]
    return np.exp(np.mean(np.log(vs))) if vs else 0
data.sort(key=key, reverse=True)

def gm(vals):
    vals = [v for v in vals if v and v > 0]
    return float(np.exp(np.mean(np.log(vals)))) if vals else float("nan")
gm_q = gm([d["kq"] for d in data]); gm_t = gm([d["kt"] for d in data])
gm_t_opt = gm([d["kt"] for d in data if d["kt"] and not d["tvm_fallback"]])

plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .3, "axes.axisbelow": True})
fig, ax = plt.subplots(figsize=(max(12, .46*len(data)), 5.4))
x = np.arange(len(data)); w = .4
q = [d["kq"] if d["kq"] else np.nan for d in data]
t = [d["kt"] if d["kt"] else np.nan for d in data]
ax.bar(x - w/2, q, w, label=f"vs QNN (vendor)   geomean {gm_q:.2f}×", color="#7570b3", edgecolor="k")
tb = ax.bar(x + w/2, t, w, label=f"vs TVM (compiler) geomean {gm_t:.2f}×", color="#d95f0e", edgecolor="k")
# hatch fallback-TVM bars
for bar, d in zip(tb, data):
    if d["tvm_fallback"]:
        bar.set_hatch("////"); bar.set_edgecolor("black")
ax.axhline(1, color="r", ls="--", lw=1.5, label="parity (1×)")
ax.axhline(gm_q, color="#7570b3", ls=":", lw=1.1)
ax.axhline(gm_t, color="#d95f0e", ls=":", lw=1.1)
ax.set_yscale("log")
ax.set_ylabel("KernelBlaster speedup (×, log) — higher = LLM faster")
ax.set_xticks(x); ax.set_xticklabels([d["name"][:24] for d in data], rotation=75, ha="right", fontsize=7)
n_fb = sum(1 for d in data if d["tvm_fallback"])
ax.set_title("KernelBlaster (LLM OpenCL) speedup over QNN vendor & TVM compiler on Adreno "
             f"(hatched = TVM fallback path, n={n_fb})")
# legend proxy for hatch
from matplotlib.patches import Patch
handles, labels = ax.get_legend_handles_labels()
handles.append(Patch(facecolor="#d95f0e", hatch="////", edgecolor="k", label="TVM via fallback compile"))
ax.legend(handles=handles, loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(OUT); plt.close(fig)
print(f"wrote {OUT}")
print(f"problems: {len(data)} | geomean vs QNN={gm_q:.2f}× | vs TVM(all)={gm_t:.2f}× | vs TVM(optimized-only)={gm_t_opt:.2f}×")
print(f"vs QNN n={sum(1 for d in data if d['kq'])} | vs TVM n={sum(1 for d in data if d['kt'])} ({n_fb} via fallback)")
