#!/usr/bin/env python3
"""Level 2: TVM-OpenCL (Adreno) compiler baseline vs our ModelBlaster/KernelBlaster results on the QRB5165 (Adreno 650, fp16).

Reads  tvm_paths.L2_RESULTS (run_tvm_opencl_baseline_l2.py)  and each problem's <results dir>/<name>/work/summary.json
(seed_e2e_us = the flow's seed model time, model_e2e_us = final model time, both whole-model replay times on the board).
Writes data/level2_comparison.csv, data/level2_comparison_summary.json, plots/level2_tvm_vs_ours.png.
tvm_over_ours > 1 means TVM is slower than our final model.
"""
from __future__ import annotations
import os, re, json, csv, glob, math
from collections import Counter
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tvm_paths import DATA, PLOTS, L2_RESULTS, L2_OURS, L2_CONFIGS

def num_of(s): return int(s.split("_")[0])
names = []
for c in L2_CONFIGS: names += json.load(open(c))["problems"]
names.sort(key=num_of)
tvm = json.load(open(L2_RESULTS))
ours = {}
for d in L2_OURS:
    for p in glob.glob(os.path.join(d, "*", "work", "summary.json")):
        nm = os.path.basename(os.path.dirname(os.path.dirname(p)))
        s = json.load(open(p))["irs"]
        v = next(iter(s.values()))
        ours[nm] = {"seed_ms": v["seed_e2e_us"] / 1e3, "final_ms": v["model_e2e_us"] / 1e3, "speedup": v["model_speedup"], "passed": v.get("model_passed")}

def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return float(np.exp(np.mean(np.log(xs)))) if xs else None

rows = []
for n in names:
    k = str(num_of(n)); t = tvm.get(k, {}); o = ours.get(n, {})
    ok = t.get("status") == "ok"
    ad = (t.get("tried") or {}).get("adreno/float16") or {}
    r = {"num": num_of(n), "name": n, "tvm_ms": t.get("avg_ms_gpu_kernel") if ok else None, "tvm_path": t.get("path"),
         "tvm_optimized": bool(ok and t.get("path") == "adreno/float16"), "tvm_status": t.get("status", "-"), "tvm_stage": t.get("stage", "-"),
         "tvm_rel_err": t.get("rel_err"), "adreno_asis_ms": ad.get("gpu_ms"), "adreno_asis_correct": ad.get("correct"),
         "ours_seed_ms": o.get("seed_ms"), "ours_final_ms": o.get("final_ms"), "ours_speedup": o.get("speedup"),
         "note": (t.get("note") or "")[:200] if not ok else ""}
    r["tvm_over_ours"] = r["tvm_ms"] / r["ours_final_ms"] if r["tvm_ms"] and r["ours_final_ms"] else None
    r["tvm_over_seed"] = r["tvm_ms"] / r["ours_seed_ms"] if r["tvm_ms"] and r["ours_seed_ms"] else None
    r["adreno_asis_over_ours"] = r["adreno_asis_ms"] / r["ours_final_ms"] if r["adreno_asis_ms"] and r["ours_final_ms"] else None
    rows.append(r)
os.makedirs(DATA, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
cols = list(rows[0].keys())
with open(os.path.join(DATA, "level2_comparison.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]

okr = [r for r in rows if r["tvm_ms"]]
opt = [r for r in okr if r["tvm_optimized"]]
summ = {
    "n_problems": len(rows), "tvm_ok": len(okr), "tvm_ok_optimized_path": len(opt), "tvm_ok_fallback": len(okr) - len(opt),
    "tvm_failed": [r["name"] for r in rows if not r["tvm_ms"]],
    "fallback_paths": dict(Counter(r["tvm_path"] for r in okr if not r["tvm_optimized"])),
    "geomean_tvm_over_ours_optimized_only": geomean([r["tvm_over_ours"] for r in opt]),
    "geomean_tvm_over_ours_with_fallbacks": geomean([r["tvm_over_ours"] for r in okr]),
    "geomean_tvm_over_seed_optimized_only": geomean([r["tvm_over_seed"] for r in opt]),
    "geomean_tvm_over_seed_with_fallbacks": geomean([r["tvm_over_seed"] for r in okr]),
    "geomean_ours_speedup_over_seed_all48": geomean([r["ours_speedup"] for r in rows]),
    "geomean_ours_speedup_over_seed_on_tvm_ok": geomean([r["ours_speedup"] for r in okr]),
    "n_tvm_faster_than_ours": sum(1 for r in okr if r["tvm_over_ours"] and r["tvm_over_ours"] < 1),
    "n_tvm_faster_than_seed": sum(1 for r in okr if r["tvm_over_seed"] and r["tvm_over_seed"] < 1),
    "adreno_asis_runs": sum(1 for r in rows if r["adreno_asis_ms"]),
    "adreno_asis_wrong_result": sum(1 for r in rows if r["adreno_asis_ms"] and not r["adreno_asis_correct"]),
    "geomean_adreno_asis_over_ours_incl_wrong": geomean([r["adreno_asis_over_ours"] for r in rows]),
}
json.dump(summ, open(os.path.join(DATA, "level2_comparison_summary.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))

# plot: per problem TVM / ours-final (log), fallback bars hatched, failures marked
fig, ax = plt.subplots(figsize=(13, 5.5))
xs = np.arange(len(rows))
for i, r in enumerate(rows):
    if r["tvm_over_ours"]:
        ax.bar(i, r["tvm_over_ours"], color="#1f77b4" if r["tvm_optimized"] else "#ff7f0e", hatch=None if r["tvm_optimized"] else "//", edgecolor="k", lw=.4)
    else:
        ax.text(i, 1.05, "x", ha="center", color="r", fontsize=9)
ax.axhline(1, color="k", lw=.8); ax.set_yscale("log")
ax.set_xticks(xs); ax.set_xticklabels([str(r["num"]) for r in rows], rotation=90, fontsize=7)
ax.set_ylabel("TVM time / our final model time  (>1: ours faster)"); ax.set_xlabel("KernelBench L2 problem (x = TVM failed)")
ax.set_title("TVM-OpenCL (Adreno) vs ModelBlaster/KernelBlaster, L2 fp16, QRB5165  (blue: adreno/fp16 path, hatched: fallback)")
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "level2_tvm_vs_ours.png"), dpi=130)
