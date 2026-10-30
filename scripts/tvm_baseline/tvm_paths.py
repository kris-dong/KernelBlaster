"""Paths and endpoints shared by the TVM-OpenCL (Adreno) baseline scripts.

Every value can be overridden from the environment; the defaults are relative
to the KernelBlaster checkout these scripts live in.
"""
import os

ROOT = os.environ.get("KB_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# the 52 OpenCL Level1 problems (names / numbering of the sweep)
L1 = os.environ.get("TVM_L1_DIR", os.path.join(ROOT, "data/benchmark-opencl/L1"))
# KernelBench level1 reference models (<num>_<name>.py), a KernelBench checkout
REFDIR = os.environ.get("TVM_REF_DIR", os.path.join(ROOT, "data/KernelBench/KernelBench/level1"))
# aarch64 cross gcc that links the board-side .so (see docs/tvm_opencl_baseline.md)
XC = os.environ.get("TVM_XCC", os.path.join(ROOT, ".xgcc/bin/aarch64-conda-linux-gnu-gcc"))
# the board's tvm_rpc, reached through an ssh tunnel (it listens on the board's localhost only)
RPC_HOST = os.environ.get("TVM_RPC_HOST", "127.0.0.1")
RPC_PORT = int(os.environ.get("TVM_RPC_PORT", "9191"))
OUT_DIR = os.environ.get("TVM_OUT_DIR", os.path.join(ROOT, "out/tvm_baseline"))
DATA = os.path.join(OUT_DIR, "data")
PLOTS = os.path.join(OUT_DIR, "plots")
RESULTS = os.path.join(DATA, "level1_tvm_opencl_results.json")
# the QNN vendor baseline and KernelBlaster's OpenCL results compared against
QNN = os.environ.get("TVM_QNN_JSON", os.path.join(ROOT, "level1_qnn_qualcomm_results_fixed.json"))
KB_SUMMARY = os.environ.get("TVM_KB_SUMMARY", os.path.join(ROOT, "out/opencl_speedup_summary.csv"))

# ---------------------------------------------------------------- Level 2 (ModelBlaster / KernelBlaster QRB5165 run)
# the ModelBlaster qrb_a6 run directory: problems_l2/, ir16_mb16/, config*.json, results*/ (see its README.txt)
QRB = os.environ.get("TVM_QRB_DIR", os.path.abspath(os.path.join(ROOT, "..", "l3_farm", "qrb_a6")))
L2_PROBLEMS = os.environ.get("TVM_L2_PROBLEMS", os.path.join(QRB, "problems_l2"))   # KernelBench L2 <num>_<name>.py
L2_IR = os.environ.get("TVM_L2_IR", os.path.join(QRB, "ir16_mb16"))                   # <name>/{graph.json,io.npz,weights.npz}
L2_CONFIGS = os.environ.get("TVM_L2_CONFIGS", ",".join(os.path.join(QRB, c) for c in
                            ("config.json", "config_2a.json", "config_2b.json", "config_2c.json"))).split(",")
L2_DIMS = os.environ.get("TVM_L2_DIMS", os.path.join(QRB, "tvm_baseline/data/l2_dims.json"))  # from l2_dims.py
L2_RESULTS = os.path.join(DATA, "level2_tvm_opencl_results.json")
L2_ART = os.environ.get("TVM_L2_ART", os.path.join(OUT_DIR, "art"))                  # compiled .so + meta per attempt
# our flow's result dirs (work/summary.json per problem)
L2_OURS = os.environ.get("TVM_L2_OURS", ",".join(os.path.join(QRB, d) for d in
                         ("results", "results_b", "results_c", "results_2a", "results_2b", "results_2c"))).split(",")
# scripts that start/stop the board's RPC server under the board lock (board side only under /root/user_mb_t)
OPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops")

# ---------------------------------------------------------------- Level 3 (same run directory, KernelBench L3 whole models, 18 problems)
L3_PROBLEMS = os.environ.get("TVM_L3_PROBLEMS", os.path.join(QRB, "problems_l3"))     # KernelBench L3 <num>_<name>.py
L3_IR = os.environ.get("TVM_L3_IR", os.path.join(QRB, "ir_l3_mb16"))                  # <name>/{graph.json,io.npz,weights.npz}
L3_CONFIGS = os.environ.get("TVM_L3_CONFIGS", ",".join(os.path.join(QRB, c) for c in
                            ("config_3a.json", "config_3b.json", "config_3c.json"))).split(",")
L3_DIMS = os.environ.get("TVM_L3_DIMS", os.path.join(QRB, "tvm_baseline/data/l3_dims.json"))  # from l3_dims.py
L3_RESULTS = os.path.join(DATA, "level3_tvm_opencl_results.json")
L3_ART = os.environ.get("TVM_L3_ART", os.path.join(OUT_DIR, "art_l3"))                # compiled .so + meta per attempt
# our flow's result dirs; per problem the best of the original and the rerun dir is taken by compare_baselines_l3.py
L3_OURS = os.environ.get("TVM_L3_OURS", ",".join(os.path.join(QRB, d) for d in
                         ("results_3a", "results_3b", "results_3c", "results_3ra", "results_3rb", "results_3rc"))).split(",")
