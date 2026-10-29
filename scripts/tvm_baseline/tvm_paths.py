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
