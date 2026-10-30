#!/bin/bash
# the L3 sweep (needs ops/tunnel.sh running in another shell): optimized path + cascade fallbacks, board runs under the board lock
cd "$(dirname "$0")/.."
. "${TVM_VENV:-/home/user/Documents/ModelBlaster-clone/l3_farm/qrb_a6/tvm_baseline/venv}/bin/activate"
export TVM_XCC=$PWD/ops/xcc_zig.sh
export TVM_OUT_DIR=${TVM_OUT_DIR:-/home/user/Documents/ModelBlaster-clone/l3_farm/qrb_a6/tvm_baseline}
exec python run_tvm_opencl_baseline_l3.py "$@"
