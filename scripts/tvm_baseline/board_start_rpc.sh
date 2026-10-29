#!/usr/bin/env bash
# Start TVM's C++ RPC server ON THE ADRENO BOARD, bound to the board's localhost.
# An unauthenticated TVM RPC server runs arbitrary uploaded code, so never bind
# it to a network interface; the host reaches it through an ssh tunnel:
#
#   host$ ssh -N -L 9191:127.0.0.1:9191 root@<board>      # then run the sweep
#
#   board$ ./board_start_rpc.sh [TVM_BUILD] [PORT]
#
# libtvm_runtime_opencl.so is preloaded so the Adreno OpenCL device registers.
# Reconstructed from the setup in docs/tvm_opencl_baseline.md; not yet re-run.
set -euo pipefail
TVM_BUILD=${1:-/root/tvm/build}
PORT=${2:-9191}
WORK=${TVM_RPC_WORK:-/root/tvm_rpc_work}     # uploaded .so files; persistent disk, not /tmp
mkdir -p "$WORK"
PRELOAD=""
for lib in "$TVM_BUILD"/libtvm_runtime_opencl.so "$TVM_BUILD"/lib/libtvm_runtime_opencl.so; do
    [ -f "$lib" ] && PRELOAD=$lib && break
done
export LD_LIBRARY_PATH="$TVM_BUILD:$TVM_BUILD/lib:${LD_LIBRARY_PATH:-}"
exec env ${PRELOAD:+LD_PRELOAD=$PRELOAD} "$TVM_BUILD/tvm_rpc" server \
    --host=127.0.0.1 --port="$PORT" --port-end="$((PORT + 1))" --work-dir="$WORK"
