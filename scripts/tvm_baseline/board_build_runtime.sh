#!/usr/bin/env bash
# Build the TVM runtime (OpenCL) and the C++ RPC server ON THE ADRENO BOARD
# (RB5 / QRB5165, Ubuntu aarch64). Runtime only: compilation happens on the x86
# host (run_tvm_opencl_baseline.py); the board only loads and runs the .so.
#
#   ./board_build_runtime.sh [TVM_DIR] [TVM_REF]
#
# Reconstructed from the setup in docs/tvm_opencl_baseline.md (TVM v0.25.0,
# USE_OPENCL=ON, USE_CPP_RPC=ON, /root/tvm/build); not yet re-run on a board.
set -euo pipefail
TVM_DIR=${1:-/root/tvm}
TVM_REF=${2:-v0.25.0}

if [ ! -d "$TVM_DIR/.git" ]; then
    git clone --recursive --branch "$TVM_REF" https://github.com/apache/tvm "$TVM_DIR"
fi
mkdir -p "$TVM_DIR/build"
cd "$TVM_DIR/build"
cp ../cmake/config.cmake .
{
    echo 'set(USE_OPENCL ON)'
    echo 'set(USE_CPP_RPC ON)'
    echo 'set(USE_LLVM OFF)'
    echo 'set(USE_CUDA OFF)'
} >> config.cmake
cmake ..
make -j"$(nproc)" runtime tvm_rpc
ls -l libtvm_runtime*.so* tvm_rpc
