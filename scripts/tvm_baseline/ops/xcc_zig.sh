#!/bin/bash
# aarch64 cross "gcc" for TVM export_library: zig cc (pip ziglang) targeting the board's glibc 2.31 (Ubuntu 20.04)
exec "${TVM_VENV:-/Users/krisd/Documents/ModelBlaster-clone/l3_farm/qrb_a6/tvm_baseline/venv}/bin/python" -m ziglang cc -target aarch64-linux-gnu.2.31 "$@"
