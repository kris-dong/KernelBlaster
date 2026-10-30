#!/bin/bash
# copy the existing runtime libs + tvm_rpc (read-only use of /root/tvm/build) into OUR dir on the board
ssh -o BatchMode=yes -o ConnectTimeout=25 qrb5165 'bash -s' <<'R'
set -e
D=/root/krisd_mb_t
df -h / | tail -1
mkdir -p $D/rt $D/work
for f in libtvm_ffi.so libtvm_runtime.so libtvm_runtime_extra.so libtvm_runtime_opencl.so; do cp /root/tvm/build/lib/$f $D/rt/; done
cp /root/tvm/build/tvm_rpc $D/rt/
cd $D/rt; ls -la
LD_LIBRARY_PATH=$D/rt ldd ./tvm_rpc
readelf -d tvm_rpc | grep -iE "rpath|runpath|needed"
du -sh $D
R
