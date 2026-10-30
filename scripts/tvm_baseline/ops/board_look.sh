#!/bin/bash
# read-only board survey
ssh -o BatchMode=yes -o ConnectTimeout=25 qrb5165 'bash -s' <<'R'
uname -a; who; echo busy=$(cat /sys/class/kgsl/kgsl-3d0/gpu_busy_percentage); df -h / /tmp | cat
ls -la /root/tvm/build | head -50
ls /root/tvm/build/*.so /root/tvm/build/lib 2>/dev/null
cat /root/tvm/version.py 2>/dev/null | grep -m3 -i version; cd /root/tvm && git describe --tags 2>/dev/null; git log -1 --format=%H%n%cd 2>/dev/null
grep -E "USE_OPENCL|USE_LLVM|USE_CPP_RPC|CMAKE_BUILD_TYPE" /root/tvm/build/CMakeCache.txt | head
ps aux | grep -Ei "tvm|opencl|python|mb_|krisd" | grep -v grep | head
ls /root | head -30; ls /root/mb_board.lock; python3 --version
R
