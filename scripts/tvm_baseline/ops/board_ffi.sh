#!/bin/bash
ssh -o BatchMode=yes -o ConnectTimeout=25 qrb5165 'cd /root/tvm; git submodule status 2>&1 | head; cat 3rdparty/tvm-ffi/pyproject.toml 2>/dev/null | grep -m2 version; ls build/lib -la; ldd build/lib/libtvm_runtime.so | head -20'
