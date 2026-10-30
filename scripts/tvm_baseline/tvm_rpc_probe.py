#!/usr/bin/env python3
"""Probe: compile a matmul on host, run on Adreno GPU via RPC, time it."""
import os, subprocess, tempfile, numpy as np
import tvm
from tvm import relax, rpc
from tvm.relax.frontend import nn
from tvm.relax.backend.adreno import get_default_pipeline

from tvm_paths import XC, RPC_HOST, RPC_PORT

def xcc(output, objects, **kwargs):
    if isinstance(objects, str):
        objects = [objects]
    cmd = [XC, "-shared", "-fPIC", "-o", output] + list(objects)
    subprocess.check_call(cmd)

def main():
    N = 512
    class MM(nn.Module):
        def forward(self, a, b): return nn.op.matmul(a, b)
    mod, _ = MM().export_tvm(spec={"forward": {
        "a": nn.spec.Tensor((N, N), "float16"),
        "b": nn.spec.Tensor((N, N), "float16")}})
    target = tvm.target.Target({"kind": "opencl", "device": "adreno"},
                               host={"kind": "llvm", "mtriple": "aarch64-linux-gnu"})
    mod = get_default_pipeline(target)(mod)
    ex = relax.build(mod, target=target)
    d = tempfile.mkdtemp()
    so = os.path.join(d, "mm.so")
    ex.export_library(so, fcompile=xcc)
    print("exported", so, os.path.getsize(so), "bytes")

    remote = rpc.connect(RPC_HOST, RPC_PORT)
    print("RPC connected")
    dev = remote.device("opencl", 0)
    print("remote opencl exist:", dev.exist)
    if dev.exist:
        print("  device:", dev.device_name, "| max_threads:", dev.max_threads_per_block,
              "| warp:", dev.warp_size)
    remote.upload(so)
    rmod = remote.load_module("mm.so")
    vm = relax.VirtualMachine(rmod, dev)
    a = tvm.runtime.tensor(np.random.randn(N, N).astype("float16"), dev)
    b = tvm.runtime.tensor(np.random.randn(N, N).astype("float16"), dev)
    vm.set_input("forward", a, b)
    vm.invoke_stateful("forward")
    out = vm.get_outputs("forward")
    dev.sync()
    print("ran matmul, output shape:", out.shape, "dtype", out.dtype)
    # time via module time_evaluator on the packed func
    ftimer = vm.time_evaluator("invoke_stateful", dev, number=10, repeat=5)
    res = ftimer("forward")
    print("Adreno GPU matmul %dx%d fp16: mean=%.4f ms (min=%.4f ms over %d runs)" %
          (N, N, res.mean*1e3, min(res.results)*1e3, len(res.results)))

if __name__ == "__main__":
    main()
