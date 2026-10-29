#!/usr/bin/env python3
"""TVM-OpenCL (Adreno) optimization-based compiler baseline over KernelBench L1.

For each of the 52 benchmark-opencl/L1 problems:
  KernelBench reference model.py -> fp16 -> ONNX -> Relax (relax.frontend.onnx)
  -> tvm.relax.backend.adreno.get_default_pipeline (Adreno OpenCL scheduling,
     incl. texture memory) -> relax.build(opencl+aarch64) -> cross-linked .so
  -> RPC upload to board -> run on Adreno GPU -> time_evaluator GPU time.

Output schema mirrors level1_qnn_qualcomm_results_fixed.json so the two baselines
can be compared directly (avg_ms_gpu_kernel).

Methodology note: TVM's time_evaluator wraps the VM invoke with a device sync and
loops `number` times per `repeat`, after a warmup — i.e. amortized on-device
latency (QNN methodology "Option B"). Reported avg_ms_gpu_kernel = min over repeats
of the per-call mean. This is the amortized on-device op latency, comparable to the
QNN per-op GPU numbers and our cl_event kernel time to within VM-dispatch overhead.
"""
import os, re, sys, glob, json, tempfile, subprocess, traceback, importlib.util
import numpy as np
import torch
import tvm
from tvm import relax, rpc
from tvm.relax.frontend.onnx import from_onnx
from tvm.relax.backend.adreno import get_default_pipeline
from tvm.relax.backend.gpu_generic import get_default_pipeline as get_generic_pipeline

# Compilation-strategy cascade: (pipeline, dtype). The first entry is the fully
# optimized Adreno-OpenCL path (texture memory etc.); the rest are conservative
# fallbacks that trade performance for validity/coverage. Every result records
# which path produced it so fallback numbers are never conflated with optimized ones.
STRATEGIES = [
    ("adreno",      "float16"),   # optimized (primary)
    ("gpu_generic", "float16"),   # fallback: simple thread-binding, valid work-groups
    ("gpu_generic", "float32"),   # fallback: fp32 (fixes fp16-constant dtype mismatches)
]
# For retrying KNOWN failures, skip the (slow, known-failing) optimized adreno path.
if os.environ.get("TVM_FALLBACK_ONLY") == "1":
    STRATEGIES = [("gpu_generic", "float16"), ("gpu_generic", "float32")]

from tvm_paths import L1, REFDIR, XC, RPC_HOST, RPC_PORT, RESULTS as OUT

TARGET = tvm.target.Target({"kind": "opencl", "device": "adreno"},
                           host={"kind": "llvm", "mtriple": "aarch64-linux-gnu"})

def xcc(output, objects, **kw):
    if isinstance(objects, str): objects = [objects]
    subprocess.check_call([XC, "-shared", "-fPIC", "-o", output] + list(objects))

def load_ref(num):
    f = glob.glob(os.path.join(REFDIR, f"{num}_*.py"))
    if not f: return None
    spec = importlib.util.spec_from_file_location(f"kb_{num}", f[0])
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m, os.path.basename(f[0])[:-3]

def build_onnx(m, path, dt="float16"):
    init = m.get_init_inputs() if hasattr(m, "get_init_inputs") else []
    init = init or []
    tdt = torch.float16 if dt == "float16" else torch.float32
    model = m.Model(*init).eval().to(tdt)
    raw = m.get_inputs()
    # cast tensor float inputs to the chosen dtype; leave non-tensor scalars as-is.
    inputs = []
    for t in raw:
        if torch.is_tensor(t):
            inputs.append(t.to(tdt) if torch.is_floating_point(t) else t)
        else:
            inputs.append(t)
    # only TENSOR inputs become named ONNX graph inputs; scalars bake in as constants
    names, shapes, dtypes = [], {}, {}
    for i, t in enumerate(inputs):
        if torch.is_tensor(t):
            n = f"in{i}"; names.append(n)
            shapes[n] = tuple(t.shape)
            dtypes[n] = (dt if torch.is_floating_point(t)
                         else str(t.dtype).replace("torch.", ""))
    torch.onnx.export(model, tuple(inputs), path, input_names=names,
                      opset_version=17, dynamo=False)
    return shapes, dtypes, inputs

def _attempt(num, tmp, pipe_name, dt):
    """One full compile+run attempt with a given (pipeline, dtype). Returns
    (gpu_ms, wall_ms, shapes) on success; raises with a stage-tagged message."""
    import onnx
    op = os.path.join(tmp, f"m{num}_{pipe_name}_{dt}.onnx")
    try:
        mod_py, _ = load_ref(num)
        shapes, dtypes, _ = build_onnx(mod_py, op, dt)
    except Exception as e:
        raise RuntimeError(f"onnx_export: {type(e).__name__}: {str(e)[:140]}")
    try:
        irmod = from_onnx(onnx.load(op), shape_dict=shapes, dtype_dict=dtypes)
    except Exception as e:
        raise RuntimeError(f"relax_import: {type(e).__name__}: {str(e)[:140]}")
    try:
        pipe = get_default_pipeline(TARGET) if pipe_name == "adreno" else get_generic_pipeline(TARGET)
        irmod = pipe(irmod)
        ex = relax.build(irmod, target=TARGET)
        so = os.path.join(tmp, f"m{num}_{pipe_name}_{dt}.so")
        ex.export_library(so, fcompile=xcc)
    except Exception as e:
        raise RuntimeError(f"compile: {type(e).__name__}: {str(e)[:140]}")
    try:
        remote = rpc.connect(RPC_HOST, RPC_PORT)
        dev = remote.device("opencl", 0)
        remote.upload(so)
        vm = relax.VirtualMachine(remote.load_module(os.path.basename(so)), dev)
        args = []
        for n in shapes:
            arr = (np.random.randn(*shapes[n]).astype(dtypes[n]) if dtypes[n].startswith("float")
                   else np.random.randint(0, 4, size=shapes[n]).astype(dtypes[n]))
            args.append(tvm.runtime.tensor(arr, dev))
        vm.set_input("main", *args)
        vm.invoke_stateful("main")   # warmup
        dev.sync()
        res = vm.time_evaluator("invoke_stateful", dev, number=3, repeat=2, min_repeat_ms=30)("main")
        return round(min(res.results) * 1e3, 4), round(res.mean * 1e3, 4), shapes
    except Exception as e:
        raise RuntimeError(f"run: {type(e).__name__}: {str(e)[:140]}")


def run_problem(num, name, tmp):
    """Try the strategy cascade; record which path produced the result. `path`
    == 'adreno/float16' is the optimized number; anything else is a fallback."""
    rec = {"name": name, "status": "fail", "stage": "", "path": None,
           "avg_ms_gpu_kernel": None, "avg_ms_wallclock": None, "note": ""}
    if load_ref(num) is None:
        rec["note"] = "no KernelBench ref"; return rec
    errs = []
    for pipe_name, dt in STRATEGIES:
        path = f"{pipe_name}/{dt}"
        try:
            gpu_ms, wall_ms, shapes = _attempt(num, tmp, pipe_name, dt)
            rec.update(status="ok", stage="run", path=path,
                       avg_ms_gpu_kernel=gpu_ms, avg_ms_wallclock=wall_ms,
                       note=("optimized" if path == "adreno/float16" else "FALLBACK path")
                            + f" | shapes={shapes}")
            return rec
        except Exception as e:
            errs.append(f"{path} -> {e}")
    rec["stage"] = errs[-1].split("-> ")[-1].split(":")[0] if errs else "unknown"
    rec["note"] = " ;; ".join(errs)[:400]
    return rec

def main():
    probs = sorted([p for p in os.listdir(L1) if os.path.isdir(os.path.join(L1, p))],
                   key=lambda p: int(re.match(r'0*(\d+)_', p).group(1)))
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    results = {}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        results = json.load(open(OUT))
    tmp = tempfile.mkdtemp(prefix="tvmocl_")
    for p in probs:
        num = int(re.match(r'0*(\d+)_', p).group(1))
        if only and str(num) not in only: continue
        rec = run_problem(num, p, tmp)
        results[str(num)] = rec
        tag = "OK  " if rec["status"] == "ok" else "FAIL"
        ms = f"{rec['avg_ms_gpu_kernel']} ms" if rec["avg_ms_gpu_kernel"] else rec["note"][:70]
        print(f"[{tag}] {num:>3} {p[:44]:44} {ms}", flush=True)
        json.dump(results, open(OUT, "w"), indent=2)
    ok = sum(1 for r in results.values() if r["status"] == "ok")
    print(f"\n=== {ok}/{len(results)} ok. saved {OUT}")

if __name__ == "__main__":
    main()
