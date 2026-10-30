#!/usr/bin/env python3
"""TVM-OpenCL (Adreno) compiler baseline over the KernelBench Level 2 problems of the ModelBlaster/KernelBlaster QRB5165 run.

Same method as run_tvm_opencl_baseline.py (L1): reference model -> fp16 -> ONNX(opset 17) -> relax.frontend.onnx ->
tvm.relax.backend.adreno pipeline -> relax.build(opencl -device=adreno, host llvm aarch64) -> cross-linked .so ->
RPC to the board -> relax.VirtualMachine on the Adreno GPU -> time_evaluator (min over repeats of the per-call mean).
Differences for L2: the problems are the 48 of the run (config*.json 'problems'); the model dims are the fp16 io-capped
dims the extractor applied (extract_graph --bench-target-mb 16; l2_dims.py records them, verified against ir16_mb16);
every compile/run attempt is a subprocess (a TVM crash or GPU hang cannot take the sweep down); every GPU session starts the
board's tvm_rpc UNDER `flock /root/mb_board.lock` (ops/board_rpc_start.sh) and stops it afterwards (ops/board_rpc_stop.sh).

usage:  run_tvm_opencl_baseline_l2.py [--stage compile|run|all] [--jobs N] [--force] [NAME_OR_NUM ...]
        run_tvm_opencl_baseline_l2.py --worker-compile NAME PIPE DT OUTDIR [MNT] [EXPORTER]   (internal)
        run_tvm_opencl_baseline_l2.py --worker-run OUTDIR                                      (internal)
needs the ssh tunnel (ops/tunnel.sh) up for the run stage.  Results: tvm_paths.L2_RESULTS (schema of level1_tvm_opencl_results.json,
keyed by problem number, + rel_err/dims/verify fields).
"""
from __future__ import annotations
import os, re, sys, json, glob, time, copy, subprocess, importlib.util, traceback
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tvm_paths import (L2_PROBLEMS, L2_IR, L2_CONFIGS, L2_DIMS, L2_RESULTS, L2_ART, XC, RPC_HOST, RPC_PORT, OPS)

# (pipeline, dtype, max_num_threads, exporter): L1's cascade, then the L1 recovery tricks (recover_tvm_failures.py)
# adreno_texture = the same Adreno pipeline with the target tag qcom/adreno-opencl-texture (keys +texture); L1's "adreno" target has no
# texture key, so the pipeline's texture-memory passes were never active there.  Kept as separate paths; see REPORT.md.
CASCADE = [("adreno", "float16", None, "script"),
           ("gpu_generic", "float16", None, "script"),
           ("gpu_generic", "float32", None, "script")]
TEXTURE = ("adreno_texture", "float16", None, "script")
RECOVER = [("gpu_generic", "float32", 64, "script"), ("gpu_generic", "float16", 64, "script"),
           ("gpu_generic", "float32", 32, "script"), ("gpu_generic", "float32", None, "dynamo"),
           ("gpu_generic", "float16", None, "dynamo")]
COMPILE_TIMEOUT = int(os.environ.get("TVM_COMPILE_TIMEOUT", "1500"))
REL_TOL = float(os.environ.get("TVM_REL_TOL", "0.05"))   # max|tvm-ref|/max|ref| vs the fp32 CPU reference
RUN_TIMEOUT = int(os.environ.get("TVM_RUN_TIMEOUT", "1200"))

def tag_of(pipe, dt, mnt=None, exporter="script"):
    return f"{pipe}/{dt}" + (f"/mnt{mnt}" if mnt else "") + (f"/{exporter}" if exporter != "script" else "")

def problems():
    names = []
    for c in L2_CONFIGS:
        names += json.load(open(c))["problems"]
    return names

def num_of(name): return int(name.split("_")[0])

# ------------------------------------------------------------------------------------------------ model (workers)
def load_model_inputs(name, dt):
    """The KernelBench model at the io-capped dims of ir16_mb16: module attrs overridden with the extractor's shrink."""
    import torch
    f = os.path.join(L2_PROBLEMS, name + ".py")
    spec = importlib.util.spec_from_file_location("kb_" + name, f); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    dims = json.load(open(L2_DIMS))[name]["changed"]
    for k, v in dims.items():
        setattr(m, k, v)
    torch.manual_seed(0)
    model = m.Model(*(m.get_init_inputs() or [])).eval()
    inputs = m.get_inputs()
    return m, model, inputs, dims

def verify_vs_ir(name, model, inputs, out):
    """The TVM model must have the shapes/params of ours: graph.json input/output shapes, io.npz and weights.npz element counts."""
    import numpy as np
    d = os.path.join(L2_IR, name)
    g = json.load(open(os.path.join(d, "graph.json")))
    ishape = tuple(g["tensors"][g["input"]["tensor"]]["shape"]); oshape = tuple(g["tensors"][g["output"]["tensor"]]["shape"])
    z = np.load(os.path.join(d, "io.npz")); w = np.load(os.path.join(d, "weights.npz"))
    ir_w = sum(int(w[k].size) for k in w.files)
    my_w = sum(int(p.numel()) for p in list(model.parameters()) + list(model.buffers()) if p.dtype.is_floating_point)
    t_in = [tuple(t.shape) for t in inputs if hasattr(t, "shape")]
    v = {"ir_input": list(ishape), "ir_output": list(oshape), "my_input": [list(s) for s in t_in], "my_output": list(out.shape),
         "ir_in_numel": int(z["input"].size), "my_in_numel": int(sum(int(np.prod(s)) for s in t_in)),
         "ir_out_numel": int(z["output"].size), "my_out_numel": int(out.numel()), "ir_weight_numel": ir_w, "my_param_numel": my_w}
    v["ok"] = (v["ir_in_numel"] == v["my_in_numel"] and v["ir_out_numel"] == v["my_out_numel"] and ishape in t_in and oshape == tuple(out.shape))
    v["weights_equal"] = ir_w == my_w   # BN running stats / folded consts can legitimately differ: informational
    return v

def worker_compile(name, pipe, dt, outdir, mnt=None, exporter="script"):
    import numpy as np, torch, onnx, tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    os.makedirs(outdir, exist_ok=True)
    meta = {"name": name, "path": tag_of(pipe, dt, mnt, exporter)}
    t0 = time.time()
    m, model, inputs, dims = load_model_inputs(name, dt)
    tdt = torch.float16 if dt == "float16" else torch.float32
    # fp16-rounded inputs/weights; fp32 CPU reference on the rounded values (a sanity check of every path, never a timing)
    ins16 = [t.half() if torch.is_floating_point(t) else t for t in inputs]
    model.half().float()
    with torch.no_grad():
        ref = model(*[t.float() if torch.is_floating_point(t) else t for t in ins16])
    meta["verify"] = verify_vs_ir(name, model, inputs, ref)
    np.save(os.path.join(outdir, "ref.npy"), ref.float().numpy())
    json.dump({"verify": meta["verify"], "dims": dims}, open(os.path.join(outdir, "verify.json"), "w"))
    ex_model = copy.deepcopy(model).to(tdt).eval()
    ex_in = [t.to(tdt) if torch.is_floating_point(t) else t for t in ins16]
    names, shapes, dtypes = [], {}, {}
    for i, t in enumerate(ex_in):
        n = f"in{i}"; names.append(n); shapes[n] = tuple(t.shape)
        dtypes[n] = dt if torch.is_floating_point(t) else str(t.dtype).replace("torch.", "")
        np.save(os.path.join(outdir, n + ".npy"), t.numpy())
    op = os.path.join(outdir, "m.onnx")
    try:
        if exporter == "dynamo":
            torch.onnx.export(ex_model, tuple(ex_in), op, input_names=names, dynamo=True, optimize=True)
        else:
            torch.onnx.export(ex_model, tuple(ex_in), op, input_names=names, opset_version=17, dynamo=False)
    except Exception as e:
        raise RuntimeError(f"onnx_export: {type(e).__name__}: {str(e)[:200]}")
    meta["t_export_s"] = round(time.time() - t0, 1)
    try:
        irmod = from_onnx(onnx.load(op), shape_dict=shapes, dtype_dict=dtypes)
    except Exception as e:
        raise RuntimeError(f"relax_import: {type(e).__name__}: {str(e)[:200]}")
    try:
        attrs = {"kind": "opencl", "device": "adreno"}
        if pipe == "adreno_texture":   # qcom/adreno-opencl-texture: the "texture" key turns on the pipeline's texture-memory passes
            attrs["keys"] = ["adreno", "opencl", "gpu", "texture"]
        if mnt: attrs["max_num_threads"] = mnt
        target = tvm.target.Target(attrs, host={"kind": "llvm", "mtriple": "aarch64-linux-gnu"})
        if pipe in ("adreno", "adreno_texture"):
            from tvm.relax.backend.adreno import get_default_pipeline as gp
        else:
            from tvm.relax.backend.gpu_generic import get_default_pipeline as gp
        irmod = gp(target)(irmod)
        ex = relax.build(irmod, target=target)
        so = os.path.join(outdir, "m.so")
        def xcc(output, objects, **kw):
            subprocess.check_call([XC, "-shared", "-fPIC", "-o", output] + ([objects] if isinstance(objects, str) else list(objects)))
        ex.export_library(so, fcompile=xcc)
    except Exception as e:
        raise RuntimeError(f"compile: {type(e).__name__}: {str(e)[:300]}")
    os.remove(op)
    meta.update(shapes={k: list(v) for k, v in shapes.items()}, dtypes=dtypes, so_bytes=os.path.getsize(so),
                t_compile_s=round(time.time() - t0, 1), dims=dims)
    json.dump(meta, open(os.path.join(outdir, "meta.json"), "w"), indent=1)

def worker_run(outdir):
    import numpy as np, tvm
    from tvm import relax, rpc
    meta = json.load(open(os.path.join(outdir, "meta.json")))
    remote = rpc.connect(RPC_HOST, RPC_PORT)
    dev = remote.device("opencl", 0)
    if not dev.exist:
        raise RuntimeError("run: remote opencl device missing")
    so = os.path.join(outdir, "m.so")
    remote.upload(so, "m.so")
    vm = relax.VirtualMachine(remote.load_module("m.so"), dev)
    args = [tvm.runtime.tensor(np.load(os.path.join(outdir, n + ".npy")), dev) for n in meta["shapes"]]
    vm.set_input("main", *args)
    vm.invoke_stateful("main"); dev.sync()
    out = vm.get_outputs("main")
    out = out.numpy() if hasattr(out, "numpy") else out[0].numpy()
    ref = np.load(os.path.join(outdir, "ref.npy"))
    np.save(os.path.join(outdir, "out.npy"), out)
    o32 = out.astype("float32"); den = float(np.abs(ref).max()) + 1e-6
    rel = float(np.abs(o32 - ref).max() / den) if o32.shape == ref.shape else None
    finite = bool(np.isfinite(o32).all())
    res = vm.time_evaluator("invoke_stateful", dev, number=3, repeat=3, min_repeat_ms=30)("main")
    print("RESULT " + json.dumps({"gpu_ms": round(min(res.results) * 1e3, 4), "wall_ms": round(res.mean * 1e3, 4),
                                  "all_ms": [round(x * 1e3, 4) for x in res.results], "rel_err": rel, "finite": finite,
                                  "out_shape": list(out.shape), "ref_shape": list(ref.shape)}), flush=True)

# ------------------------------------------------------------------------------------------------ orchestration
def sh(script, *args, timeout=120):
    p = subprocess.run([os.path.join(OPS, script)] + [str(a) for a in args], capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()

def adir(name, tag): return os.path.join(L2_ART, name, tag.replace("/", "_"))

def compile_attempt(name, strat, force=False):
    pipe, dt, mnt, exp = strat
    d = adir(name, tag_of(*strat))
    if os.path.exists(os.path.join(d, "meta.json")) and not force:
        return True, ""
    if os.path.exists(os.path.join(d, "error.txt")) and not force:
        return False, open(os.path.join(d, "error.txt")).read()
    os.makedirs(d, exist_ok=True)
    cmd = [sys.executable, os.path.abspath(__file__), "--worker-compile", name, pipe, dt, d, str(mnt or 0), exp]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT, env=dict(os.environ, TVM_XCC=XC))
        ok, err = p.returncode == 0, (p.stderr.strip().splitlines() or [""])[-1] if p.returncode else ""
        if not ok and not err: err = f"rc={p.returncode}"
        if p.returncode < 0: err = f"compile: killed by signal {-p.returncode}"
    except subprocess.TimeoutExpired:
        ok, err = False, f"compile: timeout {COMPILE_TIMEOUT}s"
    if not ok:
        open(os.path.join(d, "error.txt"), "w").write(err)
    return ok, err

def wait_session(max_wait_s=4 * 3600):
    """Start the board's RPC server under the lock (refusing while a foreign user / GPU process is present); wait to be LISTENING."""
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        rc, out = sh("board_rpc_start.sh", RPC_PORT)
        if rc == 3 or "FOREIGN" in out:
            print("  board: foreign activity, waiting:", out[:120], flush=True); time.sleep(120); continue
        if "STARTED" not in out and "ALREADY" not in out:
            print("  board: start failed:", out[:200], flush=True); time.sleep(60); continue
        while time.time() - t0 < max_wait_s:
            rc, st = sh("board_rpc_status.sh", RPC_PORT)
            if st.startswith("LISTENING"):
                return True
            if st.startswith("DEAD"):
                print("  board: server died before listening:", st[:200], flush=True); break
            time.sleep(10)
    return False

def run_attempt(name, strat):
    """One GPU session: start server under the lock, run the worker, ALWAYS stop the server. -> (ok, dict|err)"""
    d = adir(name, tag_of(*strat))
    if not wait_session():
        sh("board_rpc_stop.sh"); return False, "run: board unavailable (lock/foreign) timeout"
    try:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker-run", d], capture_output=True, text=True,
                           timeout=RUN_TIMEOUT, env=dict(os.environ, TVM_XCC=XC))
        m = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if m:
            return True, json.loads(m[-1][7:])
        err = (p.stderr.strip().splitlines() or [f"rc={p.returncode}"])[-1]
        return False, "run: " + err[:300]
    except subprocess.TimeoutExpired:
        return False, f"run: timeout {RUN_TIMEOUT}s"
    finally:
        rc, o = sh("board_rpc_stop.sh", timeout=90)
        if "STILL" in o:
            print("  WARNING: rpc server still running:", o, flush=True)

def run_problem(name, strategies):
    rec = {"name": name, "status": "fail", "stage": "", "path": None, "avg_ms_gpu_kernel": None, "avg_ms_wallclock": None, "note": ""}
    errs, tried = [], {}
    for strat in strategies:
        tag = tag_of(*strat)
        ok, err = compile_attempt(name, strat)
        if not ok:
            errs.append(f"{tag} -> {err}"); continue
        meta = json.load(open(os.path.join(adir(name, tag), "meta.json")))
        ok, r = run_attempt(name, strat)
        if not ok:
            errs.append(f"{tag} -> {r}"); continue
        opt = tag in ("adreno/float16", "adreno_texture/float16")   # an Adreno-pipeline path (numerics checked below)
        wrong = (not r["finite"]) or r["rel_err"] is None or r["rel_err"] > REL_TOL
        att = {"gpu_ms": r["gpu_ms"], "rel_err": r["rel_err"], "finite": r["finite"], "correct": not wrong}
        tried[tag] = att
        if wrong:   # ran, but the numbers are wrong vs the fp32 CPU reference: not a valid baseline, go on down the cascade
            errs.append(f"{tag} -> wrong_result: rel_err={r['rel_err']} finite={r['finite']} (ran in {r['gpu_ms']} ms)"); continue
        rec.update(status="ok", stage="run", path=tag, avg_ms_gpu_kernel=r["gpu_ms"], avg_ms_wallclock=r["wall_ms"],
                   note=("optimized" if opt else "FALLBACK path") + f" | shapes={meta['shapes']}", all_ms=r["all_ms"],
                   rel_err=r["rel_err"], finite=r["finite"], so_bytes=meta["so_bytes"], t_compile_s=meta["t_compile_s"],
                   dims=meta["dims"], verify=meta["verify"], tvm_optimized=opt, tried=tried)
        if errs: rec["earlier_failures"] = errs
        return rec
    rec["stage"] = errs[-1].split("-> ")[-1].split(":")[0] if errs else "unknown"
    rec["note"] = " ;; ".join(errs)[:900]
    rec["attempt_errors"] = errs
    rec["tried"] = tried
    return rec

def main():
    a = sys.argv[1:]
    if a and a[0] == "--worker-compile":
        try:
            worker_compile(a[1], a[2], a[3], a[4], int(a[5]) or None if len(a) > 5 else None, a[6] if len(a) > 6 else "script")
        except Exception as e:
            print(str(e).splitlines()[0][:400] if isinstance(e, RuntimeError) else f"setup: {type(e).__name__}: {str(e)[:300]}", file=sys.stderr)
            sys.exit(1)
        return
    if a and a[0] == "--worker-run":
        try:
            worker_run(a[1])
        except Exception as e:
            print(f"{type(e).__name__}: {str(e)[:300]}", file=sys.stderr); sys.exit(1)
        return
    stage, jobs, force, recover, sel, only_strat = "all", 3, False, False, [], None
    i = 0
    while i < len(a):
        if a[i] == "--stage": stage = a[i + 1]; i += 2
        elif a[i] == "--jobs": jobs = int(a[i + 1]); i += 2
        elif a[i] == "--force": force = True; i += 1
        elif a[i] == "--strategy": only_strat = a[i + 1]; i += 2   # diagnostic: ONLY this path (e.g. gpu_generic/float16); results go to a side JSON
        elif a[i] == "--recover": recover = True; i += 1     # append the L1 recovery strategies (only for problems that failed the cascade)
        else: sel.append(a[i]); i += 1
    names = [n for n in problems() if not sel or n in sel or str(num_of(n)) in sel]
    names.sort(key=num_of)
    os.makedirs(os.path.dirname(L2_RESULTS), exist_ok=True)
    results = json.load(open(L2_RESULTS)) if os.path.exists(L2_RESULTS) else {}
    strategies = CASCADE + (RECOVER if recover else [])
    out_json = L2_RESULTS
    if only_strat:
        strategies = [s for s in CASCADE + [TEXTURE] + RECOVER if tag_of(*s) == only_strat]
        out_json = L2_RESULTS.replace(".json", "." + only_strat.replace("/", "_") + ".json")
        results = json.load(open(out_json)) if os.path.exists(out_json) else {}
    if stage in ("compile", "all"):     # compile the optimized path for everything first (CPU only, no board)
        def c(n):
            ok, err = compile_attempt(n, strategies[0], force)
            print(f"[compile {'OK  ' if ok else 'FAIL'}] {n[:52]:52} {err[:90]}", flush=True)
        with ThreadPoolExecutor(jobs) as ex:
            list(ex.map(c, [n for n in names if force or str(num_of(n)) not in results or results[str(num_of(n))]["status"] != "ok"]))
    if stage == "compile":
        return
    for n in names:
        k = str(num_of(n))
        if k in results and not force and not recover and (results[k]["status"] == "ok" or
                (results[k].get("attempt_errors") and "unavailable" not in results[k]["note"])):
            continue   # done (a cascade failure is final unless --recover / --force)
        if recover and not (k in results and results[k]["status"] != "ok"):
            continue
        rec = run_problem(n, strategies)
        results[k] = rec
        ms = f"{rec['avg_ms_gpu_kernel']} ms ({rec['path']}, rel_err {rec.get('rel_err')})" if rec["status"] == "ok" else rec["note"][:110]
        print(f"[{'OK  ' if rec['status'] == 'ok' else 'FAIL'}] {k:>3} {n[:46]:46} {ms}", flush=True)
        json.dump(results, open(out_json, "w"), indent=2)
    ok = sum(1 for r in results.values() if r["status"] == "ok")
    print(f"\n=== {ok}/{len(results)} ok. saved {out_json}")

if __name__ == "__main__":
    main()
