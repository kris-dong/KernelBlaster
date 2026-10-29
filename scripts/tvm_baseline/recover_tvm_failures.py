#!/usr/bin/env python3
"""Round-2 fallback recovery for the remaining TVM failures, with targeted,
reasonable compiler/harness techniques (each result tagged with its path):

  - max_num_threads cap (128/64)  -> work-group-size failures (softmax, etc.)
  - scalar constant-baking          -> ops with a python-scalar forward arg
  - dynamo ONNX exporter            -> ops torchscript export can't emit (diag,
                                       smooth_l1_loss, cross-entropy) — decomposed
                                       into primitives the relax frontend supports

Only touches the ops passed on the CLI; merges into level1_tvm_opencl_results.json.
"""
import os, re, sys, glob, json, tempfile, subprocess, importlib.util
import numpy as np, torch, tvm, onnx
from tvm import relax, rpc
from tvm.relax.frontend.onnx import from_onnx
from tvm.relax.backend.adreno import get_default_pipeline as adreno_pipe
from tvm.relax.backend.gpu_generic import get_default_pipeline as generic_pipe

from tvm_paths import REFDIR, XC, RPC_HOST, RPC_PORT, RESULTS as OUT
def xcc(o,objs,**k): subprocess.check_call([XC,"-shared","-fPIC","-o",o]+([objs] if isinstance(objs,str) else list(objs)))

def load_ref(num):
    f=glob.glob(os.path.join(REFDIR,f"{num}_*.py"))[0]
    spec=importlib.util.spec_from_file_location(f"kb{num}",f); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

class ScalarBaked(torch.nn.Module):
    """Wrap a model so forward takes only TENSOR inputs; python-scalar args are
    captured as constants (fixes 'params 1 vs 2' when a scalar becomes a graph input)."""
    def __init__(self, model, raw):
        super().__init__(); self.model=model
        self.tensor_pos=[i for i,t in enumerate(raw) if torch.is_tensor(t)]
        self.scalars={i:t for i,t in enumerate(raw) if not torch.is_tensor(t)}
        self.n=len(raw)
    def forward(self, *tensors):
        args=[None]*self.n
        for pos,t in zip(self.tensor_pos,tensors): args[pos]=t
        for i,v in self.scalars.items(): args[i]=v
        return self.model(*args)

class CrossEntropyDecomposed(torch.nn.Module):
    """Exact decomposition of F.cross_entropy(pred, target) into
    -mean(log_softmax(pred)[i, target[i]]) — identical math, but emits only
    log_softmax + gather + mean, which the relax ONNX frontend supports."""
    def forward(self, predictions, targets):
        logp = torch.log_softmax(predictions, dim=1)
        picked = logp.gather(1, targets.long().unsqueeze(1)).squeeze(1)
        return -picked.mean()

# ops that need an exact decomposition to get through the frontend at all
DECOMP = {95: CrossEntropyDecomposed}

def make_onnx(m, path, dt, exporter="script", bake_scalar=False, decomp=None):
    init=m.get_init_inputs() or []
    tdt=torch.float16 if dt=="float16" else torch.float32
    model=(decomp() if decomp else m.Model(*init)).eval().to(tdt)
    raw=m.get_inputs()
    tensors=[t.to(tdt) if (torch.is_tensor(t) and torch.is_floating_point(t)) else t for t in raw]
    if bake_scalar:
        model=ScalarBaked(model, tensors)
        ex_inputs=tuple(t for t in tensors if torch.is_tensor(t))
    else:
        ex_inputs=tuple(tensors)
    names,shapes,dtypes=[],{},{}
    ti=0
    for t in ex_inputs:
        if torch.is_tensor(t):
            n=f"in{ti}"; ti+=1; names.append(n); shapes[n]=tuple(t.shape)
            dtypes[n]=dt if torch.is_floating_point(t) else str(t.dtype).replace("torch.","")
    if exporter=="dynamo":
        torch.onnx.export(model, ex_inputs, path, input_names=names, dynamo=True, optimize=True)
    else:
        torch.onnx.export(model, ex_inputs, path, input_names=names, opset_version=17, dynamo=False)
    return shapes,dtypes

def target_for(mnt=None):
    attrs={"kind":"opencl","device":"adreno"}
    if mnt: attrs["max_num_threads"]=mnt
    return tvm.target.Target(attrs, host={"kind":"llvm","mtriple":"aarch64-linux-gnu"})

def attempt(num, tmp, pipe_name, dt, mnt=None, exporter="script", bake=False):
    tag=f"{pipe_name}/{dt}" + (f"/mnt{mnt}" if mnt else "") + (f"/{exporter}" if exporter!="script" else "") + ("/scalar_baked" if bake else "")
    op=os.path.join(tmp,f"m{num}_{tag.replace('/','_')}.onnx")
    shapes,dtypes=make_onnx(load_ref(num),op,dt,exporter,bake,DECOMP.get(num))
    if num in DECOMP: tag += "/decomposed"
    irmod=from_onnx(onnx.load(op),shape_dict=shapes,dtype_dict=dtypes)
    tgt=target_for(mnt)
    irmod=(adreno_pipe(tgt) if pipe_name=="adreno" else generic_pipe(tgt))(irmod)
    ex=relax.build(irmod,target=tgt)
    so=os.path.join(tmp,f"m{num}_{tag.replace('/','_')}.so"); ex.export_library(so,fcompile=xcc)
    r=rpc.connect(RPC_HOST,RPC_PORT); dev=r.device("opencl",0); r.upload(so)
    vm=relax.VirtualMachine(r.load_module(os.path.basename(so)),dev)
    args=[]
    for n in shapes:
        a=np.random.randn(*shapes[n]).astype(dtypes[n]) if dtypes[n].startswith("float") else np.random.randint(0,4,shapes[n]).astype(dtypes[n])
        args.append(tvm.runtime.tensor(a,dev))
    vm.set_input("main",*args); vm.invoke_stateful("main"); dev.sync()
    res=vm.time_evaluator("invoke_stateful",dev,number=3,repeat=2,min_repeat_ms=30)("main")
    return tag, round(min(res.results)*1e3,4), round(res.mean*1e3,4), shapes

# Per-op targeted strategy lists (only reasonable techniques for each failure mode)
PLANS={
  # work-group failures at run -> cap threads
  23:[("gpu_generic","float32",64,"script",False),("gpu_generic","float32",128,"script",False)],
  24:[("gpu_generic","float32",64,"script",False),("gpu_generic","float32",128,"script",False)],
  34:[("gpu_generic","float32",64,"script",False),("gpu_generic","float32",128,"script",False)],
  # compile internal errors -> capped threads may change codegen path (one shot each)
  36:[("gpu_generic","float32",64,"script",False)],
  94:[("gpu_generic","float32",64,"script",False)],
  100:[("gpu_generic","float32",64,"script",False)],
  # scalar param mismatch -> bake scalar constant
  5:[("gpu_generic","float16",None,"script",True),("gpu_generic","float32",None,"script",True)],
  # unsupported torchscript export / relax frontend -> dynamo exporter (decomposes)
  12:[("gpu_generic","float32",None,"dynamo",False),("gpu_generic","float16",None,"dynamo",False)],
  95:[("gpu_generic","float32",None,"script",False),("gpu_generic","float16",None,"script",False)],
  96:[("gpu_generic","float32",None,"dynamo",False)],
}

if __name__=="__main__":
    tmp=tempfile.mkdtemp(prefix="rec2_")
    d=json.load(open(OUT))
    for num in [int(x) for x in sys.argv[1:]]:
        errs=[]
        for (pipe,dt,mnt,exp,bake) in PLANS.get(num,[]):
            try:
                tag,gpu,wall,shapes=attempt(num,tmp,pipe,dt,mnt,exp,bake)
                d[str(num)]={"name":d[str(num)]["name"],"status":"ok","stage":"run","path":tag,
                             "avg_ms_gpu_kernel":gpu,"avg_ms_wallclock":wall,
                             "note":f"FALLBACK path | shapes={shapes}"}
                json.dump(d,open(OUT,"w"),indent=2)
                print(f"[OK  ] {num} via {tag}: {gpu} ms",flush=True); break
            except Exception as e:
                errs.append(f"{pipe}/{dt}/mnt{mnt}/{exp}/bake{bake}: {type(e).__name__}: {str(e)[:80]}")
        else:
            print(f"[FAIL] {num}: "+" ;; ".join(errs)[:200],flush=True)
