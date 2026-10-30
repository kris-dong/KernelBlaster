#!/usr/bin/env python3
"""Record the io-capped (--bench-target-mb 16) dims the ModelBlaster extractor applies to each KernelBench L3
problem, by running the extractor's own loader (modelblaster.pipeline.extract_graph._load_kernelbench) and
capturing the shrunk module-level int attrs.  Output: l3_dims.json {problem: {attr: value}} (changed attrs only,
plus all). Needs torch + the ModelBlaster checkout on PYTHONPATH (MB_PP) ; CPU only.
usage: l3_dims.py PROBLEMS_DIR OUT_JSON [TARGET_MB=16]  (only the problems of the L3 config_3a/3b/3c.json; env TVM_L3_CONFIGS)"""
import sys, os, json, glob
sys.path.insert(0, os.environ.get("MB_PP", "/Users/krisd/Documents/ModelBlaster-clone/l3_farm/qrb_a6/pp"))
import importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tvm_paths import L3_CONFIGS
WANT = set(p for c in L3_CONFIGS for p in json.load(open(c))["problems"])
from modelblaster.pipeline import extract_graph as eg
pdir, out = sys.argv[1], sys.argv[2]
mb = int(sys.argv[3]) if len(sys.argv) > 3 else 16
captured = {}
orig = eg._shrink_for_utilization
def spy(mod, tb):
    orig(mod, tb); captured["mod"] = mod
eg._shrink_for_utilization = spy
def intattrs(m):
    return {k: v for k, v in vars(m).items() if not k.startswith("_") and
            (isinstance(v, (int, float)) and not isinstance(v, bool) or isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v))}
res = {}
for f in sorted(glob.glob(os.path.join(pdir, "*.py"))):
    name = os.path.basename(f)[:-3]
    if name not in WANT: continue
    spec = importlib.util.spec_from_file_location("o", f); o = importlib.util.module_from_spec(spec); spec.loader.exec_module(o)
    before = intattrs(o)
    captured.clear()
    try:
        eg._load_kernelbench(f, target_bytes=mb * 2**20)
    except Exception as e:
        print("FAIL", name, e); continue
    after = intattrs(captured["mod"]) if "mod" in captured else before
    res[name] = {"changed": {k: v for k, v in after.items() if before.get(k) != v}, "orig": before}
    print(name, res[name]["changed"], flush=True)
json.dump(res, open(out, "w"), indent=1)
