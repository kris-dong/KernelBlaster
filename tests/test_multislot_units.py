"""GPU-free unit tests for the multi-slot agent's pieces."""
import random
from pathlib import Path

from kernelblaster.agents.opt_multislot import (TechniqueBandit, extract_code, propose_fusions,
                                                slot_state)
from kernelblaster.knowledge.impl_store import ImplStore, shape_bucket
from kernelblaster.knowledge.slots import EvalResult, SlotProblem, parse_driver_output, wrap_slot

FIX = Path(__file__).parent / "fixtures" / "mb_lenet"


def _result(**slot_us):
    return EvalResult(ok=True, stage="ok", partition_us=100.0, slot_us=slot_us)


def test_fusion_proposals_from_structure_and_times():
    p = SlotProblem(FIX)
    props = propose_fusions(p, _result(conv2d=30.0, relu=40.0, maxpool2d=10.0, linear=20.0), "75")
    pats = {tuple(x["pattern"]): x for x in props["proposals"]}
    assert set(pats) == {("conv2d", "relu"), ("linear", "relu")}        # maxpool isn't elementwise
    assert sorted(a for a, _, _ in pats[("conv2d", "relu")]["pairs"]) == ["conv1", "conv2"]
    assert pats[("linear", "relu")]["evidence"]["est_gain_pct"] == 20.0  # 2 relu instances x 10 us
    names = [n for x in props["proposals"] for pr in x["pairs"] for n in pr[:2]]
    assert len(names) == len(set(names))                                 # each op in one pair
    assert propose_fusions(p, _result(conv2d=97.0, relu=1.0, maxpool2d=1.0, linear=1.0), "75")["proposals"] == []


def test_roofline_state_and_bandit_transfer():
    p = SlotProblem(FIX)
    st, nums = slot_state(p, "linear", 5.0, "75")
    assert st == "latency_bound" and nums["bytes"] > 0
    st, _ = slot_state(p, "conv2d", 1e6, "75")
    assert st.startswith("underutilized")
    b, rng = TechniqueBandit(), random.Random(0)
    b.update("gemm|memory_bound", "tiling", 0.5)
    b.update("gemm|memory_bound", "unroll", -0.1)
    # an unseen key inherits each technique's mean from other keys
    assert b.select("conv|memory_bound", ["tiling", "unroll"], rng) == "tiling"


def test_wrap_and_parse():
    w = wrap_slot("relu", '#include <cuda_fp16.h>\n#define X 1\nextern "C" void kernel_relu() {}\n')
    assert w.startswith("#include <cuda_fp16.h>") and "namespace mbslot_relu {" in w
    assert "#undef X" in w and 'extern "C"' not in w
    r = parse_driver_output("MB_CHECK output y max_abs_err=0 bound=1e-05 ok=1\n"
                            "MB_SLOT_CHECK relu rel_err=0.5 ok=0\nfailed\n")
    assert not r.ok and r.failing_slots() == ["relu"] and "relu" in r.error_text()
    assert extract_code("x\n```cuda\nint a;\n```\n") == "int a;\n"


def test_store_retrieval_tiers(tmp_path):
    s = ImplStore(tmp_path)
    sig = "void kernel_linear(const float *input, const float *weight, const float *bias, float *output, int M, int K, int N)"
    sh = [{"M": 1, "K": 400, "N": 120}]
    kw = dict(target="cuda", op_class="gemm", signature_sha=__import__(
        "kernelblaster.knowledge.impl_store", fromlist=["x"]).signature_sha(sig))
    s.add("// lin fast", arch="sm_75", op="linear", shapes=sh, buckets=[shape_bucket("gemm", sh[0])],
          slot_us=1.0, seed_us=4.0, speedup_vs_seed=4.0, **kw)
    s.add("// bmm fast", arch="sm_75", op="bmm", shapes=[{"batch": 8, "M": 64, "K": 64, "N": 64}],
          buckets=["gemm:x"], slot_us=1.0, seed_us=3.0, speedup_vs_seed=3.0,
          **{**kw, "signature_sha": "other"})
    r = s.retrieve(target="cuda", arch="sm_75", op="linear", op_class="gemm", signature=sig, shapes=sh)
    assert r["seed"] is not None and r["adapt"] is None and [e.op for e in r["examples"]] == ["bmm"]
    r = s.retrieve(target="cuda", arch="sm_75", op="linear", op_class="gemm", signature=sig,
                   shapes=[{"M": 4096, "K": 4096, "N": 4096}])
    assert r["seed"] is None and r["adapt"] is None                      # no shared bucket: not an adapt seed
    assert [e.op for e in r["examples"]] == ["bmm"]                      # ...and by default not shown at all
    r = s.retrieve(target="cuda", arch="sm_75", op="linear", op_class="gemm", signature=sig,
                   shapes=[{"M": 4096, "K": 4096, "N": 4096}], far_same_op=True)
    assert [e.op for e in r["examples"]][0] == "linear"                  # opt-in: ideas-only example
    r = s.retrieve(target="cuda", arch="sm_89", op="linear", op_class="gemm", signature=sig, shapes=sh)
    assert r["seed"] is None and r["adapt"] is not None                  # other arch, same bucket -> adapt
    r = s.retrieve(target="cuda", arch="sm_75", op="linear", op_class="gemm", signature=sig, shapes=sh,
                   exclude_problem=None)
    assert r["seed"] is not None
    s.add("// own baseline", arch="sm_75", op="linear", shapes=sh, buckets=[shape_bucket("gemm", sh[0])],
          slot_us=4.0, problem_id="me", **kw)
    r = s.retrieve(target="cuda", arch="sm_75", op="linear", op_class="gemm", signature=sig, shapes=sh,
                   exclude_problem="me")
    assert s.code(r["seed"]) == "// lin fast"                          # own records don't shadow others
    out = s.export_winners(tmp_path / "w")
    import json
    assert {w["op"] for w in json.loads(out.read_text())["winners"]} == {"linear", "bmm"}
