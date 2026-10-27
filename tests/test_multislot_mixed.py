"""Mixed-dtype multi-slot problems (ModelBlaster ``extract_graph --quant native``): manifest ``dtype`` "mixed",
each slot with its own storage dtype and isolated bound. Slot parsing (incl. the fallbacks for older manifests),
the mixed system prompt, and the slot dtype in bandit keys / records / roofline peaks. No GPU, no LLM.
"""
import json
import shutil
from types import SimpleNamespace

from kernelblaster.agents import hw_profile
from kernelblaster.agents.opt_multislot import (BUFFER_BYTES, MIXED_RULE, SYSTEM_PROMPT, MultiSlotAgent,
                                                _nbytes, bandit_key, slot_dtype, system_prompt, target_profile)
from kernelblaster.knowledge.impl_store import ImplStore
from kernelblaster.knowledge.slots import SLOT_BOUNDS, SlotProblem

from test_multislot_bf16 import BF16, F16, F32, _messages

# slot -> (dtype, tolerance) written into the copied fp32 LeNet; maxpool2d keeps no dtype (fallback: fp32)
MIXED = {"conv2d": ("fp32", {"slot_rel": 1e-3, "slot_abs": 0.0}),
         "relu": ("bf16", {"slot_rel": 1.6e-2, "slot_abs": 0.0}),
         "linear": ("fp16", {"slot_rel": 2e-3, "slot_abs": 2.0 ** -23})}


def _mixed(tmp_path, slots=MIXED):
    root = shutil.copytree(F32, tmp_path / "lenet_mixed")
    m = json.loads((root / "manifest.json").read_text())
    m["dtype"] = "mixed"
    m["tolerance"] = {"output_rel": 0.08, "output_abs": 1e-5, "slot_rel": 1.6e-2, "slot_abs": 0.0}   # bf16's
    for s in m["slots"]:
        if s["id"] in slots:
            s["dtype"], s["tolerance"] = slots[s["id"]]
    (root / "manifest.json").write_text(json.dumps(m))
    return SlotProblem(root)


def test_slot_dtype_and_tolerance(tmp_path):
    p = _mixed(tmp_path)
    assert p.dtype == "mixed" and p.dtypes == ["bf16", "fp16", "fp32"]
    for sid, (dt, tol) in MIXED.items():
        assert p.slots[sid].dtype == dt and p.slots[sid].tolerance == tol
    assert p.slots["maxpool2d"].dtype == "fp32" and p.slots["maxpool2d"].tolerance == SLOT_BOUNDS["fp32"]
    assert p.tolerance["slot_rel"] == 1.6e-2                     # the problem bound stays the loosest


def test_old_manifests_fall_back_to_the_problem():
    for root, dt in ((F32, "fp32"), (F16, "fp16"), (BF16, "bf16")):
        p = SlotProblem(root)
        assert p.dtype == dt and p.dtypes == [dt]
        tol = {k: v for k, v in p.tolerance.items() if k.startswith("slot_")}
        assert all(s.dtype == dt and s.tolerance == tol for s in p.slots.values())
        assert all(slot_dtype(p, sid) == dt for sid in p.slots)
    # a mixed manifest without slot dtypes: every slot fp32
    bare = SimpleNamespace(dtype="mixed", slots={"s": SimpleNamespace(op="x")})
    assert slot_dtype(bare, "s") == "fp32"


def test_mixed_prompt_has_each_dtype_rule_with_its_bound(tmp_path):
    p = _mixed(tmp_path)
    sys_ = _messages(p, "relu")[0]["content"]
    assert MIXED_RULE in sys_
    assert "fp32 in/out" in sys_ and "`__half`" in sys_ and "`__nv_bfloat16`" in sys_
    assert "max error <= 1e-3 relative" in sys_                  # fp32 slot bound
    assert "max error <= 2e-3 relative" in sys_                  # fp16
    assert "max error <= 1.6e-2 relative" in sys_                # bf16 (not the problem's bound for all)
    assert sys_.index("bf16 storage") < sys_.index("fp16 storage") < sys_.index("fp32 in/out")
    assert _messages(p, "conv2d")[0]["content"] == sys_          # one prompt for every slot of the problem
    # without an fp16 slot: no fp16 rule
    q = _mixed(tmp_path / "b", {"relu": MIXED["relu"]})
    s2 = _messages(q, "relu")[0]["content"]
    assert "`__half`" not in s2 and "fp32 in/out" in s2 and "1.6e-2" in s2 and MIXED_RULE in s2
    # the target-hardware block: the fp16 tensor-core text once any slot is fp16
    assert "fp16 tensor-core peak" in target_profile("cuda").prompt_for(p, "89")
    assert "fp16 tensor-core peak" not in target_profile("cuda").prompt_for(q, "89")


def test_single_dtype_prompts_unchanged():
    assert _messages(SlotProblem(F32), "relu")[0]["content"] == SYSTEM_PROMPT
    for root, dt, sid in ((F16, "fp16", "relu_f16"), (BF16, "bf16", "relu_bf16")):
        p = SlotProblem(root)
        assert MIXED_RULE not in _messages(p, sid)[0]["content"]
        assert _messages(p, sid)[0]["content"] == system_prompt(dt, p.tolerance)
        assert target_profile("cuda").prompt_for(p, "89") == system_prompt(dt, p.tolerance, "89")


def test_bandit_key_per_slot_dtype(tmp_path):
    p = _mixed(tmp_path)
    keys = {sid: bandit_key(p.slots[sid].op_class, "memory_bound", slot_dtype(p, sid)) for sid in p.slots}
    assert keys == {"conv2d": "conv|memory_bound", "maxpool2d": "reduction|memory_bound",      # fp32: as before
                    "relu": "elementwise|memory_bound|bf16", "linear": "gemm|memory_bound|fp16"}


def _bare_agent(p, tmp_path):
    a = MultiSlotAgent.__new__(MultiSlotAgent)
    a.p, a.tp, a.arch, a.arch_label = p, target_profile("cuda"), "89", "sm_89"
    a.problem_id, a.run_id, a.baseline = "prob", "rid", None
    a.store, a._existing = ImplStore(tmp_path / "impl"), {"impl": set(), "outcome": set()}
    return a


def test_records_carry_the_slot_dtype(tmp_path):
    p = _mixed(tmp_path)
    a = _bare_agent(p, tmp_path)
    got = {sid: a._outcome(sid=sid, technique="t", status="improved").dtype for sid in p.slots}
    assert got == {"conv2d": "fp32", "maxpool2d": "fp32", "relu": "bf16", "linear": "fp16"}
    for sid in ("relu", "linear", "conv2d"):
        a._store(sid, f"// {sid}", 1.0, 2.0, ["t"], None, 0.0)
    assert {r.slot: r.dtype for r in a.store.records()} == {"relu": "bf16", "linear": "fp16", "conv2d": "fp32"}
    # roofline peaks per slot: the fp16 GEMM slot gets the tensor-core peak, the fp32 conv the CUDA-core one
    lin, conv = a._roofline_input("linear", 10.0), a._roofline_input("conv2d", 10.0)
    assert lin.peak_gflops == hw_profile.peaks("89", "fp16", "gemm")[0]
    assert conv.peak_gflops == hw_profile.peaks("89", "fp32", "conv")[0] < lin.peak_gflops


def test_one_byte_buffers():
    assert BUFFER_BYTES["b8"] == BUFFER_BYTES["u8"] == 1 and BUFFER_BYTES["i32"] == 4
    assert _nbytes({"dtype": "b8", "numel": 10}) == 10.0 and _nbytes({"dtype": "u8", "numel": 3}) == 3.0
    assert _nbytes({"dtype": "i64", "numel": 2}) == 16.0
