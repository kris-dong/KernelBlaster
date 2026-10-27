"""bf16 multi-slot problems (ModelBlaster docs/bf16_cuda.md): dtype in the prompt, the
roofline bytes, the implementation store (bf16, fp16 and fp32 records never mix), the
bf16 knowledge seeds, and an end-to-end agent run on the bf16 LeNet fixture with a
scripted LLM (needs nvcc + GPU; sm_75 has no bf16 arithmetic, the seeds compute in fp32).
"""
import asyncio
import json
import shutil
from pathlib import Path

from kernelblaster.agents.opt_multislot import (SYSTEM_PROMPT, MultiSlotAgent, _op_cost,
                                                build_codegen_messages, extract_code, system_prompt)
from kernelblaster.agents.utils import agentic_llm
from kernelblaster.knowledge.impl_store import ImplStore, shape_bucket, signature_sha
from kernelblaster.knowledge.outcomes import OutcomeLog
from kernelblaster.knowledge.slots import LocalEvaluator, SlotProblem
from kernelblaster.knowledge.v2.ontology import Ontology
from kernelblaster.knowledge.v2.targets import TargetRegistry

from test_multislot_agent import ARCH, needs_gpu

FIX = Path(__file__).parent / "fixtures"
BF16, F16, F32 = FIX / "mb_lenet_bf16", FIX / "mb_lenet_f16", FIX / "mb_lenet"


def _messages(p, sid):
    return build_codegen_messages(problem=p, sid=sid, source="// src", technique="t", technique_desc="d",
                                  state="memory_bound", state_numbers={}, slot_us=1.0, partition_us=2.0,
                                  neighbours=[], examples=[], adapt=None, knowledge_text="",
                                  already_tried="", best_note="")


def test_fixture_is_a_bf16_problem():
    p = SlotProblem(BF16)
    assert p.dtype == "bf16"
    assert {b["dtype"] for b in p.manifest["buffers"]} == {"bf16"}
    for s in p.slots.values():
        assert "__nv_bfloat16" in s.signature and "__bf16" not in s.signature.replace("__nv_bfloat16", "")
        assert "cuda_bf16.h" in (BF16 / "walker_head.cu").read_text()


def test_prompt_rules_follow_the_manifest_dtype():
    pb, p16, p32 = SlotProblem(BF16), SlotProblem(F16), SlotProblem(F32)
    sysb = _messages(pb, "relu_bf16")[0]["content"]
    assert "`__nv_bfloat16`" in sysb and "fp32 math" in sysb and "sm_80" in sysb
    assert f"{pb.tolerance['slot_rel']:.1e}".replace("e-0", "e-") in sysb       # e.g. 1.6e-2, not 2e-2
    assert "`__half`" not in sysb and "fp32 in/out" not in sysb
    assert "`__half`" in _messages(p16, "relu_f16")[0]["content"]               # fp16 prompt still fp16
    assert _messages(p32, "relu")[0]["content"] == SYSTEM_PROMPT                # fp32 prompt unchanged
    assert "1e-3" in system_prompt("fp32", {"slot_rel": 1e-3}) and "2e-3" in system_prompt("fp16", {"slot_rel": 2e-3})


def test_roofline_counts_two_bytes_per_bf16():
    p = SlotProblem(BF16)
    op = next(o for o in p.manifest["ops"] if o["slot"] == "relu_bf16")
    numel = sum(p.manifest["buffers"][b]["numel"] for b in op["inputs"] + op["outputs"])
    assert _op_cost(p, op)[0] == 2 * numel


def test_store_separates_bf16_from_fp16_and_fp32(tmp_path):
    s = ImplStore(tmp_path)
    sh = [{"n": 4704}]
    b = [shape_bucket("elementwise", sh[0])]
    sigs = {"fp32": "void kernel_relu(const float *input, float *output, int n)",
            "fp16": "void kernel_relu_f16(const __half *input, __half *output, int n)",
            "bf16": "void kernel_relu_bf16(const __nv_bfloat16 *input, __nv_bfloat16 *output, int n)"}
    ops = {"fp32": "relu", "fp16": "relu_f16", "bf16": "relu_bf16"}
    common = dict(target="cuda", arch="sm_75", op_class="elementwise", shapes=sh, buckets=b, slot_us=1.0,
                  seed_us=3.0, speedup_vs_seed=3.0)
    for dt in ("fp32", "fp16", "bf16"):
        s.add(f"// relu {dt}", op=ops[dt], signature_sha=signature_sha(sigs[dt]), dtype=dt, **common)
        s.add(f"// sigmoid {dt}", op=ops[dt].replace("relu", "sigmoid"), signature_sha="x" + dt, dtype=dt, **common)
    for dt in ("fp32", "fp16", "bf16"):
        r = s.retrieve(target="cuda", arch="sm_75", op=ops[dt], op_class="elementwise", signature=sigs[dt],
                       shapes=sh, dtype=dt)
        assert s.code(r["seed"]) == f"// relu {dt}" and [s.code(e) for e in r["examples"]] == [f"// sigmoid {dt}"]
    assert {w.dtype for w in s.winners()} == {"fp32", "fp16", "bf16"}


def test_bf16_knowledge_seeds():
    onto = Ontology.seed(TargetRegistry())
    real = lambda tid, tk: (onto.realization(onto.get(tid), tk) or {}).get("how", "")   # noqa: E731
    hp = onto.get("half_precision")
    assert "bf16" in json.dumps(hp.preconditions)
    assert "__nv_bfloat162" in real("half_precision", "cuda:sm_75")
    assert "bf16" in real("tensor_core_mma", "cuda:sm_75")                 # no bf16 tensor cores on Turing
    assert "__nv_bfloat16" in real("tensor_core_mma", "cuda:sm_89")
    assert "__nv_bfloat162" in real("vectorized_memory_access", "cuda:sm_89")


FAST_RELU_BF16 = """```cuda
#include <cuda_bf16.h>
__global__ void k_relu_b2(const __nv_bfloat162 *__restrict__ x, __nv_bfloat162 *__restrict__ y, int n2) {
    for (long i = MB_TID; i < n2; i += (long)gridDim.x * blockDim.x) {
        float2 v = __bfloat1622float2(x[i]);
        y[i] = __floats2bfloat162_rn(fmaxf(v.x, 0.f), fmaxf(v.y, 0.f));
    }
}
__global__ void k_relu_b_tail(const __nv_bfloat16 *x, __nv_bfloat16 *y, int s, int n) {
    long i = s + MB_TID; if (i < n) y[i] = __float2bfloat16_rn(fmaxf(__bfloat162float(x[i]), 0.f));
}
void kernel_relu_bf16(const __nv_bfloat16 *input, __nv_bfloat16 *output, int n) {
    int n2 = n / 2;
    if (n2) k_relu_b2<<<MB_GRID(n2), MB_THREADS>>>((const __nv_bfloat162 *)input, (__nv_bfloat162 *)output, n2);
    if (n % 2) k_relu_b_tail<<<1, MB_THREADS>>>(input, output, n2 * 2, n);
}
```"""


def scripted_bf16(state):
    """relu_bf16 -> a bf162 kernel; linear_bf16 -> wrong (bias of +1) first, fixed on
    the fix prompt; everything else unchanged."""
    def fn(msgs):
        user = msgs[-1]["content"]
        state["systems"].add(msgs[0]["content"])
        cur = extract_code(user.split("Current source:")[-1] if "Current source:" in user
                           else user.split("Current slot source:")[-1])
        if "failed:" in user:
            state["fix_calls"] += 1
            return f"```cuda\n{state['linear_ok']}\n```"
        if "## Slot `relu_bf16`" in user:
            return FAST_RELU_BF16
        if "## Slot `linear_bf16`" in user and not state["broke"]:
            state["broke"] = True
            state["linear_ok"] = cur
            return f"```cuda\n{cur.replace('y[i] = acc;', 'y[i] = acc + 1.0;')}\n```"
        return f"```cuda\n{cur}```"
    return fn


@needs_gpu
def test_agent_end_to_end_bf16(tmp_path):
    p = SlotProblem(shutil.copytree(BF16, tmp_path / "lenet_bf16"))
    store, olog = ImplStore(tmp_path / "impl"), OutcomeLog(tmp_path / "outcomes.jsonl")
    # fp32 and fp16 winners of the same class in the shared store must not reach the bf16 prompts
    others = [store.add(f"// {dt} ONLY", target="cuda", arch=f"sm_{ARCH}", op=op, op_class="elementwise",
                        signature_sha="z" + dt, shapes=[{"n": 4704}], buckets=["elementwise:n4096"], slot_us=1.0,
                        seed_us=9.0, speedup_vs_seed=9.0, dtype=dt)
              for dt, op in (("fp32", "sigmoid"), ("fp16", "sigmoid_f16"))]
    state = {"broke": False, "fix_calls": 0, "linear_ok": None, "systems": set()}
    agentic_llm.register_stub("scripted_bf16", scripted_bf16(state))
    try:
        agent = MultiSlotAgent(p, LocalEvaluator(ARCH, work_root=tmp_path / "work"), model="stub:scripted_bf16",
                               out_dir=tmp_path / "out", arch=ARCH, impl_store=store, outcome_log=olog,
                               rollouts=2, steps=4, min_share=0.0, seed=1)
        res = asyncio.run(agent.run())
    finally:
        agentic_llm.unregister_stub("scripted_bf16")
    assert res["final_ok"], res
    assert set(res["slots"]) == {"conv2d_bf16", "relu_bf16", "maxpool2d_bf16", "linear_bf16"}
    assert all("`__nv_bfloat16`" in s for s in state["systems"])
    outcomes = list(olog)
    assert len(outcomes) == 8 and {o.dtype for o in outcomes} == {"bf16"}
    if state["broke"]:                                         # the +1 bias failed its check -> fix prompt
        assert state["fix_calls"] >= 1, [(o.slot, o.status) for o in outcomes]
    recs = [r for r in store.records() if not r.op.startswith("sigmoid")]
    assert recs and {r.dtype for r in recs} == {"bf16"}
    assert not any(o.id in s.context["examples"] for o in others for s in agent.steps_log)
    assert all(k.endswith("|bf16") for k in res["bandit"])
    if any(o.slot == "relu_bf16" for o in outcomes):     # the bf162 relu verified (faster or not: timing noise)
        assert any(r.op == "relu_bf16" and "k_relu_b2" in store.code(r) for r in recs), \
            [(o.slot, o.status, o.error) for o in outcomes if o.slot == "relu_bf16"]
