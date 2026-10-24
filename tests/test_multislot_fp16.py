"""fp16 multi-slot problems: dtype in the prompt, the roofline bytes, the
implementation store (fp16 and fp32 records never mix), and an end-to-end
agent run on the fp16 LeNet fixture with a scripted LLM (needs nvcc + GPU).
"""
import asyncio
import json
import shutil
from pathlib import Path

from kernelblaster.agents.opt_multislot import (SYSTEM_PROMPT, MultiSlotAgent, _op_cost,
                                                build_codegen_messages, extract_code)
from kernelblaster.agents.utils import agentic_llm
from kernelblaster.knowledge.impl_store import ImplStore, shape_bucket, signature_sha
from kernelblaster.knowledge.outcomes import OutcomeLog
from kernelblaster.knowledge.slots import LocalEvaluator, SlotProblem

from test_multislot_agent import ARCH, needs_gpu

FIX = Path(__file__).parent / "fixtures"
F16, F32 = FIX / "mb_lenet_f16", FIX / "mb_lenet"


def _messages(p, sid):
    return build_codegen_messages(problem=p, sid=sid, source="// src", technique="t", technique_desc="d",
                                  state="memory_bound", state_numbers={}, slot_us=1.0, partition_us=2.0,
                                  neighbours=[], examples=[], adapt=None, knowledge_text="",
                                  already_tried="", best_note="")


def test_prompt_rules_follow_the_manifest_dtype():
    p16, p32 = SlotProblem(F16), SlotProblem(F32)
    assert (p16.dtype, p32.dtype) == ("fp16", "fp32")
    sys16 = _messages(p16, "relu_f16")[0]["content"]
    assert "`__half`" in sys16 and "fp32 math" in sys16 and "2e-3" in sys16 and "fp32 in/out" not in sys16
    assert _messages(p32, "relu")[0]["content"] == SYSTEM_PROMPT          # fp32 prompt unchanged
    assert "__half" in p16.slots["relu_f16"].signature


def test_roofline_counts_two_bytes_per_half():
    p = SlotProblem(F16)
    op = next(o for o in p.manifest["ops"] if o["slot"] == "relu_f16")
    numel = sum(p.manifest["buffers"][b]["numel"] for b in op["inputs"] + op["outputs"])
    assert _op_cost(p, op)[0] == 2 * numel


def test_store_separates_dtypes(tmp_path):
    s = ImplStore(tmp_path)
    sh = [{"n": 4704}]
    b = [shape_bucket("elementwise", sh[0])]
    sig16 = "void kernel_relu_f16(const __half *input, __half *output, int n)"
    sig32 = "void kernel_relu(const float *input, float *output, int n)"
    common = dict(target="cuda", arch="sm_75", op_class="elementwise", shapes=sh, buckets=b, slot_us=1.0,
                  seed_us=3.0, speedup_vs_seed=3.0)
    s.add("// relu32", op="relu", signature_sha=signature_sha(sig32), **common)
    s.add("// relu16", op="relu_f16", signature_sha=signature_sha(sig16), dtype="fp16", **common)
    s.add("// sigmoid32", op="sigmoid", signature_sha="x", **common)
    s.add("// sigmoid16", op="sigmoid_f16", signature_sha="y", dtype="fp16", **common)
    r = s.retrieve(target="cuda", arch="sm_75", op="relu_f16", op_class="elementwise", signature=sig16,
                   shapes=sh, dtype="fp16")
    assert s.code(r["seed"]) == "// relu16" and [s.code(e) for e in r["examples"]] == ["// sigmoid16"]
    r = s.retrieve(target="cuda", arch="sm_75", op="relu", op_class="elementwise", signature=sig32, shapes=sh)
    assert s.code(r["seed"]) == "// relu32" and [s.code(e) for e in r["examples"]] == ["// sigmoid32"]
    # records written before the dtype field are fp32
    line = json.loads(s.index_path.read_text().splitlines()[0])
    line.pop("dtype")
    s.index_path.write_text(json.dumps(line) + "\n")
    assert ImplStore(tmp_path).records()[0].dtype == "fp32"


FAST_RELU_F16 = """```cuda
#include <cuda_fp16.h>
__global__ void k_relu_h2(const __half2 *__restrict__ x, __half2 *__restrict__ y, int n2) {
    for (long i = MB_TID; i < n2; i += (long)gridDim.x * blockDim.x) {
        float2 v = __half22float2(x[i]);
        y[i] = __floats2half2_rn(fmaxf(v.x, 0.f), fmaxf(v.y, 0.f));
    }
}
__global__ void k_relu_h_tail(const __half *x, __half *y, int s, int n) {
    long i = s + MB_TID; if (i < n) y[i] = __float2half_rn(fmaxf(__half2float(x[i]), 0.f));
}
void kernel_relu_f16(const __half *input, __half *output, int n) {
    int n2 = n / 2;
    if (n2) k_relu_h2<<<MB_GRID(n2), MB_THREADS>>>((const __half2 *)input, (__half2 *)output, n2);
    if (n % 2) k_relu_h_tail<<<1, MB_THREADS>>>(input, output, n2 * 2, n);
}
```"""


def scripted16(state):
    """relu_f16 -> a half2 kernel; linear_f16 -> wrong (fp16 bias of +1) first,
    fixed on the fix prompt; everything else unchanged."""
    def fn(msgs):
        user = msgs[-1]["content"]
        state["systems"].add(msgs[0]["content"])
        cur = extract_code(user.split("Current source:")[-1] if "Current source:" in user
                           else user.split("Current slot source:")[-1])
        if "failed:" in user:
            state["fix_calls"] += 1
            return f"```cuda\n{state['linear_ok']}\n```"
        if "## Slot `relu_f16`" in user:
            return FAST_RELU_F16
        if "## Slot `linear_f16`" in user and not state["broke"]:
            state["broke"] = True
            state["linear_ok"] = cur
            return f"```cuda\n{cur.replace('y[i] = acc;', 'y[i] = acc + 1.0;')}\n```"
        return f"```cuda\n{cur}```"
    return fn


@needs_gpu
def test_agent_end_to_end_fp16(tmp_path):
    p = SlotProblem(shutil.copytree(F16, tmp_path / "lenet16"))
    store, olog = ImplStore(tmp_path / "impl"), OutcomeLog(tmp_path / "outcomes.jsonl")
    # an fp32 winner of the same class in the shared store must not reach the fp16 prompts
    fp32_rec = store.add("// FP32 ONLY", target="cuda", arch=f"sm_{ARCH}", op="sigmoid", op_class="elementwise",
              signature_sha="z", shapes=[{"n": 4704}], buckets=["elementwise:n4096"], slot_us=1.0,
              seed_us=9.0, speedup_vs_seed=9.0)
    state = {"broke": False, "fix_calls": 0, "linear_ok": None, "systems": set()}
    agentic_llm.register_stub("scripted16", scripted16(state))
    try:
        agent = MultiSlotAgent(p, LocalEvaluator(ARCH, work_root=tmp_path / "work"), model="stub:scripted16",
                               out_dir=tmp_path / "out", arch=ARCH, impl_store=store, outcome_log=olog,
                               rollouts=2, steps=4, min_share=0.0, seed=1)
        res = asyncio.run(agent.run())
    finally:
        agentic_llm.unregister_stub("scripted16")
    assert res["final_ok"], res
    assert set(res["slots"]) == {"conv2d_f16", "relu_f16", "maxpool2d_f16", "linear_f16"}
    assert all("`__half`" in s for s in state["systems"])
    outcomes = list(olog)
    assert len(outcomes) == 8 and {o.dtype for o in outcomes} == {"fp16"}
    if state["broke"]:                                         # the +1 bias failed its check -> fix prompt
        assert state["fix_calls"] >= 1, [(o.slot, o.status) for o in outcomes]
    recs = [r for r in store.records() if r.op != "sigmoid"]
    assert recs and {r.dtype for r in recs} == {"fp16"}
    assert not any(fp32_rec.id in s.context["examples"] for s in agent.steps_log)
    assert all(k.endswith("|fp16") for k in res["bandit"])
    if any(o.slot == "relu_f16" for o in outcomes):      # the half2 relu verified (faster or not: timing noise)
        assert any(r.op == "relu_f16" and "k_relu_h2" in store.code(r) for r in recs), \
            [(o.slot, o.status, o.error) for o in outcomes if o.slot == "relu_f16"]
