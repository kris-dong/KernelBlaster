"""End-to-end multi-slot agent on the LeNet fixture with a scripted LLM.

Needs nvcc + a CUDA GPU; skipped otherwise.
"""
import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from kernelblaster.agents.opt_multislot import MultiSlotAgent, extract_code
from kernelblaster.agents.utils import agentic_llm
from kernelblaster.knowledge.impl_store import ImplStore
from kernelblaster.knowledge.outcomes import OutcomeLog
from kernelblaster.knowledge.slots import LocalEvaluator, SlotProblem

FIX = Path(__file__).parent / "fixtures" / "mb_lenet"


def _gpu_arch():
    if shutil.which("nvcc") is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.split()
        return out[0].replace(".", "") if out else None
    except Exception:
        return None


ARCH = _gpu_arch()
needs_gpu = pytest.mark.skipif(ARCH is None, reason="needs nvcc and a CUDA GPU")

FAST_RELU = """```cuda
__global__ void k_relu4(const float4 *__restrict__ x, float4 *__restrict__ y, int n4) {
    for (long i = MB_TID; i < n4; i += (long)gridDim.x * blockDim.x) {
        float4 v = x[i];
        v.x = fmaxf(v.x, 0.f); v.y = fmaxf(v.y, 0.f); v.z = fmaxf(v.z, 0.f); v.w = fmaxf(v.w, 0.f);
        y[i] = v;
    }
}
__global__ void k_relu_tail(const float *x, float *y, int s, int n) {
    long i = s + MB_TID; if (i < n) y[i] = fmaxf(x[i], 0.f);
}
void kernel_relu(const float *input, float *output, int n) {
    int n4 = n / 4;
    if (n4) k_relu4<<<MB_GRID(n4), MB_THREADS>>>((const float4 *)input, (float4 *)output, n4);
    if (n % 4) k_relu_tail<<<1, MB_THREADS>>>(input, output, n4 * 4, n);
}
```"""


def scripted(state):
    """relu -> a real float4 kernel; linear -> broken first, fixed on the fix
    prompt; everything else -> the current source unchanged."""
    def fn(msgs):
        user = msgs[-1]["content"]
        cur = extract_code(user.split("Current source:")[-1] if "Current source:" in user
                           else user.split("Current slot source:")[-1])
        if "failed:" in user:
            state["fix_calls"] += 1
            return f"```cuda\n{state['linear_ok']}\n```"
        if "## Slot `relu`" in user:
            return FAST_RELU
        if "## Slot `linear`" in user and not state["broke"]:
            state["broke"] = True
            state["linear_ok"] = cur
            return "```cuda\nvoid kernel_linear(const float *input) { this is not valid }\n```"
        return f"```cuda\n{cur}```"
    return fn


@needs_gpu
def test_agent_end_to_end(tmp_path):
    prob_dir = tmp_path / "lenet"
    shutil.copytree(FIX, prob_dir)
    p = SlotProblem(prob_dir)
    state = {"broke": False, "fix_calls": 0, "linear_ok": None}
    agentic_llm.register_stub("scripted", scripted(state))
    store, olog = ImplStore(tmp_path / "impl"), OutcomeLog(tmp_path / "outcomes.jsonl")
    try:
        agent = MultiSlotAgent(p, LocalEvaluator(ARCH, work_root=tmp_path / "work"), model="stub:scripted",
                               out_dir=tmp_path / "out", arch=ARCH, impl_store=store, outcome_log=olog,
                               rollouts=2, steps=4, min_share=0.0, seed=1)
        res = asyncio.run(agent.run())
    finally:
        agentic_llm.unregister_stub("scripted")

    assert res["final_ok"], res
    assert set(res["slots"]) == {"conv2d", "relu", "maxpool2d", "linear"}
    outcomes = list(olog)
    assert len(outcomes) == 8                                  # 2 rollouts x 4 steps
    assert all(o.slot in res["slots"] and o.op_class for o in outcomes)
    tried = {o.slot for o in outcomes}
    assert "linear" in tried                                   # biggest share gets picked
    lin = [o for o in outcomes if o.slot == "linear"]
    assert state["fix_calls"] >= 1 and lin[0].status != "compile_error"   # broken -> fixed
    # every verified step and the 4 baselines are in the store
    recs = store.records()
    assert len(recs) >= 4 and {r.op for r in recs} == {"conv2d", "relu", "maxpool2d", "linear"}
    assert (tmp_path / "out" / "best" / "slots" / "relu.cu").exists()
    saved = json.loads((tmp_path / "out" / "result.json").read_text())
    assert saved["llm_calls"] == agent.llm_calls >= 8
    assert set(saved["llm_usage"]) == {"prompt_tokens", "completion_tokens", "cost_usd"}


@needs_gpu
def test_store_seed_reused_in_a_later_run(tmp_path):
    """A second problem with the same op/shape starts from the stored winner."""
    p = SlotProblem(shutil.copytree(FIX, tmp_path / "a"))
    store = ImplStore(tmp_path / "impl")
    agentic_llm.register_stub("relu_only", lambda m: FAST_RELU if "## Slot `relu`" in m[-1]["content"]
                              else "no code")
    try:
        a1 = MultiSlotAgent(p, LocalEvaluator(ARCH, work_root=tmp_path / "w1"), model="stub:relu_only",
                            out_dir=tmp_path / "o1", arch=ARCH, impl_store=store, rollouts=1, steps=8,
                            min_share=0.0, max_fix_attempts=0)
        asyncio.run(a1.run())
    finally:
        agentic_llm.unregister_stub("relu_only")
    fast = [r for r in store.records() if r.op == "relu" and "k_relu4" in store.code(r)]
    assert fast, "the float4 relu was not stored"
    p2 = SlotProblem(shutil.copytree(FIX, tmp_path / "b"))
    a2 = MultiSlotAgent(p2, LocalEvaluator(ARCH, work_root=tmp_path / "w2"), model="stub:none",
                        out_dir=tmp_path / "o2", arch=ARCH, impl_store=store, rollouts=0, steps=0)
    asyncio.run(a2.initialize())
    r = store.retrieve(target="cuda", arch=f"sm_{ARCH}", op="relu", op_class="elementwise",
                       signature=p2.slots["relu"].signature, shapes=p2.slots["relu"].shapes())
    assert r["seed"] is not None
