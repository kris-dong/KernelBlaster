"""X-149: the agent on a clock-suspect GPU (needs nvcc + a CUDA GPU; skipped otherwise). The GPU is real, the guard's nvidia-smi
answer is the scripted locked state (the real locked state is reproduced by l3_farm/analysis/ncu_guard_data/node_scripts):
every timed step must come out inconclusive (neutral, no adoption, no knowledge record), flagged in result.json."""
import asyncio
import json
import shutil

import pytest

from kernelblaster.agents.opt_multislot import MultiSlotAgent
from kernelblaster.agents.utils import agentic_llm
from kernelblaster.knowledge.impl_store import ImplStore
from kernelblaster.knowledge.outcomes import OutcomeLog
from kernelblaster.knowledge.slots import LocalEvaluator, SlotProblem

from test_multislot_agent import ARCH, FIX, needs_gpu, scripted

LOCKED = ("0, GPU-locked-test, 1065 MHz, 2520 MHz, 9001 MHz, 9001 MHz, [Requested functionality has been deprecated], "
          "0x0000000000000001, 0 %, P0")


@needs_gpu
def test_agent_on_a_locked_gpu_marks_steps_inconclusive(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELBLASTER_CLOCK_GUARD", "1")
    monkeypatch.setenv("KERNELBLASTER_CLOCK_CONFIRM_S", "0")
    prob_dir = tmp_path / "lenet"
    shutil.copytree(FIX, prob_dir)
    p = SlotProblem(prob_dir)
    state = {"broke": False, "fix_calls": 0, "linear_ok": None}
    agentic_llm.register_stub("scripted", scripted(state))
    ev = LocalEvaluator(ARCH, work_root=tmp_path / "work")

    async def locked(device, cmd):
        return LOCKED
    ev._clock_run = locked
    store, olog = ImplStore(tmp_path / "impl"), OutcomeLog(tmp_path / "outcomes.jsonl")
    try:
        agent = MultiSlotAgent(p, ev, model="stub:scripted", out_dir=tmp_path / "out", arch=ARCH, impl_store=store,
                               outcome_log=olog, rollouts=1, steps=3, min_share=0.0, seed=1)
        res = asyncio.run(agent.run())
    finally:
        agentic_llm.unregister_stub("scripted")
    saved = json.loads((tmp_path / "out" / "result.json").read_text())
    assert res["final_ok"]
    assert saved["clock_suspect"]["steps"] >= 1
    assert saved["gpu"]["clock"]["suspect"] and saved["gpu"]["clock"]["state"] == "locked"
    assert all(o.status != "regressed" for o in olog)           # nothing was learned as a regression from locked-clock timings
    assert not [o for o in olog if o.status == "improved"]      # nor as an improvement
