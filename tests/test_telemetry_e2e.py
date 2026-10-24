"""Telemetry end to end: the multi-slot agent on the LeNet fixture with a
scripted LLM (needs nvcc + a CUDA GPU)."""
import asyncio
import json
import shutil

import pytest

from kernelblaster import telemetry as tel
from kernelblaster.agents.opt_multislot import MultiSlotAgent
from kernelblaster.agents.utils import agentic_llm
from kernelblaster.knowledge.impl_store import ImplStore
from kernelblaster.knowledge.outcomes import OutcomeLog
from kernelblaster.knowledge.slots import LocalEvaluator, SlotProblem
from kernelblaster.telemetry import report

from test_multislot_agent import ARCH, FIX, needs_gpu, scripted


def _agent(tmp_path, name, **kw):
    prob = tmp_path / name
    shutil.copytree(FIX, prob)
    state = {"broke": False, "fix_calls": 0, "linear_ok": None}
    fn = scripted(state)
    agentic_llm.register_stub(name, lambda m: {"text": fn(m), "usage": {"prompt_tokens": 1000 + len(m[-1]["content"]),
                                                                        "completion_tokens": 300,
                                                                        "stop_reason": "end_turn"}})
    return MultiSlotAgent(SlotProblem(prob), LocalEvaluator(ARCH, work_root=tmp_path / f"{name}_work"),
                          model=f"stub:{name}", out_dir=tmp_path / f"{name}_out", arch=ARCH,
                          impl_store=ImplStore(tmp_path / "impl"), outcome_log=OutcomeLog(tmp_path / "o.jsonl"),
                          min_share=0.0, seed=1, **kw), state


@needs_gpu
def test_lenet_step_breakdown_sums_to_step_wall_time(tmp_path):
    tel.set_enabled(True)
    tel.configure(tmp_path / "telemetry.jsonl", run_id="e2e")
    agent, state = _agent(tmp_path, "tel_e2e", rollouts=2, steps=4)
    try:
        res = asyncio.run(agent.run())
    finally:
        agentic_llm.unregister_stub("tel_e2e")
        tel.close()
    assert res["final_ok"] and state["fix_calls"] >= 1
    # every step: the breakdown covers the step's wall time
    for s in res["steps"]:
        t = s["context"]["telemetry"]
        assert abs(t["wall_s"] - s["seconds"]) <= 0.1 + 0.02 * s["seconds"], s
        assert sum(t["time_s"].values()) == pytest.approx(t["wall_s"], abs=0.05 + 0.01 * t["wall_s"]), t
        assert t["llm"]["calls"] >= 1 and t["evals"].get("candidate") == 1
        # evaluation work is in the breakdown: a build or a driver run (with the persistent
        # host a cached build and the request's own overhead can both be under 5 ms)
        assert {"build", "run.exec"} & set(t["time_s"]) or any(k.startswith("driver.") for k in t["time_s"])
    tm = json.loads((tmp_path / "tel_e2e_out" / "result.json").read_text())["telemetry"]
    assert sum(tm["time_s"].values()) == pytest.approx(tm["wall_s"], abs=0.05)
    # the agent's own LLM calls, no double counting against llm_usage
    by = {b["role"]: b for b in tm["llm"]["by_role_model"]}
    assert tm["llm"]["total"]["calls"] == res["llm_calls"]
    assert sum(b["input_tokens"] for b in by.values()) == res["llm_usage"]["prompt_tokens"]
    assert sum(b["output_tokens"] for b in by.values()) == res["llm_usage"]["completion_tokens"]
    assert by["fix"]["calls"] == state["fix_calls"]
    d = tm["derived"]
    assert d["steps"] == 8 and tm["evals"]["candidate"]["n"] == 8 and tm["evals"]["baseline"]["n"] == 1
    assert d["compile_cache_hit_rate"] is not None and 0 <= d["gpu_idle_frac"] <= 1
    assert d["time_by_category_s"]["compile"] > 0 and d["time_by_category_s"].get("driver_startup", 0) > 0
    # the JSONL rebuilds the same picture
    s = report.summarize_events(report.read_events(tmp_path / "telemetry.jsonl"))
    assert s["llm"]["total"]["calls"] == res["llm_calls"] and s["derived"]["steps"] == 8
    assert "llm_wait" in s["derived"]["time_by_category_s"]


@needs_gpu
def test_lenet_runs_with_telemetry_disabled(tmp_path):
    tel.set_enabled(False)
    try:
        agent, _ = _agent(tmp_path, "tel_off", rollouts=1, steps=2)
        try:
            res = asyncio.run(agent.run())
        finally:
            agentic_llm.unregister_stub("tel_off")
    finally:
        tel.set_enabled(True)
    assert res["final_ok"] and "telemetry" not in res
    assert all("telemetry" not in s["context"] for s in res["steps"])
    assert res["llm_calls"] >= 2 and res["llm_usage"]["prompt_tokens"] > 0
