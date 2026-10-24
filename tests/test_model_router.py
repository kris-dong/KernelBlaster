"""Model heterogeneity: role resolution, the escalation policy, and the agent's routing."""
import asyncio
import json
import shutil

import pytest

from kernelblaster.agents.model_router import ModelRouter, parse_policy, technique_tier
from kernelblaster.agents.opt_multislot import MultiSlotAgent
from kernelblaster.agents.utils import agentic_llm
from kernelblaster.knowledge.slots import LocalEvaluator, SlotProblem

from test_multislot_agent import ARCH, FIX, needs_gpu, scripted


def test_build_precedence(tmp_path):
    cfg = tmp_path / "models.json"
    cfg.write_text(json.dumps({"models": {"codegen_hard": "cfg-hard", "fix": "cfg-fix", "upkeep": "cfg-up"},
                               "policy": "stall:2"}))
    env = {"MODEL_CODEGEN_SIMPLE": "env-simple", "MODEL_CODEGEN_HARD": "env-hard", "MODEL_PLAN": "env-plan"}
    r = ModelRouter.build("base", hard="flag-hard", config=str(cfg), env=env)
    assert r.models["codegen_hard"] == "flag-hard"          # flag > config
    assert r.models["codegen_simple"] == "env-simple"       # env when neither flag nor config
    assert r.models["fix"] == "cfg-fix" and r.models["upkeep"] == "cfg-up"
    assert r.models["reflect"] is None                      # opt-in passes stay off (MODEL_PLAN ignored)
    assert [x.kind for x in r.policy] == ["stall"] and r.heterogeneous
    plain = ModelRouter.build("base", env={})
    assert set(plain.models.values()) - {None} == {"base"} and not plain.heterogeneous
    with pytest.raises(ValueError):
        cfg.write_text(json.dumps({"models": {"planner": "x"}}))
        ModelRouter.build("base", config=str(cfg), env={})
    with pytest.raises(ValueError):
        parse_policy("stall")
    with pytest.raises(ValueError):
        parse_policy("bogus")


def test_policy_rules():
    r = ModelRouter({"codegen_simple": "S", "codegen_hard": "H", "fix": "F"},
                    parse_policy("category,stall:3,share:0.5,explore,fix-escalate"))
    assert r.codegen(technique="memory_coalescing", tier="simple")[:2] == ("S", "codegen_simple")
    assert r.codegen(technique="shared_memory_tiling", tier="hard")[0] == "H"
    assert "stalled" in r.codegen(technique="x", tier="simple", stall=3)[2]
    assert "partition" in r.codegen(technique="x", tier="simple", share=0.6)[2]
    assert r.codegen(technique="x", tier="simple", kind="hypothesis")[0] == "H"
    assert r.fix(attempt=1, max_attempts=2) == ("F", "fix")
    assert r.fix(attempt=2, max_attempts=2) == ("H", "fix_escalated")
    never = ModelRouter({"codegen_simple": "S", "codegen_hard": "H"}, parse_policy("never,category"))
    assert never.codegen(technique="t", tier="hard")[0] == "S"
    assert never.fix(attempt=5, max_attempts=2)[0] == "S"   # fix defaults to codegen_simple


def test_technique_tier():
    assert technique_tier("shared_memory_tiling") == technique_tier("implicit_gemm") == "hard"
    assert technique_tier("vectorized_memory_access") == "simple"

    class T:
        params = {"tier": "hard"}
    assert technique_tier("vectorized_memory_access", T()) == "hard"   # a v2 technique's own tier wins


@needs_gpu
def test_agent_routes_by_role(tmp_path):
    """Two stub models: every step's model follows the policy and usage is split per role/model."""
    p = SlotProblem(shutil.copytree(FIX, tmp_path / "lenet"))
    state = {"broke": False, "fix_calls": 0, "linear_ok": None}
    calls = []
    fn = scripted(state)
    for name in ("cheap", "big"):
        agentic_llm.register_stub(name, (lambda n: lambda m: (calls.append(n), fn(m))[1])(name))
    try:
        router = ModelRouter({"codegen_simple": "stub:cheap", "codegen_hard": "stub:big", "fix": "stub:cheap"},
                             parse_policy("category,fix-escalate"))
        agent = MultiSlotAgent(p, LocalEvaluator(ARCH, work_root=tmp_path / "work"), model="stub:cheap",
                               out_dir=tmp_path / "out", arch=ARCH, rollouts=2, steps=4, min_share=0.0, seed=1,
                               max_fix_attempts=1, router=router)
        res = asyncio.run(agent.run())
    finally:
        for name in ("cheap", "big"):
            agentic_llm.unregister_stub(name)
    assert res["final_ok"]
    for s in res["steps"]:
        want = "stub:big" if technique_tier(s["technique"]) == "hard" else "stub:cheap"
        assert s["context"]["model"] == want, s
        for fm in s["context"].get("fix_models", []):
            assert fm == "stub:big"                          # max_fix=1: the only fix attempt is the last
    by = {(d["role"], d["model"]): d["calls"] for d in res["llm_usage_by_model"]}
    assert sum(by.values()) == res["llm_calls"] == len(calls)
    assert res["models"]["heterogeneous"] and res["models"]["policy"] == "category,fix-escalate"
