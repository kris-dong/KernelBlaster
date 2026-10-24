"""The CUDA RL rollout records outcomes and injects knowledge only when configured."""
import asyncio
import json
import logging

from kernelblaster.agents.database import StateProfile
from kernelblaster.agents.opt_ncu_rl_optimized import OptimizedRLNCUAgent
from kernelblaster.agents.rl import UCB1Bandit
from kernelblaster.agents.utils import FeedbackError, agentic_llm
from kernelblaster.knowledge.semantic import SemanticKnowledge, classify_error


class FakeDB:
    def __init__(self):
        self.plan_kwargs = []
        self.updates = []
        self.optimization_strategies = {"s0": {"optimizations": []}}

    async def analyze_performance_state(self, *a, **k):
        return StateProfile(state_name="s0", primary_bottleneck="memory_bound",
                               secondary_characteristics=[], performance_signature="",
                               context_description="", relative_patterns={})

    async def generate_optimization_plan(self, analysis, code, top_n=4, **kw):
        self.plan_kwargs.append(kw)
        return [{"technique": "tiling", "relevance_score": 0.9, "description": "tile it"},
                {"technique": "vectorize", "relevance_score": 0.5, "description": "float4"}]

    async def match_state_against_database(self, profile):
        return "s0"

    def update_optimization_result(self, *a, **kw):
        self.updates.append(kw)


def make_agent(tmp_path, knowledge, outcomes):
    a = object.__new__(OptimizedRLNCUAgent)
    a._trajectory_lock = asyncio.Lock()
    a.total_trajectories = 0
    a.folder = tmp_path
    a.initial_cycles = 1000
    a.last_ncu_log = ""
    a.max_rollout_steps = len(outcomes)
    a.database = FakeDB()
    a.bandit = UCB1Bandit()
    a.agent_logger = logging.getLogger("rollout")
    a.prune_regression_pct = -1e9
    a.prune_patience = 99
    a.progress_writer = None
    a.problem_id = "p1"
    a.knowledge = knowledge
    seq = iter(outcomes)

    async def fake_apply(code, tech, desc, cycles, traj_dir, step, *, state=None):
        o = next(seq)
        if isinstance(o, Exception):
            raise o
        return code + "//", o, "", {}
    a._apply_optimization = fake_apply
    return a


def test_rollout_without_knowledge_is_unchanged(tmp_path):
    a = make_agent(tmp_path, None, [800, 900])
    traj = asyncio.run(a._run_rollout(0, "code", "s0"))
    assert len(traj.steps) == 2
    assert a.database.plan_kwargs == [{}, {}]          # no knowledge kwargs passed at all
    assert all("op_class" not in u for u in a.database.updates)
    assert not (tmp_path / "kb").exists()


def test_rollout_records_outcomes_and_reflects(tmp_path):
    calls = []
    agentic_llm.register_stub("roll", lambda m: (calls.append(m), '{"edits": []}')[1])
    try:
        kb = SemanticKnowledge(tmp_path / "kb", target="cuda", arch="sm_89", op_class="gemm",
                               problem_id="p1", reflect_model="stub:roll", every_n=3)
        a = make_agent(tmp_path, kb, [800, 880, FeedbackError("nvcc error: expected ';'")])
        asyncio.run(a._run_rollout(0, "code", "s0"))
    finally:
        agentic_llm.unregister_stub("roll")
    recs = list(kb.outcomes)
    assert [r.status for r in recs] == ["improved", "regressed", "compile_error"]
    assert recs[0].speedup == 1.25 and recs[0].arch == "sm_89" and recs[0].op_class == "gemm"
    assert recs[0].problem_id == "p1" and recs[0].state_before == "s0" and recs[2].error
    assert a.database.updates[0]["op_class"] == "gemm" and a.database.updates[0]["arch"] == "sm_89"
    kw = a.database.plan_kwargs
    assert kw[0]["stats_context"] == {"op_class": "gemm", "arch": "sm_89"}
    assert kw[0]["knowledge_notes"] is None              # nothing known yet
    assert "Already tried" in kw[1]["knowledge_notes"]   # tallies visible to the very next step
    assert len(calls) == 1                               # one reflection at rollout end (3 >= every_n)
    assert "vectorize" in a._kb_notes("codegen", technique="tiling") or "tiling" in a._kb_notes(
        "codegen", technique="tiling")
    assert a._kb_notes("fix", technique="tiling") is None   # no pitfall claims yet
    a.knowledge = None
    assert a._kb_notes("plan") is None


def test_classify_error():
    assert classify_error("kernel.cu(12): error: identifier undefined") == "compile_error"
    assert classify_error("Output mismatch: max abs diff 0.3") == "wrong_output"
    assert classify_error("illegal memory access was encountered") == "runtime_error"
