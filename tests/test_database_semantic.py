"""Finer stats, old-JSON load, and knowledge injection in GPUOptimizationDatabase."""
import asyncio
import json
import logging
import shutil
from pathlib import Path

import pytest

from kernelblaster.agents.database import GPUOptimizationDatabase

REPO = Path(__file__).resolve().parents[1]
OLD_JSON = REPO / "data" / "kernelblaster" / "optimization_database.json"
LOG = logging.getLogger("test_db")


class FakeLLM:
    """Captures prompts; answers plans with a fixed JSON list."""

    def __init__(self, n=4):
        self.logger = LOG
        self.prompts = []
        self.layered = []
        self.n = n

    def is_available(self):
        return True

    def _plan(self):
        return json.dumps([{"technique": f"t{i}", "relevance_score": 0.5, "description": "d"}
                           for i in range(self.n)])

    async def query(self, prompt, max_tokens=1000, temperature=0.1):
        self.prompts.append(prompt)
        return self._plan()

    async def query_layered(self, system, user, **kw):
        self.layered.append((system, user))
        return self._plan()


def make_db(tmp_path, *, tiered=False, llm=None):
    llm = llm or FakeLLM()
    shutil.copy2(OLD_JSON, tmp_path / "optimization_database.json")   # pre-semantic snapshot
    return GPUOptimizationDatabase(tmp_path / "optimization_database.md", tmp_path / "no_report.md",
                                   llm, cheap_llm=llm if tiered else None), llm


def test_old_json_loads_and_new_json_is_back_compatible(tmp_path):
    assert "fine_stats" not in json.loads(OLD_JSON.read_text())
    db, _ = make_db(tmp_path)
    assert db.fine_stats == {} and len(db.optimization_strategies) > 5
    state = next(iter(db.optimization_strategies))
    tech = db.optimization_strategies[state]["optimizations"][0].technique
    db.update_optimization_result(state, tech, 20.0, op_class="gemm", arch="sm_89")
    data = json.loads((tmp_path / "optimization_database.json").read_text())
    assert set(data) >= {"known_states", "optimization_strategies", "fine_stats"}
    # entries keep exactly the legacy schema
    assert set(data["optimization_strategies"][state]["optimizations"][0]) == set(
        json.loads(OLD_JSON.read_text())["optimization_strategies"][state]["optimizations"][0])
    db2 = GPUOptimizationDatabase(tmp_path / "optimization_database.md", tmp_path / "no_report.md", FakeLLM())
    assert db2.fine_stats == db.fine_stats and db2.fine_stats


def test_fine_stats_fallback(tmp_path):
    db, _ = make_db(tmp_path)
    state = next(iter(db.optimization_strategies))
    tech = db.optimization_strategies[state]["optimizations"][0].technique
    assert db.get_speedup_estimate(state, tech, "gemm", "sm_89")[2] == "prior"
    # 50% improvement per step == 2x; record on a *different* op class first
    for _ in range(3):
        db.update_optimization_result(state, tech, 50.0, op_class="conv", arch="sm_89")
    sp, n, src = db.get_speedup_estimate(state, tech, "gemm", "sm_89")
    assert src == "coarse" and n == 3 and sp == pytest.approx(2.0)
    db.update_optimization_result(state, tech, -25.0, op_class="gemm", arch="sm_89")   # 0.8x
    db.update_optimization_result(state, tech, -25.0, op_class="gemm", arch="sm_89")
    assert db.get_speedup_estimate(state, tech, "gemm", "sm_89")[2] == "coarse"        # n=2 < 3
    db.update_optimization_result(state, tech, -25.0, op_class="gemm", arch="sm_89")
    sp, n, src = db.get_speedup_estimate(state, tech, "gemm", "sm_89")
    assert src == "fine" and n == 3 and sp == pytest.approx(0.8)
    assert db.get_speedup_estimate(state, tech, "conv", "sm_89")[0] == pytest.approx(2.0)
    coarse = db.fine_stats[db.fine_stats_key(state, tech)]
    assert coarse["n"] == 6 and coarse["n_improved"] == 3
    # legacy coarse fields still updated
    assert db.optimization_strategies[state]["optimizations"][0].usage_count == 6


def test_fine_stats_recorded_for_technique_outside_state(tmp_path):
    db, _ = make_db(tmp_path)
    db.update_optimization_result("memory_bandwidth_saturated", "not_in_this_state", 10.0,
                                  op_class="gemm", arch="sm_89")
    assert db.fine_stats_key("memory_bandwidth_saturated", "not_in_this_state", "gemm", "sm_89") in db.fine_stats


def test_codegen_and_fix_prompts_unchanged_without_knowledge(tmp_path):
    db, _ = make_db(tmp_path)
    kw = dict(technique_name="tiling", kernel_source="__global__ void k(){}", ncu_metrics_json={"a": 1})
    base = db.build_codegen_messages(**kw)
    assert db.build_codegen_messages(**kw, knowledge_notes=None) == base
    assert db.build_codegen_messages(**kw, knowledge_notes="  ") == base
    withk = db.build_codegen_messages(**kw, knowledge_notes="- (+3/-0) [pitfall] use __syncthreads")
    assert withk[0] == base[0]                       # cache-stable system prompt untouched
    assert "LEARNED NOTES FOR THIS TECHNIQUE" in withk[1]["content"]
    assert "LEARNED" not in base[1]["content"]
    fb = db.build_fix_messages(broken_kernel="x", compiler_error="e")
    assert db.build_fix_messages(broken_kernel="x", compiler_error="e", knowledge_notes=None) == fb
    fk = db.build_fix_messages(broken_kernel="x", compiler_error="e", knowledge_notes="- FIX: include mma.h")
    assert "KNOWN PITFALLS" in fk[1]["content"] and fk[1]["content"].endswith("```cpp``` block.")


def test_plan_prompt_injection_legacy_and_tiered(tmp_path):
    for tiered in (False, True):
        d = tmp_path / str(tiered)
        d.mkdir()
        db, llm = make_db(d, tiered=tiered)
        state = next(iter(db.optimization_strategies))
        tech = db.optimization_strategies[state]["optimizations"][0].technique
        for _ in range(3):
            db.update_optimization_result(state, tech, 50.0, op_class="gemm", arch="sm_89")
        get = (lambda: llm.layered[-1][1]) if tiered else (lambda: llm.prompts[-1])
        asyncio.run(db.generate_optimization_plan("state", "code", top_n=4))
        plain = get()
        asyncio.run(db.generate_optimization_plan("state", "code", top_n=4,
                                                  knowledge_notes=None, stats_context=None))
        assert get() == plain and "LEARNED KNOWLEDGE" not in plain
        plan = asyncio.run(db.generate_optimization_plan(
            "state", "code", top_n=4, knowledge_notes="- (+2/-0) [helps | tiling] big K",
            stats_context={"op_class": "gemm", "arch": "sm_89"}))
        assert len(plan) == 4
        p = get()
        assert "LEARNED KNOWLEDGE" in p and "big K" in p
        assert f"{tech} in {state}: 2.00x geomean per step over 3 tries" in p
        if tiered:
            assert llm.layered[-1][0] == llm.layered[0][0]      # system prompt unchanged
