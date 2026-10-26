"""kb-experiment/1, v1.1 (X-149 / X-145 follow-up): validity + lineage, decision, unchanged source + noise, tokens / $ per
turn, absolute vs paired time, probe-time clocks, nvcc. All optional fields; old records / lines stay learnable."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from kernelblaster.agents import experiment_tool as xt
from kernelblaster.knowledge import clock_guard as cg
from kernelblaster.knowledge.v2 import experiments as E

from test_experiments import (B, FINAL, PHASE_OUT, PROG_PHASE, R1, R2, VARIANT_OUT, FakeTool, _bare_agent, _legacy_dir,
                              _llm, _run_dir, _step, run)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (E.ENV_RECORD, E.ENV_INGEST, E.ENV_PROMPT, E.ENV_PROMPT_CHARS, xt.ENV, E.ENV_ARM, E.ENV_SEED, E.ENV_FORCE,
              E.ENV_VALID, E.ENV_NOTE):
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------- (f) decision

@pytest.mark.parametrize("text,dec", [
    ("The wait dominates, so I will use persistent state and adopt the register tiling.", "adopted"),
    ("This is slower and does not help; I'll abandon the idea.", "rejected"),
    ("The numbers are noisy, inconclusive; I will use the baseline anyway.", "inconclusive"),
    ("Variant 1 is 2x slower than variant 0.", "rejected"),
    ("Here are the results.", None), ("", None), (None, None),
    ("I will use it, but it is slower.", None),                      # mixed cues: no statement
])
def test_parse_decision(text, dec):
    assert E.parse_decision(text) == dec


# ---------------------------------------------------------------- (b) lineage / validity

def test_lineage_and_validity(monkeypatch):
    assert E.lineage_from_env() == {}
    monkeypatch.setenv(E.ENV_ARM, "S8_first_r1")
    monkeypatch.setenv(E.ENV_FORCE, "step_sync, tile")
    assert E.lineage_from_env(seed=3) == {"arm": "S8_first_r1", "seed": "3", "forced_order": ["step_sync", "tile"]}
    assert E.validity_of({}) == {"valid": True, "reasons": [], "ncu_disabled": None}
    assert E.validity_of({"clock_suspect": True})["reasons"] == ["clock_suspect"]
    v = E.validity_of({"ncu_disabled": "ncu timed out"})
    assert not v["valid"] and v["reasons"] == ["ncu_timed_out"]
    assert E.validity_of({"ncu_disabled": "ncu timed out", "clock_clean_after_ncu": True})["valid"]   # checked clean
    assert E.validity_of({"ncu_disabled": "no GPU performance-counter permission"})["valid"]          # not a contamination
    monkeypatch.setenv(E.ENV_VALID, "0")
    assert E.validity_of({})["reasons"] == ["operator_marked_invalid"]


def _step_flags(rec, flags_keep=None):
    ids, log = _step(rec)
    return ids


def test_recorder_v11_fields_and_ingest_skips_invalid(tmp_path):
    state = {"ncu_disabled": None}
    rec = E.ExperimentRecorder(tmp_path / "ok", run_id="run1", problem="p1", arch="sm_89", dtype="fp16",
                               lineage={"arm": "A", "seed": "1"}, flags=lambda: state)
    _step(rec)
    rec.record_outcome("r0s1", status="improved", speedup=1.1, reward=0.1, slot_us_before=10.0, slot_us_after=9.0,
                       time={"abs_candidate_us": 4300.0, "paired_after_us": 4360.0}, noise_pct=None)
    lines = E.read_jsonl(tmp_path / "ok" / "experiments.jsonl")
    ex = [x for x in lines if x["type"] == "experiment"]
    assert all(x["schema"] == "kb-experiment/1" and x["schema_v"] == "1.1" and x["lineage"] == {"arm": "A", "seed": "1"}
               and x["validity"]["valid"] for x in ex)
    assert ex[1]["decision"] in ("adopted", None) and "decision_src" in ex[1]
    recs = E.finalize(lines)
    assert all(r["valid"] and r["outcome"]["time"]["abs_candidate_us"] == 4300.0 and r["outcome"]["unchanged"] is False
               for r in recs)
    # the same run with an ncu timeout recorded: invalid, kept in the records, not learned from
    state["ncu_disabled"] = "ncu timed out"
    bad = E.ExperimentRecorder(tmp_path / "bad", run_id="run2", problem="p2", arch="sm_89", dtype="fp16", flags=lambda: state)
    _step(bad)
    bad.record_outcome("r0s1", status="improved", speedup=1.1, reward=0.1)
    brecs = E.finalize(E.read_jsonl(tmp_path / "bad" / "experiments.jsonl"))
    assert brecs and all(r["valid"] is False and r["validity"]["reasons"] == ["ncu_timed_out"] for r in brecs)
    assert E.facts_from_records(brecs) == {} and E.probes_from_records(brecs) == []
    assert E.facts_from_records(recs) and E.probes_from_records(recs)                    # the valid run does count
    mixed = E.facts_from_records(recs + brecs)
    assert {i for f in mixed.values() for i in f["support"]} <= {r["id"] for r in recs}
    kd = tmp_path / "kd"
    rep = E.ingest(tmp_path / "bad", kd)
    assert rep["facts"] == 0 and rep["skipped_invalid"] == len(brecs) and rep["records"] == len(brecs)
    assert E.read_jsonl(kd / "experiments" / "records.jsonl")                            # visible, flagged


def test_old_lines_without_v11_fields_are_learnable(tmp_path):
    d = _run_dir(tmp_path, "run", "run1", "p1")
    lines = E.read_jsonl(d / "experiments.jsonl")
    for ln in lines:                                                                      # strip every v1.1 key
        for k in ("schema_v", "decision", "decision_src", "lineage", "validity", "cost", "nvcc", "unchanged", "noise_pct", "time"):
            ln.pop(k, None)
    recs = E.finalize(lines)
    assert recs and all("valid" not in r and E.learnable(r) for r in recs)
    assert E.facts_from_records(recs)


# ---------------------------------------------------------------- (d) unchanged source + noise

def test_unchanged_outcome_is_not_evidence_and_noise_is_estimated(tmp_path):
    assert E.same_source("void k(){ /* a */ int x=1; }", "void k() { int x = 1; } // c") and not E.same_source("a", "b")
    rec = E.ExperimentRecorder(tmp_path, run_id="run1", problem="p", arch="sm_89", dtype="fp16")
    _step(rec, key="r0s1")
    rec.record_outcome("r0s1", status="unchanged", speedup=None, reward=-0.02, slot_us_before=100.0, slot_us_after=102.7,
                       unchanged=True, noise_pct=2.7)
    recs = E.finalize(E.read_jsonl(tmp_path / "experiments.jsonl"))
    assert recs[0]["outcome"]["status"] == "unchanged" and recs[0]["outcome"]["noise_pct"] == 2.7 and E.learnable(recs[0])
    probes = E.probes_from_records(recs)
    assert sum(p["next_regressed"] + p["next_improved"] + p["next_neutral"] for p in probes) == 0
    assert sum(p["next_unchanged"] for p in probes) == len(recs)                         # counted apart
    n = E.noise_stats(recs)
    assert n["n"] == 1 and n["median_abs_pct"] == 2.7
    rep = E.report(recs)
    assert rep["noise"]["n"] == 1 and "unchanged-source steps: 1" in E._fmt(rep)
    assert rep["step_outcomes"] == {"unchanged": 2}


# ---------------------------------------------------------------- (a)(c)(g) clocks, tokens / $, nvcc through converse

def test_converse_carries_clock_nvcc_and_recorder_cost(tmp_path):
    class Tool(FakeTool):
        async def run(self, code, left):
            r = await super().run(code, left)
            pre = cg.ClockVerdict(state="ok", sample=cg.ClockSample(0, "GPU-x", 2520, 2520, 9001, 9001)).to_dict()
            bad = cg.ClockVerdict(state="locked", reasons=["held"], sample=cg.ClockSample(0, "GPU-x", 1065, 2520)).to_dict()
            return {**r, "clock": {"pre": pre, "post": bad}, "gpu_uuid": "GPU-x",
                    "nvcc": {"version": "12.6.85", "flags": ["-O3", "-std=c++17", "-arch=sm_89", "-lineinfo"]}}
    seen = []
    _, log = run(xt.converse(_llm([R1, R2, FINAL], seen), [{"role": "user", "content": "u"}], Tool([PHASE_OUT, VARIANT_OUT]),
                             B, key="r0s1c0", record=True))
    assert log.records[0]["clock"]["pre"]["state"] == "ok" and log.records[0]["nvcc"]["version"] == "12.6.85"
    rec = E.ExperimentRecorder(tmp_path, run_id="run1", problem="p", arch="sm_89")
    usage = {0: {"prompt_tokens": 1000, "completion_tokens": 200, "usd": 0.01, "key": "k0"},
             1: {"prompt_tokens": 1500, "completion_tokens": 250, "usd": 0.02, "key": "k0t1"},
             2: {"prompt_tokens": 2000, "completion_tokens": 900, "usd": 0.05, "key": "k0t2"}}
    rec.record_step("r0s1", log, slot="gru", technique="t", op_class="recurrent", usage=usage)
    lines = E.read_jsonl(tmp_path / "experiments.jsonl")
    ex = [x for x in lines if x["type"] == "experiment"]
    assert ex[0]["gpu"]["uuid"] == "GPU-x" and ex[0]["gpu"]["clock"]["post"]["state"] == "locked"
    assert ex[0]["nvcc"]["flags"][0] == "-O3" and ex[0]["cost"] == {"prompt_tokens": 1000, "completion_tokens": 200,
                                                                    "usd": 0.01, "key": "k0", "turn": 0,
                                                                    "experiments_in_turn": 1}
    assert ex[0]["validity"] == {"valid": False, "reasons": ["clock_suspect"], "ncu_disabled": None, "clock_suspect": True}
    st = [x for x in lines if x["type"] == "step"][0]
    assert st["cost"] == {"prompt_tokens": 4500, "completion_tokens": 1350, "usd": 0.08} and len(st["turns_usage"]) == 3
    recs = E.finalize(lines)
    assert not any(E.learnable(r) for r in recs)                                           # a locked clock at probe time
    rep = E.report(recs)
    assert rep["cost"]["experiment_turns"] == 2 and rep["nvcc"] == {"12.6.85": 2} and rep["invalid"] == {"clock_suspect": 2}
    # the global store keeps the record but not the UUID
    s = E._slim(recs[0])
    assert "uuid" not in s["gpu"] and "uuid" not in s["gpu"]["clock"]["pre"] and s["gpu"]["clock"]["post"]["state"] == "locked"
    assert recs[0]["gpu"]["uuid"] == "GPU-x"                                               # the run's own record keeps it


# ---------------------------------------------------------------- the agent: usage per turn, unchanged, time, suspect

def test_agent_records_cost_lineage_unchanged_time_and_suspect(tmp_path, monkeypatch):
    monkeypatch.setenv(E.ENV_RECORD, "1")
    monkeypatch.setenv(E.ENV_ARM, "armX")
    a, seen = _bare_agent(tmp_path, monkeypatch, [R1, R2, FINAL])
    a.ncu_disabled, a.seed = None, 7
    a._usage_key = {"r0s1c0": {"prompt_tokens": 10, "completion_tokens": 2, "usd": 0.001, "key": "r0s1c0"},
                    "r0s1c0t1": {"prompt_tokens": 20, "completion_tokens": 4, "usd": 0.002, "key": "r0s1c0t1"}}
    run(a._multiturn([{"role": "user", "content": "u"}], "m", "codegen", "r0s1", "gru", "persistent_recurrence"))
    lines = E.read_jsonl(a.out / "experiments.jsonl")
    ex = [x for x in lines if x["type"] == "experiment"]
    assert ex[0]["lineage"] == {"arm": "armX", "seed": "7"} and ex[0]["cost"]["prompt_tokens"] == 10 and ex[1]["cost"]["usd"] == 0.002
    # the step's final source equals the seed: an unchanged outcome, never a regression
    res = SimpleNamespace(ok=True, deployed_us=4400.0, ref_e2e_us=4300.0, e2e_basis="graph", e2e_ratio={"med": 1.023},
                          gpu={"clock": {"suspect": False}})
    d = {"ctx": {"gain": -0.023, "before_us": 4300.0, "pairing": "paired"}, "e2e_after": 4399.0}
    seed = "void kernel_x() { int a = 1; }"
    a._experiment_outcome(0, 1, "gru", seed, "void kernel_x(){ int a=1; } // same", "regressed", 0.97, -0.02, d, 100.0, 102.7,
                          None, "o1", res=res)
    out = [x for x in E.read_jsonl(a.out / "experiments.jsonl") if x["type"] == "outcome"][0]
    assert out["status"] == "unchanged" and out["unchanged"] and out["noise_pct"] == 2.7 and out["speedup"] is None
    assert out["time"] == {"basis": "graph", "abs_candidate_us": 4400.0, "abs_incumbent_us": 4300.0,
                           "paired_before_us": 4300.0, "paired_after_us": 4399.0, "pairing": "paired", "ratio": 1.023}
    # a step timed on a clock-suspect GPU
    res2 = SimpleNamespace(ok=True, deployed_us=11000.0, ref_e2e_us=None, e2e_basis="eager", e2e_ratio=None, _clock_suspect=True)
    a._experiment_outcome(0, 2, "gru", seed, "void kernel_x() { int b = 2; }", "neutral", None, 0.0, None, 100.0, 220.0,
                          None, "o2", res=res2)
    o2 = [x for x in E.read_jsonl(a.out / "experiments.jsonl") if x["type"] == "outcome"][1]
    assert o2["status"] == "clock_suspect" and o2["validity"]["reasons"] == ["clock_suspect"] and not o2["unchanged"]


# ---------------------------------------------------------------- legacy import, CLI

def test_legacy_import_v11_fields_and_cli(tmp_path, capsys):
    d = _legacy_dir(tmp_path)
    r = json.loads((d / "result.json").read_text())
    r.update(gpu={"name": "NVIDIA L40S", "uuid": "GPU-1", "index": 0, "clock": {"suspect": False, "sm_mhz": 2520}},
             ncu={"disabled": "ncu timed out"}, llm_usage={"prompt_tokens": 100, "completion_tokens": 10, "cost_usd": 0.5})
    (d / "result.json").write_text(json.dumps(r))
    recs = E.import_legacy(d, lineage={"arm": "S8_first_r1"}, invalid=["clock_locked_after_ncu_timeout"])
    assert recs and all(x["lineage"] == {"arm": "S8_first_r1"} and x["valid"] is False and x["gpu"]["uuid"] == "GPU-1"
                        and x["cost"]["run_total"]["usd"] == 0.5 and x["legacy"] for x in recs)
    assert set(recs[0]["validity"]["reasons"]) == {"ncu_timed_out", "clock_locked_after_ncu_timeout"}
    assert E.facts_from_records(recs) == {}                                  # contaminated: nothing learned
    key = recs[0]["step_key"]
    u = E.import_legacy(d, unchanged=[key])
    assert any(x["outcome"] and x["outcome"]["status"] == "unchanged" for x in u)
    plain = E.import_legacy(_legacy_dir(tmp_path / "x") if (tmp_path / "x").mkdir() is None else d)
    assert plain
    k = tmp_path / "kd"
    assert E.main(["import-legacy", str(d), "--kdir", str(k), "--arm", "A", "--invalid", "locked"]) == 0
    assert "validity:" in capsys.readouterr().out
    assert all(x["lineage"] == {"arm": "A"} for x in E.read_jsonl(k / "experiments" / "records.jsonl"))
