# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""X-145 experiment recording pipeline (knowledge/v2/experiments.py): parser, recorder, ingest, merge / consolidation /
global round trip, flag-off identity, old-store compatibility. Stub LLM and stub experiment tool: no GPU, no nvcc."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernelblaster.agents import experiment_tool as xt
from kernelblaster.knowledge.claims import CLAIM_KINDS, Claim, EXPERIMENT
from kernelblaster.knowledge.v2 import KnowledgeV2
from kernelblaster.knowledge.v2 import experiments as E
from kernelblaster.knowledge.v2.consolidate import consolidate_store
from kernelblaster.knowledge.v2.merge import GlobalStore, Policy, materialize

from test_knowledge_global import P89, gk, merge, outcome, start_run, tech

PHASE_OUT = ("v0 it0 1853.4 us | blk0 wait 1988 mma 1078 gate 446 | blkL 2174 934 416 ns\n"
             "v0 it1 1811.2 us | blk0 wait 1998 mma 1074 gate 430 | blkL 2216 836 442 ns\n")
VARIANT_OUT = "variant 0: 836.9 us  err=no error\nvariant 1: 1600.5 us  err=no error\nmaxdiff 0\n"
PROG_PHASE = ("#include <cstdio>\n#define H 256\nconstexpr int T = 512;\n__global__ void k(){ long t=clock64(); }\n"
              "int main(){ printf(\"x\"); }\n")
PROG_VAR = ("#include <cstdio>\nconst int N = 1024;\nint main(){ float ms; cudaEventElapsedTime(&ms,0,0);"
            " printf(\"variant\"); }\n")
FINAL = "I will use the register tiling.\n```cuda\nvoid kernel_x() { /* tile_regs */ }\n```"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (E.ENV_RECORD, E.ENV_INGEST, E.ENV_PROMPT, E.ENV_PROMPT_CHARS, xt.ENV):
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------- parser

def test_parser_real_a2_outputs():
    ms = E.parse_measurements(VARIANT_OUT)
    assert [(m["name"], m["value"], m["unit"]) for m in ms[:2]] == [("variant 0", 836.9, "us"),
                                                                    ("variant 1", 1600.5, "us")]
    ms = E.parse_measurements(PHASE_OUT)
    ph = {m["name"]: m for m in ms if m["line"] == 0 and m["group"] == 1}
    assert set(ph) == {"blk0 wait", "mma", "gate"}
    assert ph["blk0 wait"]["value"] == 1988 and ph["gate"]["unit"] == "ns" and ph["gate"].get("inferred_unit")
    assert E._stem("blk0 wait") == "wait"
    ms = E.parse_measurements("err no error\nmode 1 median 32.5 us\nmode 3 median 1345.5 us\n")
    assert [m["name"] for m in ms] == ["mode 1 median", "mode 3 median"]


def test_parser_tolerant_forms_and_garbage():
    ms = E.parse_measurements("bw = 812.5 GB/s, occupancy: 6 blocks/SM\n3.2e3 ns\nno numbers here\n\x00\xff junk 12")
    d = {m["name"]: m for m in ms}
    assert d["bw"]["unit"] == "GB/s" and d["bw"]["value"] == 812.5
    assert any(m["unit"] == "ns" and m["value"] == 3200 for m in ms)
    assert E.parse_measurements("") == [] and E.parse_measurements(None) == []
    assert len(E.parse_measurements("a 1 us\n" * 1000)) == 200            # capped


def test_probe_kind_and_params():
    assert E.classify_probe(PROG_PHASE) == "phase_breakdown"
    assert E.classify_probe(PROG_VAR, VARIANT_OUT) == "variant_timing"
    assert E.classify_probe("int main(){ cudaOccupancyMaxActiveBlocksPerMultiprocessor(0,0,0,0); }") == "occupancy"
    assert E.classify_probe(None, PHASE_OUT) == "phase_breakdown"            # legacy: from the output
    tmpl, params = E.extract_params(PROG_PHASE)
    assert params == {"H": 256, "T": 512} and "@@H@@" in tmpl and "@@T@@" in tmpl
    t2, _ = E.extract_params(PROG_PHASE.replace("256", "1024").replace("512", "64"))
    assert t2 == tmpl                                                         # same template at another shape


def test_reference_overlap():
    seed = "void kernel_x(){ for(int i=0;i<n;i++){} }"
    final = "void kernel_x(){ register_tile_cols(); wait_spin_flag(); }"
    prog = "int main(){ wait_spin_flag(); register_tile_cols(); }"
    ov = E.reference_overlap(seed, final, prog)
    assert ov["shared"] == 2 and ov["score"] > 0.5
    assert E.reference_overlap(None, final, prog) is None


# ---------------------------------------------------------------- converse (stub LLM / tool)

class FakeTool:
    arch = "89"

    def __init__(self, outs):
        self.outs, self.calls = list(outs), []

    async def run(self, code, left):
        self.calls.append(code)
        out = self.outs.pop(0)
        return {"ok": True, "stage": "run", "exit": 0, "out": out, "wall_s": 0.1, "device": "0"}


def _llm(replies, seen):
    async def llm(conv, key, extra=None):
        seen.append((key, [m["content"] for m in conv]))
        return replies.pop(0)
    return llm


B = {"turns": 3, "runs": 4, "seconds": 60}
R1 = "Is the step dominated by the wait?\n```experiment\n" + PROG_PHASE + "```"
R2 = "The wait dominates (about 2 us).\n```experiment\n" + PROG_VAR + "```"


def test_converse_records_purpose_conclusion_and_source():
    seen = []
    text, log = run(xt.converse(_llm([R1, R2, FINAL], seen), [{"role": "user", "content": "u"}],
                                FakeTool([PHASE_OUT, VARIANT_OUT]), B, key="r0s1c0", record=True))
    assert text == FINAL and len(log.records) == 2
    a, b = log.records
    assert a["purpose"] == "Is the step dominated by the wait?" and a["program"].startswith("#include")
    assert a["conclusion"] == "The wait dominates (about 2 us)."          # prose opening the next reply
    assert b["purpose"] == "The wait dominates (about 2 us)." and b["conclusion"] == "I will use the register tiling."
    assert log.rationale == "I will use the register tiling."
    assert "records" not in log.to_json() and "rationale" not in log.to_json()


def test_converse_default_is_unchanged():
    """record=False / no extra prompt: same conversation, same step log, nothing extra kept."""
    s1, s2 = [], []
    t1, l1 = run(xt.converse(_llm([R1, FINAL], s1), [{"role": "user", "content": "u"}], FakeTool([PHASE_OUT]), B,
                             key="k"))
    t2, l2 = run(xt.converse(_llm([R1, FINAL], s2), [{"role": "user", "content": "u"}], FakeTool([PHASE_OUT]), B,
                             key="k", record=True))
    assert s1 == s2 and t1 == t2 and l1.to_json() == l2.to_json()         # recording does not change the dialogue
    assert l1.records == [] and l1.rationale == ""
    assert s1[0][1][-1] == "u\n\n" + xt.prompt_section(B, "89")           # the exact old prompt
    s3 = []
    run(xt.converse(_llm([FINAL], s3), [{"role": "user", "content": "u"}], FakeTool([]), B, key="k",
                    extra_prompt="HINT"))
    assert s3[0][1][-1] == "u\n\n" + xt.prompt_section(B, "89") + "\n\nHINT"


# ---------------------------------------------------------------- recorder / finalize

def _step(rec, key="r0s1", outs=(PHASE_OUT, VARIANT_OUT), replies=None, tech_="persistent_recurrence", problem="p"):
    seen = []
    _, log = run(xt.converse(_llm(list(replies or [R1, R2, FINAL]), seen), [{"role": "user", "content": "u"}],
                             FakeTool(outs), B, key=key + "c0", record=True))
    return rec.record_step(key, log, slot="gru", technique=tech_, op="gru_seq_f16", op_class="recurrent",
                           shape_bucket="B10_T512_H256", model="stub", role="codegen_hard", rationale=log.rationale,
                           extra_prompt="", fact_ids=[]), log


def test_recorder_lines_blobs_and_outcome_join(tmp_path):
    rec = E.ExperimentRecorder(tmp_path, run_id="run1", problem="p1", arch="sm_89", dtype="fp16")
    ids, _ = _step(rec)
    assert len(ids) == 2 and rec.pointer()["records"] == 2
    rec.record_outcome("r0s1", status="improved", speedup=1.14, reward=0.115, e2e_gain=0.115, slot_us_before=4972.0,
                       slot_us_after=4358.0, impl_id="impl1", outcome_id="o1", overlap=None)
    lines = E.read_jsonl(tmp_path / "experiments.jsonl")
    assert [x["type"] for x in lines] == ["experiment", "experiment", "step", "outcome"]
    blob = tmp_path / "experiments" / "blobs" / (E.sha256(PROG_PHASE) + ".cu")
    assert blob.read_text() == PROG_PHASE                                  # content-addressed
    recs = E.finalize(lines, gpu={"name": "NVIDIA L40S", "uuid": "secret"}, blob_dir=tmp_path / "experiments" / "blobs")
    assert len(recs) == 2
    r = recs[0]
    assert r["gpu"]["name"] == "NVIDIA L40S" and "uuid" not in r["gpu"] and r["arch"] == "sm_89"
    assert r["probe_kind"] == "phase_breakdown" and r["outcome"]["status"] == "improved"
    assert r["purpose"] and r["conclusion"] and r["program_bytes"] == len(PROG_PHASE)
    assert r["out_tail"].startswith("v0 it0") and r["measurements"][0]["unit"] == "us"
    # deterministic ids: the same step recorded again (a resumed step) gives the same ids
    rec2 = E.ExperimentRecorder(tmp_path, run_id="run1", problem="p1", arch="sm_89")
    ids2, _ = _step(rec2)
    assert ids2 == ids
    assert len(E.finalize(E.read_jsonl(tmp_path / "experiments.jsonl"))) == 2      # last line per id wins, no dups


def test_output_cap(tmp_path):
    rec = E.ExperimentRecorder(tmp_path, run_id="r", problem="p", arch="sm_89")
    big = "x 1 us\n" * 5000
    _step(rec, outs=(big, big))
    r = E.finalize(E.read_jsonl(tmp_path / "experiments.jsonl"))[0]
    assert len(r["out_tail"]) <= E.OUT_TAIL


# ---------------------------------------------------------------- ingest

def _run_dir(tmp, name, run_id, problem, *, outs=(PHASE_OUT, VARIANT_OUT), status="improved", tech_="persistent_recurrence",
            prog_phase=PROG_PHASE):
    d = tmp / name
    rec = E.ExperimentRecorder(d, run_id=run_id, problem=problem, arch="sm_89", dtype="fp16")
    r1 = R1.replace(PROG_PHASE, prog_phase)
    _step(rec, outs=outs, tech_=tech_, replies=[r1, R2, FINAL])
    rec.record_outcome("r0s1", status=status, speedup=1.1, reward=0.1, impl_id=f"impl_{run_id}", outcome_id=f"o_{run_id}")
    return d


def _store_dir(tmp, name):
    k = tmp / name
    kv = KnowledgeV2(k, seed_mined=False)
    return k, kv


def test_ingest_idempotent_and_facts(tmp_path):
    d = _run_dir(tmp_path, "run", "run1", "p1")
    k, kv = _store_dir(tmp_path, "k")
    r1 = E.ingest(d, k, kv.claims, gpu={"name": "NVIDIA L40S"})
    files1 = {p.name: p.read_bytes() for p in (k / "experiments").glob("*.jsonl")}
    c1 = {c.id: (c.support, c.contradict, json.dumps(c.data, sort_keys=True)) for c in kv.claims.claims.values()
          if c.kind == EXPERIMENT}
    r2 = E.ingest(d, k, kv.claims, gpu={"name": "NVIDIA L40S"})
    files2 = {p.name: p.read_bytes() for p in (k / "experiments").glob("*.jsonl")}
    c2 = {c.id: (c.support, c.contradict, json.dumps(c.data, sort_keys=True)) for c in kv.claims.claims.values()
          if c.kind == EXPERIMENT}
    assert r1["records"] == 2 and r1["added"] >= 1 and r2["added"] == 0 and r2["updated"] == 0
    assert files1 == files2 and c1 == c2 and c1
    ph = [c for c in kv.claims.claims.values() if c.data.get("fact_type") == "phase_profile"][0]
    assert ph.kind == EXPERIMENT and ph.arch == "sm_89" and ph.op_class == "recurrent" and ph.support == 1
    assert "wait-dominated" in ph.text and "wait ~1993ns" in ph.data["summary"]
    assert ph.data["experiment_ids"] and ph.data["gpus"] == ["NVIDIA L40S"]
    # a persisted copy reloads (SQLite claim store)
    kv2 = KnowledgeV2(k, seed_mined=False)
    assert kv2.claims.claims[ph.id].data["summary"] == ph.data["summary"]
    # probes: template shared across shapes; usage stats from the step outcome
    probes = E.read_jsonl(k / "experiments" / "probes.jsonl")
    pb = {p["question"]: p for p in probes}
    assert pb["phase_breakdown"]["next_improved"] == 1 and pb["phase_breakdown"]["params"] == {"H": 256, "T": 512}
    assert pb["phase_breakdown"]["ontology"]["in_technique_menu"] is False


def test_confidence_rule_agreement_and_contradiction(tmp_path):
    k, kv = _store_dir(tmp_path, "k")
    a = _run_dir(tmp_path, "a", "runA", "pA")
    b = _run_dir(tmp_path, "b", "runB", "pB", prog_phase=PROG_PHASE.replace("256", "1024"))
    E.ingest(a, k, kv.claims)
    E.ingest(b, k, kv.claims)
    ph = [c for c in kv.claims.claims.values() if c.data.get("fact_type") == "phase_profile"]
    assert len(ph) == 1 and ph[0].support == 2 and sorted(ph[0].data["problems"]) == ["pA", "pB"]
    # the same program twice inside one run is ONE independent experiment
    d = tmp_path / "c"
    rec = E.ExperimentRecorder(d, run_id="runC", problem="pC", arch="sm_89")
    _step(rec, key="r0s1", outs=(PHASE_OUT, VARIANT_OUT))
    _step(rec, key="r0s2", outs=(PHASE_OUT, VARIANT_OUT))
    E.ingest(d, k, kv.claims)
    assert [c for c in kv.claims.claims.values() if c.data.get("fact_type") == "phase_profile"][0].support == 3
    # a probe finding another dominant component contradicts it
    mma = PHASE_OUT.replace("wait 1988", "wait 100").replace("wait 1998", "wait 100")
    e = _run_dir(tmp_path, "e", "runE", "pE", outs=(mma, VARIANT_OUT), prog_phase=PROG_PHASE.replace("256", "99"))
    E.ingest(e, k, kv.claims)
    ph = {c.data["subject"]: c for c in kv.claims.claims.values() if c.data.get("fact_type") == "phase_profile"}
    assert ph["wait"].contradict == 1 and ph["mma"].support == 1


def test_experiment_claims_hidden_from_default_retrieval(tmp_path):
    d = _run_dir(tmp_path, "run", "run1", "p1")
    k, kv = _store_dir(tmp_path, "k")
    E.ingest(d, k, kv.claims)
    kw = dict(target="cuda", arch="sm_89", op_class="recurrent")
    assert not [c for c in kv.claims.relevant(**kw, include_ineligible=True) if c.kind == EXPERIMENT]
    assert [c for c in kv.claims.relevant(**kw, kinds=[EXPERIMENT]) if c.kind == EXPERIMENT]
    assert EXPERIMENT not in CLAIM_KINDS                                      # the reflection schema is untouched


# ---------------------------------------------------------------- prompt

def test_prompt_text_cap_and_scope(tmp_path):
    d = _run_dir(tmp_path, "run", "run1", "p1")
    k, kv = _store_dir(tmp_path, "k")
    E.ingest(d, k, kv.claims)
    t, ids = E.prompt_text(kv.claims, k, arch="sm_89", op_class="recurrent", n=3, max_chars=800)
    assert "wait-dominated" in t and len(t) <= 800 and ids
    assert "probe template '" in t
    assert E.prompt_text(kv.claims, k, arch="sm_100", op_class="recurrent", n=3)[0].count("wait-dominated") == 0
    assert E.prompt_text(kv.claims, k, arch="sm_89", op_class="gemm", n=3) == ("", [])
    assert E.prompt_text(kv.claims, k, arch="sm_89", op_class="recurrent", n=0) == ("", [])
    short, _ = E.prompt_text(kv.claims, k, arch="sm_89", op_class="recurrent", n=3, max_chars=120)
    assert len(short) <= 120
    assert E.prompt_n() == 0                                                  # off unless the env asks


# ---------------------------------------------------------------- agent hooks (no GPU: a bare agent object)

def _bare_agent(tmp_path, monkeypatch, replies):
    from kernelblaster.agents import opt_multislot as om
    a = om.MultiSlotAgent.__new__(om.MultiSlotAgent)
    slot = SimpleNamespace(op="gru_seq_f16", op_class="recurrent", instances=[1],
                           shapes=lambda: [{"B": 10, "T": 512, "H": 256}])
    a.p = SimpleNamespace(name="prob", target="cuda", dtype="fp16", slots={"gru": slot}, root=tmp_path)
    a.ev, a.arch, a.arch_label, a.run_id, a.out, a.kv = object(), "89", "sm_89", "rid", tmp_path / "out", None
    a.out.mkdir()
    a.log = SimpleNamespace(warning=lambda *x: None)
    seen = []

    async def _llm(conv, model, role, key=None, extra=None, digest=None):
        seen.append(conv[-1]["content"])
        return replies.pop(0)
    a._llm = _llm
    monkeypatch.setattr(xt, "ExperimentTool", lambda *x, **y: FakeTool([PHASE_OUT, VARIANT_OUT]))
    return a, seen


def test_agent_multiturn_flag_off_writes_nothing_new(tmp_path, monkeypatch):
    a, seen = _bare_agent(tmp_path, monkeypatch, [R1, R2, FINAL])
    text = run(a._multiturn([{"role": "user", "content": "u"}], "m", "codegen", "r0s1", "gru", "tech"))
    assert text == FINAL
    assert sorted(p.name for p in a.out.iterdir()) == ["multiturn.jsonl"]    # no experiments.jsonl / experiments dir
    assert getattr(a, "_exp", None) is None and "experiments" not in json.dumps(a.mt_stats)
    row = json.loads((a.out / "multiturn.jsonl").read_text())
    assert set(row) == {"key", "slot", "technique", "turns", "runs", "refused", "compile_failed", "seconds",
                        "forced_final", "items"} and row["runs"] == 2
    assert set(row["items"][0]) == {"turn", "stage", "exit", "wall_s", "out_head"}


def test_agent_multiturn_records_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv(E.ENV_RECORD, "1")
    a, seen = _bare_agent(tmp_path, monkeypatch, [R1, R2, FINAL])
    run(a._multiturn([{"role": "user", "content": "u"}], "m", "codegen", "r0s1", "gru", "persistent_recurrence"))
    lines = E.read_jsonl(a.out / "experiments.jsonl")
    ex = [x for x in lines if x["type"] == "experiment"]
    assert len(ex) == 2 and ex[0]["op_class"] == "recurrent" and ex[0]["shape_bucket"]
    assert ex[0]["model"] == "m" and ex[0]["role"] == "codegen" and ex[0]["ridx"] == 0 and ex[0]["step"] == 1
    # multiturn.jsonl is exactly what it is with the flag off
    row = json.loads((a.out / "multiturn.jsonl").read_text())
    assert set(row) == {"key", "slot", "technique", "turns", "runs", "refused", "compile_failed", "seconds",
                        "forced_final", "items"}
    a._experiment_outcome(0, 1, "gru", "seed", "void kernel_x(){ register_tile_cols(); }", "improved", 1.1, 0.1,
                          None, 100.0, 90.0, "impl9", "oid9")
    out = [x for x in E.read_jsonl(a.out / "experiments.jsonl") if x["type"] == "outcome"][0]
    assert out["status"] == "improved" and out["impl_id"] == "impl9"
    assert a._exp.pointer()["records"] == 2


def test_agent_prompt_injection_only_when_asked(tmp_path, monkeypatch):
    k, kv = _store_dir(tmp_path, "k")
    E.ingest(_run_dir(tmp_path, "prev", "run0", "p0"), k, kv.claims)
    monkeypatch.setenv(E.ENV_RECORD, "1")
    a, seen = _bare_agent(tmp_path, monkeypatch, [FINAL])
    a.kv = SimpleNamespace(claims=kv.claims, root=k)
    run(a._multiturn([{"role": "user", "content": "u"}], "m", "codegen", "r0s0", "gru", "t"))
    base_len = len(seen[0])
    assert "Experiments already run" not in seen[0]                           # record on, prompt flag off
    monkeypatch.setenv(E.ENV_PROMPT, "3")
    a2, seen2 = _bare_agent(tmp_path / "x", monkeypatch, [FINAL]) if (tmp_path / "x").mkdir() is None else None
    a2.kv = SimpleNamespace(claims=kv.claims, root=k)
    run(a2._multiturn([{"role": "user", "content": "u"}], "m", "codegen", "r0s0", "gru", "t"))
    assert "Experiments already run" in seen2[0] and 0 < len(seen2[0]) - base_len <= 800 + 10
    step = [x for x in E.read_jsonl(a2.out / "experiments.jsonl") if x["type"] == "step"][0]
    assert 0 < step["prompt_extra_chars"] <= 800 and step["prompt_extra_ids"]


# ---------------------------------------------------------------- global store

@pytest.fixture()
def store(tmp_path):
    return GlobalStore(tmp_path / "store").ensure()


def _tables(store):
    return {r[0] for r in gk(store, "SELECT name FROM sqlite_master WHERE type='table'")}


def _run_with_experiments(store, tmp_path, name, problem, prog_phase=PROG_PHASE):
    kv, seed, kdir = start_run(store, tmp_path, name, problem=problem)
    outcome(kv, tech(kv), 0.2)
    d = _run_dir(tmp_path, name + "_out", "run_" + name, problem, prog_phase=prog_phase)
    E.ingest(d, kdir, kv.claims, gpu={"name": "NVIDIA L40S"})
    kv.claims.save()
    return kv, seed, kdir


def test_default_policy_writes_no_experiment_tables(store, tmp_path):
    kv, seed, kdir = _run_with_experiments(store, tmp_path, "a", "pa")
    merge(store, kdir, seed, "a")                                              # default Policy: no experiments
    assert not {"gk_experiments", "gk_probes"} & _tables(store)
    man = materialize(store, P89, tmp_path / "s2")
    assert "experiments" not in man and not (tmp_path / "s2" / "experiments").exists()
    ks = KnowledgeV2(tmp_path / "s2", seed_mined=False)
    assert not [c for c in ks.claims.claims.values() if c.kind == EXPERIMENT]  # claims reached the store, not the seed


def test_global_roundtrip_provenance_and_sharing(store, tmp_path):
    pol = Policy(experiments=True, seed_experiments=True)
    kv, seed, kdir = _run_with_experiments(store, tmp_path, "a", "pa")
    rep = merge(store, kdir, seed, "a", policy=pol)
    assert rep["experiments"]["new"] == 2 and rep["experiments"]["probes"] >= 2
    assert {"gk_experiments", "gk_probes"} <= _tables(store)
    rows = gk(store, "SELECT id, part, problem, run_uid, rec FROM gk_experiments")
    assert len(rows) == 2 and rows[0][1] == P89 and rows[0][2] == "a" and rows[0][3] == "uid-a"
    assert (store.root / "experiment_code" / f"{E.sha256(PROG_PHASE)}.cu").read_text() == PROG_PHASE
    cl = [json.loads(d) for (d,) in gk(store, "SELECT data FROM gk_claims WHERE data LIKE '%\"experiment\"%'")
          if json.loads(d)["kind"] == EXPERIMENT]
    ph = [c for c in cl if c["data"]["fact_type"] == "phase_profile"]
    assert len(ph) == 1 and ph[0]["support"] == 1 and ph[0]["data"]["problems"] == ["pa"]
    # a second run on another problem, seeded from the store, finds the same fact and adds its own evidence
    seed2 = tmp_path / "b_seed"                                                # a seed with the experiment policy
    materialize(store, P89, seed2, problem="pb", run_uid="uid-b", policy=pol)
    kdir2 = tmp_path / "b_k"
    shutil.copytree(seed2, kdir2)
    kv2 = KnowledgeV2(kdir2)
    assert E.read_jsonl(kdir2 / "experiments" / "seed_probes.jsonl")           # probes shipped by the seed
    seeded = [c for c in kv2.claims.claims.values() if c.kind == EXPERIMENT] if False else None
    outcome(kv2, tech(kv2), 0.1, problem="kb_problem__y__r00")
    d2 = _run_dir(tmp_path, "b_out", "run_b", "pb", prog_phase=PROG_PHASE.replace("256", "1024"))
    E.ingest(d2, kdir2, kv2.claims)
    kv2.claims.save()
    merge(store, kdir2, seed2, "b", policy=pol)
    cl = [json.loads(d) for (d,) in gk(store, "SELECT data FROM gk_claims")]
    ph = [c for c in cl if c.get("kind") == EXPERIMENT and c["data"]["fact_type"] == "phase_profile"]
    assert len(ph) == 1 and ph[0]["support"] == 2                                # one shared fact, two independent probes
    assert sorted(ph[0]["data"]["problems"]) == ["pa", "pb"] and ph[0]["data"]["n_obs"] == 2
    assert len(gk(store, "SELECT 1 FROM gk_experiments")) == 4
    # probes aggregate usage across both runs (one template, two shapes)
    probes = [json.loads(d) for (d,) in gk(store, "SELECT data FROM gk_probes")]
    pp = [p for p in probes if p["question"] == "phase_breakdown"]
    assert len(pp) == 1 and pp[0]["uses"] == 2 and pp[0]["next_improved"] == 2
    # re-merging the same run is a no-op (ledger duplicate); the table does not grow
    assert merge(store, kdir2, seed2, "b", policy=pol)["status"] == "duplicate"
    # a seed for a third run carries the fact and the probes
    man = materialize(store, P89, tmp_path / "s3", policy=pol, problem="pc")
    assert man["experiments"]["probes"] >= 2
    k3 = KnowledgeV2(tmp_path / "s3", seed_mined=False)
    sc = [c for c in k3.claims.claims.values() if c.kind == EXPERIMENT]
    assert sc and any(c.support == 2 for c in sc)
    t, ids = E.prompt_text(k3.claims, tmp_path / "s3", arch="sm_89", op_class="recurrent", n=3)
    assert "wait-dominated" in t and "+2/-0" in t
    # and the store-level report works on the store itself
    recs, info = E.load_records(store.root)
    assert info["kind"] == "store" and len(recs) == 4


def test_links_experiments_before_winner(store, tmp_path):
    from kernelblaster.knowledge.impl_links import ImplLinks
    kv, seed, kdir = _run_with_experiments(store, tmp_path, "a", "pa")
    merge(store, kdir, seed, "a", policy=Policy(experiments=True))
    lk = ImplLinks.from_db(kdir / "knowledge.db")
    L = lk.links("impl_run_a")
    assert L["experiments"] and L["experiments"][0]["probe_kind"] in ("phase_breakdown", "variant_timing")
    assert "preceded by a" in lk.render(["impl_run_a"], max_chars=2000)
    snap = store.path(store.current_id())
    lk2 = ImplLinks.from_db(snap, part=P89)
    assert lk2.exp is not None and lk2.links("impl_run_a")["experiments"]


def test_consolidation_folds_paraphrased_experiment_facts(store, tmp_path):
    """Two runs word the same finding differently (e.g. an older ingest version): A5's consolidation folds them
    (same kind / scope / fact type / subject), unions the observations, and a second pass changes nothing."""
    pol = Policy(experiments=True)
    kv, seed, kdir = _run_with_experiments(store, tmp_path, "a", "pa")
    merge(store, kdir, seed, "a", policy=pol)
    kv2, seed2, kdir2 = _run_with_experiments(store, tmp_path, "b", "pb",
                                              prog_phase=PROG_PHASE.replace("256", "1024"))
    for c in kv2.claims.claims.values():                                       # another wording of the same fact
        if c.kind == EXPERIMENT and c.data.get("fact_type") == "phase_profile":
            c.text = "the wait phase dominates each recurrence step of recurrent kernels (wait, mma, gate)"
            c.id = "cx_paraphrase"
    kv2.claims.claims = {c.id: c for c in kv2.claims.claims.values()}
    kv2.claims.save()
    merge(store, kdir2, seed2, "b", policy=pol)

    def active():
        return [json.loads(d) for (d,) in gk(store, "SELECT data FROM gk_claims")
                if json.loads(d).get("kind") == EXPERIMENT and json.loads(d)["status"] == "active"
                and json.loads(d)["data"]["fact_type"] == "phase_profile"]
    assert len(active()) == 2
    r1 = consolidate_store(store, threshold=0.2)
    ph = active()
    assert len(ph) == 1 and ph[0]["support"] == 2 and ph[0]["data"]["n_obs"] == 2
    assert sorted(ph[0]["data"]["problems"]) == ["pa", "pb"] and ph[0]["data"]["consolidated_from"]
    r2 = consolidate_store(store, threshold=0.2)
    assert r2.get("status") == "no_change" or r2.get("merged", 0) == 0, r2


def test_old_store_compat(store, tmp_path):
    """A store written without the flag (and a snapshot from before) loads, seeds and links as before; an old reader
    skips experiment claims it does not know."""
    kv, seed, kdir = start_run(store, tmp_path, "old", problem="po")
    t = tech(kv)
    ids = [outcome(kv, t, 0.2) for _ in range(2)]
    c, _ = kv.claims.add(Claim(kind="applicability", text=f"{t} helps", arch="sm_89", op_class="gemm", technique=t,
                               data={"effect": "helps"}), ids)
    kv.claims.save()
    merge(store, kdir, seed, "old")
    # flag on later against the old store: migration is just the next merge
    kv2, seed2, kdir2 = _run_with_experiments(store, tmp_path, "new", "pn")
    merge(store, kdir2, seed2, "new", policy=Policy(experiments=True))
    assert {"gk_experiments", "gk_probes"} <= _tables(store)
    man = materialize(store, P89, tmp_path / "s9", policy=Policy())          # flag-off seed from a flagged store
    k9 = KnowledgeV2(tmp_path / "s9", seed_mined=False)
    assert c.id in k9.claims.claims and not [x for x in k9.claims.claims.values() if x.kind == EXPERIMENT]
    from kernelblaster.knowledge.impl_links import ImplLinks
    assert ImplLinks.from_db(tmp_path / "s9" / "knowledge.db").exp is None   # nothing recorded there
    # an experiment claim row read by code that only knows the old kinds is skipped, not fatal
    row = {"kind": "experiment", "text": "x", "id": "cx_1"}
    import kernelblaster.knowledge.claims as cm
    old = cm.VALID_KINDS
    try:
        cm.VALID_KINDS = cm.CLAIM_KINDS
        with pytest.raises(ValueError):
            Claim.from_dict(row)
    finally:
        cm.VALID_KINDS = old


# ---------------------------------------------------------------- legacy import and CLI

LEGACY = Path(__file__).resolve().parents[3].parent / "l3_farm" / "analysis" / "capabilities" / "results" / "5299224" / \
    "ab5" / "M_rec" / "a2_41_gru_rev_l1_5"


def _legacy_dir(tmp_path):
    if LEGACY.exists():
        d = tmp_path / "legacy"
        d.mkdir()
        for f in ("multiturn.jsonl", "result.json"):
            shutil.copyfile(LEGACY / f, d / f)
        return d
    d = tmp_path / "legacy"
    d.mkdir()
    (d / "multiturn.jsonl").write_text(json.dumps({
        "key": "r0s1", "slot": "gru_seq_f16__v1", "technique": "t", "turns": 2, "runs": 1, "items": [
            {"turn": 0, "stage": "run", "exit": 0, "wall_s": 1.0, "out_head": PHASE_OUT}]}) + "\n")
    (d / "result.json").write_text(json.dumps({"arch": "sm_89", "run_id": "x", "problem": "p",
                                               "steps": [{"rollout": 0, "step": 1, "status": "improved",
                                                          "slot_us_before": 10.0, "slot_us_after": 5.0}]}))
    return d


def test_import_legacy_marks_missing_and_joins_outcomes(tmp_path):
    d = _legacy_dir(tmp_path)
    recs = E.import_legacy(d)
    assert recs and all(r["legacy"] and "program" in r["missing"] and r["purpose"] is None for r in recs)
    assert all(r["arch"] == "sm_89" and r["op_class"] == "recurrent" for r in recs)
    assert any(r["outcome"] and r["outcome"]["status"] == "improved" for r in recs)
    assert recs == E.import_legacy(d)                                         # deterministic
    facts = E.facts_from_records(recs)
    assert any(f["fact_type"] == "phase_profile" and f["subject"] == "wait" for f in facts.values())
    rep = E.report(recs)
    assert rep["missing_fields"]["program"] == len(recs) and rep["probes"]


def test_cli_report_ingest_show(tmp_path, capsys):
    d = _legacy_dir(tmp_path)
    assert E.main(["report", str(d)]) == 0
    assert "phase_profile" in capsys.readouterr().out
    k = tmp_path / "kd"
    assert E.main(["import-legacy", str(d), "--kdir", str(k)]) == 0
    assert (k / "experiments" / "records.jsonl").exists()
    kv = KnowledgeV2(k, seed_mined=False)
    assert [c for c in kv.claims.claims.values() if c.kind == EXPERIMENT]
    # run-dir ingest + show
    rd = _run_dir(tmp_path, "run", "run1", "p1")
    k2 = tmp_path / "k2"
    assert E.main(["ingest", str(rd), "--kdir", str(k2)]) == 0
    rid = E.read_jsonl(k2 / "experiments" / "records.jsonl")[0]["id"]
    capsys.readouterr()
    assert E.main(["show", str(k2), rid[:8]]) == 0
    assert "#define H 256" in capsys.readouterr().out
    assert E.main(["report", str(k2), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["experiments"] == 2
