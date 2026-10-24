import asyncio
import json

import pytest

from kernelblaster.agents.utils import agentic_llm
from kernelblaster.knowledge.claims import Claim, ClaimStore
from kernelblaster.knowledge.outcomes import OutcomeLog, OutcomeRecord
from kernelblaster.knowledge.reflect import (
    EditError,
    SemanticUpdater,
    parse_json_response,
    validate_edit,
)


@pytest.fixture
def env(tmp_path):
    log = OutcomeLog(tmp_path / "o.jsonl")
    store = ClaimStore(tmp_path / "c.json")
    R = {}

    def add(key, **kw):
        base = dict(target="cuda", arch="sm_89", op_class="gemm", problem_id="p1",
                    state_before="compute_throughput_saturated")
        base.update(kw)
        R[key] = log.append(OutcomeRecord(**base))
        return R[key]

    add("tc1", technique="tensor_core_utilization", status="improved", speedup=1.8, impl_id="i1")
    add("tc2", technique="tensor_core_utilization", status="improved", speedup=1.5)
    add("tc3", technique="tensor_core_utilization", status="compile_error",
        error="error: identifier 'wmma' is undefined")
    add("ti1", technique="shared_memory_tiling", status="regressed", speedup=0.8)
    add("rv1", technique="vectorize", status="improved", speedup=1.3, target="riscv", arch="rvv-vlen256",
        op_class="elementwise", problem_id="p2")
    index = {r.id: r for r in log}
    return log, store, R, index


def V(edit, store, index, **kw):
    return validate_edit(edit, index=index, claims=store, **kw)


def test_valid_applicability_inherits_scope(env):
    log, store, R, index = env
    e = V({"kind": "applicability", "text": "wmma helps fp16 gemm when K%16==0",
           "evidence": [R["tc1"].id, R["tc2"].id], "data": {"effect": "helps"}}, store, index)
    sc = e["scope"]
    assert sc["target"] == "cuda" and sc["arch"] == "sm_89" and sc["op_class"] == "gemm"
    assert sc["technique"] == "tensor_core_utilization" and sc["state"] == "compute_throughput_saturated"
    assert sc["problem_id"] is None            # general lesson, not problem-local
    e2 = V({"kind": "applicability", "text": "x", "evidence": [R["tc1"].id], "data": {"effect": "helps"},
            "scope": {"arch": "*"}}, store, index)
    assert e2["scope"]["arch"] is None


@pytest.mark.parametrize("edit,msg", [
    ("not a dict", "not an object"),
    ({"kind": "rumour", "text": "x", "evidence": ["?"]}, "unknown edit kind"),
    ({"kind": "applicability", "text": "x", "data": {"effect": "helps"}}, "evidence is empty"),
    ({"kind": "applicability", "text": "x", "evidence": ["deadbeef0000"], "data": {"effect": "helps"}},
     "not in the outcome log"),
    ({"kind": "applicability", "text": "x", "evidence": "tc1", "data": {"effect": "helps"}}, "list of record ids"),
    ({"kind": "applicability", "text": "", "evidence": ["@tc1"], "data": {"effect": "helps"}}, "missing text"),
    ({"kind": "applicability", "text": "x", "evidence": ["@tc1"], "data": {"effect": "great"}}, "effect must be"),
    ({"kind": "applicability", "text": "x", "evidence": ["@ti1"], "data": {"effect": "helps"}}, "no improved"),
    ({"kind": "applicability", "text": "x", "evidence": ["@tc1", "@rv1"], "data": {"effect": "helps"}},
     "several targets"),
    ({"kind": "applicability", "text": "x", "evidence": ["@tc1"], "data": {"effect": "helps"},
      "scope": {"target": "riscv"}}, "target"),
    ({"kind": "applicability", "text": "x", "evidence": ["@tc1"], "data": {"effect": "helps"},
      "scope": {"arch": "sm_90"}}, "scope.arch"),
    ({"kind": "applicability", "text": "x", "evidence": ["@tc1"], "data": {"effect": "helps"},
      "scope": {"op_class": "quantum"}}, "scope.op_class"),
    ({"kind": "pitfall", "text": "x", "evidence": ["@tc1"], "data": {"symptom": "s"}}, "no compile"),
    ({"kind": "pitfall", "text": "x", "evidence": ["@tc3"], "data": {}}, "data.symptom"),
    ({"kind": "interaction", "text": "x", "evidence": ["@tc1"], "data": {"other": "teleport", "relation": "undoes"}},
     "not a known technique"),
    ({"kind": "parameter", "text": "x", "evidence": ["@tc1"], "data": {"params": {}}}, "params"),
    ({"kind": "parameter", "text": "x", "evidence": ["@tc1"], "data": {"params": {"a": [1]}}}, "params"),
    ({"kind": "state_split", "text": "x", "evidence": ["@tc1"], "data": {"proposed_state": "a_b", "criterion": "c"}},
     ">= 2 records"),
    ({"kind": "state_split", "text": "x", "evidence": ["@tc1", "@tc3"], "data": {"proposed_state": "Bad Name",
                                                                               "criterion": "c"}}, "snake_case"),
    ({"kind": "impl_note", "text": "x", "evidence": ["@tc1"], "data": {"note_type": "gossip"}}, "note_type"),
    ({"kind": "impl_note", "text": "x", "evidence": ["@tc1"], "data": {"note_type": "why_fast"},
      "scope": {"impl_id": "nope"}}, "impl_id"),
    ({"kind": "support", "claim_id": "c_missing", "evidence": ["@tc1"]}, "unknown claim"),
])
def test_rejections(env, edit, msg):
    log, store, R, index = env

    def sub(v):
        if isinstance(v, str) and v.startswith("@"):
            return R[v[1:]].id
        if isinstance(v, list):
            return [sub(x) for x in v]
        if isinstance(v, dict):
            return {k: sub(x) for k, x in v.items()}
        return v
    with pytest.raises(EditError, match=msg):
        V(sub(edit), store, index)


def test_support_contradict_and_cross_target(env):
    log, store, R, index = env
    c, _ = store.add(Claim(kind="applicability", text="vectorize helps", target="riscv",
                           technique="vectorize", data={"effect": "helps"}), [R["rv1"].id])
    with pytest.raises(EditError, match="claim target"):
        V({"kind": "contradict", "claim_id": c.id, "evidence": [R["ti1"].id]}, store, index)
    e = V({"kind": "support", "claim_id": c.id, "evidence": [R["rv1"].id]}, store, index)
    assert e == {"kind": "support", "claim_id": c.id, "evidence": [R["rv1"].id]}


def test_impl_note_scoping(env):
    log, store, R, index = env
    e = V({"kind": "impl_note", "text": "wmma fragments with 2-stage cp.async are why i1 is fast",
           "evidence": [R["tc1"].id], "data": {"note_type": "why_fast"}}, store, index)
    assert e["scope"]["impl_id"] == "i1" and e["scope"]["problem_id"] == "p1"
    e = V({"kind": "impl_note", "text": "smem tiling regressed on p1", "evidence": [R["ti1"].id],
           "data": {"note_type": "dead_end"}}, store, index)
    assert e["scope"]["problem_id"] == "p1" and e["scope"]["impl_id"] is None


def test_known_technique_whitelist(env):
    log, store, R, index = env
    with pytest.raises(EditError, match="scope.technique"):
        V({"kind": "applicability", "text": "x", "evidence": [R["tc1"].id], "data": {"effect": "helps"},
           "scope": {"technique": "made_up"}}, store, index)
    e = V({"kind": "applicability", "text": "x", "evidence": [R["tc1"].id], "data": {"effect": "helps"},
           "scope": {"technique": "made_up"}}, store, index, known_techniques={"made_up"})
    assert e["scope"]["technique"] == "made_up"


def test_parse_json_response():
    assert parse_json_response('```json\n{"edits": []}\n```') == {"edits": []}
    assert parse_json_response('Sure! {"edits": [1]} hope this helps') == {"edits": [1]}
    with pytest.raises(ValueError):
        parse_json_response("no json here")


def _stub(name, fn):
    agentic_llm.register_stub(name, fn)


def test_reflect_applies_valid_and_rejects_bad(env):
    log, store, R, index = env
    seen = []

    def fn(msgs):
        seen.append(msgs)
        return json.dumps({"edits": [
            {"kind": "applicability", "text": "wmma helps fp16 gemm with K%16==0",
             "evidence": [R["tc1"].id, R["tc2"].id], "data": {"effect": "helps"}},
            {"kind": "pitfall", "text": "wmma needs <mma.h> and nvcuda namespace",
             "evidence": [R["tc3"].id], "data": {"symptom": "identifier wmma undefined",
                                                 "fix": "#include <mma.h>; using namespace nvcuda;"}},
            {"kind": "applicability", "text": "hallucinated", "evidence": ["ffffffffffff"],
             "data": {"effect": "helps"}},
            {"kind": "support", "claim_id": "c_nope", "evidence": [R["tc1"].id]},
        ]})
    _stub("reflect_ok", fn)
    try:
        up = SemanticUpdater(log, store, model="stub:reflect_ok", every_n=3)
        applied = asyncio.run(up.reflect([R["tc1"], R["tc2"], R["tc3"]]))
    finally:
        agentic_llm.unregister_stub("reflect_ok")
    assert [a["result"] for a in applied] == ["added", "added"]
    assert len(up.last_rejected) == 2
    user = seen[0][-1]["content"]
    assert R["tc1"].id in user and "OUTCOME RECORDS (3)" in user
    assert store.meta["reflections"] == 1 and store.epoch == 1
    # persisted, and the eligible claim is retrievable for prompts
    s2 = ClaimStore(store.path)
    rel = s2.relevant(target="cuda", arch="sm_89", op_class="gemm", technique="tensor_core_utilization")
    assert [c.kind for c in rel] == ["applicability"]       # the pitfall has 1 support < 2
    assert "FIX:" in s2.render(s2.relevant(target="cuda", kinds=["pitfall"], include_ineligible=True))


def test_reflect_garbage_response_is_harmless(env):
    log, store, R, index = env
    _stub("garbage", lambda m: "I think tiling is great!!")
    try:
        up = SemanticUpdater(log, store, model="stub:garbage")
        assert asyncio.run(up.reflect([R["tc1"]])) == []
    finally:
        agentic_llm.unregister_stub("garbage")
    assert store.claims == {}


def test_existing_claims_shown_and_supported(env):
    log, store, R, index = env
    c, _ = store.add(Claim(kind="applicability", text="wmma helps gemm", target="cuda", arch="sm_89",
                           op_class="gemm", technique="tensor_core_utilization",
                           data={"effect": "helps"}), [R["tc1"].id])
    seen = []

    def fn(msgs):
        seen.append(msgs[-1]["content"])
        return json.dumps({"edits": [{"kind": "support", "claim_id": c.id, "evidence": [R["tc2"].id]}]})
    _stub("sup", fn)
    try:
        applied = asyncio.run(SemanticUpdater(log, store, model="stub:sup").reflect([R["tc2"]]))
    finally:
        agentic_llm.unregister_stub("sup")
    assert c.id in seen[0]                       # weak claims are shown so they can grow
    assert applied[0]["support"] == 2 and store.eligible(store.get(c.id))


def test_maybe_reflect_cadence(tmp_path):
    log = OutcomeLog(tmp_path / "o.jsonl")
    store = ClaimStore(tmp_path / "c.json")
    calls = []
    _stub("count", lambda m: (calls.append(1), '{"edits": []}')[1])
    try:
        up = SemanticUpdater(log, store, model="stub:count", every_n=4)
        for i in range(3):
            log.append(OutcomeRecord(technique="t", status="neutral"))
        assert asyncio.run(up.maybe_reflect()) == [] and calls == []
        log.append(OutcomeRecord(technique="t", status="neutral"))
        asyncio.run(up.maybe_reflect())
        assert len(calls) == 1 and store.meta["reflect_cursor"] == 4
        asyncio.run(up.maybe_reflect())          # nothing new
        assert len(calls) == 1
        log.append(OutcomeRecord(technique="t", status="neutral"))
        asyncio.run(up.maybe_reflect())          # 1 < every_n
        assert len(calls) == 1
        asyncio.run(up.maybe_reflect(force=True))
        assert len(calls) == 2
        # cursor survives a restart
        assert ClaimStore(store.path).meta["reflect_cursor"] == 5
    finally:
        agentic_llm.unregister_stub("count")


def test_maybe_reflect_concurrent_calls_run_once(tmp_path):
    log = OutcomeLog(tmp_path / "o.jsonl")
    store = ClaimStore(tmp_path / "c.json")
    calls = []

    async def slow(m):
        calls.append(1)
        await asyncio.sleep(0.05)
        return '{"edits": []}'
    _stub("slow", slow)
    try:
        for _ in range(8):
            log.append(OutcomeRecord(technique="t", status="neutral"))
        up = SemanticUpdater(log, store, model="stub:slow", every_n=8)

        async def go():
            await asyncio.gather(*[up.maybe_reflect() for _ in range(5)])
        asyncio.run(go())
    finally:
        agentic_llm.unregister_stub("slow")
    assert len(calls) == 1


def test_max_batch_bounds_prompt(tmp_path):
    log = OutcomeLog(tmp_path / "o.jsonl")
    store = ClaimStore(tmp_path / "c.json")
    seen = []
    _stub("cap", lambda m: (seen.append(m[-1]["content"]), '{"edits": []}')[1])
    try:
        for _ in range(30):
            log.append(OutcomeRecord(technique="t", status="neutral"))
        up = SemanticUpdater(log, store, model="stub:cap", every_n=8, max_batch=10)
        asyncio.run(up.maybe_reflect())
    finally:
        agentic_llm.unregister_stub("cap")
    assert "OUTCOME RECORDS (10)" in seen[0] and store.meta["reflect_cursor"] == 30


def test_consolidate_merges_only_compatible(env):
    log, store, R, index = env
    mk = lambda text, eff="helps", ids=(): store.claims.setdefault(
        (c := Claim(kind="applicability", text=text, target="cuda", technique="tensor_core_utilization",
                    data={"effect": eff})).id, c) and c
    a = mk("wmma is good for gemm"); a.add_support([R["tc1"].id, R["tc2"].id])
    b = mk("tensor cores speed up matrix multiply"); b.add_support([R["tc2"].id])
    h = mk("wmma hurts", eff="hurts"); h.add_support([R["ti1"].id])
    for i in range(4):
        mk(f"filler claim number {i} about something else entirely {i}")

    _stub("cons", lambda m: json.dumps({"merges": [
        {"keep": a.id, "absorb": [b.id], "text": "wmma / tensor cores speed up gemm"},
        {"keep": a.id, "absorb": [h.id]},           # polarity differs -> rejected
    ]}))
    try:
        res = asyncio.run(SemanticUpdater(log, store, model="stub:cons").consolidate())
    finally:
        agentic_llm.unregister_stub("cons")
    assert [r["result"] for r in res] == ["merged", "rejected"]
    assert a.text == "wmma / tensor cores speed up gemm" and a.support == 2
    assert b.status == "retired" and h.status == "active"
