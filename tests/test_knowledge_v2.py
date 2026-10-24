"""Knowledge architecture v2: DSL, classifier, adapters, ontology, mechanism check,
shrinkage, SQLite store (incl. concurrency), claim levels + lift / mechanism edits,
hypothesis lifecycle, state discovery validation, upkeep with a stub LLM, the legacy
adapter path and a scripted end-to-end multi-slot run."""
import asyncio
import json
import multiprocessing as mp
import random
import shutil
from pathlib import Path

import pytest

from kernelblaster.agents.utils import agentic_llm
from kernelblaster.knowledge.claims import APPLICABILITY, Claim
from kernelblaster.knowledge.outcomes import OutcomeRecord
from kernelblaster.knowledge.reflect import EditError
from kernelblaster.knowledge.v2 import Choice, KnowledgeV2
from kernelblaster.knowledge.v2 import discovery as D
from kernelblaster.knowledge.v2 import metrics as M
from kernelblaster.knowledge.v2 import predicates as P
from kernelblaster.knowledge.v2.hypotheses import Hypothesis, validate_spec
from kernelblaster.knowledge.v2.ontology import (CONFIRMED, CONTRADICTED, MECH_UNKNOWN, Ontology,
                                                 mechanism_check, seed_legacy)
from kernelblaster.knowledge.v2.reflect_v2 import SemanticUpdaterV2, validate_lift, validate_mechanism
from kernelblaster.knowledge.v2.states import State, StateHierarchy
from kernelblaster.knowledge.v2.stats import HierStats, path as stat_path
from kernelblaster.knowledge.v2.store import KnowledgeDB, SQLiteClaimStore, SQLiteOutcomeLog
from kernelblaster.knowledge.v2.targets import TargetRegistry


def m(v, src="measured"):
    return {"v": v, "src": src, "from": "test"}


def vec(**kw):
    return {k: m(v) for k, v in kw.items()}


REG = TargetRegistry()


# ---------------------------------------------------------------- DSL

def test_predicate_validation_and_three_valued_logic():
    p = {"all": [{"bw_util": ">=", "v": 50}, {"compute_util": "<", "ref": "bw_util"}]}
    P.validate(p)
    assert P.evaluate(p, {"bw_util": 60, "compute_util": 10}) is True
    assert P.evaluate(p, {"bw_util": 40, "compute_util": 10}) is False
    assert P.evaluate(p, {"compute_util": 10}) is None                 # absent metric -> unknown
    assert P.evaluate({"all": [{"bw_util": "<", "v": 1}, {"l1_hit": ">", "v": 1}]}, {"bw_util": 5}) is False
    assert P.evaluate({"any": [{"bw_util": "<", "v": 1}, {"l1_hit": ">", "v": 1}]}, {"bw_util": 5}) is None
    assert P.evaluate({"not": {"l1_hit": ">", "v": 1}}, {}) is None
    assert P.evaluate({"op_class": "in", "v": ["gemm"]}, {}, {"op_class": "gemm"}) is True
    assert P.metrics_used(p) == {"bw_util", "compute_util"}
    for bad in ({"bogus": ">", "v": 1}, {"bw_util": "~", "v": 1}, {"all": []}, {"bw_util": ">"},
                {"op_class": "in", "v": "gemm"}, {"bw_util": ">", "v": 1, "l1_hit": "<"}):
        with pytest.raises(P.PredicateError):
            P.validate(bad)


# ---------------------------------------------------------------- classifier

def test_seed_classification_chains():
    h = StateHierarchy.seed(REG)
    c = h.classify(vec(mem_util=80, bw_util=75, compute_util=10, l2_hit=20, parallel_fill=1.0,
                       launch_frac=0.0), target="cuda:sm_89", op_class="elementwise")
    assert c.chain == ["memory_bound", "dram_streaming"] and c.status == "classified"
    assert c.decided_by["bw_util"] == 75
    # child metrics absent -> stays at the parent
    c = h.classify(vec(mem_util=80, compute_util=10, parallel_fill=1, launch_frac=0), target="cuda:sm_89")
    assert c.chain == ["memory_bound"]
    c = h.classify(vec(mem_util=10, compute_util=20, parallel_fill=1.0, launch_frac=0.1, occupancy=12),
                   target="cuda:sm_75")
    assert c.chain == ["underutilized", "underutilized_compute", "stalled_low_occupancy"]
    assert h.classify(vec(launch_frac=0.9, mem_util=5, compute_util=5), target="cuda:sm_75").chain == \
        ["latency_bound", "launch_overhead"]
    assert h.classify({}, target="cuda:sm_75").status == "unknown"
    assert h.map_legacy("memory_bank_conflicts") == "on_chip_memory_bound"
    assert h.ancestors("stalled_low_occupancy") == ["stalled_low_occupancy", "underutilized_compute", "underutilized"]


def test_unexplained_and_target_level_states():
    h = StateHierarchy([State("hot", None, {"bw_util": ">", "v": 90})], REG)
    assert h.classify(vec(bw_util=10), target="cuda:sm_75").status == "unexplained"
    h.add(State("adreno_tex", "hot", {"l1_hit": "<", "v": 50}, level="target", scope="opencl:adreno_740"))
    v = vec(bw_util=95, l1_hit=10)
    assert h.classify(v, target="cuda:sm_75").chain == ["hot"]
    assert h.classify(v, target="opencl:adreno_740").chain == ["hot", "adreno_tex"]


# ---------------------------------------------------------------- adapters

def test_adapters_ncu_summary_text_roofline():
    t = REG.get("cuda", "89")
    s = {"dram_throughput": 70.0, "compute_throughput": 20.0, "memory_throughput": 80.0, "achieved_occupancy": 45.0,
         "l1_hit_rate": 30.0, "l2_hit_rate": 60.0, "registers_per_thread_max": 128, "waves_per_sm_heaviest": 0.5,
         "duration_us": 90.0}
    v = M.adapter_for(t).vector(profile=s, estimate=M.RooflineInput(bytes=1e8, flops=1e8, time_us=100.0,
                                                                     work_items=1e6))
    assert v["bw_util"]["src"] == "measured" and v["bw_util"]["v"] == 70.0       # measured beats estimate
    assert v["arith_intensity"]["src"] == "estimated" and v["arith_intensity"]["v"] == 1.0
    assert v["parallel_fill"]["v"] == 0.5 and round(v["reg_pressure"]["v"], 1) == 50.2
    assert round(v["launch_frac"]["v"], 2) == 0.10                               # 1 - 90/100
    text = ("    DRAM Throughput                %        63.45\n    Memory Throughput   %   71.2\n"
            "    Compute (SM) Throughput        %        12.10\n    Achieved Occupancy  %  33.3\n"
            "    L2 Hit Rate    %   48.0\n    Registers Per Thread   register/thread   40\n")
    tv = M.adapter_for(t, "ncu_text").from_profile({"log": text})
    assert tv["bw_util"]["v"] == 63.45 and tv["mem_util"]["v"] == 71.2 and tv["compute_util"]["v"] == 12.1
    assert tv["occupancy"]["v"] == 33.3 and tv["l2_hit"]["v"] == 48.0 and "l1_hit" not in tv
    est = M.roofline_vector(REG.get("cuda:sm_75"), M.RooflineInput(bytes=672e3, flops=0, time_us=1.0))
    assert est["bw_util"]["v"] == 100.0 and all(x["src"] == "estimated" for x in est.values())
    # OpenCL / RVV stubs share the interface
    assert M.adapter_for(REG.get("opencl:adreno_740")).from_profile({"alu_util": 40})["compute_util"]["v"] == 40
    assert "compute_util" in M.adapter_for(REG.get("rvv:vlen256")).from_profile({"vector_util": 10})
    assert M.canonical_ncu_delta({"dram_throughput": -3.0, "registers_per_thread_max": 51}) == \
        {"bw_util": -3.0, "reg_pressure": 20.0}


# ---------------------------------------------------------------- ontology + mechanism check

def test_ontology_seed_aliases_realizations_candidates():
    o = Ontology.seed(REG, legacy=False)
    r = seed_legacy(o)
    assert r["added"] >= 10 and r["folded"] >= 40
    assert o.resolve("coalesced_access") == "memory_coalescing"
    assert o.resolve("local_memory_tiling") == "shared_memory_tiling"
    assert o.resolve("cuda_graphs") == "launch_batching" and o.get("transfer_optimization").slot is False
    t = o.get("shared_memory_tiling")
    assert "cp.async" in o.realization(t, "cuda:sm_89")["how"]
    assert "__shared__" in o.realization(t, "cuda:sm_75")["how"]
    assert "__local" in o.realization(t, "opencl:adreno_740")["how"]
    assert "VLEN" in o.realization(t, "rvv:vlen256")["how"]
    assert o.realization(o.get("warp_shuffle_reduction"), "rvv:vlen256") is None
    ctx = {"op_class": "gemm", "dtype": "fp32"}
    c1 = {t.id for t in o.candidates(target_key="cuda:sm_89", op_class="gemm", chain=["compute_bound", "matrix_compute"],
                                     values={"arith_intensity": 20}, ctx=ctx)}
    c2 = {t.id for t in o.candidates(target_key="cuda:sm_89", op_class="gemm", chain=["memory_bound"],
                                     values={"arith_intensity": 20}, ctx=ctx)}
    assert "tensor_core_mma" in c1 and "tensor_core_mma" not in c2          # state-gated
    assert "half_precision" not in c1                                          # fp16-only precondition
    c3 = {t.id for t in o.candidates(target_key="cuda:sm_89", op_class="gemm", chain=["memory_bound"],
                                     values={"arith_intensity": 0.2}, ctx=ctx)}
    assert "shared_memory_tiling" not in c3                                    # no reuse to tile


def test_mechanism_check():
    pred = {"bw_util": "down", "l1_hit": "up", "occupancy": "down_or_flat"}
    b = vec(bw_util=80, l1_hit=20, occupancy=50)
    assert mechanism_check(pred, b, vec(bw_util=40, l1_hit=70, occupancy=50))["label"] == CONFIRMED
    assert mechanism_check(pred, b, vec(bw_util=95, l1_hit=10, occupancy=60))["label"] == CONTRADICTED
    est = {k: {"v": x["v"], "src": "estimated"} for k, x in b.items()}
    assert mechanism_check(pred, est, vec(bw_util=40, l1_hit=70))["label"] == MECH_UNKNOWN   # estimates don't count
    assert mechanism_check(pred, b, vec(bw_util=40, l1_hit=10, occupancy=50))["label"] == CONFIRMED  # 2:1


# ---------------------------------------------------------------- shrinkage

def test_hierarchical_shrinkage():
    s = HierStats(REG, kappa=3.0)
    p75 = stat_path(REG, target_key="cuda:sm_75", op_class="conv", chain=["underutilized", "underutilized_compute"])
    p89 = stat_path(REG, target_key="cuda:sm_89", op_class="conv", chain=["underutilized", "underutilized_compute"])
    for _ in range(6):
        s.update("tiling", p75, 0.4)
    assert s.count("tiling", p89[0]) == 0 and s.count("tiling", p89[1]) == 6     # shares the family node
    e89 = s.estimate("tiling", p89)
    assert 0.2 < e89 < 0.4                                     # starts from the family / universal evidence
    assert s.evidence_level("tiling", p89)[0].startswith("simt_gpu family")
    for _ in range(12):
        s.update("tiling", p89, -0.2)
    assert s.estimate("tiling", p89) < 0 < s.estimate("tiling", p75)          # own data takes over
    # exact formula on a two-node path
    s2 = HierStats(REG, kappa=2.0)
    nodes = [("target", "cuda:sm_75", "gemm", "x"), ("universal", "", "*", "*")]
    s2.update("a", nodes[1:], 1.0)                                             # parent: n=1, mean=1
    s2.update("a", nodes, 0.0)                                                 # both: child n=1 mean 0
    parent = (2 * 0.5 + 2 * 0.0) / 4                                            # n=2 mean 0.5 -> 0.25
    assert abs(s2.estimate("a", nodes) - (0.0 + 2 * parent) / 3) < 1e-9


# ---------------------------------------------------------------- SQLite store

def _rec(i, tech="tiling", status="improved", arch="sm_75", speedup=1.5, pid="p1", **kw):
    return OutcomeRecord(id=f"r{i}", technique=tech, status=status, target="cuda", arch=arch, problem_id=pid,
                         slot="s", op_class="conv", speedup=speedup, metric_before=10.0, **kw)


def test_outcome_log_and_claim_store_roundtrip(tmp_path):
    db = KnowledgeDB(tmp_path / "k.db")
    log = SQLiteOutcomeLog(db)
    for i in range(3):
        log.append(_rec(i), {"reward": 0.1 * i, "chain": ["underutilized"]})
    assert [r.id for r in log] == ["r0", "r1", "r2"] and len(log) == 3
    new, cur = log.since(1)
    assert [r.id for r in new] == ["r1", "r2"] and cur == 3
    assert log.rows("id=?", ("r2",))[0]["v2"]["reward"] == 0.2
    cs = SQLiteClaimStore(db, registry=REG)
    c, _ = cs.add(Claim(kind=APPLICABILITY, text="tiling helps conv", technique="tiling", data={"effect": "helps"},
                        arch="sm_75"), ["r0", "r1"])
    cs.save()
    cs2 = SQLiteClaimStore(db, registry=REG)
    assert cs2.get(c.id).support == 2
    assert "tiling: 3 tries" in cs2.already_tried("p1", "s")


def _proc_support(path, cid, ids, q):
    db = KnowledgeDB(Path(path))
    cs = SQLiteClaimStore(db)
    for i in ids:
        cs.support(cid, [i])
        cs.save()
    db.add_stats([("t", "target", "x", "conv", "s")] * 1, 1.0)
    for _ in range(19):
        db.add_stats([("t", "target", "x", "conv", "s")], 1.0)
    q.put("ok")


def test_sqlite_concurrent_writers_union_evidence(tmp_path):
    db = KnowledgeDB(tmp_path / "k.db")
    cs = SQLiteClaimStore(db)
    c, _ = cs.add(Claim(kind=APPLICABILITY, text="x helps", technique="t", data={"effect": "helps"}), ["base"])
    cs.save()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_proc_support, args=(str(tmp_path / "k.db"), c.id, [f"a{w}_{i}" for i in range(10)], q))
          for w in range(3)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(120)
    assert [q.get(timeout=5) for _ in ps] == ["ok"] * 3
    final = SQLiteClaimStore(KnowledgeDB(tmp_path / "k.db")).get(c.id)
    assert final.support == 31 and len(set(final.support_ids)) == 31        # nobody's evidence lost
    assert db.all_stats()[("t", "target", "x", "conv", "s")][0] == 60


# ---------------------------------------------------------------- claim levels, lift, mechanism edits

def _index(*recs):
    return {r.id: r for r in recs}


def test_claim_levels_retrieval_and_lift(tmp_path):
    db = KnowledgeDB(tmp_path / "k.db")
    h = StateHierarchy.seed(REG)
    cs = SQLiteClaimStore(db, registry=REG, hierarchy=h)
    r75 = [_rec(i, arch="sm_75", pid=f"p{i}") for i in range(2)]
    r89 = [_rec(10 + i, arch="sm_89", pid=f"q{i}") for i in range(2)]
    idx = _index(*r75, *r89)
    c, _ = cs.add(Claim(kind=APPLICABILITY, text="tiling helps conv", technique="tiling", op_class="conv",
                        arch="sm_75", state="underutilized", data={"effect": "helps"}), [r.id for r in r75])
    q89 = dict(target="cuda", arch="sm_89", op_class="conv", technique="tiling", state="underutilized_compute")
    assert cs.relevant(**q89) == []                                           # sm_75-only claim
    assert cs.relevant(**{**q89, "arch": "sm_75"}) == [c]                      # parent state matches child query
    with pytest.raises(EditError, match=">= 2 of its targets"):
        validate_lift({"kind": "lift", "claim_id": c.id, "to": "family", "evidence": []}, index=idx, claims=cs,
                      registry=REG)
    e = validate_lift({"kind": "lift", "claim_id": c.id, "to": "family", "evidence": [r.id for r in r89]},
                      index=idx, claims=cs, registry=REG)
    assert e["targets"] == ["cuda:sm_75", "cuda:sm_89"]
    from kernelblaster.knowledge.v2.reflect_v2 import apply_lift
    apply_lift(e, cs)
    assert c.level == "family" and c.family == "simt_gpu" and c.support == 4
    assert cs.relevant(**q89) == [c]
    assert cs.relevant(target="opencl", arch="adreno_740", op_class="conv", technique="tiling") == [c]
    assert "simt_gpu family" in cs.render([c])
    with pytest.raises(EditError, match="2 families"):
        validate_lift({"kind": "lift", "claim_id": c.id, "to": "universal", "evidence": []}, index=idx, claims=cs,
                      registry=REG)
    # contradictions on a target cancel its support
    c2, _ = cs.add(Claim(kind=APPLICABILITY, text="unroll helps conv", technique="tiling", op_class="conv",
                         arch="sm_75", data={"effect": "helps"}), [r75[0].id], [r89[0].id])
    with pytest.raises(EditError):
        validate_lift({"kind": "lift", "claim_id": c2.id, "to": "family", "evidence": [r89[1].id]}, index=idx,
                      claims=cs, registry=REG)


def test_mechanism_edit_validation(tmp_path):
    db = KnowledgeDB(tmp_path / "k.db")
    cs = SQLiteClaimStore(db)
    o = Ontology.seed(REG, legacy=False)
    recs = [_rec(i, tech="shared_memory_tiling", metrics_delta={"bw_util": -20.0, "reg_pressure": 15.0})
            for i in range(2)]
    idx = _index(*recs)
    ok = {"kind": "mechanism", "text": "tiling cut DRAM traffic as predicted", "evidence": ["r0", "r1"],
          "scope": {"technique": "shared_memory_tiling"}, "data": {"metric": "bw_util", "observed": "down",
                                                                   "verdict": "confirmed"}}
    e = validate_mechanism(ok, index=idx, claims=cs, ontology=o)
    assert e["data"]["predicted"] == "down"
    for bad, msg in (({"observed": "up"}, "mostly"), ({"verdict": "contradicted"}, "means confirmed"),
                     ({"metric": "l2_hit"}, "measured l2_hit")):
        with pytest.raises(EditError, match=msg):
            validate_mechanism({**ok, "data": {**ok["data"], **bad}}, index=idx, claims=cs, ontology=o)


# ---------------------------------------------------------------- hypotheses

def _kv(tmp_path, **kw):
    return KnowledgeV2(tmp_path / "kv", **kw)


def test_hypothesis_spec_validation_and_dedupe(tmp_path):
    kv = _kv(tmp_path)
    good = {"name": "streaming_store_hints", "source": "port", "mechanism":
            "write-once outputs bypass L1 with streaming stores so they do not evict reused input lines",
            "how": "store outputs with __stcs() (cache-streaming) on CUDA", "predicts": {"l1_hit": "up"},
            "preconditions": {"bw_util": ">=", "v": 30}, "states": ["memory_bound"]}
    h = validate_spec(good, ontology=kv.ontology, existing=[], target_key="cuda:sm_89",
                      known_states=set(kv.hierarchy.states))
    assert h.status == "proposed" and h.states == ["memory_bound"]
    with pytest.raises(ValueError, match="duplicates hypothesis"):
        validate_spec({**good, "name": "other_name"}, ontology=kv.ontology, existing=[h], target_key="cuda:sm_89",
                      known_states=set(kv.hierarchy.states))
    with pytest.raises(ValueError, match="already technique"):
        validate_spec({**good, "name": "coalesced_access"}, ontology=kv.ontology, existing=[],
                      target_key="cuda:sm_89", known_states=set(kv.hierarchy.states))
    with pytest.raises(ValueError, match="duplicates technique"):
        validate_spec({**good, "name": "tile_again", "mechanism": kv.ontology.get("shared_memory_tiling").mechanism},
                      ontology=kv.ontology, existing=[], target_key="cuda:sm_89", known_states=set(kv.hierarchy.states))
    with pytest.raises(P.PredicateError):
        validate_spec({**good, "preconditions": {"vibes": ">", "v": 1}}, ontology=kv.ontology, existing=[],
                      target_key="cuda:sm_89", known_states=set(kv.hierarchy.states))


def _hyp(kv, name="h_one", budget=4, **kw):
    kw.setdefault("op_classes", ["conv"])
    h = Hypothesis(name=name, mechanism=f"mechanism text for {name} that is long enough", source="gap",
                   how="how to do it on the target, concretely", target_key=kw.pop("target_key", "cuda:sm_75"),
                   budget=budget, **kw)
    return kv.hypotheses.add(h)


def test_hypothesis_exploration_budget_and_lifecycle(tmp_path):
    kv = _kv(tmp_path)
    h = _hyp(kv, predicts={"bw_util": "down"})
    cls = kv.classify(vec(mem_util=10, compute_util=20, parallel_fill=1, launch_frac=0), target_key="cuda:sm_75",
                      op_class="conv")
    rng = random.Random(0)
    picks = [kv.select(target_key="cuda:sm_75", op_class="conv", dtype="fp32", cls=cls,
                       vec=vec(mem_util=10, compute_util=20), rng=rng) for _ in range(40)]
    n_h = sum(1 for c in picks if c.kind == "hypothesis")
    assert 4 <= n_h <= 6 and picks[0].kind == "technique"                     # ~15% of steps
    # promote: wins on 2 problems on 2 targets of one family, LCB > 0, mechanism confirmed
    for i, (arch, pid) in enumerate([("sm_75", "p1"), ("sm_89", "p2"), ("sm_75", "p3")]):
        kv.hypotheses.record_attempt(h.id, outcome_id=f"o{i}", problem_id=pid, target_key=f"cuda:{arch}",
                                     reward=0.2 + 0.01 * i, status="improved", mechanism=CONFIRMED, measured=True)
    ev = kv.hypotheses.review()
    assert ev and ev[0]["event"] == "promoted" and ev[0]["level"] == "family"
    t = kv.ontology.get("h_one")
    assert t is not None and t.scope == "simt_gpu" and kv.ontology.realization(t, "cuda:sm_86")
    assert KnowledgeV2(tmp_path / "kv").ontology.get("h_one") is not None       # persisted
    # retire: budget spent, one problem only
    h2 = _hyp(kv, name="h_two", budget=2)
    for i in range(2):
        kv.hypotheses.record_attempt(h2.id, outcome_id=f"x{i}", problem_id="p1", target_key="cuda:sm_75",
                                     reward=0.1, status="improved", mechanism=MECH_UNKNOWN, measured=False)
    ev = kv.hypotheses.review()
    assert ev[0]["event"] == "retired" and "1 problem" in ev[0]["reason"]
    # mechanism gate: wins but contradicted mechanism -> not promoted
    h3 = _hyp(kv, name="h_three", budget=3)
    for i in range(3):
        kv.hypotheses.record_attempt(h3.id, outcome_id=f"y{i}", problem_id=f"p{i}", target_key="cuda:sm_75",
                                     reward=0.3, status="improved", mechanism=CONTRADICTED, measured=True)
    ev = kv.hypotheses.review()
    assert ev[0]["event"] == "retired" and "mechanism confirmed in 0%" in ev[0]["reason"]
    kinds = {r["kind"] for r in kv.db.review_items()}
    assert {"hypothesis_proposed", "hypothesis_promoted", "hypothesis_retired"} <= kinds


def test_graph_hypotheses_follow_mb_graph_hints(tmp_path):
    kv = _kv(tmp_path)
    props = {"proposals": [{"kind": "fuse", "pattern": ["conv2d", "relu"], "pairs": [["a", "b", "t"]],
                            "evidence": {"est_gain_pct": 5.0}, "rationale": "relu only reads the conv output"},
                           {"kind": "split", "slot": "conv2d", "groups": [], "evidence": {}, "rationale": "x"}]}
    hs = kv.hypotheses.from_proposals(props, "cuda:sm_75")
    assert {h.id for h in hs} == {"fuse:fused__conv2d__relu", "split:conv2d"}
    n = kv.hypotheses.sync_graph_hints({"hints": [{"id": "fuse:fused__conv2d__relu", "status": "accepted",
                                                   "evidence": [{}, {}]}]})
    assert n == 1 and kv.hypotheses.items["fuse:fused__conv2d__relu"].status == "promoted"


# ---------------------------------------------------------------- state discovery

def _rows(n_hi=5, n_lo=5, sep_metric="l1_hit"):
    rows = []
    for i in range(n_hi + n_lo):
        hi = i < n_hi
        rows.append({"id": f"d{i:02d}", "seq": i + 1, "technique": "shared_memory_tiling", "status": "improved",
                     "op_class": "conv", "target": "cuda", "arch": "sm_75", "problem_id": f"p{i % 3}",
                     "v2": {"reward": (0.3 + 0.01 * i) if hi else (-0.05 - 0.01 * i),
                            "chain": ["underutilized", "underutilized_compute"],
                            "item_id": "shared_memory_tiling",
                            "metrics_before": vec(**{sep_metric: 20 if hi else 80, "mem_util": 10,
                                                     "compute_util": 20})}})
    return rows


def test_state_discovery_trigger_validation_probation(tmp_path):
    kv = _kv(tmp_path)
    h = kv.hierarchy
    rows = _rows()
    trigs = D.find_triggers(rows, h)
    assert len(trigs) == 1 and trigs[0]["state"] == "underutilized_compute" and trigs[0]["diff"] > 0.1
    msgs = D.build_discovery_messages(trigs[0], h, [])
    shown = msgs[1]["content"]
    assert all((r["id"] in shown) != D.held_out(r["id"]) for r in rows)        # held-out rows not shown
    good = {"id": "low_l1_reuse", "parent": "underutilized_compute", "predicate": {"l1_hit": "<", "v": 50},
            "mechanism": "reuse is lost in L1, so explicit tiling recovers it"}
    ok, why, nums = D.validate_proposal(good, trigs[0], h)
    assert ok, why
    for bad, msg in (({"predicate": {"l2_hit": "<", "v": 50}}, "decidable on 0%"),
                     ({"predicate": {"mem_util": "<", "v": 50}}, "sides 10/0"),
                     ({"predicate": {"compute_util": ">=", "v": 20}}, "sides"),
                     ({"parent": "memory_bound"}, "not underutilized_compute"),
                     ({"id": "Bad Id"}, "snake_case"), ({"predicate": {"bogus": "<", "v": 1}}, "bad predicate")):
        ok2, why2, _ = D.validate_proposal({**good, **bad}, trigs[0], h)
        assert not ok2 and msg in why2, (bad, why2)
    # a predicate that splits sides but not rewards: variance not reduced
    noisy = _rows()
    for i, r in enumerate(noisy):
        r["v2"]["metrics_before"]["l2_hit"] = m(10 if i % 2 else 90)
    ok3, why3, _ = D.validate_proposal({**good, "predicate": {"l2_hit": "<", "v": 50}}, D.find_triggers(noisy, h)[0], h)
    assert not ok3 and "variance reduced" in why3
    s = D.admit(good, trigs[0], h, kv.db, nums, seq=10)
    assert s.status == "probationary"
    c = h.classify(vec(mem_util=10, compute_util=20, l1_hit=10, parallel_fill=1, launch_frac=0), target="cuda:sm_75")
    assert c.chain[-1] == "low_l1_reuse"
    later = _rows(6, 6)
    for i, r in enumerate(later):
        r["seq"], r["id"] = 100 + i, f"late{i}"
    ev = D.review_probation(h, later, kv.db)
    assert ev and ev[0]["status"] == "core"
    assert KnowledgeV2(tmp_path / "kv").hierarchy.states["low_l1_reuse"].status == "core"
    kinds = [r["kind"] for r in kv.db.review_items()]
    assert "state_admitted" in kinds and "state_core" in kinds


def test_upkeep_with_stub_llm(tmp_path):
    """Hypothesis generation + state discovery through the LLM path, rejected and admitted."""
    calls = []

    def fn(msgs):
        sys_ = msgs[0]["content"]
        calls.append(sys_[:40])
        if "refine a hierarchy" in sys_:
            # first a proposal that fails validation, the second (after seeing the rejection) passes
            if "REJECTED" in msgs[1]["content"]:
                return json.dumps({"proposal": {"id": "low_l1_reuse", "parent": "underutilized_compute",
                                                "predicate": {"l1_hit": "<", "v": 50}, "mechanism": "lost reuse"}})
            return json.dumps({"proposal": {"id": "big_l2", "parent": "underutilized_compute",
                                            "predicate": {"l2_hit": "<", "v": 50}, "mechanism": "?"}})
        return json.dumps({"hypotheses": [{"trigger": 0, "source": "gap", "name": "conv_epilogue_vector_stores",
                                           "mechanism": "pack the epilogue writes of four output channels into one "
                                                        "vector store to cut store instructions",
                                           "how": "accumulate 4 OC per thread and store a float4",
                                           "predicts": {"compute_util": "down"}, "op_classes": ["conv"]}]})

    agentic_llm.register_stub("upkeep", fn)
    try:
        kv = _kv(tmp_path, upkeep_model="stub:upkeep", upkeep_every=1)
        for r in _rows():
            rec = OutcomeRecord(id=r["id"], technique="shared_memory_tiling", status="regressed"
                                if r["v2"]["reward"] < 0 else "improved", target="cuda", arch="sm_75",
                                problem_id=r["problem_id"], slot="conv2d", op_class="conv")
            cls = kv.classify(r["v2"]["metrics_before"], target_key="cuda:sm_75", op_class="conv")
            kv.record(rec, choice=Choice("shared_memory_tiling", "technique", ""), reward=r["v2"]["reward"], cls=cls,
                      before=r["v2"]["metrics_before"])
        for t in ("memory_coalescing", "loop_unrolling_and_ilp", "launch_configuration_tuning"):
            for i in range(2):
                rec = OutcomeRecord(id=f"{t}{i}", technique=t, status="regressed", target="cuda", arch="sm_75",
                                    problem_id="p0", slot="conv2d", op_class="conv")
                kv.record(rec, choice=Choice(t, "technique", ""), reward=-0.1, cls=cls, before=_rows()[0]["v2"]["metrics_before"])
        ev = asyncio.run(kv.maybe_upkeep(force=True))
        kinds = [e["event"] for e in ev]
        assert "state_rejected" in kinds and "hypothesis_proposed" in kinds, ev
        rej = next(e for e in ev if e["event"] == "state_rejected")
        assert "decidable" in rej["reason"]
        # more evidence -> re-trigger; the model sees the rejection and proposes a valid split
        for i in range(4):
            r = _rows()[i]
            rec = OutcomeRecord(id=f"more{i}", technique="shared_memory_tiling", status="improved", target="cuda",
                                arch="sm_75", problem_id="p9", slot="conv2d", op_class="conv")
            kv.record(rec, choice=Choice("shared_memory_tiling", "technique", ""), reward=r["v2"]["reward"],
                      cls=cls, before=r["v2"]["metrics_before"])
        ev2 = asyncio.run(kv.maybe_upkeep(force=True))
        assert any(e["event"] == "state_admitted" and e["state"] == "low_l1_reuse" for e in ev2), ev2
        assert kv.hypotheses.open() and kv.hypotheses.open()[0].name == "conv_epilogue_vector_stores"
    finally:
        agentic_llm.unregister_stub("upkeep")


def test_reflection_v2_lift_through_updater(tmp_path):
    kv = _kv(tmp_path)
    recs = [_rec(i, arch=a, pid=f"p{i}", tech="memory_coalescing") for i, a in enumerate(["sm_75", "sm_75", "sm_89"])]
    for r in recs:
        kv.outcomes.append(r, {"reward": 0.1})
    c, _ = kv.claims.add(Claim(kind=APPLICABILITY, text="coalescing helps conv", technique="memory_coalescing",
                               op_class="conv", arch="sm_75", data={"effect": "helps"}), ["r0", "r1"])
    up = SemanticUpdaterV2(kv, model="stub:none")
    applied, rejected = up.validate_and_apply({"edits": [
        {"kind": "lift", "claim_id": c.id, "to": "family", "evidence": ["r2"]},
        {"kind": "lift", "claim_id": c.id, "to": "universal", "evidence": ["r2"]}]},
        index={r.id: r for r in recs})
    assert [a["result"] for a in applied] == ["lifted target->family"] and "2 families" in rejected[0]["reason"]
    assert any(r["kind"] == "claim_lifted" for r in kv.db.review_items())
    msgs = up.build_messages(recs, [])
    assert "lift" in msgs[0]["content"] and "mechanism" in msgs[0]["content"]


def test_legacy_single_file_adapter_path(tmp_path, monkeypatch):
    from kernelblaster.knowledge.semantic import SemanticKnowledge
    monkeypatch.setenv("KB_KNOWLEDGE_DIR", str(tmp_path / "k"))
    monkeypatch.setenv("KB_KNOWLEDGE", "v2")
    sk = SemanticKnowledge.from_env(target="cuda", arch="sm_75", op_class="gemm", problem_id="p")
    assert sk.is_v2
    before = ("Memory Throughput  %  85.0\nDRAM Throughput  %  80.0\nCompute (SM) Throughput  %  20.0\n"
              "L2 Hit Rate  %  30.0\n")
    after = ("Memory Throughput  %  60.0\nDRAM Throughput  %  45.0\nCompute (SM) Throughput  %  55.0\n"
             "L2 Hit Rate  %  70.0\n")
    assert sk.state_chain(before) == ["memory_bound", "dram_streaming"]
    sk.record(technique="shared_memory_blocking", status="improved", state_before="memory_bandwidth_saturated",
              metric_before=100, metric_after=60, speedup=100 / 60, ncu_log_before=before, ncu_log_after=after)
    row = sk.outcomes.rows()[0]
    assert row["state_before"] == "dram_streaming" and row["v2"]["item_id"] == "shared_memory_tiling"
    assert row["v2"]["legacy_state_mapped"] == "dram_streaming"
    assert row["v2"]["mechanism"]["label"] == CONFIRMED                       # bw down, as predicted
    assert row["metrics_delta"]["bw_util"] == -35.0


# ---------------------------------------------------------------- end to end (needs a GPU)

from test_multislot_agent import ARCH, FIX, needs_gpu, scripted  # noqa: E402


@needs_gpu
def test_agent_end_to_end_v2(tmp_path):
    from kernelblaster.agents.opt_multislot import MultiSlotAgent
    from kernelblaster.knowledge.impl_store import ImplStore
    from kernelblaster.knowledge.slots import LocalEvaluator, SlotProblem
    p = SlotProblem(shutil.copytree(FIX, tmp_path / "lenet"))
    state = {"broke": False, "fix_calls": 0, "linear_ok": None}
    agentic_llm.register_stub("scripted_v2", scripted(state))
    kv = KnowledgeV2(tmp_path / "k")
    _hyp(kv, name="lenet_probe", op_classes=[], budget=2, target_key=f"cuda:sm_{ARCH}")
    try:
        agent = MultiSlotAgent(p, LocalEvaluator(ARCH, work_root=tmp_path / "work"), model="stub:scripted_v2",
                               out_dir=tmp_path / "out", arch=ARCH, impl_store=ImplStore(tmp_path / "impl"),
                               outcome_log=kv.outcomes, claims=kv.claims, rollouts=2, steps=6, min_share=0.0,
                               seed=1, knowledge_v2=kv)
        res = asyncio.run(agent.run())
    finally:
        agentic_llm.unregister_stub("scripted_v2")
    assert res["final_ok"]
    rows = kv.outcomes.rows()
    assert len(rows) == 12
    assert all(r["v2"]["chain"] and r["v2"]["metrics_before"] for r in rows)      # every step classified
    assert all(r["state_before"] == r["v2"]["chain"][-1] for r in rows)
    assert all(s["context"]["chain"] for s in res["steps"])
    assert any(r["v2"]["kind"] == "hypothesis" for r in rows)                     # exploration budget used
    assert kv.hypotheses.items[[h for h in kv.hypotheses.items if kv.hypotheses.items[h].name == "lenet_probe"][0]] \
        .attempts
    assert res["knowledge"]["selected"] == 12
    assert any(h.kind == "graph" for h in kv.hypotheses.items.values()) or "proposals" in res
    # the stats DB has per-target nodes for every recorded item
    assert any(k[1] == "target" and k[2] == f"cuda:sm_{ARCH}" for k in kv.db.all_stats())
