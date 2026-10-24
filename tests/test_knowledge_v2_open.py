"""Knowledge v2, open scope tree + automatic promotion: deep chains, runtime nodes, LCA and
stepwise lifts, legacy DB compatibility, automatic rollback of lifts and promoted
techniques, and the minimum-effect promotion gate."""
import json
import sqlite3

import pytest

from kernelblaster.knowledge.claims import APPLICABILITY, Claim
from kernelblaster.knowledge.outcomes import OutcomeRecord
from kernelblaster.knowledge.reflect import EditError
from kernelblaster.knowledge.v2 import Choice, KnowledgeV2
from kernelblaster.knowledge.v2.hypotheses import MIN_EFFECT, Hypothesis
from kernelblaster.knowledge.v2.ontology import CONFIRMED, Technique
from kernelblaster.knowledge.v2.reflect_v2 import apply_lift, revalidate_lifts, validate_lift
from kernelblaster.knowledge.v2.stats import HierStats, path as stat_path
from kernelblaster.knowledge.v2.store import KnowledgeDB, SQLiteClaimStore
from kernelblaster.knowledge.v2.targets import TargetRegistry


def _rec(i, arch, status="improved", tech="tiling", pid=None, target="cuda"):
    return OutcomeRecord(id=f"r{i}", technique=tech, status=status, target=target, arch=arch,
                         problem_id=pid or f"p{i}", slot="s", op_class="conv", speedup=1.5)


def test_deep_chain_and_backend_map():
    r = TargetRegistry()
    assert r.chain("cuda:sm_89") == ["cuda:sm_89", "ada", "nvidia_gpu", "simt_gpu", "universal"]
    assert r.chain("opencl:adreno650") == ["opencl:adreno650", "adreno", "simt_gpu", "universal"]
    assert r.chain("cuda:sm_120")[1] == "nvidia_gpu"                 # unknown target: backend -> node (data)
    assert r.chain("weird:thing") == ["weird:thing", "universal"]
    assert r.family_of("cuda:sm_75") == "simt_gpu"                   # v1 alias
    assert r.covers("family", "simt_gpu", "cuda:sm_75") and r.covers("node", "turing", "cuda:sm_75")
    assert not r.covers("node", "ampere", "cuda:sm_75") and r.covers("universal", None, "rvv:vlen256")
    assert r.lca(["cuda:sm_80", "cuda:sm_86"]) == "ampere"
    assert r.lca(["cuda:sm_75", "cuda:sm_89"]) == "nvidia_gpu"
    assert r.lca(["cuda:sm_75", "opencl:adreno740"]) == "simt_gpu"
    assert r.lca(["cuda:sm_75", "rvv:vlen256"]) == "universal"
    assert r.lca(["cuda:sm_75"]) == "cuda:sm_75"
    # stats follow the whole chain; kappa per edge
    s = HierStats(r, kappa=3.0)
    p = stat_path(r, target_key="cuda:sm_80", op_class="gemm", chain=["compute_bound"])
    assert [n[1] for n in p[:4]] == ["cuda:sm_80", "ampere", "nvidia_gpu", "simt_gpu"]
    s.update("t", p[2:], 1.0)                                       # nvidia_gpu and up: mean 1
    e3 = s.estimate("t", p)
    r.nodes["ampere"].kappa = 30.0                                    # ampere pulls hard toward its parent
    s.update("t", p[:2], -1.0)
    e_strong = s.estimate("t", p)
    r.nodes["ampere"].kappa = None
    e_weak = s.estimate("t", p)
    assert e_strong > e_weak and 0 < e3 <= 1


def test_runtime_node_and_target_persist(tmp_path):
    kv = KnowledgeV2(tmp_path / "k")
    kv.add_scope("blackwell", "nvidia_gpu", "sm_100: tcgen05, TMEM")
    kv.add_target("cuda:sm_100", "blackwell", peak_gflops=80000.0, peak_gbps=8000.0, sms=148)
    with pytest.raises(ValueError, match="unknown parent"):
        kv.add_scope("orphan", "nope")
    kv2 = KnowledgeV2(tmp_path / "k")                                 # reload from the DB
    assert kv2.registry.chain("cuda:sm_100") == ["cuda:sm_100", "blackwell", "nvidia_gpu", "simt_gpu", "universal"]
    assert kv2.registry.get("cuda:sm_100").peak_gbps == 8000.0
    kinds = [a["kind"] for a in kv2.db.audit_items()]
    assert "scope_added" in kinds and "target_added" in kinds
    # knowledge at the new node applies to its targets only
    assert kv2.registry.covers("node", "blackwell", "cuda:sm_100") and not kv2.registry.covers(
        "node", "blackwell", "cuda:sm_90")


def _store(tmp_path):
    return SQLiteClaimStore(KnowledgeDB(tmp_path / "k.db"), registry=TargetRegistry())


def test_lca_and_stepwise_lift(tmp_path):
    cs = _store(tmp_path)
    reg = cs.registry
    recs = [_rec(0, "sm_80"), _rec(1, "sm_86"), _rec(2, "sm_75"), _rec(3, "adreno740", target="opencl"),
            _rec(4, "rvv-vlen256", target="riscv")]
    idx = {r.id: r for r in recs}
    c, _ = cs.add(Claim(kind=APPLICABILITY, text="tiling helps conv", technique="tiling", op_class="conv",
                        arch="sm_80", data={"effect": "helps"}), ["r0"])
    steps = []
    for ev in (["r1"], ["r2"], ["r3"], ["r4"]):
        e = validate_lift({"kind": "lift", "claim_id": c.id, "evidence": ev}, index=idx, claims=cs, registry=reg)
        apply_lift(e, cs)
        steps.append(c.family or c.level)
    assert steps == ["ampere", "nvidia_gpu", "simt_gpu", "universal"]      # one node at a time, up the chain
    # a cap: evidence spans simt_gpu but only nvidia_gpu was asked for
    c2, _ = cs.add(Claim(kind=APPLICABILITY, text="unroll helps conv", technique="tiling", op_class="conv",
                         arch="sm_80", data={"effect": "helps"}), ["r0"])
    e = validate_lift({"kind": "lift", "claim_id": c2.id, "to": "nvidia_gpu", "evidence": ["r3", "r2"]},
                      index=idx, claims=cs, registry=reg)
    assert e["to"] == "nvidia_gpu"
    with pytest.raises(EditError, match="supports simt_gpu, not universal|2 families"):
        validate_lift({"kind": "lift", "claim_id": c2.id, "to": "universal", "evidence": ["r3"]},
                      index=idx, claims=cs, registry=reg)
    # no lift when the evidence stays under the claim's scope
    c5, _ = cs.add(Claim(kind=APPLICABILITY, text="split-k helps conv", technique="tiling", op_class="conv",
                         arch="sm_80", data={"effect": "helps"}), ["r0"])
    apply_lift(validate_lift({"kind": "lift", "claim_id": c5.id, "evidence": ["r1"]}, index=idx, claims=cs,
                             registry=reg), cs)
    assert c5.family == "ampere"
    with pytest.raises(EditError, match="above ampere"):
        validate_lift({"kind": "lift", "claim_id": c5.id, "evidence": ["r1"]}, index=idx, claims=cs, registry=reg)


def test_lift_rollback_when_targets_disagree(tmp_path):
    cs = _store(tmp_path)
    reg = cs.registry
    recs = [_rec(0, "sm_75"), _rec(1, "sm_89"), _rec(2, "sm_80"),
            _rec(3, "sm_89", status="regressed"), _rec(4, "sm_89", status="regressed")]
    idx = {r.id: r for r in recs}
    c, _ = cs.add(Claim(kind=APPLICABILITY, text="tiling helps conv", technique="tiling", op_class="conv",
                        arch="sm_75", data={"effect": "helps"}), ["r0"])
    apply_lift(validate_lift({"kind": "lift", "claim_id": c.id, "evidence": ["r1", "r2"]}, index=idx, claims=cs,
                             registry=reg), cs)
    assert c.family == "nvidia_gpu"
    assert revalidate_lifts(cs, idx, reg) == []                         # nothing disagrees yet
    cs.contradict(c.id, ["r3", "r4"])                                    # sm_89 now: 1 support, 2 contradictions
    ev = revalidate_lifts(cs, idx, reg, db=cs.db)
    # sm_75 (turing) and sm_80 (ampere) still agree, but every node covering both also covers
    # sm_89: back to the best-supported single target
    assert ev[0]["event"] == "claim_demoted" and ev[0]["disagreeing"] == ["cuda:sm_89"]
    assert ev[0]["to"] == "cuda:sm_75" and c.level == "target" and c.arch == "sm_75"
    assert cs.relevant(target="cuda", arch="sm_89", op_class="conv", technique="tiling") == []
    assert "claim_demoted" in [a["kind"] for a in cs.db.audit_items()]
    # everything contradicted -> retired
    c4, _ = cs.add(Claim(kind=APPLICABILITY, text="shuffles help conv", technique="tiling", op_class="conv",
                         arch="sm_75", data={"effect": "helps"}), ["r1", "r2"])
    apply_lift(validate_lift({"kind": "lift", "claim_id": c4.id, "evidence": []}, index=idx, claims=cs,
                             registry=reg), cs)
    cs.contradict(c4.id, ["r3", "r4"])
    recs.append(_rec(5, "sm_80", status="regressed"))
    idx[recs[-1].id] = recs[-1]
    cs.contradict(c4.id, ["r5"])
    ev = revalidate_lifts(cs, idx, reg, db=cs.db)
    assert any(e["claim_id"] == c4.id and e["event"] == "claim_retired" for e in ev) and c4.status == "retired"


def test_lift_rollback_moves_to_lower_common_ancestor(tmp_path):
    cs = _store(tmp_path)
    reg = cs.registry
    recs = [_rec(0, "sm_80"), _rec(1, "sm_86"), _rec(2, "adreno740", target="opencl"),
            _rec(3, "adreno740", status="regressed", target="opencl"),
            _rec(4, "adreno740", status="regressed", target="opencl")]
    idx = {r.id: r for r in recs}
    c, _ = cs.add(Claim(kind=APPLICABILITY, text="tiling helps conv", technique="tiling", op_class="conv",
                        arch="sm_80", data={"effect": "helps"}), ["r0", "r1", "r2"])
    apply_lift(validate_lift({"kind": "lift", "claim_id": c.id, "evidence": []}, index=idx, claims=cs,
                             registry=reg), cs)
    assert c.family == "simt_gpu"
    cs.contradict(c.id, ["r3", "r4"])
    ev = revalidate_lifts(cs, idx, reg)
    assert ev[0]["to"] == "ampere" and c.family == "ampere"           # sm_80 + sm_86 still agree


def _hyp(kv, rewards, targets, name="h_gate"):
    h = Hypothesis(name=name, mechanism=f"mechanism text for {name} that is long enough", source="gap",
                   how="how to do it on the target, concretely", target_key=targets[0], op_classes=["conv"],
                   budget=len(rewards))
    kv.hypotheses.add(h)
    for i, (rw, tk) in enumerate(zip(rewards, targets)):
        kv.hypotheses.record_attempt(h.id, outcome_id=f"{name}{i}", problem_id=f"p{i}", target_key=tk, reward=rw,
                                     status="improved", mechanism=CONFIRMED, measured=False)
    return h


def test_min_effect_gate(tmp_path):
    kv = KnowledgeV2(tmp_path / "k")
    small = _hyp(kv, [0.004, 0.006, 0.005], ["cuda:sm_75"] * 3, name="tiny_wins")        # consistent but noise-sized
    big = _hyp(kv, [0.05, 0.08, 0.06], ["cuda:sm_75", "cuda:sm_75", "cuda:sm_80"], name="real_wins")
    ev = {e["id"]: e for e in kv.hypotheses.review()}
    assert ev[small.id]["event"] == "retired" and "minimum effect" in ev[small.id]["reason"]
    assert ev[big.id]["event"] == "promoted" and kv.ontology.get("real_wins").scope == "nvidia_gpu"
    assert MIN_EFFECT == 0.01


def test_promoted_technique_auto_demotion(tmp_path):
    kv = KnowledgeV2(tmp_path / "k")
    _hyp(kv, [0.05, 0.08, 0.06], ["cuda:sm_75", "cuda:sm_89", "cuda:sm_75"], name="promo")
    kv.hypotheses.review()
    t = kv.ontology.get("promo")
    assert t.scope == "nvidia_gpu" and t.status == "active"            # used at once, no gate

    def outcome(i, arch, reward):
        rec = OutcomeRecord(id=f"o{arch}{i}", technique="promo", status="improved" if reward > 0 else "regressed",
                            target="cuda", arch=arch, problem_id=f"p{i}", slot="s", op_class="conv")
        cls = kv.classify({}, target_key=f"cuda:{arch}", op_class="conv")
        kv.record(rec, choice=Choice("promo", "technique", ""), reward=reward, cls=cls, before={})

    for i in range(3):
        outcome(i, "sm_89", -0.05)                                       # sm_89 now says no
        outcome(i, "sm_75", 0.04)
    ev = kv.hypotheses.revalidate(kv.outcomes.rows())
    assert ev[0]["event"] == "technique_demoted" and ev[0]["to"] == "cuda:sm_75"
    assert kv.ontology.get("promo").scope == "cuda:sm_75"
    assert not kv.registry.covers(t.level, t.scope, "cuda:sm_89")
    for i in range(3, 6):
        outcome(i, "sm_75", -0.05)                                       # and later sm_75 too
    ev = kv.hypotheses.revalidate(kv.outcomes.rows())
    assert ev[0]["event"] == "technique_retired" and kv.ontology.get("promo").status == "retired"
    assert "promo" not in {x.id for x in kv.ontology.candidates(
        target_key="cuda:sm_75", op_class="conv", chain=[], values={}, ctx={"op_class": "conv"})}
    k2 = KnowledgeV2(tmp_path / "k")
    assert k2.ontology.get("promo").status == "retired"                 # persisted
    assert {"technique_demoted", "technique_retired"} <= {a["kind"] for a in k2.db.audit_items()}


def test_legacy_v1_database_stays_readable(tmp_path):
    """A seed-version-1 DB: family-level claims / techniques / stats rows map onto nodes."""
    root = tmp_path / "k"
    kv = KnowledgeV2(root)
    db = kv.db
    db.set_meta("seed_version", 1)
    old = Claim(kind=APPLICABILITY, text="coalescing helps conv on GPUs", technique="memory_coalescing",
                op_class="conv", level="family", family="simt_gpu", data={"effect": "helps"})
    old.support_ids, old.support = ["x1", "x2"], 2
    from dataclasses import asdict
    with db.tx() as c:
        c.execute("INSERT INTO claims VALUES (?, ?, ?, ?)", (old.id, "active", json.dumps(asdict(old)), old.updated_ts))
        c.execute("INSERT INTO stats VALUES ('memory_coalescing', 'family', 'simt_gpu', 'conv', 'underutilized', 4, 0.4, 0.1)")
        c.execute("INSERT OR REPLACE INTO techniques VALUES ('old_tech', 'seed', ?, 'x')", (json.dumps(
            Technique(id="old_tech", mechanism="an old family technique", level="family", scope="simt_gpu",
                      realizations=[{"scope": "family:simt_gpu", "how": "x"}]).to_dict()),))
        c.execute("INSERT OR REPLACE INTO targets VALUES ('cuda:sm_61', 'simt_gpu', ?)",
                  (json.dumps({"key": "cuda:sm_61", "family": "simt_gpu", "peak_gbps": 480.0}),))
    kv2 = KnowledgeV2(root)
    got = kv2.claims.relevant(target="cuda", arch="sm_89", op_class="conv", technique="memory_coalescing")
    assert [g.id for g in got] == [old.id] and "simt_gpu (" in kv2.claims.render(got)
    assert not kv2.claims.relevant(target="riscv", arch="rvv-vlen256", op_class="conv", technique="memory_coalescing")
    p = stat_path(kv2.registry, target_key="cuda:sm_89", op_class="conv", chain=["underutilized"])
    assert kv2.stats.count("memory_coalescing", ("node", "simt_gpu", "conv", "underutilized")) == 4
    assert abs(kv2.stats.estimate("memory_coalescing", p) - 0.1 * 4 / 7) < 0.05
    t = kv2.ontology.get("old_tech")
    assert t is not None and kv2.ontology.realization(t, "cuda:sm_75") and \
        kv2.registry.covers(t.level, t.scope, "opencl:adreno650")
    assert kv2.registry.chain("cuda:sm_61")[1] == "simt_gpu" and kv2.registry.get("cuda:sm_61").peak_gbps == 480.0
    assert kv2.db.get_meta("seed_version") >= 2
    # the seed refresh did not touch the curated technique scopes the old DB had at family level
    assert kv2.ontology.get("warp_shuffle_reduction").level == "node"


def test_probationary_states_are_live_without_review(tmp_path):
    """No code path withholds an admitted (probationary) state: it classifies at once."""
    from kernelblaster.knowledge.v2.states import PROBATIONARY, State
    kv = KnowledgeV2(tmp_path / "k")
    kv.hierarchy.add(State("tiny_kernel", "latency_bound", {"parallel_fill": "<", "v": 0.01}, status=PROBATIONARY,
                           order=-1))
    v = {"parallel_fill": {"v": 0.001, "src": "measured"}, "launch_frac": {"v": 0.1, "src": "measured"}}
    assert kv.classify(v, target_key="cuda:sm_75", op_class="conv").chain == ["latency_bound", "tiny_kernel"]
    con = sqlite3.connect(str(tmp_path / "k" / "knowledge.db"))
    assert con.execute("SELECT name FROM sqlite_master WHERE name='review'").fetchone()   # audit log table
