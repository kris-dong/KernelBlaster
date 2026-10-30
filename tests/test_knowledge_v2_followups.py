"""Knowledge v2 follow-ups: access metrics from NCU, prediction refinement with rollback,
hypothesis triggers (gap fires, port capped, access-metric profiler triggers), graph-hint sync."""
import json
from pathlib import Path

from kernelblaster.knowledge import ncu
from kernelblaster.knowledge.claims import Claim
from kernelblaster.knowledge.outcomes import OutcomeRecord
from kernelblaster.knowledge.v2 import Choice, KnowledgeV2
from kernelblaster.knowledge.v2 import metrics as M
from kernelblaster.knowledge.v2.predictions import MIN_N, PredictionBook

FIX = Path(__file__).parent / "fixtures" / "ncu"


def test_ncu_access_metrics_real_l40s_capture():
    by = ncu.profile_by_slot((FIX / "lenet_sm89_access_metrics.csv").read_text())
    # the float4 relu is coalesced, the naive linear reads its weights with a stride
    assert by["relu"]["global_excess_sector_pct"] < 5
    assert by["linear"]["global_excess_sector_pct"] > 300
    assert all(v["smem_bank_conflict_pct"] == 0.0 for v in by.values())
    assert "--metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum" in ncu.ncu_command()
    from kernelblaster.knowledge.v2.targets import TargetRegistry
    v = M.NCUAdapter(TargetRegistry().get("cuda:sm_89")).from_profile(by["linear"])
    assert v["uncoalesced"]["src"] == "measured" and v["uncoalesced"]["v"] > 300 and v["bank_conflicts"]["v"] == 0
    assert M.canonical_ncu_delta({"global_excess_sector_pct": -40.0}) == {"uncoalesced": -40.0}
    d = ncu.delta(by["linear"], by["relu"])
    assert "global_excess_sector_pct" in d and "smem_bank_conflict_pct" in d


def test_access_metrics_switch(monkeypatch):
    monkeypatch.setenv("KB_NCU_ACCESS_METRICS", "0")
    assert "--metrics" not in ncu.ncu_command()


def _kv(tmp_path):
    return KnowledgeV2(tmp_path / "k", seed_mined=False)


def _outcome(kv, i, tech, chain, before, after, reward=0.2, pid="p", slot="s", status="improved"):
    rec = OutcomeRecord(id=f"{tech}_{i}_{pid}", technique=tech, status=status, target="cuda", arch="sm_89",
                        problem_id=pid, slot=slot, op_class="conv")
    from kernelblaster.knowledge.v2.states import Classification
    cls = Classification(list(chain), "classified")
    kv.record(rec, choice=Choice(tech, "technique", ""), reward=reward, cls=cls, before=before, after=after)


def _vec(**kw):
    return {k: {"v": v, "src": "measured", "from": "t"} for k, v in kw.items()}


def test_prediction_refinement_conditioned_on_state_and_rollback(tmp_path):
    kv = _kv(tmp_path)
    t = kv.ontology.get("thread_coarsening")
    t.predicts = {"compute_util": "up"}                         # the prior the evidence will contradict
    naive = ["underutilized", "underutilized_compute"]
    tuned = ["compute_bound", "matrix_compute"]
    for i in range(MIN_N):                                       # on naive kernels compute_util falls with a win
        _outcome(kv, i, "thread_coarsening", naive, _vec(compute_util=80), _vec(compute_util=40))
    for i in range(MIN_N):                                       # on tuned ones it rises as predicted
        _outcome(kv, 10 + i, "thread_coarsening", tuned, _vec(compute_util=50), _vec(compute_util=70))
    rows = kv.outcomes.rows()
    assert kv.predictions.refine(rows, kv.ontology, kv.claims) == []      # 4 rows, no claim: not enough
    c = Claim(kind="mechanism", text="coarsening a naive kernel lowers SM throughput while it wins",
              technique="thread_coarsening", op_class="conv",
              data={"metric": "compute_util", "observed": "down", "verdict": "contradicted",
                    "technique_id": "thread_coarsening"})
    kv.claims.add(c, [r["id"] for r in rows[:2]])
    ev = kv.predictions.refine(rows, kv.ontology, kv.claims)
    assert len(ev) == 1 and ev[0]["state"] == "underutilized" and ev[0]["dir"] == "down" and ev[0]["claims"] == [c.id]
    assert kv.predictions.effective("thread_coarsening", t.predicts, naive)["compute_util"] == "down"
    assert kv.predictions.effective("thread_coarsening", t.predicts, tuned)["compute_util"] == "up"   # conditioned
    assert "compute_util down" in kv.technique_text(t, "cuda:sm_89", naive)
    # the mechanism check now uses the refined prediction
    _outcome(kv, 99, "thread_coarsening", naive, _vec(compute_util=70), _vec(compute_util=30))
    assert kv.outcomes.rows()[-1]["v2"]["mechanism"]["label"] == "confirmed"
    assert PredictionBook(kv.db, kv.hierarchy).overrides                    # persisted
    # later evidence disagrees with the override -> reverted automatically
    for i in range(MIN_N):
        _outcome(kv, 200 + i, "thread_coarsening", naive, _vec(compute_util=40), _vec(compute_util=75))
    ev = kv.predictions.refine(kv.outcomes.rows(), kv.ontology, kv.claims)
    assert any(e["event"] == "prediction_reverted" for e in ev)
    assert kv.predictions.effective("thread_coarsening", t.predicts, naive)["compute_util"] == "up"
    kinds = [a["kind"] for a in kv.db.audit_items()]
    assert "prediction_refined" in kinds and "prediction_reverted" in kinds


def test_unconditional_refinement_from_data_alone(tmp_path):
    kv = _kv(tmp_path)
    t = kv.ontology.get("loop_unrolling_and_ilp")
    t.predicts = {"compute_util": "up_or_flat"}
    for i in range(2 * MIN_N):
        chain = ["underutilized", "underutilized_compute"] if i % 2 else ["memory_bound"]
        _outcome(kv, i, "loop_unrolling_and_ilp", chain, _vec(compute_util=60), _vec(compute_util=30))
    ev = kv.predictions.refine(kv.outcomes.rows(), kv.ontology, kv.claims)
    assert len(ev) == 1 and ev[0]["state"] is None and ev[0]["dir"] == "down"        # holds everywhere


def test_gap_trigger_fires_and_port_is_capped(tmp_path):
    kv = _kv(tmp_path)
    chain = ["underutilized", "underutilized_compute"]
    for i, tech in enumerate(["memory_coalescing", "loop_unrolling_and_ilp", "thread_coarsening",
                              "vectorized_memory_access", "memory_coalescing"]):
        _outcome(kv, i, tech, chain, _vec(compute_util=30, mem_util=10), _vec(compute_util=30, mem_util=10),
                 reward=-0.01 if i % 2 else 0.002, status="neutral")
    trig = kv.hypothesis_triggers(kv.outcomes.rows())
    srcs = [t["source"] for t in trig]
    assert srcs[0] == "gap" and srcs.count("port") <= 1
    gaps = [t for t in trig if t["source"] == "gap"]
    assert any("stalled_slot" in t for t in gaps) and any("tried" in t and "stalled_slot" not in t for t in gaps)


def test_access_metric_profiler_trigger(tmp_path):
    kv = _kv(tmp_path)
    _outcome(kv, 0, "memory_coalescing", ["underutilized", "underutilized_memory", "uncoalesced_access"],
             _vec(uncoalesced=690.0, mem_util=20, compute_util=5), _vec(uncoalesced=5.0, mem_util=40, compute_util=5))
    trig = kv.hypothesis_triggers(kv.outcomes.rows())
    assert any(t["source"] == "profiler" and t["rule"]["rule"] == "UncoalescedGlobalAccess" for t in trig)


def test_graph_hint_sync(tmp_path):
    kv = _kv(tmp_path)
    kv.hypotheses.from_proposals({"proposals": [{"kind": "fuse", "pattern": ["conv2d", "relu"], "pairs": [],
                                                 "evidence": {}, "rationale": "x"}]}, "cuda:sm_89")
    hints = {"format": "modelblaster.graph_hints/v1", "hints": [
        {"id": "fuse:fused__conv2d__relu", "kind": "fuse", "status": "accepted", "evidence": [{"gain": 0.1}]},
        {"id": "split:conv2d", "kind": "split_slot", "status": "rejected", "evidence": [{}, {}]},
        {"id": "fuse:fused__a__b", "kind": "fuse", "status": "proposed"}]}
    fp = tmp_path / "graph_hints.json"
    fp.write_text(json.dumps(hints))
    assert kv.sync_graph_hints(fp, "cuda:sm_89") == 2
    h = kv.hypotheses.items
    assert h["fuse:fused__conv2d__relu"].status == "promoted" and h["split:conv2d"].status == "retired"
    assert "fuse:fused__a__b" not in h                              # undecided hints from others are not added
    assert kv.sync_graph_hints(fp, "cuda:sm_89") == 0               # idempotent
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[1] / "scripts" / "sync_graph_hints.py"
    out = json.loads(subprocess.run([sys.executable, str(script), str(tmp_path / "k"), str(fp)],
                                    capture_output=True, text=True, check=True).stdout)
    assert out["graph_hypotheses"] == 2 and out["synced"] == 0
