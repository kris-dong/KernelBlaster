# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Global knowledge store (knowledge/v2/merge.py): seed / merge round trips with stub runs."""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from kernelblaster.knowledge.claims import Claim
from kernelblaster.knowledge.outcomes import OutcomeRecord
from kernelblaster.knowledge.v2 import KnowledgeV2
from kernelblaster.knowledge.v2.hypotheses import Hypothesis
from kernelblaster.knowledge.v2.merge import (GlobalStore, MergeError, Policy, RunInfo, materialize,
                                              merge_knowledge_dir, part_key)
from kernelblaster.knowledge.v2.salvage import parse_rendered_predicate
from kernelblaster.knowledge.v2.states import State
from kernelblaster.knowledge.v2.stats import path as stat_path
from kernelblaster.knowledge.v2 import predicates as P

P89 = "cuda:sm_89|fp16"


def start_run(store: GlobalStore, tmp: Path, name: str, part: str = P89, problem: str = None):
    """A run as the farm starts it: seed dir materialized from the current snapshot, knowledge dir a copy."""
    seed = tmp / f"{name}_seed"
    materialize(store, part, seed, problem=problem or name, run_uid=f"uid-{name}")
    kdir = tmp / f"{name}_k"
    shutil.copytree(seed, kdir)
    return KnowledgeV2(kdir), seed, kdir


def root_state(kv) -> str:
    return sorted(s.id for s in kv.hierarchy.states.values() if s.parent is None)[0]


def outcome(kv, tech, reward, *, problem="kb_problem__x__r00", status="improved", arch="sm_89", op="gemm",
            dtype="fp16", oid=None):
    chain = [root_state(kv)]
    rec = OutcomeRecord(technique=tech, status=status, target="cuda", arch=arch, problem_id=problem, slot="s0",
                        op_class=op, dtype=dtype, speedup=1.0 + reward)
    if oid:
        rec.id = oid
    kv.outcomes.append(rec, {"item_id": tech, "reward": reward, "chain": chain})
    nodes = stat_path(kv.registry, target_key=f"cuda:{arch}", op_class=op, chain=chain, dtype=dtype)
    kv.stats.update(tech, nodes, reward)
    return rec.id


def tech(kv) -> str:
    return sorted(kv.ontology.techniques)[0]


def merge(store, kdir, seed, name, **kw):
    info = RunInfo(run_uid=kw.pop("uid", f"uid-{name}"), problem=kw.pop("problem", name), attempt="1",
                   part=kw.pop("part", None))
    return merge_knowledge_dir(store, kdir, seed, info=info, **kw)


def gk(store, sql, args=()):
    c = sqlite3.connect(store.path(store.current_id()))
    try:
        return c.execute(sql, args).fetchall()
    finally:
        c.close()


def stat_n(store, part, item):
    r = gk(store, "SELECT SUM(n) FROM gk_stats WHERE part=? AND item=? AND level='target' AND op LIKE 'gemm%'", (part, item))
    return r[0][0] or 0


@pytest.fixture()
def store(tmp_path):
    return GlobalStore(tmp_path / "store").ensure()


def test_roundtrip_no_double_count(store, tmp_path):
    kv, seed, kdir = start_run(store, tmp_path, "a")
    t = tech(kv)
    ids = [outcome(kv, t, 0.2) for _ in range(3)]
    c, _ = kv.claims.add(Claim(kind="applicability", text=f"{t} helps small gemm on sm_89", arch="sm_89",
                               op_class="gemm", technique=t, data={"effect": "helps"}), ids[:2])
    kv.claims.save()
    r1 = merge(store, kdir, seed, "a")
    assert r1["status"] == "merged" and r1["version"] == "v0001"
    assert r1["outcomes"] == 3
    n1 = stat_n(store, P89, t)
    assert n1 == 3 * 2                                  # target op|fp16 and target op rows, 3 outcomes each
    # run B starts from v0001: the claim and its evidence arrive in the seed
    kv2, seed2, kdir2 = start_run(store, tmp_path, "b")
    got = [x for x in kv2.claims.claims.values() if x.text == c.text]
    assert len(got) == 1 and got[0].support == 2 and got[0].id == c.id
    assert kv2.claims.meta["reflect_cursor"] == len(kv2.outcomes) >= 2     # imported records are not re-reflected
    new = outcome(kv2, t, 0.1, problem="kb_problem__y__r00")
    kv2.claims.support(c.id, [new])
    kv2.claims.save()
    r2 = merge(store, kdir2, seed2, "b")
    assert r2["outcomes"] == 1
    row = json.loads(gk(store, "SELECT data FROM gk_claims WHERE id=?", (c.id,))[0][0])
    assert row["support"] == 3                          # 2 + the new one, not 2 + 3
    assert stat_n(store, P89, t) == n1 + 2              # only the delta was added


def test_idempotent_and_incremental(store, tmp_path):
    kv, seed, kdir = start_run(store, tmp_path, "a")
    t = tech(kv)
    outcome(kv, t, 0.3)
    r1 = merge(store, kdir, seed, "a")
    again = merge(store, kdir, seed, "a")
    assert again["status"] == "duplicate" and store.current_id() == r1["version"]
    merged_copy = tmp_path / "a_merged"
    shutil.copytree(kdir, merged_copy)
    outcome(kv, t, 0.3)
    with pytest.raises(MergeError):                     # against its seed again would count the first part twice
        merge(store, kdir, seed, "a")
    r2 = merge(store, kdir, merged_copy, "a")           # incremental: against the copy merged last
    assert r2["status"] == "merged" and r2["outcomes"] == 1
    assert stat_n(store, P89, t) == 4
    assert gk(store, "SELECT COUNT(*) FROM gk_ledger")[0][0] == 2


def test_partition_isolation(store, tmp_path):
    kv, seed, kdir = start_run(store, tmp_path, "a")
    t = tech(kv)
    ids = [outcome(kv, t, 0.2) for _ in range(2)]
    kv.claims.add(Claim(kind="applicability", text="sm_89 fp16 only lesson about tiles", arch="sm_89",
                        op_class="gemm", technique=t, data={"effect": "helps"}), ids)
    kv.claims.save()
    merge(store, kdir, seed, "a")
    for other in ("cuda:sm_100|fp32", "cuda:sm_89|fp32", "cuda:sm_100|fp16"):
        m = materialize(store, other, tmp_path / f"seed_{other.replace(':', '_').replace('|', '_')}")
        assert m["counts"]["claims"] == 0 and m["counts"]["stats_rows"] == 0 and m["counts"]["outcomes"] == 0
    m = materialize(store, P89, tmp_path / "seed_same")
    assert m["counts"]["claims"] == 1 and m["counts"]["stats_rows"] > 0


def test_conflicts_dedupe_and_renames(store, tmp_path):
    runs = []
    for name in ("a", "b"):
        runs.append((name, *start_run(store, tmp_path, name)))
    for i, (name, kv, seed, kdir) in enumerate(runs):
        t = tech(kv)
        ids = [outcome(kv, t, 0.2, problem=f"kb_problem__{name}__r00") for _ in range(2)]
        # the same lesson in both runs (folded), a state with the same id but different predicates (renamed)
        kv.claims.add(Claim(kind="pitfall", text="half2 loads need 4-byte alignment of the row pitch", arch="sm_89",
                            op_class="gemm", data={"symptom": "misaligned address", "fix": "pad rows"}), ids)
        kv.claims.save()
        root = root_state(kv)
        s = State(id="wide_rows", parent=root, predicate={"bw_util": "<", "v": 30 + 10 * i}, status="probationary",
                  source="discovered:test")
        kv.hierarchy.add(s)
        kv.db.put("states", s.id, s.to_dict(), status=s.status)
        kv.db.set_meta("prediction_overrides", {t: [{"state": None, "metric": "l1_hit", "dir": "up" if i else "down",
                                                     "n": 5 + i, "since_seq": 1}]})
    for name, kv, seed, kdir in runs:
        merge(store, kdir, seed, name)
    rows = gk(store, "SELECT data FROM gk_claims WHERE status='active'")
    same = [json.loads(r[0]) for r in rows if "4-byte alignment" in json.loads(r[0])["text"]]
    assert len(same) == 1 and same[0]["support"] == 4
    sids = sorted(r[0] for r in gk(store, "SELECT id FROM gk_states"))
    assert len(sids) == 2 and "wide_rows" in sids and any(s.startswith("wide_rows_") for s in sids)
    ov = json.loads(gk(store, "SELECT data FROM gk_overrides")[0][0])
    assert len(ov) == 1 and ov[0]["dir"] == "up"        # the better-supported override wins the conflict


def test_problem_local_items_stay_local(store, tmp_path):
    kv, seed, kdir = start_run(store, tmp_path, "a", problem="1_MLP")
    t = tech(kv)
    i = outcome(kv, t, 0.2, problem="kb_problem__relu__r00")
    kv.claims.add(Claim(kind="impl_note", text="the fused relu wins by vectorising half2", problem_id="kb_problem__relu__r00",
                        slot="s0", data={"note_type": "why_fast"}), [i])
    kv.claims.save()
    merge(store, kdir, seed, "1_MLP", problem="1_MLP")
    pid = json.loads(gk(store, "SELECT rec FROM gk_outcomes")[0][0])["problem_id"]
    assert pid == "1_MLP::kb_problem__relu__r00"
    other = materialize(store, P89, tmp_path / "s_other", problem="2_Other")
    mine = materialize(store, P89, tmp_path / "s_mine", problem="1_MLP")
    assert other["counts"]["claims"] == 0 and mine["counts"]["claims"] == 1
    kv3 = KnowledgeV2(tmp_path / "s_mine")
    assert "vectorising half2" in kv3.already_tried("kb_problem__relu__r00", "s0")
    kv4 = KnowledgeV2(tmp_path / "s_other")
    assert kv4.already_tried("kb_problem__relu__r00", "s0") == ""


def test_concurrent_merges_serialise(store, tmp_path):
    """Four merges started at once through the CLI (as the farm runs them): all land, one after another."""
    import os
    import subprocess
    import sys
    procs = []
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for i in range(4):
        kv, seed, kdir = start_run(store, tmp_path, f"r{i}")
        outcome(kv, tech(kv), 0.1 * (i + 1), problem=f"kb_problem__{i}__r00")
        kv.db.conn.close()
        procs.append(subprocess.Popen([sys.executable, "-m", "kernelblaster.knowledge.v2.merge", "--store",
                                       str(store.root), "merge", "--run", str(kdir), "--base", str(seed),
                                       "--problem", f"r{i}"], env=env, stdout=subprocess.PIPE, text=True))
    versions = [json.loads(p.communicate(timeout=120)[0])["version"] for p in procs]
    assert sorted(versions) == ["v0001", "v0002", "v0003", "v0004"]
    assert gk(store, "SELECT COUNT(*) FROM gk_outcomes")[0][0] == 4       # no lost update
    assert gk(store, "SELECT COUNT(*) FROM gk_ledger")[0][0] == 4
    assert all(v == "ok" for v in store.verify().values())


def test_corrupt_run_leaves_store_untouched(store, tmp_path):
    kv, seed, kdir = start_run(store, tmp_path, "a")
    outcome(kv, tech(kv), 0.1)
    merge(store, kdir, seed, "a")
    before = (store.current_id(), store.versions())
    kv2, seed2, kdir2 = start_run(store, tmp_path, "b")
    del kv2
    db = kdir2 / "knowledge.db"
    for sfx in ("-wal", "-shm"):
        Path(str(db) + sfx).unlink(missing_ok=True)
    db.write_bytes(db.read_bytes()[:5000])               # truncated: a partial copy from a dead node
    with pytest.raises(Exception):
        merge(store, kdir2, seed2, "b")
    assert (store.current_id(), store.versions()) == before
    assert not list((store.root / "tmp").glob("*.db"))


def test_rollback_then_remerge(store, tmp_path):
    kv, seed, kdir = start_run(store, tmp_path, "a")
    outcome(kv, tech(kv), 0.1)
    v1 = merge(store, kdir, seed, "a")["version"]
    kv2, seed2, kdir2 = start_run(store, tmp_path, "b")
    outcome(kv2, tech(kv2), 0.1)
    v2 = merge(store, kdir2, seed2, "b")["version"]
    store.rollback(v1, reason="test")
    assert store.current_id() == v1
    r = merge(store, kdir2, seed2, "b")                  # v1's ledger does not have b: merged again
    assert r["status"] == "merged" and r["parent"] == v1 and r["version"] != v2


def test_store_promotes_hypothesis_across_problems(store, tmp_path):
    for name in ("a", "b"):
        kv, seed, kdir = start_run(store, tmp_path, name)
        h = Hypothesis(name="warp_specialised_epilogue", mechanism="overlap the epilogue stores with the next tile's "
                       "mma by giving them to a dedicated producer warp", source="gap",
                       how="split the CTA into a producer warp and consumer warps", target_key="cuda:sm_89",
                       id="h_" + name * 10)
        kv.hypotheses.add(h)
        for k in range(2):
            oid = outcome(kv, tech(kv), 0.05, problem=f"kb_problem__{name}{k}__r00")
            kv.hypotheses.record_attempt(h.id, outcome_id=oid, problem_id=f"kb_problem__{name}{k}__r00",
                                         target_key="cuda:sm_89", reward=0.04 + 0.01 * k, status="improved",
                                         mechanism="confirmed", measured=True)
        merge(store, kdir, seed, name)
    hs = gk(store, "SELECT status, data FROM gk_hypotheses")
    assert len(hs) == 1                                  # the second run's copy folded into the first
    d = json.loads(hs[0][1])
    assert len(d["attempts"]) == 4 and hs[0][0] == "promoted"
    assert gk(store, "SELECT COUNT(*) FROM gk_techniques WHERE id='warp_specialised_epilogue'")[0][0] == 1
    m = materialize(store, P89, tmp_path / "s3")
    kv3 = KnowledgeV2(tmp_path / "s3")
    assert "warp_specialised_epilogue" in kv3.ontology.techniques and m["counts"]["techniques"] == 1


def test_seed_claims_reach_the_prompt_and_caps(store, tmp_path):
    kv, seed, kdir = start_run(store, tmp_path, "a")
    t = tech(kv)
    ids = [outcome(kv, t, 0.2) for _ in range(3)]
    for k in range(5):
        kv.claims.add(Claim(kind="applicability", text=f"lesson number {k} on gemm tiles {k * 16}", arch="sm_89",
                            op_class="gemm", data={"effect": "helps"}), ids[: 2 + (k % 2)])
    kv.claims.save()
    merge(store, kdir, seed, "a")
    materialize(store, P89, tmp_path / "s2", policy=Policy(seed_max_claims=3))
    kv2 = KnowledgeV2(tmp_path / "s2")
    learned = [c for c in kv2.claims.claims.values() if c.text.startswith("lesson number")]
    assert sorted(c.support for c in learned) == [2, 3, 3]                # the best ranked first
    cls = kv2.classify({}, target_key="cuda:sm_89", op_class="gemm")
    text = kv2.retrieve(target_key="cuda:sm_89", op_class="gemm", dtype="fp16", cls=cls, vec={})
    assert "lesson number" in text


def test_part_key_and_rendered_predicates():
    assert part_key("89", "fp16") == "cuda:sm_89|fp16" and part_key("cuda:sm_100", None) == "cuda:sm_100|fp32"
    for p in ({"bw_util": "<", "v": 30}, {"all": [{"launch_frac": "<", "v": 0.1}, {"not": {"occupancy": ">=", "v": 50}}]},
              {"any": [{"op_class": "in", "v": ["conv", "gemm"]}, {"l2_hit": "<", "ref": "l1_hit"}]}):
        assert parse_rendered_predicate(P.render(p)) == p


def test_fresh_claims_beat_old_well_supported_ones_into_a_capped_seed(store, tmp_path):
    kv, seed, kdir = start_run(store, tmp_path, "old")
    t = tech(kv)
    ids = [outcome(kv, t, 0.2) for _ in range(6)]
    for k in range(6):                                   # well supported, older
        kv.claims.add(Claim(kind="applicability", text=f"old lesson {k} about gemm tile {k * 32}", arch="sm_89",
                            op_class="gemm", data={"effect": "helps"}), ids)
    kv.claims.save()
    merge(store, kdir, seed, "old")
    for i in range(3):                                   # merges in between age them
        kv_, s_, k_ = start_run(store, tmp_path, f"mid{i}")
        outcome(kv_, tech(kv_), 0.1, problem=f"kb_problem__m{i}__r00")
        merge(store, k_, s_, f"mid{i}")
    kv2, seed2, kdir2 = start_run(store, tmp_path, "new")
    ids2 = [outcome(kv2, t, 0.2, problem="kb_problem__n__r00") for _ in range(2)]
    c, _ = kv2.claims.add(Claim(kind="applicability", text="new lesson on relu launch overhead", arch="sm_89",
                                op_class="elementwise", data={"effect": "no_effect"}), ids2)
    kv2.claims.save()
    merge(store, kdir2, seed2, "new")
    pol = Policy(seed_max_claims=3, seed_fresh_merges=1, seed_fresh_max=1)
    materialize(store, P89, tmp_path / "s", policy=pol)
    texts = [x.text for x in KnowledgeV2(tmp_path / "s").claims.claims.values() if "lesson" in x.text]
    assert "new lesson on relu launch overhead" in texts and len(texts) == 3


def test_remerge_under_another_uid_does_not_double_count_stats(store, tmp_path):
    kv, seed, kdir = start_run(store, tmp_path, "a")
    t = tech(kv)
    outcome(kv, t, 0.2)
    merge(store, kdir, seed, "a")
    n = stat_n(store, P89, t)
    r = merge(store, kdir, seed, "a", uid="another-uid")         # e.g. a bootstrapped run merged again by hand
    assert r["status"] == "merged" and r["outcomes"] == 0 and "skipped" in r["stats"]
    assert stat_n(store, P89, t) == n
