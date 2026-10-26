# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A5 global-store additions: pools in snapshots (merge / seed / migration / GC), claim consolidation
(provenance, idempotence, evidence redirect, concurrency with the merge ledger), structural priors."""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from kernelblaster.knowledge.claims import Claim
from kernelblaster.knowledge.impl_store import ImplStore, shape_bucket, signature_sha
from kernelblaster.knowledge.v2.consolidate import consolidate_store
from kernelblaster.knowledge.v2.merge import GlobalStore, Policy, materialize
from kernelblaster.knowledge.v2.pool_store import build_from_runs
from kernelblaster.knowledge.v2.priors import StructPriors, parse_hint, rows_from_state

from test_knowledge_global import P89, gk, merge, outcome, start_run, tech

SIG = "void k(const half* x, half* y, int N)"


@pytest.fixture()
def store(tmp_path):
    return GlobalStore(tmp_path / "store").ensure()


def add_impl(kdir: Path, code: str, n: int, us: float, sp: float = 2.0, problem="kb_problem__x__r00"):
    s = ImplStore(kdir / "impl")
    shape = {"n": n}
    return s.add(code, target="cuda", arch="sm_89", op="relu_f16", op_class="elementwise",
                 signature_sha=signature_sha(SIG), shapes=[shape], buckets=[shape_bucket("elementwise", shape)],
                 slot_us=us, seed_us=us * sp, speedup_vs_seed=sp, problem_id=problem, dtype="fp16")


def tables(store) -> set:
    return {r[0] for r in gk(store, "SELECT name FROM sqlite_master WHERE type='table'")}


# ---------------------------------------------------------------- pools

def test_default_policy_writes_no_new_tables_or_files(store, tmp_path):
    kv, seed, kdir = start_run(store, tmp_path, "a")
    outcome(kv, tech(kv), 0.2)
    add_impl(kdir, "k_a", 1 << 20, 10.0)
    merge(store, kdir, seed, "a")
    assert not {"gk_pool", "gk_pool_obs", "gk_struct"} & tables(store)
    man = materialize(store, P89, tmp_path / "s2")
    assert "pool" not in man and "struct" not in man and not (tmp_path / "s2" / "impl" / "pool").exists()


def test_pool_merge_retention_seed_and_gc(store, tmp_path):
    kv, seed, kdir = start_run(store, tmp_path, "a")
    outcome(kv, tech(kv), 0.2)
    sha = signature_sha(SIG)
    r1 = add_impl(kdir, "fast on small", 1 << 16, 5.0)
    r2 = add_impl(kdir, "fast on big", 1 << 24, 50.0)
    add_impl(kdir, "slow everywhere", 1 << 16, 9.0)
    add_impl(kdir, "seed copy", 1 << 16, 20.0, sp=1.0 - 1e-6)            # below pool_min_speedup: not pooled
    st = ImplStore(kdir / "impl")
    st.observe("relu_f16", sha, "sm_89", "fp16", r1.code_sha, {"n": 1 << 24}, us=80.0)   # cross-shape cell
    pol = Policy(impl_pool=True, pool_max_kernels=2)
    rep = merge(store, kdir, seed, "a", policy=pol)
    assert rep["pool"]["records"] == 3 and rep["pool"]["observations"] == 1 and rep["pool"]["kept"] == 2
    codes = {json.loads(d)["code_sha"] for (d,) in gk(store, "SELECT data FROM gk_pool")}
    assert codes == {r1.code_sha, r2.code_sha}
    assert all((store.codedir / f"{c}.cu").exists() for c in codes)
    with store.lock():
        store.gc_code()
    assert all((store.codedir / f"{c}.cu").exists() for c in codes)   # GC keeps pool-only sources
    man = materialize(store, P89, tmp_path / "s2", policy=Policy(seed_pool=True))
    assert man["pool"]["kernels"] == 2
    p = ImplStore(tmp_path / "s2" / "impl").pool("relu_f16", sha, "sm_89", "fp16")
    assert set(p.kernels) == {r1.code_sha[:16], r2.code_sha[:16]}
    big = {"n": 1 << 24}
    assert p.best_for(big)["kernel"] == r2.code_sha[:16]                  # 50 < 80 on the big shape
    assert p.route([{"n": 1 << 16}, big])["sum_us"] == 55.0
    assert store.status()["parts"][P89]["pool"] == {"pools": 1, "kernels": 2}


def test_migration_builds_pools_and_old_snapshots_still_load(store, tmp_path):
    kv, seed, kdir = start_run(store, tmp_path, "a")
    outcome(kv, tech(kv), 0.2)
    add_impl(kdir, "winner", 1 << 20, 10.0)
    v1 = merge(store, kdir, seed, "a")["version"]                       # an old-format snapshot (no pool tables)
    other = tmp_path / "results" / "p9" / "work" / "knowledge"
    add_impl(other, "another run's kernel", 1 << 20, 12.0, problem="kb_problem__y__r00")
    rep = build_from_runs(store, P89, [other / "impl"], label="migrate")
    assert rep["status"] == "built" and store.current_id() != v1
    assert rep["from_gk_impl"] == 1 and rep["runs"]["records"] == 1
    assert {"gk_pool", "gk_pool_obs"} <= tables(store)
    probs = [json.loads(d).get("problems") for (d,) in gk(store, "SELECT data FROM gk_pool")]
    assert any(p and p[0].startswith("p9::") for p in probs)            # namespaced like gk_impl rows
    assert all(v == "ok" for v in store.verify().values())
    materialize(store, P89, tmp_path / "old_seed", snapshot=v1, policy=Policy(seed_pool=True))   # old one loads
    store.rollback(v1)
    assert store.current_id() == v1 and "gk_pool" not in tables(store)


# ---------------------------------------------------------------- consolidation

TXT_A = "On sm_89 skinny gemm with a long K, vectorized loads give large wins (M16 K4096: 28x) by widening loads"
TXT_B = "vectorized loads give large wins on skinny gemm with long K on sm_89 (M4,K4096: 12x): wider loads per lane"


def claim_run(store, tmp_path, name, text, n_out, *, seed_from=None):
    kv, seed, kdir = start_run(store, tmp_path, name)
    t = tech(kv)
    ids = [outcome(kv, t, 0.2, problem=f"kb_problem__{name}__r00") for _ in range(n_out)]
    c, _ = kv.claims.add(Claim(kind="applicability", text=text, arch="sm_89", op_class="gemm", technique=t,
                               data={"effect": "helps"}), ids)
    kv.claims.save()
    return kv, seed, kdir, c, t


def test_consolidation_merges_with_provenance_and_is_idempotent(store, tmp_path):
    _, sa, ka, ca, t = claim_run(store, tmp_path, "a", TXT_A, 3)
    merge(store, ka, sa, "a")
    # run c starts from this snapshot (before consolidation) and will add evidence to A's claim later
    kv_c, sc, kc = start_run(store, tmp_path, "c")
    _, sb, kb, cb, _ = claim_run(store, tmp_path, "b", TXT_B, 2)
    rb = merge(store, kb, sb, "b")
    assert rb["claims"]["folded_into_existing"] == 0                     # the merge's own rule does not fold them
    act = lambda: {r[0]: json.loads(r[1]) for r in gk(store, "SELECT id, data FROM gk_claims WHERE status='active' "
                                                             "AND json_extract(data,'$.technique')=?", (t,))}
    assert set(act()) == {ca.id, cb.id}
    rep = consolidate_store(store, P89)
    assert rep["status"] == "consolidated" and rep["parts"][P89]["merged_members"] == 1
    a = act()
    assert set(a) == {ca.id}                                             # higher support survives
    assert a[ca.id]["support"] == 5 and len(a[ca.id]["support_ids"]) == 5
    assert a[ca.id]["data"]["consolidated_from"] == [cb.id]
    prov = json.loads(gk(store, "SELECT prov FROM gk_claims WHERE id=?", (ca.id,))[0][0])
    assert prov["consolidated"][0]["id"] == cb.id and {"a", "b"} <= set(prov["problems"])
    mem = json.loads(gk(store, "SELECT data FROM gk_claims WHERE id=?", (cb.id,))[0][0])
    assert mem["status"] == "retired" and mem["data"]["gk_consolidated_into"] == ca.id
    v = store.current_id()
    again = consolidate_store(store, P89)
    assert again["status"] == "no_change" and store.current_id() == v
    # a run seeded before the consolidation: its new evidence on B's claim? (it never had it) -- A's claim
    # evidence lands normally; a seeded-before run holding B's claim passes evidence to the survivor
    kv_d, sd, kd = start_run(store, tmp_path, "d")                       # seeded AFTER: only the survivor
    assert cb.id not in kv_d.claims.claims and ca.id in kv_d.claims.claims
    new = outcome(kv_c, t, 0.1, problem="kb_problem__c__r00")
    kv_c.claims.support(ca.id, [new])
    kv_c.claims.save()
    merge(store, kc, sc, "c")
    assert act()[ca.id]["support"] == 6


def test_evidence_on_a_consolidated_member_goes_to_the_survivor(store, tmp_path):
    _, sa, ka, ca, t = claim_run(store, tmp_path, "a", TXT_A, 3)
    merge(store, ka, sa, "a")
    _, sb, kb, cb, _ = claim_run(store, tmp_path, "b", TXT_B, 2)
    merge(store, kb, sb, "b")
    kv_c, sc, kc = start_run(store, tmp_path, "c")                       # holds both claims (pre-consolidation)
    consolidate_store(store, P89)
    new = outcome(kv_c, t, 0.1, problem="kb_problem__c__r00")
    kv_c.claims.support(cb.id, [new])                                   # evidence on the member
    kv_c.claims.save()
    merge(store, kc, sc, "c")
    rows = {r[0]: (r[1], json.loads(r[2])) for r in gk(store, "SELECT id, status, data FROM gk_claims WHERE id IN "
                                                               "(?, ?)", (ca.id, cb.id))}
    assert rows[cb.id][0] == "retired" and rows[cb.id][1]["support"] == 2   # member unchanged
    assert rows[ca.id][0] == "active" and rows[ca.id][1]["support"] == 6    # 3 + 2 + 1
    assert rows[ca.id][1]["text"] == TXT_A                                 # survivor keeps its own text


def test_consolidation_serialises_with_concurrent_merges(store, tmp_path):
    _, sa, ka, ca, _ = claim_run(store, tmp_path, "a", TXT_A, 3)
    merge(store, ka, sa, "a")
    _, sb, kb, cb, _ = claim_run(store, tmp_path, "b", TXT_B, 2)
    merge(store, kb, sb, "b")
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))
    procs = []
    for i in range(3):
        kv, seed, kdir = start_run(store, tmp_path, f"r{i}")
        outcome(kv, tech(kv), 0.1, problem=f"kb_problem__{i}__r00")
        kv.db.conn.close()
        procs.append(subprocess.Popen([sys.executable, "-m", "kernelblaster.knowledge.v2.merge", "--store",
                                       str(store.root), "merge", "--run", str(kdir), "--base", str(seed),
                                       "--problem", f"r{i}"], env=env, stdout=subprocess.PIPE, text=True))
        if i == 1:
            procs.append(subprocess.Popen([sys.executable, "-m", "kernelblaster.knowledge.v2.merge", "--store",
                                           str(store.root), "consolidate"], env=env, stdout=subprocess.PIPE, text=True))
    outs = [json.loads(p.communicate(timeout=180)[0]) for p in procs]
    assert all("error" not in o for o in outs)
    vs = sorted(o["version"] for o in outs)
    assert vs == ["v0003", "v0004", "v0005", "v0006"]                    # one after another, none lost
    assert gk(store, "SELECT COUNT(*) FROM gk_ledger")[0][0] == 5
    assert gk(store, "SELECT status FROM gk_claims WHERE id=?", (cb.id,))[0][0] == "retired"
    assert all(v == "ok" for v in store.verify().values())


def test_contradicted_weak_claim_is_retired(store, tmp_path):
    from kernelblaster.knowledge.v2.consolidate import consolidate
    from kernelblaster.knowledge.v2.merge import _connect
    _, sa, ka, ca, t = claim_run(store, tmp_path, "a", TXT_A, 6)
    merge(store, ka, sa, "a")
    kv, sb, kb = start_run(store, tmp_path, "b")
    ids = [outcome(kv, t, -0.1, problem="kb_problem__b__r00")]
    cb, _ = kv.claims.add(Claim(kind="applicability", text=TXT_B.replace("give large wins", "hurt"), arch="sm_89",
                                op_class="gemm", technique=t, data={"effect": "hurts"}), ids)
    kv.claims.save()
    merge(store, kb, sb, "b")
    tmp = store._start_from(store.current_id())
    g = _connect(tmp)
    rep = consolidate(g, P89, seq=2)
    assert rep["contradictions_retired"] == 1
    assert json.loads(g.execute("SELECT data FROM gk_claims WHERE id=?", (cb.id,)).fetchone()[0])["status"] == "retired"
    g.close()


# ---------------------------------------------------------------- structural priors

STATE = {"rounds": [
    {"round": 0, "decisions": [
        {"hint": "fuse:fused__batchnorm2d_f16__relu_f16", "status": "accepted", "gain": 0.4, "ops": ["a", "b"]},
        {"hint": "fuse:fused__conv2d_f16__relu_f16", "status": "rejected", "gain": -0.2},
        {"hint": "fuse_attention:attention_f16", "status": "accepted", "gain": 0.6}]},
    {"round": 1, "decisions": [
        {"hint": "fuse:fused__conv2d_f16__relu_f16", "status": "accepted", "gain": 0.1},
        {"hint": "fuse:fused__par__gru_seq_f16__gru_seq_f16", "status": "rejected", "gain": -0.01},
        {"hint": "split_slot:conv2d_f16", "status": "deferred"}]}]}


def test_struct_rows_priors_and_store_roundtrip(store, tmp_path):
    assert parse_hint("fuse:fused__conv2d_f16__batchnorm2d_f16__add_f16__a")["classes"] == "conv+norm+ewise"
    assert parse_hint("fuse:fused__par__gru_seq_f16__gru_seq_f16")["kind"] == "fuse_siblings"
    rows = rows_from_state(STATE, part=P89, problem="m", run_uid="u")
    assert len(rows) == 5 and rows[3]["rounds_to_decide"] == 2                 # conv+relu seen in round 0
    sp = StructPriors(rows)
    p = sp.prior("fuse", ["conv2d_f16", "relu_f16"])
    assert (p.level, p.n, p.accepted) == ("exact", 2, 1) and 0 < p.lo < p.rate < p.hi <= 1
    assert p.alpha > 1 and p.beta > 1 and 0 <= p.sample() <= 1 and p.ucb() > p.rate
    q = sp.prior("fuse", ["conv2d_f16", "sigmoid_f16"])                         # unseen: backs off to classes
    assert q.level == "classes" and q.n == 2
    assert sp.prior("fuse", "fused__layernorm_f16__gelu_f16").level == "classes"         # = bn+relu's classes
    assert sp.prior("fuse", "fused__softmax_f16__dropout_f16").level == "kind"
    assert sp.prior("fold_layout", []).level == "all"
    assert sp.prior("fuse_attention", ["attention_f16"]).gain_mean == 0.6
    # store: merged with the run (opt-in), idempotent, shipped in a seed
    kv, seed, kdir = start_run(store, tmp_path, "a")
    outcome(kv, tech(kv), 0.2)
    (kdir.parent / "state.json").write_text(json.dumps(STATE))                  # the MB work dir next to knowledge/
    rep = merge(store, kdir, seed, "a", policy=Policy(struct_outcomes=True))
    assert rep["struct"] == {"decisions": 5, "added": 5}
    man = materialize(store, P89, tmp_path / "s2", policy=Policy(seed_struct=True))
    assert man["struct"]["rows"] == 5
    sp2 = StructPriors.load(tmp_path / "s2")
    assert sp2.prior("fuse", ["batchnorm2d", "relu"]).accepted == 1
    assert StructPriors.load(store, part=P89).prior("fuse", ["conv2d", "relu"]).n == 2
