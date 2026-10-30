# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mid-run sharing (knowledge/v2/refresh.py + merge.py): round-boundary merges of a running run and the
reverse import of other runs' knowledge, with stub runs (no LLM, no GPU)."""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from kernelblaster.knowledge.claims import Claim
from kernelblaster.knowledge.outcomes import OutcomeRecord
from kernelblaster.knowledge.v2 import KnowledgeV2
from kernelblaster.knowledge.v2.merge import GlobalStore, MergeError, RunInfo, materialize, merge_knowledge_dir
from kernelblaster.knowledge.v2.refresh import build_package, import_package, run_sync_point
from kernelblaster.knowledge.v2.stats import path as stat_path

P89 = "cuda:sm_89|fp16"
P89F = "cuda:sm_89|fp32"


class Run:
    """A run as the farm drives it: seed, live knowledge dir, the controller's last merged copy."""

    def __init__(self, store, tmp: Path, name: str, part: str = P89):
        self.store, self.tmp, self.name, self.part = store, tmp, name, part
        self.seed = tmp / f"{name}_seed"
        materialize(store, part, self.seed, problem=name, run_uid=f"uid-{name}")
        self.kdir = tmp / f"{name}_k"
        shutil.copytree(self.seed, self.kdir)
        self.merged = None                       # the copy merged last (incremental base)
        self.kv = KnowledgeV2(self.kdir)

    def close(self):                             # a round boundary: no process holds the db
        if self.kv is not None:
            self.kv.claims.save()
            self.kv.db.conn.close()
            self.kv = None

    def open(self):
        if self.kv is None:
            self.kv = KnowledgeV2(self.kdir)
        return self.kv

    def tech(self):
        return sorted(self.open().ontology.techniques)[0]

    def outcome(self, reward, *, op="gemm", problem="kb_problem__x__r00", tech=None):
        kv = self.open()
        chain = [sorted(s.id for s in kv.hierarchy.states.values() if s.parent is None)[0]]
        t = tech or self.tech()
        dt = self.part.split("|")[1]
        rec = OutcomeRecord(technique=t, status="improved", target="cuda", arch="sm_89", problem_id=problem, slot="s0",
                            op_class=op, dtype=dt, speedup=1.0 + reward)
        kv.outcomes.append(rec, {"item_id": t, "reward": reward, "chain": chain})
        kv.stats.update(t, stat_path(kv.registry, target_key="cuda:sm_89", op_class=op, chain=chain, dtype=dt), reward)
        return rec.id

    def claim(self, text, ids, **kw):
        kv = self.open()
        c, _ = kv.claims.add(Claim(kind="applicability", text=text, arch="sm_89", op_class="gemm",
                                   technique=self.tech(), data={"effect": "helps"}, **kw), ids)
        kv.claims.save()
        return c.id

    def merge(self, why="round"):
        """The controller's round merge: pull (copy) the live db, merge against the copy merged last."""
        self.close()
        pull = self.tmp / f"{self.name}_pull_{why}"
        shutil.rmtree(pull, ignore_errors=True)
        shutil.copytree(self.kdir, pull)
        base = self.merged or self.seed
        res = merge_knowledge_dir(self.store, pull, base,
                                  info=RunInfo(run_uid=f"uid-{self.name}", problem=self.name, attempt="1"))
        if res.get("status") == "merged":
            self.merged = self.tmp / f"{self.name}_merged_{self.store.current_id()}"
            shutil.copytree(pull, self.merged)
        return res

    def refresh(self, **kw):
        """Controller builds a package from the run's last pulled db; the node imports it at round start."""
        self.close()
        pkg = self.tmp / f"{self.name}_pkg_{self.store.current_id()}_{len(list(self.tmp.glob(self.name + '_pkg_*')))}"
        sp = run_sync_point(self.kdir / "knowledge.db")
        man = build_package(self.store, self.part, pkg, run_uid=f"uid-{self.name}", problem=self.name,
                            since_snapshot=sp.get("snapshot"), since_seq=sp.get("seq"))
        if man["status"] != "ready":
            return man, None
        return man, import_package(self.kdir, pkg, **kw)


def gk(store, sql, args=()):
    c = sqlite3.connect(store.path(store.current_id()))
    try:
        return c.execute(sql, args).fetchall()
    finally:
        c.close()


def stat_n(store, part, item):
    r = gk(store, "SELECT SUM(n) FROM gk_stats WHERE part=? AND item=? AND level='target' AND op LIKE 'gemm%'",
           (part, item))
    return r[0][0] or 0


def run_stat_n(run, item):
    run.close()
    c = sqlite3.connect(run.kdir / "knowledge.db")
    try:
        r = c.execute("SELECT SUM(n) FROM stats WHERE item=? AND level='target' AND op LIKE 'gemm%'", (item,)).fetchone()
    finally:
        c.close()
    return r[0] or 0


def run_claim(run, cid):
    run.close()
    c = sqlite3.connect(run.kdir / "knowledge.db")
    try:
        r = c.execute("SELECT data FROM claims WHERE id=?", (cid,)).fetchone()
    finally:
        c.close()
    return json.loads(r[0]) if r else None


@pytest.fixture()
def store(tmp_path):
    return GlobalStore(tmp_path / "store").ensure()


def test_round_merges_are_incremental_and_idempotent(store, tmp_path):
    a = Run(store, tmp_path, "a")
    t = a.tech()
    a.outcome(0.2)
    a.outcome(0.3)
    r1 = a.merge("r0")
    assert r1["status"] == "merged" and r1["outcomes"] == 2
    n1 = stat_n(store, P89, t)
    again = merge_knowledge_dir(store, a.merged, a.seed, info=RunInfo(run_uid="uid-a", problem="a"))
    assert again["status"] == "duplicate"                       # same state again: no new version
    with pytest.raises(MergeError):                              # against the seed after a merge: refused
        a.outcome(0.1)
        a.close()
        merge_knowledge_dir(store, a.kdir, a.seed, info=RunInfo(run_uid="uid-a", problem="a"))
    r2 = a.merge("r1")                                          # incremental: only the new outcome
    assert r2["status"] == "merged" and r2["outcomes"] == 1
    assert stat_n(store, P89, t) == n1 + n1 / 2
    r3 = a.merge("final")                                       # the final merge of an unchanged run
    assert r3["status"] == "duplicate"


def test_refresh_imports_other_runs_and_merge_does_not_count_them_again(store, tmp_path):
    a, b = Run(store, tmp_path, "a"), Run(store, tmp_path, "b")
    t = a.tech()
    ids = [a.outcome(0.2) for _ in range(3)]
    ca = a.claim(f"{t} helps small gemm on sm_89 (run a)", ids[:2])
    a.merge("r0")
    n_a = stat_n(store, P89, t)
    assert n_a > 0
    man, rep = b.refresh()                                       # b's round 0 ended with everything reflected
    assert man["counts"]["claims"] == 1 and man["counts"]["stats"] > 0
    assert rep["status"] == "applied" and rep["counts"]["claims"] == 1 and rep["counts"]["outcomes"] >= 2
    got = run_claim(b, ca)
    assert got and got["support"] == 2
    assert run_stat_n(b, t) == n_a                               # a's 3 outcomes, exactly
    b.outcome(0.1)
    # imported outcomes are never reflected on by b
    b.close()
    c = sqlite3.connect(b.kdir / "knowledge.db")
    cm = json.loads(c.execute("SELECT value FROM meta WHERE key='claims_meta'").fetchone()[0])
    n_out = c.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
    c.close()
    assert cm["reflect_cursor"] == n_out - 1                     # b's own new record is still to be reflected
    b.outcome(0.4)
    r = b.merge("r1")
    assert r["status"] == "merged" and r["outcomes"] == 2            # b's own two only, not a's three
    assert r.get("refresh_base_adjust", {}).get("rids") == 1
    assert stat_n(store, P89, t) == n_a + 2 * n_a / 3               # a (3) + b (2): nothing twice
    row = json.loads(gk(store, "SELECT data FROM gk_claims WHERE id=?", (ca,))[0][0])
    assert row["support"] == 2                                       # b did not re-add a's evidence


def test_a_run_never_imports_its_own_contributions(store, tmp_path):
    a, b = Run(store, tmp_path, "a"), Run(store, tmp_path, "b")
    t = a.tech()
    ida = [a.outcome(0.2) for _ in range(2)]
    ca = a.claim(f"{t} helps (run a)", ida)
    a.merge("r0")                                                    # v0001: a's round 0
    man, rep = a.refresh()                                           # only a's own merge since its seed
    assert man["counts"]["claims"] == 0 and man["counts"]["stats"] == 0 and man["counts"]["outcomes"] == 0
    assert man["counts"]["evidence"] == 0 and man["counts"]["impl"] == 0
    idb = [b.outcome(0.3) for _ in range(2)]
    b.close()
    b.open()
    kvb = b.open()
    # b adds evidence to a claim it does not have: create its own one instead
    cb = b.claim(f"{t} helps large gemm too (run b)", idb)
    b.merge("r0")                                                    # v0002: b
    a.outcome(0.1)
    man, rep = a.refresh()
    assert rep["status"] == "applied"
    assert rep["counts"]["claims"] == 1 and run_claim(a, cb) is not None
    assert run_claim(a, ca)["support"] == 2                          # its own claim did not come back as new evidence
    r = a.merge("r1")
    n_total = stat_n(store, P89, t)
    # a: 3 outcomes, b: 2 -> per-level rows add up exactly once each
    per = n_total / 5
    assert n_total == per * 5 and r["outcomes"] == 1
    assert stat_n(store, P89, t) == run_stat_n(a, t)                 # a now sees exactly the global sum
    del kvb


def test_overlapping_packages_add_statistics_once(store, tmp_path):
    a, b = Run(store, tmp_path, "a"), Run(store, tmp_path, "b")
    t = a.tech()
    a.outcome(0.2)
    a.merge("r0")
    sp = run_sync_point(b.kdir / "knowledge.db")
    b.close()
    p1 = tmp_path / "p1"
    build_package(store, P89, p1, run_uid="uid-b", problem="b", since_snapshot=sp["snapshot"], since_seq=sp["seq"])
    a.outcome(0.2)
    a.merge("r1")
    p2 = tmp_path / "p2"                                             # built from the same (older) sync point
    build_package(store, P89, p2, run_uid="uid-b", problem="b", since_snapshot=sp["snapshot"], since_seq=sp["seq"])
    assert import_package(b.kdir, p1)["status"] == "applied"
    assert import_package(b.kdir, p1)["status"] == "already_applied"
    r2 = import_package(b.kdir, p2)
    assert r2["status"] == "applied"
    assert run_stat_n(b, t) == stat_n(store, P89, t)                 # a's two merges, each exactly once


def test_scope_problem_local_and_partition_are_excluded(store, tmp_path):
    a = Run(store, tmp_path, "a")
    t = a.tech()
    i = a.outcome(0.2, problem="kb_problem__relu__r00")
    a.open().claims.add(Claim(kind="impl_note", text="the fused relu wins by vectorising half2",
                              problem_id="kb_problem__relu__r00", slot="s0", data={"note_type": "why_fast"}), [i])
    a.claim(f"{t} helps general gemm", [i, a.outcome(0.3)])
    f = Run(store, tmp_path, "f", part=P89F)                         # another partition
    f.claim(f"{t} helps fp32 gemm", [f.outcome(0.5), f.outcome(0.5)])
    b = Run(store, tmp_path, "b")
    a.merge("r0")
    f.merge("r0")
    man, rep = b.refresh()
    texts = [json.loads(r[0])["text"] for r in gk(store, "SELECT data FROM gk_claims")]
    assert any("vectorising" in x for x in texts) and any("fp32" in x for x in texts)
    pkg = json.loads(next(tmp_path.glob("b_pkg_*")).joinpath("refresh.json").read_text())
    got = [c["text"] for c in pkg["claims"]]
    assert got == [f"{t} helps general gemm"]                        # not the impl note, not the fp32 claim
    assert all(s[1][0] for s in pkg["stats"])
    assert rep["counts"]["claims"] == 1


def test_failed_import_rolls_back(store, tmp_path):
    a, b = Run(store, tmp_path, "a"), Run(store, tmp_path, "b")
    t = a.tech()
    a.claim(f"{t} helps (a)", [a.outcome(0.2), a.outcome(0.2)])
    a.merge("r0")
    b.close()
    db = b.kdir / "knowledge.db"
    before = {tb: sqlite3.connect(db).execute(f"SELECT COUNT(*) FROM {tb}").fetchone()[0]
              for tb in ("claims", "outcomes", "stats")}
    sp = run_sync_point(db)
    pkg = tmp_path / "pf"
    build_package(store, P89, pkg, run_uid="uid-b", problem="b", since_snapshot=sp["snapshot"], since_seq=sp["seq"])
    idx = b.kdir / "impl" / "index.jsonl"
    idx_before = idx.read_bytes() if idx.exists() else None
    for step in ("stats", "claims", "impl"):
        rep = import_package(b.kdir, pkg, fail_after=step)
        assert rep["status"] == "error"
        c = sqlite3.connect(db)
        after = {tb: c.execute(f"SELECT COUNT(*) FROM {tb}").fetchone()[0] for tb in ("claims", "outcomes", "stats")}
        st = c.execute("SELECT value FROM meta WHERE key='gk_refresh_state'").fetchone()
        c.close()
        assert after == before and st is None
        assert (idx.read_bytes() if idx.exists() else None) == idx_before
    assert import_package(b.kdir, pkg)["status"] == "applied"        # and then it applies cleanly
    assert run_stat_n(b, t) == stat_n(store, P89, t)


def test_refresh_refused_after_store_rollback(store, tmp_path):
    a, b, c = Run(store, tmp_path, "a"), Run(store, tmp_path, "b"), Run(store, tmp_path, "c")
    a.outcome(0.2)
    a.merge("r0")                                                    # v0001
    a.outcome(0.2)
    a.merge("r1")                                                    # v0002
    _, rep = b.refresh()
    assert rep["status"] == "applied" and rep["sync"]["snapshot"] == "v0002"
    store.rollback("v0001", reason="test")
    c.outcome(0.3)
    c.merge("r0")                                                    # v0003, parent v0001
    man, rep = b.refresh()
    assert man["status"] == "refused" and rep is None                # v0002 is not an ancestor of v0003


def test_partial_run_db_is_refused_and_outcomes_wait_for_the_cursor(store, tmp_path):
    a, b = Run(store, tmp_path, "a"), Run(store, tmp_path, "b")
    a.claim(f"{a.tech()} helps (a)", [a.outcome(0.2), a.outcome(0.2)])
    a.merge("r0")
    before = (store.current_id(), store.versions())
    a.close()
    bad = tmp_path / "a_partial"
    shutil.copytree(a.kdir, bad)
    db = bad / "knowledge.db"
    for sfx in ("-wal", "-shm"):
        Path(str(db) + sfx).unlink(missing_ok=True)
    db.write_bytes(db.read_bytes()[:4096])                            # a truncated pull from a dying node
    with pytest.raises(Exception):
        merge_knowledge_dir(store, bad, a.merged, info=RunInfo(run_uid="uid-a", problem="a"))
    assert (store.current_id(), store.versions()) == before
    # b has an unreflected outcome of its own: imported outcomes are skipped, claims still come
    b.outcome(0.1)
    b.close()
    c = sqlite3.connect(b.kdir / "knowledge.db")
    n = c.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
    cm = json.loads(c.execute("SELECT value FROM meta WHERE key='claims_meta'").fetchone()[0])
    cm["reflect_cursor"] = n - 1
    c.execute("UPDATE meta SET value=? WHERE key='claims_meta'", (json.dumps(cm),))
    c.commit()
    c.close()
    _, rep = b.refresh()
    assert rep["status"] == "applied" and rep["counts"]["outcomes"] == 0 and rep["counts"]["outcomes_skipped"] >= 2
    assert rep["counts"]["claims"] == 1
