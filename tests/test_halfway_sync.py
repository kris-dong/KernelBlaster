# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Halfway sync (ModelBlaster ``--knowledge-global-halfway``): the knowledge store side of a mid-round exchange, with
stub runs (no LLM, no GPU). A halfway sync is the round-end machinery used once more in the middle of a round:
controller merge ``why="round<n>h"`` (incremental against the copy merged last), a refresh package of what other runs
merged, imported while no process has the db open; the round-end merge then must not count any of it twice."""
from __future__ import annotations

import json
import sqlite3

import pytest

from kernelblaster.knowledge.v2.merge import GlobalStore, RunInfo, merge_knowledge_dir
from kernelblaster.knowledge.impl_store import ImplStore

from test_knowledge_refresh import P89, Run, gk, run_claim, run_stat_n, stat_n
from test_knowledge_global_a5 import add_impl


@pytest.fixture()
def store(tmp_path):
    return GlobalStore(tmp_path / "store").ensure()


def _impl_ids(kdir):
    return {r.id for r in ImplStore(kdir / "impl").records()}


def test_halfway_then_round_end_merge_counts_everything_once(store, tmp_path):
    a, b = Run(store, tmp_path, "a"), Run(store, tmp_path, "b")
    t = a.tech()
    # b merged earlier (another run's round end): 2 outcomes, a claim, an impl winner
    idb = [b.outcome(0.3) for _ in range(2)]
    cb = b.claim(f"{t} helps gemm (run b)", idb)
    ib = add_impl(b.kdir, "k_from_b", 1 << 16, 5.0)
    assert b.merge("r0")["status"] == "merged"
    unit = stat_n(store, P89, t) / 2                                          # one outcome's weight over all levels
    # a: first half of its round -> halfway merge
    ida = [a.outcome(0.2) for _ in range(3)]
    ca = a.claim(f"{t} helps gemm (run a)", ida[:2])
    ia = add_impl(a.kdir, "k_from_a", 1 << 20, 7.0)
    h = a.merge("round1h")
    assert h["status"] == "merged" and h["outcomes"] == 3
    assert stat_n(store, P89, t) == pytest.approx(5 * unit)
    # the halfway read: b's contribution arrives, never a's own
    man, rep = a.refresh()
    assert rep["status"] == "applied" and rep["counts"]["claims"] == 1
    assert run_claim(a, cb) is not None and run_claim(a, ca)["support"] == 2
    assert ib.id in _impl_ids(a.kdir) and ia.id in _impl_ids(a.kdir)
    assert run_stat_n(a, t) == pytest.approx(5 * unit)                      # a's 3 + b's 2, once
    # second half: two more outcomes, then the ordinary round-end merge
    a.outcome(0.2)
    a.outcome(0.2)
    e = a.merge("round1")
    assert e["status"] == "merged" and e["outcomes"] == 2                    # only the second half's, not b's, not the first
    assert e.get("refresh_base_adjust", {}).get("rids") == 1
    assert stat_n(store, P89, t) == pytest.approx(7 * unit)                  # b 2 + a 3 + a 2: exactly once each
    assert run_stat_n(a, t) == pytest.approx(7 * unit)
    assert gk(store, "SELECT COUNT(*) FROM gk_impl")[0][0] == 2             # b's winner and a's, no copy of b's under a
    assert a.merge("final")["status"] == "duplicate"                         # the final merge of an unchanged run
    # a second halfway-style pass is idempotent too
    man, rep = a.refresh()
    assert man["status"] in ("up_to_date", "ready")
    if rep is not None:
        assert rep["status"] in ("applied", "already_applied")
        assert run_stat_n(a, t) == pytest.approx(7 * unit)


def test_two_runs_share_at_halfway_and_b_is_otherwise_unchanged(store, tmp_path):
    a, b = Run(store, tmp_path, "a"), Run(store, tmp_path, "b")
    t = a.tech()
    ida = [a.outcome(0.2) for _ in range(3)]
    ca = a.claim(f"{t} helps small gemm on sm_89 (run a)", ida[:2])
    ia = add_impl(a.kdir, "winner_of_a", 1 << 20, 4.0)
    a.merge("round1h")                                                       # run A reached halfway and merged
    idb = [b.outcome(0.1) for _ in range(2)]
    cb = b.claim(f"{t} helps (b's own)", idb)
    ib = add_impl(b.kdir, "winner_of_b", 1 << 16, 6.0)
    b.close()
    own = {"claim": run_claim(b, cb), "impl": [x.id for x in ImplStore(b.kdir / "impl").records()]}
    c = sqlite3.connect(b.kdir / "knowledge.db")
    out_before = {r[0] for r in c.execute("SELECT id FROM outcomes")}
    idx_before = (b.kdir / "impl" / "index.jsonl").read_text()
    c.close()
    man, rep = b.refresh()                                                   # B's halfway package
    assert rep["status"] == "applied"
    assert man["counts"]["claims"] == 1 and man["counts"]["impl"] == 1 and man["counts"]["stats"] > 0
    got = run_claim(b, ca)
    assert got and got["support"] == 2                                       # A's claim and its evidence
    assert ia.id in _impl_ids(b.kdir)                                        # A's impl winner is retrievable
    assert (b.kdir / "impl" / "index.jsonl").read_text().startswith(idx_before)   # B's own index lines untouched
    assert run_claim(b, cb) == own["claim"]                                  # B's own claim unchanged
    assert set(own["impl"]) <= _impl_ids(b.kdir)
    c = sqlite3.connect(b.kdir / "knowledge.db")
    out_after = {r[0] for r in c.execute("SELECT id FROM outcomes")}
    c.close()
    assert out_before <= out_after                                           # nothing of B's was dropped
    # B's later merge adds only B's own learning
    b.outcome(0.1)
    r = b.merge("round1")
    assert r["status"] == "merged" and r["outcomes"] == 3 and r.get("refresh_base_adjust", {}).get("rids") == 1
    assert gk(store, "SELECT COUNT(*) FROM gk_impl")[0][0] == 2


def test_halfway_with_nothing_new_is_a_no_op(store, tmp_path):
    a = Run(store, tmp_path, "a")
    a.outcome(0.2)
    a.merge("round1h")
    man, rep = a.refresh()                                                   # only its own merge since its seed
    assert man["counts"]["claims"] == 0 and man["counts"]["stats"] == 0 and man["counts"]["impl"] == 0
    assert a.merge("round1")["status"] in ("duplicate", "merged")
