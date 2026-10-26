"""KERNELBLASTER_SEED_MINED=0: a fresh knowledge.db holds technique definitions only (no mined outcomes / claims)."""
import os

from kernelblaster.knowledge.v2.knowledge import KnowledgeV2, seed_mined_enabled


def _n_outcomes(k):
    return k.db.conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]


def test_env_switch(tmp_path, monkeypatch):
    monkeypatch.delenv("KERNELBLASTER_SEED_MINED", raising=False)
    assert seed_mined_enabled()
    on = KnowledgeV2(tmp_path / "on")
    monkeypatch.setenv("KERNELBLASTER_SEED_MINED", "0")
    assert not seed_mined_enabled()
    off = KnowledgeV2(tmp_path / "off")
    assert off.db.get_meta("mined_seed") is None
    assert on.db.get_meta("mined_seed")
    assert len(off.ontology.techniques) == len(on.ontology.techniques) > 10
    assert _n_outcomes(off) == 0
