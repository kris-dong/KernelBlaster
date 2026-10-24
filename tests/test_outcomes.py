import pytest

from kernelblaster.knowledge.outcomes import OutcomeLog, OutcomeRecord, classify_speedup


def test_log_roundtrip_and_cursor(tmp_path):
    log = OutcomeLog(tmp_path / "o.jsonl")
    a = log.append(OutcomeRecord(technique="t1", status="improved", speedup=1.3, slot="s0"))
    log.append(OutcomeRecord(technique="t2", status="compile_error", error="x" * 5000))
    recs = list(log)
    assert [r.technique for r in recs] == ["t1", "t2"] and recs[0].id == a.id
    assert len(recs[1].error) < 2100
    new, cur = log.since(1)
    assert [r.technique for r in new] == ["t2"] and cur == 2
    assert log.get([a.id])[0].slot == "s0"


def test_status_validation_and_classify():
    with pytest.raises(ValueError):
        OutcomeRecord(technique="t", status="great")
    assert classify_speedup(1.10) == "improved"
    assert classify_speedup(0.90) == "regressed"
    assert classify_speedup(1.01) == "neutral"
