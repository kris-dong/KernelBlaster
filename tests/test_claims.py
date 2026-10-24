import json

import pytest

from kernelblaster.knowledge import claims as C
from kernelblaster.knowledge.claims import Claim, ClaimStore
from kernelblaster.knowledge.outcomes import OutcomeRecord


def _claim(**kw):
    base = dict(kind="applicability", text="tiling helps when K is large", target="cuda",
                arch="sm_89", op_class="gemm", technique="tiling", data={"effect": "helps"})
    base.update(kw)
    return Claim(**base)


def test_claim_validation():
    with pytest.raises(ValueError):
        _claim(kind="opinion")
    with pytest.raises(ValueError):
        _claim(text="   ")
    assert len(_claim(text="x" * 1000).text) == C.MAX_TEXT_CHARS


def test_support_counts_distinct_ids_and_threshold(tmp_path):
    s = ClaimStore(tmp_path / "c.json")
    c, res = s.add(_claim(), ["r1"])
    assert res == "added" and c.support == 1 and not s.eligible(c)   # below MIN_SUPPORT=2
    s.support(c.id, ["r1", "r1"])          # same record again: no double count
    assert c.support == 1
    s.support(c.id, ["r2"])
    assert c.support == 2 and s.eligible(c)
    # a record that supports cannot also contradict
    assert s.contradict(c.id, ["r2"]) == 0
    s.contradict(c.id, ["r3", "r4"])
    assert c.contradict == 2 and not s.eligible(c)   # support must exceed contradictions


def test_contradicting_claims_coexist(tmp_path):
    s = ClaimStore(tmp_path / "c.json")
    a, _ = s.add(_claim(), ["r1", "r2"])
    b, res = s.add(_claim(text="tiling hurts when K is large", data={"effect": "hurts"}), ["r3", "r4"])
    assert res == "added" and a.id != b.id          # opposite polarity never merges
    assert {x.id for x in s.relevant(target="cuda", op_class="gemm")} == {a.id, b.id}
    out = s.render([a, b])
    assert "(+2/-0) [helps | tiling | gemm | sm_89]" in out and "[hurts" in out


def test_near_duplicate_merges_on_add_but_numbers_matter(tmp_path):
    s = ClaimStore(tmp_path / "c.json")
    a, _ = s.add(_claim(text="Tiling helps when K is large."), ["r1"])
    b, res = s.add(_claim(text="tiling helps when the K is large"), ["r2"])
    assert res == "merged" and b.id == a.id and a.support == 2
    p1, _ = s.add(_claim(kind="parameter", text="block 128 threads wins", data={"params": {"block": 128}}), ["r3"])
    p2, res2 = s.add(_claim(kind="parameter", text="block 256 threads wins", data={"params": {"block": 256}}), ["r4"])
    assert res2 == "added" and p1.id != p2.id
    assert not C.similar(C.normalise("tile 64 wins"), C.normalise("tile 128 wins"))


def test_dedupe_merges_existing(tmp_path):
    s = ClaimStore(tmp_path / "c.json")
    a = _claim(text="vectorised float4 loads help elementwise kernels", op_class="elementwise")
    b = _claim(text="vectorised float4 loads help the elementwise kernels!", op_class="elementwise")
    s.claims[a.id], s.claims[b.id] = a, b          # bypass add() to force duplicates
    a.add_support(["r1", "r2"]); b.add_support(["r3"])
    assert s.dedupe() == 1
    assert a.support == 3 and b.status == C.RETIRED and b.data["merged_into"] == a.id


def test_scoping_target_arch_op_class_and_local(tmp_path):
    s = ClaimStore(tmp_path / "c.json")
    cuda, _ = s.add(_claim(), ["r1", "r2"])
    rv, _ = s.add(_claim(target="riscv", arch="rvv-vlen256", text="tiling helps riscv"), ["r3", "r4"])
    general, _ = s.add(_claim(arch=None, op_class=None, text="tiling generally fine"), ["r5", "r6"])
    note, _ = s.add(_claim(kind="impl_note", text="warp-shuffle reduction is a dead end here",
                           problem_id="p1", slot="s0", data={"note_type": "dead_end"}), ["r7"])
    ids = lambda **q: {c.id for c in s.relevant(**q)}
    assert ids(target="riscv") == {rv.id}                       # no CUDA leak
    assert ids(target="cuda", arch="sm_80") == {general.id}     # arch mismatch excluded
    assert ids(target="cuda", arch="sm_89", op_class="gemm") == {cuda.id, general.id}
    assert ids(target="cuda", op_class="conv") == {general.id}
    assert note.id not in ids(target="cuda")                    # impl notes stay problem-local
    assert note.id in ids(target="cuda", problem_id="p1", slot="s0")
    assert note.id not in ids(target="cuda", problem_id="p1", slot="s1")
    # most specific first
    assert s.relevant(target="cuda", arch="sm_89", op_class="gemm", technique="tiling")[0].id == cuda.id


def test_decay_and_age_out(tmp_path):
    s = ClaimStore(tmp_path / "c.json")
    weak, _ = s.add(_claim(text="weak claim about unrolling", technique="unroll"), ["r1"])
    strong, _ = s.add(_claim(), ["r2", "r3"])
    r0 = strong.rank(s.epoch)
    for _ in range(C.AGE_OUT_EPOCHS):
        s.tick()
    assert weak.status == C.RETIRED              # ineligible + untouched -> retired
    assert strong.status == C.ACTIVE             # eligible claims only decay in rank
    assert strong.rank(s.epoch) < r0
    s.support(strong.id, ["r9"])                  # touching refreshes
    assert strong.last_epoch == s.epoch
    for _ in range(C.PURGE_RETIRED_EPOCHS):
        s.tick()
    assert weak.id not in s.claims


def test_persistence_roundtrip_and_tolerates_unknown_fields(tmp_path):
    p = tmp_path / "c.json"
    s = ClaimStore(p)
    c, _ = s.add(_claim(), ["r1", "r2"])
    s.observe(OutcomeRecord(technique="tiling", status="compile_error", problem_id="p1",
                            error="error: identifier TILE undefined"))
    s.meta["reflect_cursor"] = 7
    s.save()
    raw = json.loads(p.read_text())
    raw["claims"][0]["future_field"] = 1
    raw["claims"].append({"kind": "bogus", "text": "x"})       # dropped, not fatal
    p.write_text(json.dumps(raw))
    s2 = ClaimStore(p)
    assert s2.get(c.id).support == 2 and s2.meta["reflect_cursor"] == 7 and len(s2.claims) == 1
    assert "tiling: 1 tries" in s2.already_tried("p1")


def test_already_tried_tallies_and_notes(tmp_path):
    s = ClaimStore(tmp_path / "c.json")
    assert s.already_tried("p1", "s0") == ""
    for st, sp in [("compile_error", None), ("regressed", 0.8), ("regressed", 0.9)]:
        s.observe(OutcomeRecord(technique="tensor_cores", status=st, speedup=sp, problem_id="p1", slot="s0"))
    s.observe(OutcomeRecord(technique="tiling", status="improved", speedup=1.4, problem_id="p1", slot="s0"))
    s.observe(OutcomeRecord(technique="tiling", status="improved", speedup=2.0, problem_id="p1", slot="s1"))
    s.add(_claim(kind="impl_note", text="wmma path needs K%16==0; K=1000 here", problem_id="p1",
                 slot="s0", data={"note_type": "dead_end"}), ["r1"])
    out = s.already_tried("p1", "s0")
    assert "tensor_cores: 3 tries (regressed 2, compile_error 1); best 0.90x -- no improvement yet" in out
    assert "tiling: 1 tries (improved 1); best 1.40x" in out and "2.00x" not in out
    assert "wmma path needs" in out
