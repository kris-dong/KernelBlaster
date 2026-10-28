"""Beam search (``run_multislot.py --search beam``): GPU-free, with the stub LLM and the fake
evaluator of tests/fixtures/beam_plugin.py (an end-to-end time per run, so the default e2e reward
decides). Plus the pure selection function."""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from kernelblaster.agents.opt_multislot import beam_key, beam_score, select_beam

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "mb_lenet"
PLUGIN = Path(__file__).parent / "fixtures" / "beam_plugin.py"
PROB = "lenet__conv1__fc3"
B, K, STEPS = 3, 2, 4
BEAM = ["--search", "beam", "--beam-width", str(B), "--beam-expand", str(K)]


def _run(tmp, out, *extra, fault=None, rollouts=1, steps=STEPS):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "KB_TEST_LLM_LOG": str(tmp / "llm.log"),
           "KB_TEST_EVAL_LOG": str(tmp / "eval.log")}
    env.pop("KB_FAULT", None)
    if fault:
        env["KB_FAULT"] = fault
    cmd = [sys.executable, str(ROOT / "scripts" / "run_multislot.py"), str(FIX), "--out", str(out),
           "--model", "stub:resume", "--plugin", str(PLUGIN), "--evaluator", "plugin",
           "--rollouts", str(rollouts), "--steps", str(steps), "--stop-grace", "30",
           "--efficiency-rules", "off", *extra]        # (E7 patience: its own test below)
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)


def _result(out):
    return json.loads((out / PROB / "result.json").read_text())


def _digest(out):
    r = _result(out)
    best = {p.name: p.read_text() for p in sorted((out / PROB / "best" / "slots").iterdir())}
    return {"steps": [(s["rollout"], s["step"], s["slot"], s["technique"], s["status"], s["slot_us_before"],
                       s["slot_us_after"], s["e2e_after"], s["context"].get("beam")) for s in r["steps"]],
            "beams": (r.get("search") or {}).get("beams"), "final": (r["final_ok"], r["final_e2e_us"]),
            "llm_calls": r["llm_calls"], "best": best}


@pytest.fixture(scope="module")
def beam_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("beam")
    out = tmp / "out"
    r = _run(tmp, out, *BEAM)
    assert r.returncode == 0, r.stderr[-3000:]
    return tmp, out, _digest(out), _result(out)


# ---- (a) independent mode is unchanged
def test_independent_default_unchanged(tmp_path):
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    assert _run(tmp_path, out_a, rollouts=2).returncode == 0
    assert _run(tmp_path, out_b, "--search", "independent", rollouts=2).returncode == 0
    ra, rb = _result(out_a), _result(out_b)
    assert "search" not in ra and "search" not in rb            # nothing beam-specific in the output
    assert _digest(out_a) == _digest(out_b)
    assert sorted((s["rollout"], s["step"]) for s in ra["steps"]) == [(i, j) for i in range(2) for j in range(STEPS)]
    assert all("beam" not in s["context"] for s in ra["steps"])
    cfg = json.loads((out_a / "run.json").read_text())["config"]
    assert not {"search", "beam_width", "beam_expand"} & set(cfg)    # a journal of an older run still matches


# ---- (b) width and monotonicity
def test_beam_width_and_never_worse(beam_run):
    _, _, _, r = beam_run
    beams = r["search"]["beams"]
    assert [b["step"] for b in beams] == list(range(STEPS))
    assert beams[0]["candidates"] == max(B * K, 1)
    assert all(b["candidates"] == b["parents"] * K for b in beams[1:])
    assert all(1 <= len(b["beam_e2e"]) <= B for b in beams)
    assert all(b["beam_e2e"] == sorted(b["beam_e2e"]) for b in beams)
    tops = [b["beam_e2e"][0] for b in beams]
    assert all(b <= a + 1e-9 for a, b in zip(tops, tops[1:]))        # the best state is never lost
    assert tops[0] <= r["baseline_e2e_us"]
    assert tops[-1] < r["baseline_e2e_us"]                           # the stub finds wins
    # the exported composite is at least as good as the baseline and (within the accept threshold) the best member
    assert r["final_ok"] and r["final_e2e_us"] < r["baseline_e2e_us"]
    assert r["final_e2e_us"] <= tops[-1] / (1 - 0.01) + 1e-6
    assert (r["steps"][0]["reward_basis"], r["reward"]) == ("e2e", "e2e")


# ---- (c) after step 0 every parent is a member of the previous beam
def test_parents_come_from_previous_beam(beam_run):
    _, _, _, r = beam_run
    beams = r["search"]["beams"]
    by_step = {}
    for s in r["steps"]:
        by_step.setdefault(s["step"], []).append(s)
    assert {s["context"]["beam"]["parent_e2e"] for s in by_step[0]} == {r["baseline_e2e_us"]}
    assert len(by_step[0]) == B * K
    for step in range(1, STEPS):
        prev = beams[step - 1]["beam_e2e"]
        assert len(by_step[step]) == len(prev) * K
        for s in by_step[step]:
            b = s["context"]["beam"]
            assert b["parent_e2e"] == pytest.approx(prev[b["parent"]], abs=1e-3)   # that member, not the seed
            assert s["rollout"] // K == b["parent"]
        assert {s["context"]["beam"]["parent"] for s in by_step[step]} == set(range(len(prev)))
        assert any(s["context"]["beam"]["parent_e2e"] < r["baseline_e2e_us"] for s in by_step[step])


# ---- (d) distinct techniques per parent
def test_distinct_techniques_per_parent(beam_run):
    _, _, _, r = beam_run
    groups = {}
    for s in r["steps"]:
        groups.setdefault((s["step"], s["context"]["beam"]["parent"]), []).append(s["technique"])
    assert len(groups) > B
    for (step, parent), techs in groups.items():
        assert len(set(techs)) == len(techs), (step, parent, techs)
    # all step-0 candidates share the baseline as parent: all different while the catalogue lasts
    assert len(set(groups[(0, 0)])) == len(groups[(0, 0)])


# ---- (e) a killed beam run resumes to the same result
@pytest.fixture(scope="module")
def reference(tmp_path_factory, beam_run):
    cache = {0: beam_run[2]}

    def get(seed):
        if seed not in cache:
            tmp = tmp_path_factory.mktemp(f"ref{seed}")
            r = _run(tmp, tmp / "out", *BEAM, "--seed", str(seed))
            assert r.returncode == 0, r.stderr[-3000:]
            cache[seed] = _digest(tmp / "out")
        return cache[seed]
    return get


# (fault, seed): seed 1 makes the best member beat the composite, so a promote run happens
FAULTS = [("eval:^r\\d+_s2_.*_f0:3", 0), ("save:11", 0), ("llm:15", 0), ("eval:_confirm:2", 0),
          ("eval:beam_s0_promote:1", 1), ("save:3", 1)]


@pytest.mark.parametrize("fault,seed", FAULTS, ids=[f"{f}-seed{s}" for f, s in FAULTS])
def test_kill_and_resume_matches_uninterrupted(tmp_path, reference, fault, seed):
    ref = reference(seed)
    out = tmp_path / "out"
    r1 = _run(tmp_path, out, *BEAM, "--seed", str(seed), fault=fault)
    assert r1.returncode == -signal.SIGKILL, (r1.returncode, r1.stderr[-2000:])
    r2 = _run(tmp_path, out, *BEAM, "--seed", str(seed))
    assert r2.returncode == 0, r2.stderr[-3000:]
    got = _digest(out)
    assert got["steps"] == ref["steps"]
    assert got["beams"] == ref["beams"]
    assert got["final"] == ref["final"] and got["best"] == ref["best"]
    assert _result(out)["resume"]["segments"] == 2


def test_beam_refuses_resume_with_other_search(tmp_path):
    out = tmp_path / "out"
    assert _run(tmp_path, out, *BEAM, fault="save:5").returncode == -signal.SIGKILL
    assert _run(tmp_path, out).returncode == 3                       # config mismatch (EXIT_CONFIG_MISMATCH)


# ---- the selection function
def _m(**sources):
    cur = {s: c for s, c in sources.items() if s != "e2e"}
    return {"cur": cur, "cur_us": {s: 1.0 for s in cur}, "meta": {"e2e": sources.get("e2e")}}


def test_select_beam_unit():
    prev = [_m(a="p0", e2e=10.0), _m(a="p1", e2e=12.0)]
    kids = [_m(a="c0", e2e=9.0), _m(a="c1", e2e=11.0), _m(a="c0", e2e=9.0), _m(a="p0", e2e=10.0)]   # duplicates
    nxt = select_beam(prev, kids, 3)
    assert [beam_score(m) for m in nxt] == [9.0, 10.0, 11.0]
    assert len({beam_key(m["cur"]) for m in nxt}) == len(nxt)
    # no child kept: the previous beam (and its best) survives
    assert select_beam(prev, [], 4) == sorted(prev, key=beam_score)
    # a tie keeps the child first
    assert select_beam([_m(a="p", e2e=5.0)], [_m(a="c", e2e=5.0)], 1)[0]["cur"] == {"a": "c"}
    # without an end-to-end time the slot times decide
    a = {"cur": {"s": "x"}, "cur_us": {"s": 3.0}, "meta": {}}
    b = {"cur": {"s": "y"}, "cur_us": {"s": 2.0}, "meta": {}}
    assert select_beam([a], [b], 1) == [b]


# ---- beam_plan (budget per step), refill, cap
def test_beam_plan_unit():
    from kernelblaster.agents.opt_multislot import beam_plan
    assert beam_plan(0, 1, 2, 2, 4) == [0, 0, 0, 0]                  # max(B*K, rollouts)
    assert beam_plan(0, 1, 2, 2, 6) == [0] * 6
    assert beam_plan(1, 2, 2, 2, 4) == [0, 0, 1, 1]
    assert beam_plan(1, 1, 2, 2, 4) == [0, 0]                        # a beam of one: only K candidates (default)
    assert beam_plan(1, 1, 2, 2, 4, refill=True) == [0, 0, 0, 0]     # refill: the budgeted B*K
    assert sorted(beam_plan(1, 3, 4, 2, 4, refill=True)) == [0, 0, 0, 1, 1, 1, 2, 2]   # 8 over 3, best first
    assert beam_plan(1, 2, 2, 2, 4, refill=True) == [0, 0, 1, 1]     # full beam: unchanged
    assert beam_plan(0, 1, 2, 2, 1, cap=1) == [0]                    # mb_budget rollouts=1 caps the beam
    assert beam_plan(1, 2, 2, 2, 1, cap=3) == [0, 0, 1]


# ---- --steps 2, refill, E7 patience
def test_beam_two_steps_and_budget(tmp_path):
    out = tmp_path / "out"
    r = _run(tmp_path, out, "--search", "beam", "--beam-width", "2", "--beam-expand", "2", rollouts=4, steps=2)
    assert r.returncode == 0, r.stderr[-3000:]
    res = _result(out)
    beams = res["search"]["beams"]
    assert [b["step"] for b in beams] == [0, 1] and beams[0]["candidates"] == 4
    assert beams[1]["candidates"] == beams[1]["parents"] * 2
    assert res["final_ok"] and len(res["steps"]) == sum(b["candidates"] for b in beams) <= 8   # 4 rollouts x 2 steps
    assert res["efficiency"]["steps_planned"] == 8
    assert all(b["llm_calls"] > 0 for b in beams) and beams[1]["llm_calls"] >= beams[0]["llm_calls"]


def test_beam_refill_keeps_candidates_per_step(tmp_path):
    out = tmp_path / "out"
    r = _run(tmp_path, out, *BEAM, "--beam-refill", "on")
    assert r.returncode == 0, r.stderr[-3000:]
    res = _result(out)
    assert res["search"]["beam_refill"] is True
    assert all(b["candidates"] == B * K for b in res["search"]["beams"])
    assert all(sum(b["per_parent"]) == B * K for b in res["search"]["beams"])


def test_beam_patience_stops_at_a_step_boundary(tmp_path):
    out = tmp_path / "out"
    r = _run(tmp_path, out, *BEAM, "--stop-patience", "4")
    assert r.returncode == 0, r.stderr[-3000:]
    res = _result(out)
    n = len(res["search"]["beams"])
    assert res["efficiency"]["stopped"] == "patience" and n < STEPS and res["final_ok"]
    assert len(res["steps"]) == sum(b["candidates"] for b in res["search"]["beams"])   # whole beam steps only


def test_kill_and_resume_with_refill(tmp_path_factory):
    args = (*BEAM, "--beam-refill", "on", "--stop-patience", "6")
    t1, t2 = tmp_path_factory.mktemp("rf1"), tmp_path_factory.mktemp("rf2")
    assert _run(t1, t1 / "out", *args).returncode == 0
    assert _run(t2, t2 / "out", *args, fault="eval:^r\\d+_s1_.*_f0:3").returncode == -signal.SIGKILL
    r = _run(t2, t2 / "out", *args)
    assert r.returncode == 0, r.stderr[-3000:]
    a, b = _digest(t1 / "out"), _digest(t2 / "out")
    assert a["steps"] == b["steps"] and a["beams"] == b["beams"] and a["final"] == b["final"]
