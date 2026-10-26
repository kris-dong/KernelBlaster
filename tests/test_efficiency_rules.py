"""E7 efficiency rules in KernelBlaster (X-129): the patience stop (``--stop-patience`` / ``--efficiency-rules``).

* ``patience_trigger`` on scripted step outcomes: commit order across parallel rollouts, ties, fix attempts,
  failures, improved-but-not-better steps, multi-slot baselines; and equivalence with the E7 replay's rule
  (``l3_farm/analysis/e7_data/e7_analyze.py: kb_kept``) on random trajectories.
* ``run_multislot.py`` end to end with a GPU-free fake evaluator: the stop, in-flight steps finishing and no
  new step started, result.json / summary.json telemetry, presets and overrides, the legacy (off) path, and
  resume (a journal's patience wins; a pre-X-129 out dir resumes with the rule off).
"""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from kernelblaster.agents.opt_multislot import StepLog, patience_trigger

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "mb_lenet"
PLUGIN = Path(__file__).parent / "fixtures" / "patience_plugin.py"
PROB = "lenet__conv1__fc3"


def S(slot, step, status, before, after, ro=0):
    return StepLog(rollout=ro, step=step, slot=slot, technique="t", status=status, slot_us_before=before,
                   slot_us_after=after, reward=0.0, fix_attempts=0, seconds=0.0)


# ------------------------------------------------------------------ the rule on scripted outcomes

def test_off_and_no_steps():
    assert patience_trigger([], 8) is None
    assert patience_trigger([S("a", 0, "neutral", 10, 10)] * 20, 0) is None


def test_consecutive_misses_in_commit_order():
    st = [S("a", 0, "improved", 10, 8, ro=0)] + [S("a", 0, "neutral", 10, 10, ro=r) for r in (1, 2, 3)] + \
         [S("a", 1, "regressed", 8, 9, ro=0), S("a", 1, "compile_error", 10, None, ro=1),
          S("a", 1, "wrong_output", 10, None, ro=2), S("a", 1, "runtime_error", 10, None, ro=3)]
    assert patience_trigger(st, 8) is None                         # 7 misses after the best
    assert patience_trigger(st, 7) == {"trigger_commit": 8, "misses": 7}
    st2 = st + [S("a", 2, "improved", 9, 7.9, ro=1)] + [S("a", 2, "neutral", 7.9, 7.9)] * 3
    assert patience_trigger(st2, 8) is None                         # the 9th commit was a new best: reset
    assert patience_trigger(st2, 3) == {"trigger_commit": 4, "misses": 3}   # first run of 3 misses
    assert patience_trigger(st2[8:], 3) == {"trigger_commit": 4, "misses": 3}


def test_ties_and_improved_but_not_better_are_misses():
    st = [S("a", 0, "improved", 10, 8)] + [S("a", 0, "improved", 10, 8, ro=1)] + \
         [S("a", 0, "improved", 10, 9, ro=2)] + [S("a", 1, "improved", 8, 8.5)]
    # a tie (8 == best 8) and an 'improved' step whose time is not below the running best do not reset
    assert patience_trigger(st, 3) == {"trigger_commit": 4, "misses": 3}


def test_improved_needs_a_time_and_the_improved_status():
    st = [S("a", 0, "improved", 10, None), S("a", 0, "neutral", 10, 5)]   # no time / not 'improved'
    assert patience_trigger(st, 2) == {"trigger_commit": 2, "misses": 2}


def test_a_step_with_fix_attempts_is_one_step():
    fixed_win = S("a", 0, "improved", 10, 6)
    fixed_win.fix_attempts = 2                                     # broken, fixed, faster: a new best
    failed = S("a", 0, "compile_error", 10, None, ro=1)
    failed.fix_attempts = 2                                        # still broken after its fixes: one miss
    assert patience_trigger([failed, fixed_win, failed], 2) is None
    assert patience_trigger([failed, failed], 2) == {"trigger_commit": 2, "misses": 2}


def test_slot_baselines_are_the_rollout_start_times():
    # two slots; the running best per slot starts at the slot's first step-0 'before' time (also when a later
    # step of the same slot committed first), so a step that beats only an intermediate state is no new best
    st = [S("b", 1, "improved", 20, 19, ro=0),                     # step 1 committed before b's step 0
          S("b", 0, "neutral", 18, 18, ro=1),                      # b's rollout-start time: 18
          S("a", 2, "improved", 10, 9, ro=2)]                      # a: no step 0 -> its first 'before' (10)
    assert patience_trigger(st, 2) == {"trigger_commit": 2, "misses": 2}   # 19 > 18: miss; then miss
    assert patience_trigger(st[:1] + st[2:], 2) is None


def _replay_kept(steps, patience):
    """e7_analyze.py kb_kept(..., patience) with the defaults (all rollouts / steps), on dict steps
    -> (kept steps, whether the rule fired)."""
    base = {}
    for s in steps:                                                 # e7_build.py: slot_base_us
        if s["step"] == 0 and s["slot"] not in base:
            base[s["slot"]] = s["before"]
    for s in steps:
        base.setdefault(s["slot"], s["before"])
    best, out, miss = dict(base), [], 0
    for s in steps:
        out.append(s)
        if s["status"] == "improved" and s["after"] is not None and s["after"] < best.get(s["slot"], 1e18):
            best[s["slot"]] = s["after"]
            miss = 0
        else:
            miss += 1
            if miss >= patience:
                return out, True
    return out, False


@pytest.mark.parametrize("seed", range(40))
def test_matches_the_e7_replay_rule(seed):
    rng = random.Random(seed)
    n_ro, n_st = rng.choice([(4, 4), (2, 8), (3, 5)])
    slots = rng.choice([["a"], ["a", "b", "c"]])
    cur = {(r, s): 100.0 for r in range(n_ro) for s in slots}
    steps, nxt = [], {r: 0 for r in range(n_ro)}
    while any(v < n_st for v in nxt.values()):                      # random commit interleaving
        r = rng.choice([x for x in nxt if nxt[x] < n_st])
        sl = rng.choice(slots)
        st = rng.choice(["improved", "improved", "neutral", "regressed", "compile_error", "wrong_output"])
        before = cur[(r, sl)]
        after = None if st in ("compile_error", "wrong_output") else round(before * rng.uniform(0.7, 1.1), 1)
        if after is not None and st != "regressed":
            cur[(r, sl)] = min(before, after)
        steps.append({"slot": sl, "step": nxt[r], "status": st, "before": before, "after": after, "ro": r})
        nxt[r] += 1
    logs = [S(s["slot"], s["step"], s["status"], s["before"], s["after"], s["ro"]) for s in steps]
    for p in (1, 2, 3, 4, 6, 8):
        kept, fired = _replay_kept(steps, p)
        t = patience_trigger(logs, p)
        assert (t is not None) == fired
        if fired:
            assert t["trigger_commit"] == len(kept)


# ------------------------------------------------------------------ run_multislot end to end (GPU-free)

def _cmd(out, *extra, rollouts=4, steps=6):
    return [sys.executable, str(ROOT / "scripts" / "run_multislot.py"), str(FIX), "--out", str(out),
            "--model", "stub:flat", "--plugin", str(PLUGIN), "--evaluator", "plugin",
            "--rollouts", str(rollouts), "--steps", str(steps), "--stop-grace", "30", "--trace", "off", *extra]


def _run(tmp, out, *extra, wins="", slow="0.02", **kw):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "KB_TEST_LLM_LOG": str(tmp / f"llm_{out.name}.log"),
           "KB_TEST_WIN_CALLS": wins, "KB_TEST_SLOW": slow, "KB_TELEMETRY": "0"}
    for k in ("KB_FAULT",):
        env.pop(k, None)
    return subprocess.run(_cmd(out, *extra, **kw), env=env, capture_output=True, text=True, timeout=300)


def _res(out):
    return json.loads((out / PROB / "result.json").read_text())


def _codegen_calls(tmp, out):
    p = tmp / f"llm_{out.name}.log"
    return sum(1 for l in p.read_text().splitlines() if l.strip()) if p.exists() else 0


def test_default_patience_stops_and_inflight_steps_finish(tmp_path):
    out = tmp_path / "on"
    r = _run(tmp_path, out, rollouts=4, steps=6)                    # default: --efficiency-rules on = patience 8
    assert r.returncode == 0, r.stderr[-3000:]
    res = _res(out)
    e = res["efficiency"]
    assert e["stop_patience"] == 8 and e["stopped"] == "patience" and e["trigger_commit"] == 8
    assert e["steps_planned"] == 24 and e["steps_committed"] == len(res["steps"])
    # in-flight steps of the other rollouts finish; none is started after the trigger
    assert 8 <= e["steps_committed"] <= 8 + 3 and e["steps_after_trigger"] == e["steps_committed"] - 8
    assert _codegen_calls(tmp_path, out) == e["steps_committed"]
    assert "efficiency: patience 8 reached at commit 8" in r.stderr
    summ = json.loads((out / "summary.json").read_text())
    row = next(x for x in summ if x.get("problem") == PROB)
    assert row["efficiency"]["stopped"] == "patience"
    run_eff = next(x["efficiency"] for x in summ if "efficiency" in x and "problem" not in x)
    assert run_eff["stop_patience"] == 8 and run_eff["rules"] == "on" and run_eff["patience_stopped"] == 1
    assert run_eff["steps_not_run"] == 24 - e["steps_committed"]


def test_new_bests_reset_the_count(tmp_path):
    out = tmp_path / "wins"
    r = _run(tmp_path, out, "--stop-patience", "3", rollouts=1, steps=10, wins="2,4")
    assert r.returncode == 0, r.stderr[-3000:]
    e = _res(out)["efficiency"]
    # rollouts=1: commits 1 (miss) 2 (win) 3 (miss) 4 (win) 5 6 7 (misses) -> stop after commit 7
    assert e["stopped"] == "patience" and e["trigger_commit"] == 7 and e["steps_committed"] == 7


@pytest.mark.parametrize("flags,patience", [(["--efficiency-rules", "off"], 0), (["--efficiency-rules", "safe"], 0),
                                            (["--efficiency-rules", "off", "--stop-patience", "5"], 5),
                                            (["--stop-patience", "0"], 0), (["--efficiency-rules", "on"], 8)])
def test_presets_and_overrides(tmp_path, flags, patience):
    out = tmp_path / "p"
    r = _run(tmp_path, out, *flags, rollouts=2, steps=6, slow="0")
    assert r.returncode == 0, r.stderr[-3000:]
    e = _res(out)["efficiency"]
    assert e["stop_patience"] == patience
    if patience == 0:                                               # legacy: every planned step runs
        assert e["stopped"] is None and e["steps_committed"] == 12
        cfg = json.loads((out / "run.json").read_text())["config"]
        assert "stop_patience" not in cfg                           # the legacy run config (resume-compatible)
    else:
        assert e["stopped"] == "patience"


def test_legacy_out_dir_resumes_with_the_rule_off(tmp_path):
    out = tmp_path / "legacy"
    # a run as before X-129: run.json config without stop_patience (rules off); killed before the end
    r = _run(tmp_path, out, "--efficiency-rules", "off", rollouts=2, steps=6, slow="0")
    assert r.returncode == 0, r.stderr[-3000:]
    run = json.loads((out / "run.json").read_text())
    assert "stop_patience" not in run["config"]
    st = json.loads((out / PROB / "journal" / "state.json").read_text())
    st.pop("stop_patience", None)
    st.pop("patience_stop", None)                                  # a journal written before the rule
    st["phase"] = "rollouts"
    (out / PROB / "journal" / "state.json").write_text(json.dumps(st))
    (out / PROB / "result.json").unlink()
    r2 = _run(tmp_path, out, rollouts=2, steps=6, slow="0")        # no flag: adopts the legacy setting
    assert r2.returncode == 0, r2.stderr[-3000:]
    assert "legacy" in r2.stderr and "patience off" in r2.stderr
    e = _res(out)["efficiency"]
    assert e["stop_patience"] == 0 and e["stopped"] is None
    # an explicit, different setting is a different configuration: refused like any other
    r3 = _run(tmp_path, out, "--stop-patience", "8", rollouts=2, steps=6, slow="0")
    assert r3.returncode == 3 and "stop_patience" in r3.stderr


def test_resume_keeps_the_recorded_patience(tmp_path):
    out = tmp_path / "rec"
    r = _run(tmp_path, out, "--stop-patience", "4", rollouts=2, steps=6, slow="0")
    assert r.returncode == 0, r.stderr[-3000:]
    assert _res(out)["efficiency"]["stopped"] == "patience"
    r2 = _run(tmp_path, out, rollouts=2, steps=6, slow="0")         # no flag: the recorded 4, done -> skipped
    assert r2.returncode == 0, r2.stderr[-3000:]
    assert "stop_patience 4" in r2.stderr or "patience 4" in r2.stderr
