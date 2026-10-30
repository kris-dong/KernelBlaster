"""Halfway sync, KernelBlaster side (run_multislot ``--pause-spec``; agents/opt_multislot.py ``pause_at``): GPU-free.

A run with a pause spec parks every named problem once all its rollouts completed ceil(steps*frac) steps, flushes the
journals, writes OUT/paused.json and exits 77; the same command without the flag continues with each rollout's OWN
state, the composite, the counters and the LLM journal restored (the finding the halfway sync rests on): the result
is identical to an uninterrupted run and no journaled LLM response is paid twice.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from test_resume import FIX, PLUGIN, PROB, ROOT, _cmd, _digest, _env, _lines


def _spec(tmp, frac=0.5, problems=(PROB,)):
    f = tmp / "pause_spec.json"
    f.write_text(json.dumps({"frac": frac, "problems": list(problems)}))
    return f


def _go(tmp, out, *extra, tag="x", rollouts=1, steps=6, probs=(FIX,)):
    return subprocess.run(_cmd(probs, out, *extra, rollouts=rollouts, steps=steps), env=_env(tmp, tag=tag),
                          capture_output=True, text=True, timeout=300)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pref")
    out = tmp / "out"
    r = _go(tmp, out, tag="ref")
    assert r.returncode == 0, r.stderr[-3000:]
    return _digest(out), len(_lines(tmp / "llm_ref.log"))


def test_pause_then_resume_equals_uninterrupted(tmp_path, reference):
    ref, ref_calls = reference
    out = tmp_path / "out"
    r1 = _go(tmp_path, out, "--pause-spec", str(_spec(tmp_path)))
    assert r1.returncode == 77, r1.stderr[-3000:]
    assert not (out / PROB / "result.json").exists() and not (out / "summary.json").exists()
    pj = json.loads((out / "paused.json").read_text())
    assert pj["problems"][PROB]["pause_at"] == 3 and pj["problems"][PROB]["steps_done"] == 3
    st = json.loads((out / PROB / "journal" / "state.json").read_text())
    assert st["phase"] == "rollouts" and st["rollout_state"]["0"]["next_step"] == 3     # the rollout's own state is saved
    calls1 = len(_lines(tmp_path / "llm_x.log"))
    r2 = _go(tmp_path, out)                                                          # no flag: continues
    assert r2.returncode == 0, r2.stderr[-3000:]
    assert _digest(out) == ref
    # no journaled / completed step is paid again: the two segments together call the LLM exactly as one run
    assert len(_lines(tmp_path / "llm_x.log")) == ref_calls
    assert calls1 < ref_calls
    res = json.loads((out / PROB / "result.json").read_text())
    assert res["resume"]["segments"] == 2
    assert res["resume"]["verify"] == {"skipped": "own pause, same host"}      # nothing changed: no re-measurement


def test_two_rollouts_keep_their_own_state(tmp_path):
    out = tmp_path / "out"
    r1 = _go(tmp_path, out, "--pause-spec", str(_spec(tmp_path)), rollouts=2, steps=4)
    assert r1.returncode == 77, r1.stderr[-3000:]
    st = json.loads((out / PROB / "journal" / "state.json").read_text())
    assert {k: v["next_step"] for k, v in st["rollout_state"].items()} == {"0": 2, "1": 2}
    assert len(st["steps_log"]) == 4
    cur0 = st["rollout_state"]["0"]["cur"]
    r2 = _go(tmp_path, out, rollouts=2, steps=4)
    assert r2.returncode == 0, r2.stderr[-3000:]
    res = json.loads((out / PROB / "result.json").read_text())
    assert len(res["steps"]) == 8 and sorted({(s["rollout"], s["step"]) for s in res["steps"]}) == \
        sorted((r, s) for r in range(2) for s in range(4))
    assert cur0                                                                       # saved, per rollout


def test_not_named_or_too_short_runs_through(tmp_path):
    out = tmp_path / "o1"
    assert _go(tmp_path, out, "--pause-spec", str(_spec(tmp_path, problems=("other",)))).returncode == 0
    out = tmp_path / "o2"
    assert _go(tmp_path, out, "--pause-spec", str(_spec(tmp_path)), steps=1).returncode == 0     # pause_at == steps
    assert not (out / "paused.json").exists()


def test_killed_while_paused_resumes(tmp_path, reference):
    ref, ref_calls = reference
    out = tmp_path / "out"
    assert _go(tmp_path, out, "--pause-spec", str(_spec(tmp_path))).returncode == 77
    # the pause was re-entered after a crash (the controller restarts the half): parks again at once, no LLM call
    n = len(_lines(tmp_path / "llm_x.log"))
    assert _go(tmp_path, out, "--pause-spec", str(_spec(tmp_path))).returncode == 77
    assert len(_lines(tmp_path / "llm_x.log")) == n
    assert _go(tmp_path, out).returncode == 0
    assert _digest(out) == ref and len(_lines(tmp_path / "llm_x.log")) == ref_calls


def test_pause_flag_is_not_part_of_the_run_config(tmp_path):
    out = tmp_path / "out"
    assert _go(tmp_path, out, "--pause-spec", str(_spec(tmp_path))).returncode == 77
    run = json.loads((out / "run.json").read_text())
    assert "pause" not in json.dumps(run["config"])
    assert run["segments"][-1]["status"] == "paused"


def test_patience_counters_survive_the_pause(tmp_path):
    """E7 patience is recomputed from the committed steps: a run that parks at step 3 and continues stops at the same
    commit as an uninterrupted one (no step lost, none repeated)."""
    plugin = Path(__file__).parent / "fixtures" / "patience_plugin.py"

    def run(out, *extra):
        cmd = [sys.executable, str(ROOT / "scripts" / "run_multislot.py"), str(FIX), "--out", str(out), "--model", "stub:flat",
               "--plugin", str(plugin), "--evaluator", "plugin", "--rollouts", "1", "--steps", "8", "--stop-grace", "30",
               "--trace", "off", "--stop-patience", "5", *extra]
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "KB_TEST_LLM_LOG": str(tmp_path / f"{out.name}.llm"),
               "KB_TEST_WIN_CALLS": "", "KB_TEST_SLOW": "0", "KB_TELEMETRY": "0"}
        return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)

    ref = tmp_path / "ref"
    assert run(ref).returncode == 0
    r = json.loads((ref / PROB / "result.json").read_text())
    n_ref = len(r["steps"])
    assert r["efficiency"]["stopped"] == "patience" and 0 < n_ref < 8
    pause_at = 2                                                  # 4 steps planned x 0.5 would be 4: use frac to park at 2
    spec = _spec(tmp_path, frac=0.25)                             # ceil(8 * 0.25) = 2
    out = tmp_path / "p"
    assert run(out, "--pause-spec", str(spec)).returncode == 77
    assert json.loads((out / "paused.json").read_text())["problems"][PROB]["pause_at"] == pause_at
    assert run(out).returncode == 0
    r2 = json.loads((out / PROB / "result.json").read_text())
    assert len(r2["steps"]) == n_ref and r2["efficiency"]["stopped"] == "patience"
    assert r2["efficiency"].get("trigger_commit") == r["efficiency"].get("trigger_commit")
    assert len(_lines(tmp_path / "p.llm")) == len(_lines(tmp_path / "ref.llm"))


def test_resume_after_a_pause_can_be_forced_to_verify_and_a_crash_resume_always_does(tmp_path, reference):
    ref, ref_calls = reference
    out = tmp_path / "out"
    assert _go(tmp_path, out, "--pause-spec", str(_spec(tmp_path))).returncode == 77
    env = _env(tmp_path)
    env["KB_PAUSE_VERIFY"] = "1"
    r = subprocess.run(_cmd([FIX], out, rollouts=1, steps=6), env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads((out / PROB / "result.json").read_text())
    assert "composite_ok" in res["resume"]["verify"]                          # verified as after a crash
    assert _digest(out) == ref
