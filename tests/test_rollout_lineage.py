"""Rollout acceptance and lineage (ModelBlaster search fixes item 4; l3_farm/analysis/search_fixes.md).

* ``rollout_decision`` / ``should_resync`` on scripted step outcomes (composite / local / lineage semantics).
* ``run_multislot.py`` end to end (GPU-free fake evaluator, scripted LLM): a rollout whose first step improves its own
  state but not the composite best (another rollout merged a bigger win meanwhile) -- composite: discarded, the
  rollout's next step starts from its old state; local: kept, the next step builds on it, nothing merged; lineage:
  kept, and after a non-improving step the lagging rollout restarts from the composite best (resync recorded).
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kernelblaster.agents.opt_multislot import (rollout_decision, rollout_mode, rollout_resync_factor,
                                                should_resync)
from kernelblaster.knowledge.outcomes import IMPROVED, NEUTRAL, REGRESSED

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "mb_lenet"
PLUGIN = Path(__file__).parent / "fixtures" / "lineage_plugin.py"
PROB = "lenet__conv1__fc3"            # the problem name in its manifest
ACC = 0.01


# ------------------------------------------------------------------ the rule on scripted outcomes

@pytest.mark.parametrize("gain,g,mode,want", [
    # own gain 30%, confirmed vs composite -60% (a lagging rollout found a real improvement)
    (0.30, -0.60, "composite", (-0.60, REGRESSED, False)),
    (0.30, -0.60, "local", (0.30, IMPROVED, True)),
    (0.30, -0.60, "lineage", (0.30, IMPROVED, True)),
    # a win that also beats the composite: every mode agrees
    (0.20, 0.10, "composite", (0.10, IMPROVED, True)),
    (0.20, 0.10, "local", (0.20, IMPROVED, True)),
    # neutral / regression of the own state: every mode agrees on the status
    (0.002, 0.002, "local", (0.002, NEUTRAL, True)),
    (-0.05, -0.05, "lineage", (-0.05, REGRESSED, False)),
    # unpaired, no gain measured: falls back to g
    (None, 0.0, "local", (0.0, NEUTRAL, True)),
    # a small own gain, a small composite loss: composite keeps it as neutral (> -acc/2), local improves
    (0.02, -0.004, "composite", (-0.004, NEUTRAL, True)),
    (0.02, -0.004, "local", (0.02, IMPROVED, True)),
])
def test_rollout_decision(gain, g, mode, want):
    assert rollout_decision(gain, g, ACC, mode) == want


def test_should_resync():
    assert should_resync(30.0, 10.0, 1.5, True)
    assert not should_resync(30.0, 10.0, 1.5, False)          # still improving: keep the lineage
    assert not should_resync(14.0, 10.0, 1.5, True)           # close enough
    assert not should_resync(30.0, 10.0, 0.0, True)           # off
    assert not should_resync(None, 10.0, 1.5, True)


def test_modes_from_env(monkeypatch):
    monkeypatch.delenv("KERNELBLASTER_MULTISLOT_ROLLOUT_ACCEPT", raising=False)
    monkeypatch.delenv("KERNELBLASTER_MULTISLOT_ROLLOUT_RESYNC", raising=False)
    assert rollout_mode() == "composite" and rollout_resync_factor() == 0.0
    monkeypatch.setenv("KERNELBLASTER_MULTISLOT_ROLLOUT_ACCEPT", "lineage")
    assert rollout_mode() == "lineage" and rollout_resync_factor() == 1.5
    monkeypatch.setenv("KERNELBLASTER_MULTISLOT_ROLLOUT_RESYNC", "2.5")
    assert rollout_resync_factor() == 2.5
    monkeypatch.setenv("KERNELBLASTER_MULTISLOT_ROLLOUT_ACCEPT", "bogus")
    assert rollout_mode() == "composite"


# ------------------------------------------------------------------ end to end (GPU-free)

def _one_slot(tmp):
    """mb_lenet with one slot (linear): every step works on the same slot, so call 2 cannot merge as an
    independent win on another slot."""
    d = tmp / "mb_lenet1"
    if not d.exists():
        shutil.copytree(FIX, d)
        mf = d / "manifest.json"
        m = json.loads(mf.read_text())
        m["slots"] = [s for s in m["slots"] if s["id"] == "linear"]
        m["ops"] = [o for o in m["ops"] if o["slot"] == "linear"]
        mf.write_text(json.dumps(m, indent=1))
    return d


def _run(tmp, name, mode, resync=None):
    out = tmp / name
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "KB_TEST_LLM_LOG": str(tmp / f"llm_{name}.log"),
           # call 1 (one rollout's step 0) wins big and commits first; call 2 (the other rollout's step 0, slow)
           # improves its own state by 20% but is behind the merged composite; every later call is flat
           "KB_TEST_SPEEDS": "0.5,0.8", "KB_TEST_SLEEPS": "0.02,0.6,0.02", "KB_TELEMETRY": "0", "KB_TEST_SLOW": "0"}
    for k in ("KB_FAULT", "KERNELBLASTER_MULTISLOT_ROLLOUT_ACCEPT", "KERNELBLASTER_MULTISLOT_ROLLOUT_RESYNC"):
        env.pop(k, None)
    if mode:
        env["KERNELBLASTER_MULTISLOT_ROLLOUT_ACCEPT"] = mode
    if resync is not None:
        env["KERNELBLASTER_MULTISLOT_ROLLOUT_RESYNC"] = str(resync)
    cmd = [sys.executable, str(ROOT / "scripts" / "run_multislot.py"), str(_one_slot(tmp)), "--out", str(out),
           "--model", "stub:lineage", "--plugin", str(PLUGIN), "--evaluator", "plugin", "--rollouts", "2",
           "--steps", "3", "--stop-grace", "30", "--trace", "off", "--efficiency-rules", "off"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads((out / PROB / "result.json").read_text())
    calls = [json.loads(x) for x in (tmp / f"llm_{name}.log").read_text().splitlines() if x.strip()]
    return res, calls, r.stderr


def _step_of_call2(res):
    # the step whose candidate carried call 2's marker: the slow rollout's step 0 (committed second)
    st = [s for s in res["steps"] if s["step"] == 0]
    assert len(st) == 2
    return st[1]


def test_composite_discards_a_lagging_rollouts_gain(tmp_path):
    res, calls, _ = _run(tmp_path, "comp", None)
    s = _step_of_call2(res)
    assert s["status"] == REGRESSED and (s["context"]["e2e"]["gain"] or 0) > 0.05
    assert not s["context"]["e2e"]["accept"]
    assert all("c2" not in c["cur"] for c in calls)             # nobody ever continued from call 2's candidate
    assert res["rollout_accept"]["mode"] == "composite" and res["rollout_accept"]["resyncs"] == []


def test_local_keeps_the_lineage_but_merges_only_composite_wins(tmp_path):
    res, calls, _ = _run(tmp_path, "local", "local")
    s = _step_of_call2(res)
    assert s["status"] == IMPROVED and s["context"]["e2e"]["rollout_accept"] == "local"
    assert not s["context"]["e2e"]["accept"]                    # not merged: it does not beat the composite
    assert any("c2" in c["cur"] for c in calls)                 # the rollout continued from it
    assert res["rollout_accept"]["resyncs"] == []


def test_lineage_resyncs_a_stalled_lagging_rollout(tmp_path):
    res, calls, err = _run(tmp_path, "lineage", "lineage", resync=1.01)
    s = _step_of_call2(res)
    assert s["status"] == IMPROVED
    assert any("c2" in c["cur"] for c in calls)                 # first it continues its own lineage ...
    rs = res["rollout_accept"]["resyncs"]
    assert res["rollout_accept"]["mode"] == "lineage" and res["rollout_accept"]["resync"] == 1.01
    assert rs and rs[0]["before_step"] == 2 and rs[0]["state_e2e"] > rs[0]["composite_e2e"] * 1.01
    last = calls[-1]                                            # ... then restarts from the composite best
    assert "c2" not in last["cur"] and "c1" in last["cur"]
    assert "lineage resync" in err
