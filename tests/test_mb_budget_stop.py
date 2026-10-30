"""ModelBlaster graph-search hooks in KernelBlaster (GPU-free): a problem's manifest ``mb_budget``
(per-problem rollouts / steps / stop_patience, run_multislot ``_budget``) and the external stop file
``<out>/<problem>/mb_stop.json`` (opt_multislot ``_external_stop``: no new steps). Both absent = unchanged."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "mb_lenet"
PLUGIN = Path(__file__).parent / "fixtures" / "patience_plugin.py"
PROB = "lenet__conv1__fc3"


def _run(tmp, prob, out, *extra, rollouts=4, steps=6):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "KB_TEST_LLM_LOG": str(tmp / f"llm_{out.name}.log"),
           "KB_TEST_WIN_CALLS": "", "KB_TEST_SLOW": "0", "KB_TELEMETRY": "0"}
    cmd = [sys.executable, str(ROOT / "scripts" / "run_multislot.py"), str(prob), "--out", str(out),
           "--model", "stub:flat", "--plugin", str(PLUGIN), "--evaluator", "plugin", "--rollouts", str(rollouts),
           "--steps", str(steps), "--stop-grace", "30", "--trace", "off", "--stop-patience", "0", *extra]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)


def _copy(tmp, budget=None):
    d = tmp / "prob"
    shutil.copytree(FIX, d)
    if budget is not None:
        m = json.loads((d / "manifest.json").read_text())
        m["mb_budget"] = budget
        (d / "manifest.json").write_text(json.dumps(m, indent=1))
    return d


def test_manifest_budget_overrides_the_flags(tmp_path):
    d = _copy(tmp_path, {"rollouts": 1, "steps": 2})
    out = tmp_path / "b"
    r = _run(tmp_path, d, out)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads((out / PROB / "result.json").read_text())
    assert len(res["steps"]) == 2
    assert res["efficiency"]["steps_planned"] == 2


def test_no_budget_is_unchanged(tmp_path):
    d = _copy(tmp_path)
    out = tmp_path / "n"
    r = _run(tmp_path, d, out, rollouts=2, steps=2)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads((out / PROB / "result.json").read_text())
    assert len(res["steps"]) == 4 and res["efficiency"].get("external_stop") is None


def test_external_stop_file_starts_no_step(tmp_path):
    d = _copy(tmp_path)
    out = tmp_path / "s"
    (out / PROB).mkdir(parents=True)
    (out / PROB / "mb_stop.json").write_text(json.dumps({"hint": "fuse:x", "why": "test"}))
    r = _run(tmp_path, d, out, rollouts=2, steps=3)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads((out / PROB / "result.json").read_text())
    assert res["steps"] == []
    e = res["efficiency"]
    assert e["stopped"] == "external" and e["external_stop"]["hint"] == "fuse:x"
    assert "external stop" in r.stderr
