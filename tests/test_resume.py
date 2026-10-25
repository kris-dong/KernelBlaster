"""Resumable multi-slot runs: fault injection with SIGKILL at chosen and random
points, then resume (scripts/run_multislot.py; agents/journal.py).

GPU-free: a deterministic stub LLM and a fake evaluator (tests/fixtures/resume_plugin.py).
"""
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "mb_lenet"
PLUGIN = Path(__file__).parent / "fixtures" / "resume_plugin.py"
PROB = "lenet__conv1__fc3"


def _cmd(probs, out, *extra, rollouts=1, steps=6):
    return [sys.executable, str(ROOT / "scripts" / "run_multislot.py"), *map(str, probs), "--out", str(out),
            "--model", "stub:resume", "--plugin", str(PLUGIN), "--evaluator", "plugin",
            "--rollouts", str(rollouts), "--steps", str(steps), "--stop-grace", "30", *extra]


def _env(tmp, fault=None, slow=None, tag="x"):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "KB_TEST_LLM_LOG": str(tmp / f"llm_{tag}.log"),
           "KB_TEST_EVAL_LOG": str(tmp / f"eval_{tag}.log")}
    env.pop("KB_FAULT", None)
    if fault:
        env["KB_FAULT"] = fault
    if slow:
        env["KB_TEST_SLOW"] = str(slow)
    return env


def _run(tmp, out, fault=None, *extra, probs=(FIX,), tag="x", **kw):
    return subprocess.run(_cmd(probs, out, *extra, **kw), env=_env(tmp, fault, tag=tag), capture_output=True,
                          text=True, timeout=300)


def _lines(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()] if Path(p).exists() else []


def _digest(out, prob=PROB):
    """What must not depend on interruptions (ids and timings excluded)."""
    r = json.loads((out / prob / "result.json").read_text())
    outcomes = _lines(out / "knowledge" / "outcomes.jsonl")
    impl = _lines(out / "knowledge" / "impl" / "index.jsonl")
    claims = json.loads((out / "knowledge" / "claims.json").read_text())
    best = {p.name: p.read_text() for p in sorted((out / prob / "best" / "slots").iterdir())}
    return {
        "steps": [(s["rollout"], s["step"], s["slot"], s["technique"], s["status"], s["slot_us_before"],
                   s["slot_us_after"], round(s["reward"], 9), s["fix_attempts"]) for s in r["steps"]],
        "llm_calls": r["llm_calls"], "llm_usage": r["llm_usage"], "by": r["llm_usage_by_model"],
        "final": (r["final_ok"], r["final_partition_us"], r["speedup"]),
        "outcomes": sorted((o["slot"], o["technique"], o["status"], o["metric_after"]) for o in outcomes
                           if o["problem_id"] == prob),
        "impl": sorted((x["slot"], x["code_sha"], x["slot_us"]) for x in impl if x["problem_id"] == prob),
        "tried": {k: {t: (v["n"], v["statuses"]) for t, v in b.items()} for k, b in claims["tried"].items()},
        "best": best, "bandit": r["bandit"],
    }


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ref")
    out = tmp / "out"
    r = _run(tmp, out, tag="ref")
    assert r.returncode == 0, r.stderr[-3000:]
    return tmp, out, _digest(out), len(_lines(tmp / "llm_ref.log"))


FAULTS = [
    ("save:4", 0),                      # between steps (right after the checkpoint of step 1)
    ("eval:baseline:1", 0),             # during the baseline
    ("llm:3", 1),                       # inside an LLM call (paid, response lost)
    ("llmfix:1", 1),                    # inside a fix call
    ("eval:_s2_.*_f0:1", 0),            # evaluating a candidate (response journaled)
    ("eval:_confirm:2", 0),             # in the confirm run of a win
    ("eval:_f1:1", 0),                  # evaluating a fix attempt
    ("eval:final_composite:1", 0),      # in the final composite run
]


@pytest.mark.parametrize("fault,lost", FAULTS, ids=[f for f, _ in FAULTS])
def test_kill_and_resume_matches_uninterrupted(tmp_path, reference, fault, lost):
    _, _, ref, ref_calls = reference
    out = tmp_path / "out"
    r1 = _run(tmp_path, out, fault)
    assert r1.returncode == -signal.SIGKILL, (r1.returncode, r1.stderr[-2000:])
    assert not (out / "summary.json").exists()
    r2 = _run(tmp_path, out)
    assert r2.returncode == 0, r2.stderr[-3000:]
    got = _digest(out)
    assert got == ref
    # an LLM response that made it into the journal is never paid for again
    assert len(_lines(tmp_path / "llm_x.log")) == ref_calls + lost
    res = json.loads((out / PROB / "result.json").read_text())
    assert res["resume"]["segments"] == 2
    run = json.loads((out / "run.json").read_text())
    assert [s["status"] for s in run["segments"]] == ["running", "done"]    # the killed one never ended
    summary = json.loads((out / "summary.json").read_text())
    assert any(s.get("problem") == PROB and s.get("final_ok") for s in summary)


def test_kill_twice_and_resume(tmp_path, reference):
    _, _, ref, ref_calls = reference
    out = tmp_path / "out"
    assert _run(tmp_path, out, "eval:_s1_.*_f0:1").returncode == -signal.SIGKILL
    assert _run(tmp_path, out, "eval:_s4_.*_f0:1").returncode == -signal.SIGKILL
    r = _run(tmp_path, out)
    assert r.returncode == 0, r.stderr[-3000:]
    assert _digest(out) == ref
    assert len(_lines(tmp_path / "llm_x.log")) == ref_calls
    assert json.loads((out / PROB / "result.json").read_text())["resume"]["segments"] == 3


def test_random_kills_parallel_rollouts(tmp_path):
    """Two concurrent rollouts, SIGKILL at random times, resume until done: no step
    lost or repeated, no record twice, usage = the journaled responses' usage."""
    out = tmp_path / "out"
    rng = random.Random(1234)
    kills = 0
    for attempt in range(8):
        p = subprocess.Popen(_cmd([FIX], out, rollouts=2, steps=5), env=_env(tmp_path, slow=0.03),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            rc = p.wait(timeout=rng.uniform(1.5, 4.0) if kills < 3 else 300)
        except subprocess.TimeoutExpired:
            p.send_signal(signal.SIGKILL)
            p.wait()
            kills += 1
            continue
        assert rc == 0
        break
    else:
        pytest.fail("never finished")
    r = json.loads((out / PROB / "result.json").read_text())
    keys = [(s["rollout"], s["step"]) for s in r["steps"]]
    assert sorted(keys) == [(i, j) for i in range(2) for j in range(5)]
    outcomes = [o for o in _lines(out / "knowledge" / "outcomes.jsonl")]
    assert len(outcomes) == 10 and len({o["id"] for o in outcomes}) == 10
    impl = _lines(out / "knowledge" / "impl" / "index.jsonl")
    assert len({x["id"] for x in impl}) == len(impl)
    tried = json.loads((out / "knowledge" / "claims.json").read_text())["tried"]
    assert sum(t["n"] for b in tried.values() for t in b.values()) == 10
    journal = _lines(out / PROB / "journal" / "llm.jsonl")
    assert len({j["key"] for j in journal}) == len(journal) == r["llm_calls"]
    assert r["llm_usage"]["prompt_tokens"] == sum(j["usage"]["prompt_tokens"] for j in journal)
    assert r["llm_usage"]["completion_tokens"] == sum(j["usage"]["completion_tokens"] for j in journal)
    paid = len(_lines(tmp_path / "llm_x.log"))
    assert r["llm_calls"] <= paid <= r["llm_calls"] + 2 * kills     # at most the in-flight calls of each kill
    assert r["resume"]["segments"] == kills + 1


def test_config_mismatch_refused_and_force_restart(tmp_path):
    out = tmp_path / "out"
    assert _run(tmp_path, out, "save:4").returncode == -signal.SIGKILL
    r = _run(tmp_path, out, None, steps=7)
    assert r.returncode == 3 and "another configuration" in r.stderr
    assert _run(tmp_path, out, None, "--seed", "5").returncode == 3
    r = _run(tmp_path, out, None, "--force-restart", steps=7)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads((out / PROB / "result.json").read_text())
    assert len(res["steps"]) == 7 and res["resume"]["segments"] == 1


def test_changed_problem_refused(tmp_path):
    prob = tmp_path / "p"
    shutil.copytree(FIX, prob)
    out = tmp_path / "out"
    assert _run(tmp_path, out, "save:4", probs=(prob,)).returncode == -signal.SIGKILL
    sl = prob / "slots" / "relu.cu"
    sl.write_text(sl.read_text() + "\n// edited\n")
    r = _run(tmp_path, out, None, probs=(prob,))
    assert r.returncode == 3 and "changed" in r.stderr


def test_completed_problem_skipped(tmp_path):
    probs = []
    for n in ("a", "b"):
        d = tmp_path / "probs" / n
        shutil.copytree(FIX, d)
        m = json.loads((d / "manifest.json").read_text())
        m["name"] = f"lenet_{n}"
        (d / "manifest.json").write_text(json.dumps(m))
        probs.append(d)
    out = tmp_path / "out"
    env = _env(tmp_path, "save:12")      # a: init, rollouts, 6 steps, final, done = 10 saves; b: 2 more
    p = subprocess.run(_cmd(probs, out, "--parallel-problems", "1"), env=env, capture_output=True, text=True,
                       timeout=300)
    assert p.returncode == -signal.SIGKILL
    a_res = (out / "lenet_a" / "result.json").read_text()
    assert json.loads((out / "lenet_a" / "journal" / "state.json").read_text())["phase"] == "done"
    (tmp_path / "eval_x.log").unlink()
    r = _run(tmp_path, out, None, "--parallel-problems", "1", probs=probs)
    assert r.returncode == 0, r.stderr[-3000:]
    evals = (tmp_path / "eval_x.log").read_text().split("\n")
    assert not any(e.startswith("lenet_a ") for e in evals)          # nothing re-run for the done problem
    assert (out / "lenet_a" / "result.json").read_text() == a_res
    summary = json.loads((out / "summary.json").read_text())
    rows = {s["problem"]: s for s in summary if "problem" in s}
    assert set(rows) == {"lenet_a", "lenet_b"} and all(x["final_ok"] for x in rows.values())
    assert rows["lenet_a"]["resume"] == "done in an earlier segment"


def test_sigterm_stops_cleanly_and_resumes(tmp_path, reference):
    _, _, ref, ref_calls = reference
    out = tmp_path / "out"
    p = subprocess.Popen(_cmd([FIX], out), env=_env(tmp_path, slow=0.15), stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 60
    while time.time() < deadline and len(_lines(tmp_path / "llm_x.log")) < 3:
        time.sleep(0.05)
    p.send_signal(signal.SIGTERM)
    _, err = p.communicate(timeout=120)
    assert p.returncode == 75, err[-2000:]
    st = json.loads((out / PROB / "journal" / "state.json").read_text())
    assert st["phase"] == "rollouts" and not st["inflight"]             # the in-flight step finished
    assert json.loads((out / "run.json").read_text())["segments"][0]["status"] == "interrupted"
    r = _run(tmp_path, out)
    assert r.returncode == 0, r.stderr[-3000:]
    assert _digest(out) == ref
    assert len(_lines(tmp_path / "llm_x.log")) == ref_calls            # graceful: nothing lost in flight


def test_second_signal_exits_at_once(tmp_path):
    out = tmp_path / "out"
    p = subprocess.Popen(_cmd([FIX], out), env=_env(tmp_path, slow=1.0), stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    deadline = time.time() + 60
    while time.time() < deadline and not (out / PROB / "journal" / "state.json").exists():
        time.sleep(0.05)
    time.sleep(0.5)
    p.send_signal(signal.SIGINT)
    time.sleep(0.3)
    p.send_signal(signal.SIGINT)
    assert p.wait(timeout=30) == 76
    r = _run(tmp_path, out)
    assert r.returncode == 0, r.stderr[-3000:]


def _v2_digest(out):
    import sqlite3
    db = sqlite3.connect(out / "knowledge" / "knowledge.db")
    outcomes = sorted(db.execute("SELECT slot, technique, status, reward FROM outcomes WHERE problem_id=?", (PROB,)))
    stats = sorted(db.execute("SELECT * FROM stats"))
    db.close()
    r = json.loads((out / PROB / "result.json").read_text())
    return {"outcomes": outcomes, "stats": stats, "llm_calls": r["llm_calls"], "llm_usage": r["llm_usage"],
            "steps": [(s["slot"], s["technique"], s["status"], s["slot_us_after"]) for s in r["steps"]]}


@pytest.mark.parametrize("fault", ["eval:_s3_.*_f0:1", "save:5"])
def test_knowledge_v2_kill_and_resume(tmp_path, fault):
    ref = tmp_path / "ref"
    r = _run(tmp_path, ref, None, "--knowledge", "v2", tag="ref")
    assert r.returncode == 0, r.stderr[-3000:]
    out = tmp_path / "out"
    assert _run(tmp_path, out, fault, "--knowledge", "v2").returncode == -signal.SIGKILL
    r = _run(tmp_path, out, None, "--knowledge", "v2")
    assert r.returncode == 0, r.stderr[-3000:]
    a, b = _v2_digest(ref), _v2_digest(out)
    assert len(b["outcomes"]) == 6
    assert a == b
    assert len(_lines(tmp_path / "llm_x.log")) == len(_lines(tmp_path / "llm_ref.log"))


def test_telemetry_segments_sum(tmp_path, reference):
    _, ref_out, _, _ = reference
    out = tmp_path / "out"
    assert _run(tmp_path, out, "eval:_s3_.*_f0:1").returncode == -signal.SIGKILL
    assert _run(tmp_path, out).returncode == 0
    from kernelblaster.telemetry import report
    evs = report.read_events(out / "telemetry.jsonl")
    assert sum(1 for e in evs if e.get("t") == "resume") == 2           # run level + the problem
    runs = {e.get("run") for e in evs if e.get("t") == "span"}
    assert len(runs) == 2 and any(r.endswith(".s1") for r in runs)      # one telemetry run per segment
    comb = json.loads(report.report([out], as_json=True))["_combined"]
    ref = json.loads(report.report([ref_out], as_json=True))
    ref_run = next(v["run"] for k, v in ref.items() if k != "_combined")
    assert comb["steps"] == ref_run["derived"]["steps"] == 6            # replayed step counted once
    assert comb["calls"] == ref_run["llm"]["total"]["calls"]            # no journaled call paid twice


def test_ssh_cleanup_previous_removes_only_own_leftovers(tmp_path, monkeypatch):
    """A resumed run removes the files its killed segment staged and stops that
    segment's orphan persistent host; other runs' files and hosts stay."""
    import asyncio
    from test_loadbalance import _fake_ssh_env, _fake_ssh_evaluator
    from kernelblaster.agents.journal import append_jsonl, read_jsonl

    _fake_ssh_env(tmp_path, monkeypatch, {"u@h1": 0})
    rdir = tmp_path / "remote_u@h1"
    other = tmp_path / "other_run"
    other.mkdir()
    for n in ("p_problem_aaaaaa", "p_problem_bbbbbb", "someone_elses_file", "kbhost_abc"):
        (rdir / n).write_text("x")
    (rdir / "data_123").mkdir()
    (rdir / "data_123" / ".data.bin.part.me.1").write_text("partial")
    (rdir / "data_123" / "data.bin").write_text("valid copy")
    mine = subprocess.Popen(["bash", "-c", "exec -a kbhost_abc sleep 60"], cwd=rdir)
    theirs = subprocess.Popen(["bash", "-c", "exec -a kbhost_abc sleep 60"], cwd=other)
    try:
        time.sleep(0.3)
        man = tmp_path / "remote.jsonl"
        for rec in ({"kind": "segment", "tmp": ".data.bin.part.me.1"}, {"kind": "file", "name": "p_problem_aaaaaa"},
                    {"kind": "file", "name": "p_problem_bbbbbb"}, {"kind": "host", "host_pid": str(mine.pid)},
                    {"kind": "host", "host_pid": str(theirs.pid)}):
            append_jsonl(man, {"pid": 1, "host": "u@h1", "rdir": str(rdir), **rec})
        ev = _fake_ssh_evaluator(tmp_path, ["u@h1"], remote_manifest=man)
        ev._note_remote("u@h1", "file", name="p_problem_cccccc")          # this process's own
        done = asyncio.run(ev.cleanup_previous())
        assert done == {"hosts": 1, "files": 2, "pids": 2}
        assert not (rdir / "p_problem_aaaaaa").exists() and not (rdir / "p_problem_bbbbbb").exists()
        assert (rdir / "someone_elses_file").exists() and (rdir / "kbhost_abc").exists()
        assert not (rdir / "data_123" / ".data.bin.part.me.1").exists() and (rdir / "data_123" / "data.bin").exists()
        assert mine.wait(timeout=10) != 0                                   # the orphan host is stopped
        assert theirs.poll() is None                                        # another dir's host is not
        left = read_jsonl(man)
        assert left and all(r["pid"] == os.getpid() for r in left)
        assert asyncio.run(ev.cleanup_previous()) == {}                      # once per evaluator
    finally:
        for p in (mine, theirs):
            if p.poll() is None:
                p.kill()


def test_stops_when_parent_dies(tmp_path):
    """KB_STOP_WITH_PARENT=1 (ModelBlaster's kb_graph_opt): the parent is SIGKILLed, the run
    stops cleanly by itself (segment "interrupted") and a rerun completes it."""
    out = tmp_path / "out"
    env = {**_env(tmp_path, slow=0.3), "KB_STOP_WITH_PARENT": "1"}
    cmd = " ".join(f"'{c}'" for c in _cmd([FIX], out))
    parent = subprocess.Popen(["bash", "-c", f"{cmd} > {tmp_path}/child.log 2>&1 & echo $! > {tmp_path}/pid; wait"],
                              env=env)
    deadline = time.time() + 60
    while time.time() < deadline and not (out / PROB / "journal" / "state.json").exists():
        time.sleep(0.05)
    parent.kill()
    parent.wait()
    child = int((tmp_path / "pid").read_text())
    deadline = time.time() + 90
    while time.time() < deadline and os.path.exists(f"/proc/{child}") and \
            "zombie" not in Path(f"/proc/{child}/status").read_text().lower():
        time.sleep(0.2)
    run = json.loads((out / "run.json").read_text())
    assert run["segments"][0]["status"] == "interrupted", (tmp_path / "child.log").read_text()[-2000:]
    r = _run(tmp_path, out)
    assert r.returncode == 0, r.stderr[-3000:]
