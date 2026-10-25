"""Trace store (kernelblaster/trace, docs/trace.md): blobs, dedup, caps, levels,
concurrency, secret scrubbing, the LLM / evaluator hooks, the viewer, and an
end-to-end run of run_multislot on the LeNet fixture (needs nvcc + a GPU)."""
import asyncio
import gzip
import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from kernelblaster import trace
from kernelblaster.trace import core, view

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _closed():
    trace.close()
    yield
    trace.close()


def _recs(root):
    return [json.loads(l) for l in (Path(root) / "index.jsonl").read_text().splitlines() if l.strip()]


def _blobs(root):
    return sorted((Path(root) / "blobs").rglob("*.gz"))


# ---------------------------------------------------------------- store

def test_dedup_compression_inline(tmp_path):
    st = trace.configure(tmp_path / "t", level="full")
    big = "line of a prompt\n" * 2000
    a, b = trace.put(big), trace.put(big)
    assert a == b and a["n"] == len(big) and "sha" in a
    files = _blobs(tmp_path / "t")
    assert len(files) == 1 and files[0].stat().st_size < len(big) / 20
    assert gzip.decompress(files[0].read_bytes()).decode() == big
    assert trace.read_ref(tmp_path / "t", a) == big
    small = trace.put("ok")
    assert small == {"text": "ok"} and len(_blobs(tmp_path / "t")) == 1
    s = trace.close()
    assert s["blobs"] == 1 and s["blob_hits"] == 1 and s["bytes_in"] >= 2 * len(big)
    kinds = [r["t"] for r in _recs(tmp_path / "t")]
    assert kinds[0] == "run" and kinds[-1] == "close"


def test_blob_cap_keeps_head_and_tail_and_total_cap_drops(tmp_path, monkeypatch):
    st = trace.configure(tmp_path / "t", level="full", blob_max=3000)
    text = "HEAD\n" + "x" * 20000 + "\nTAIL"
    ref = trace.put(text)
    got = trace.read_ref(tmp_path / "t", ref)
    assert ref["trunc"] > 0 and got.startswith("HEAD") and got.endswith("TAIL") and "bytes cut" in got
    assert len(got) < 3200
    trace.close()
    monkeypatch.setenv("KB_TRACE_MAX_MB", "0.004")          # ~4 KB of compressed blobs
    st = trace.configure(tmp_path / "t2", level="full")
    import random
    rng = random.Random(0)
    refs = [trace.put("".join(rng.choice("abcdefgh") for _ in range(4000))) for _ in range(5)]
    assert refs[0].get("dropped") is None and any(r.get("dropped") for r in refs)
    assert st.capped and trace.close()["dropped"] >= 1


def test_levels_meta_and_off(tmp_path, monkeypatch):
    trace.configure(tmp_path / "m", level="meta")
    ref = trace.put("some prompt text " * 100)
    assert ref["meta"] and ref["sha"] and ref["n"] > 0 and trace.read_ref(tmp_path / "m", ref) is None
    trace.record("llm", rid="x", messages=[{"role": "user", "content": ref}])
    trace.close()
    assert not _blobs(tmp_path / "m") and [r["t"] for r in _recs(tmp_path / "m")] == ["run", "llm", "close"]
    assert trace.configure(tmp_path / "off", level="off") is None
    trace.put("x")
    trace.record("llm", rid="y")
    assert not (tmp_path / "off").exists() and not trace.enabled()
    monkeypatch.setenv("KB_TRACE", "0")
    assert trace.configure(tmp_path / "off2") is None and not (tmp_path / "off2").exists()


def test_concurrent_scopes_and_threads(tmp_path):
    trace.configure(tmp_path / "t", level="full")

    async def rollout(r):
        with trace.scope(run="R", problem="p", rollout=r):
            for s in range(5):
                with trace.scope(step=s):
                    trace.step_input(slot=f"slot{r}", technique="t", source=f"src {r} {s}\n" * 40)
                    await asyncio.sleep(0.001 * ((r + s) % 3))
                    trace.attempt(1)
                    await asyncio.sleep(0)
                    trace.record("probe", rid=trace.key("probe"))

    async def main():
        await asyncio.gather(*[rollout(r) for r in range(6)])
    th = [threading.Thread(target=lambda i=i: [trace.put(f"thread {i} blob {k}" * 30) for k in range(50)])
          for i in range(4)]
    for t in th:
        t.start()
    asyncio.run(main())
    for t in th:
        t.join()
    trace.close()
    recs = _recs(tmp_path / "t")
    probes = [r for r in recs if r["t"] == "probe"]
    assert len(probes) == 30
    for p in probes:          # every record carries its own task's rollout / step / slot / attempt
        assert p["id"] == f"R/p/r{p['rollout']}/s{p['step']}/probe" and p["slot"] == f"slot{p['rollout']}"
        assert p["attempt"] == 1
    assert len(_blobs(tmp_path / "t")) == 30 + 4 * 50


def test_secrets_are_scrubbed(tmp_path, monkeypatch):
    secret = "abcDEF1234567890secretvalue"
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", secret)
    tok = tmp_path / "tok"
    tok.write_text("relaytoken-0123456789\n")
    monkeypatch.setenv("KB_LLM_SERVER_TOKEN_FILE", str(tok))
    trace.configure(tmp_path / "t", level="full")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-zzzzzzzzzzzzzzzzzzzzzzzzzz")      # exported after configure
    with trace.scope(run="R", problem="p"):
        ref = trace.put(f"export AWS_BEARER_TOKEN_BEDROCK={secret}\nAuthorization: Bearer {'q' * 30}\n" + "x" * 300)
        trace.tool("nvcc", ["env", f"KEY={secret}", "nvcc", "relaytoken-0123456789"], 0,
                   "using sk-proj-zzzzzzzzzzzzzzzzzzzzzzzzzz")
        trace.output("driver", "MB_E2E_US 1.0 basis=graph\n", env={"MB_SCREEN": "1", "MY_API_KEY": "hunter2hunter2"})
    trace.close()
    blob = trace.read_ref(tmp_path / "t", ref)
    assert secret not in blob and "q" * 30 not in blob and blob.count("***") == 2
    raw = (tmp_path / "t" / "index.jsonl").read_text() + "".join(
        gzip.decompress(p.read_bytes()).decode() for p in _blobs(tmp_path / "t"))
    for s in (secret, "relaytoken-0123456789", "sk-proj-zzzz", "hunter2hunter2"):
        assert s not in raw
    drv = next(r for r in _recs(tmp_path / "t") if r["t"] == "driver")
    assert drv["env"] == {"MB_SCREEN": "1", "MY_API_KEY": "***"}


# ---------------------------------------------------------------- hooks

def test_llm_hook_records_messages_response_and_request_id(tmp_path):
    from kernelblaster.agents.utils import agentic_llm
    from kernelblaster.agents.utils.query import generate_code

    seen = {}

    def stub(msgs):
        seen["rid"] = trace.request_id()            # what llm_relay would send as request_id
        return "here:\n```cuda\nint x;\n```"
    agentic_llm.register_stub("trace_llm", stub)
    trace.configure(tmp_path / "t", level="full")
    try:
        msgs = [{"role": "system", "content": "SYSTEM " * 100}, {"role": "user", "content": "do it " * 50}]
        with trace.scope(run="R", problem="p", rollout=0, step=3):
            trace.step_input(slot="s", technique="t", source="old\n")
            asyncio.run(generate_code(msgs, 1, "stub:trace_llm"))
            asyncio.run(generate_code([msgs[0], {"role": "user", "content": "another"}], 1, "stub:trace_llm"))
    finally:
        agentic_llm.unregister_stub("trace_llm")
    trace.close()
    llm = [r for r in _recs(tmp_path / "t") if r["t"] == "llm"]
    assert len(llm) == 2 and llm[0]["id"] == "R/p/r0/s3/a0/llm" and llm[0]["ok"]
    assert llm[1]["request_id"] == seen["rid"] and llm[0]["model"] == "stub:trace_llm"
    root = tmp_path / "t"
    assert trace.read_ref(root, llm[0]["messages"][0]["content"]) == "SYSTEM " * 100
    assert trace.read_ref(root, llm[0]["response"]).endswith("```")
    # the shared system prompt is stored once
    assert llm[0]["messages"][0]["content"] == llm[1]["messages"][0]["content"]


def test_extract_hook_keeps_responses_without_code(tmp_path):
    from kernelblaster.agents.opt_multislot import extract_code
    trace.configure(tmp_path / "t", level="full")
    with trace.scope(run="R", problem="p", rollout=1, step=0):
        trace.step_input(slot="s", technique="t", source="old\n")
        assert extract_code("I would rather not.") is None
        trace.attempt(1)
        assert extract_code("```cuda\nint y;\n```") == "int y;\n"
    extract_code("```cuda\nnot in a step\n```")
    trace.close()
    c = {r["attempt"]: r for r in _recs(tmp_path / "t") if r["t"] == "candidate"}
    assert c[0]["no_code"] and c[1]["code"] == {"text": "int y;\n"} and len(c) == 2


def test_ncu_and_driver_output_hooks(tmp_path):
    from kernelblaster.knowledge.slots import _ncu_result, parse_driver_output
    trace.configure(tmp_path / "t", level="full")
    csv = (FIX / "ncu" / "lenet_sm89_all_slots.csv").read_text()
    with trace.scope(run="R", problem="p", eval="R/p/evaluate/baseline", eval_occ="o1"):
        by, err = _ncu_result(csv)
        parse_driver_output("MB_PARTITION_US 3.5\npassed\n")
    trace.close()
    recs = {r["t"]: r for r in _recs(tmp_path / "t")}
    assert err is None and trace.read_ref(tmp_path / "t", recs["ncu"]["output"]) == csv
    assert recs["driver"]["id"] == "R/p/evaluate/baseline/driver1" and recs["driver"]["eval_occ"] == "o1"


def test_evaluator_hook_and_viewer(tmp_path):
    """A fake evaluator through the real wrappers -> eval / build / driver records the viewer renders."""
    from kernelblaster.knowledge.slots import EvalResult, Evaluator, parse_driver_output

    class FakeProblem:
        name = "prob"

        def assemble(self, sources):
            return "".join(sources.values())

    class FakeEval(Evaluator):
        async def evaluate(self, problem, sources, tag, *, only_slots=None, screen=False, incumbent=None):
            bad = "broken" in sources["a"]
            trace.tool("nvcc", ["nvcc", "-c", "a.cu"], 2 if bad else 0, "a.cu(1): error: nope" if bad else "")
            if bad:
                return EvalResult(ok=False, stage="compile", errors=["a.cu(1): error: nope"])
            return parse_driver_output("MB_E2E_US 10.0 basis=graph\nMB_PARTITION_US 12.0\npassed\n")

    assert getattr(FakeEval.evaluate, "__kb_trace__", False)
    trace.configure(tmp_path / "t", level="full")
    ev, p = FakeEval(), FakeProblem()

    async def go():
        with trace.scope(run="R", problem="prob"):
            await ev.evaluate(p, {"a": "seed\n"}, "baseline")
            with trace.scope(rollout=0, step=0):
                trace.step_input(slot="a", technique="tile", source="seed\n")
                await ev.evaluate(p, {"a": "broken\n"}, "r0_s0_a_tile_f0", only_slots=["a"])
                trace.attempt(1)
                await ev.evaluate(p, {"a": "fixed\n"}, "r0_s0_a_tile_f1", only_slots=["a"])
    asyncio.run(go())
    trace.close()
    tr = view.Trace(tmp_path / "t")
    (s,) = tr.steps()
    assert [e["attempt"] for e in s.evals] == [0, 1] and s.evals[0]["purpose"] == "candidate"
    assert s.evals[0]["result"]["stage"] == "compile" and s.evals[1]["result"]["ok"]
    assert trace.read_ref(tr.root, s.evals[1]["sources"]["a"]) == "fixed\n"
    txt = view.render_step(tr, s, diff=True)
    assert "FAILED (compile)" in txt and "error: nope" in txt and "fix attempt 1" in txt and "e2e_us=10.00" in txt
    assert "(no decision record" in txt
    assert [e["tag"] for e in tr.problem_evals()] == ["baseline"]
    assert "tile" in view.list_table([s])
    out = view.export(tr, tmp_path / "exp")
    assert (out / "prob" / "r0_s0" / "00_input.cu").read_text() == "seed\n"
    assert (out / "prob" / "r0_s0" / "story.txt").exists()
    h = view.export_html(tr, tmp_path / "t.html").read_text()
    assert h.startswith("<!doctype html>") and "<script" not in h and "http" not in h.split("</style>")[1]


def test_resumed_step_replaces_its_records(tmp_path):
    """Stable ids: a step re-run after a kill (same run / problem / rollout / step / attempt) is one step."""
    for seg in range(2):
        trace.configure(tmp_path / "t", level="full")
        with trace.scope(run="R", problem="p", rollout=0, step=1):
            trace.step_input(slot="a", technique="t", source=f"seed {seg}\n")
            trace.record("candidate", rid=trace.key("a0", "candidate"), code=trace.put(f"code {seg}\n"))
        trace.close()
    tr = view.Trace(tmp_path / "t")
    (s,) = tr.steps()
    assert tr.text(s.candidates[0]["code"]) == "code 1\n" and tr.dupes


def test_trace_view_cli(tmp_path):
    trace.configure(tmp_path / "run" / "trace", level="full")
    with trace.scope(run="R", problem="lenet", rollout=0, step=0):
        trace.step_input(slot="relu", technique="vec", source="a\n")
        trace.record("llm", rid=trace.key("a0", "llm"), request_id="feedbeef" * 4, ok=True, role="codegen",
                     messages=[{"role": "user", "content": trace.put("PROMPT")}], response=trace.put("no code"))
        trace.on_extract("no code", None)
    trace.close()
    run = lambda *a: subprocess.run([sys.executable, str(ROOT / "scripts" / "trace_view.py"), *a],
                                    capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    r = run(str(tmp_path / "run"), "--list")
    assert r.returncode == 0 and "relu" in r.stdout and "vec" in r.stdout, r.stderr
    r = run(str(tmp_path / "run"), "--problem", "len", "--rollout", "0", "--step", "0")
    assert "candidate: NONE" in r.stdout and "PROMPT" in r.stdout
    r = run(str(tmp_path / "run"), "--llm", "feedbeef")
    assert "=== LLM call feedbeef" in r.stdout and "no code" in r.stdout


# ---------------------------------------------------------------- end to end

from test_multislot_agent import ARCH, needs_gpu  # noqa: E402

PLUGIN = '''
import sys
sys.path.insert(0, {tests!r})
from test_multislot_agent import scripted
from kernelblaster.agents.utils import agentic_llm
state = {{"broke": False, "fix_calls": 0, "linear_ok": None}}
agentic_llm.register_stub("trace_e2e", scripted(state))
'''


def _run_multislot(tmp_path, name, *extra, rollouts=2, steps=3):
    import shutil
    prob = tmp_path / f"{name}_prob"
    shutil.copytree(FIX / "mb_lenet", prob)
    plugin = tmp_path / f"{name}_plugin.py"
    plugin.write_text(PLUGIN.format(tests=str(Path(__file__).parent)))
    out = tmp_path / f"{name}_out"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_multislot.py"), str(prob), "--out", str(out),
                        "--model", "stub:trace_e2e", "--plugin", str(plugin), "--arch", ARCH, "--rollouts",
                        str(rollouts), "--steps", str(steps), "--seed", "1", *extra],
                       capture_output=True, text=True, timeout=900,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "KERNELBLASTER_ALLOW_SHARED_GPU": "1"})
    assert r.returncode == 0, r.stderr[-3000:]
    return out


@needs_gpu
def test_e2e_every_step_is_traced(tmp_path):
    out = _run_multislot(tmp_path, "full")
    res = json.loads(next(out.glob("*/result.json")).read_text())
    tr = view.Trace(out / "trace")
    steps = tr.steps()
    assert len(steps) == len(res["steps"]) == 6
    for s in steps:
        assert s.input and tr.text(s.input["source"]) is not None
        assert s.llm.get(0) and tr.text(s.llm[0]["messages"][-1]["content"]).startswith("## Slot")
        assert tr.text(s.llm[0]["response"]) is not None
        assert 0 in s.candidates
        cand = [e for e in s.evals if e.get("attempt") == 0]
        assert cand and cand[0]["purpose"] == "candidate" and cand[0]["sources"]
        assert any(c["t"] == "driver" for e in s.evals for c in tr.children.get(e["occ"], [])) or \
            cand[0]["result"]["stage"] == "compile"
        r = s.record
        assert r and r["steps_index"] is not None and r["decision"] in ("merged", "not_merged", "rejected")
        sl = res["steps"][r["steps_index"]]
        assert (sl["rollout"], sl["step"], sl["slot"]) == (s.rollout, s.step, r["steplog"]["slot"])
        assert r["tel_span"] and r["outcome_id"]
    # the broken linear candidate (a loser) is kept with its compiler output, and its fix
    fixed = [s for s in steps if s.record["fix_attempts"]]
    assert fixed, "the scripted LLM breaks linear once"
    s = fixed[0]
    assert "this is not valid" in tr.text(s.candidates[0]["code"])
    builds = [c for e in s.evals if e.get("attempt") == 0 for c in tr.children.get(e["occ"], []) if c["t"] == "build"]
    assert any(b["rc"] != 0 and "error" in tr.text(b["output"]) for b in builds)
    assert s.llm[1]["role"] == "fix" and 1 in s.candidates
    # baseline and final composite are traced at problem level; telemetry links resolve
    assert {"baseline", "final_composite"} <= {e["tag"] for e in tr.problem_evals()}
    spans = {json.loads(l).get("id") for l in (out / "telemetry.jsonl").read_text().splitlines()
             if '"t":"span"' in l}
    assert all(s.record["tel_span"] in spans for s in steps)
    assert all(s.llm[0]["tel_span"] in spans for s in steps)
    close = [r for r in tr.raw if r["t"] == "close"][-1]["stats"]
    assert close["bytes_stored"] < 2_000_000


@needs_gpu
def test_e2e_trace_off_writes_nothing(tmp_path):
    out = _run_multislot(tmp_path, "off", "--trace", "off", rollouts=1, steps=1)
    assert (out / "summary.json").exists() and not (out / "trace").exists()


def test_kill_and_resume_one_trace_per_step(tmp_path):
    """Killed while a journaled candidate is evaluated, then resumed: the step's records keep
    their ids (persisted run id), its response is an llm_replay of the original call, and the
    trace has exactly the steps of result.json."""
    import signal
    import test_resume as R
    out = tmp_path / "out"
    r1 = R._run(tmp_path, out, "eval:_s2_.*_f0:1")
    assert r1.returncode == -signal.SIGKILL, r1.stderr[-2000:]
    r2 = R._run(tmp_path, out)
    assert r2.returncode == 0, r2.stderr[-3000:]
    res = json.loads((out / R.PROB / "result.json").read_text())
    tr = view.Trace(out / "trace")
    steps = tr.steps()
    assert len(steps) == len(res["steps"]) and all(s.record for s in steps)
    assert len({r["run"] for r in tr.of("step")}) == 1 == len({s.run for s in steps})
    s2 = next(s for s in steps if s.step == 2)
    rp = s2.replays[0]
    assert rp["replay_of"] == s2.llm[0]["id"]                # the original call, from the killed segment
    assert rp["response"] == s2.llm[0]["response"]           # deduplicated: the same blob
    assert "replayed from the resume journal" in view.render_step(tr, s2)
    assert tr.dupes                                          # step 2's input / candidate re-recorded, one step
