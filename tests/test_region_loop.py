"""Region loop (agents/region_loop.py, ModelBlaster ``--region-loop on``): the tool-using loop on REGION slots, over plain chat.

No GPU, no LLM: a fake backend (the loop's GPU abstraction: compile / check / paired timing as a function of the candidate's text) and a
scripted LLM (a list of replies; the messages it was shown are recorded). Covers the structured-text protocol, every tool, the budgets
(turns / tool calls / usd / wall), verify rejection, the commit rule, rollback, context compaction, knowledge records through a real
MultiSlotAgent run (fake evaluator), and flag-off identity (no key in the snapshot / result, the rollouts pick as before).
"""
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelblaster.agents import region_loop as rl
from kernelblaster.agents.opt_multislot import MultiSlotAgent
from kernelblaster.knowledge.impl_store import ImplStore
from kernelblaster.knowledge.outcomes import OutcomeLog
from kernelblaster.knowledge.slots import EvalResult, Evaluator, SlotProblem

SID = "region__ops__abc123_f16"
ENTRY = f"kernel_{SID}"
SIG = f"void {ENTRY}(const __half *in_x, __half *out_y, __half *scratch_t, const __half *w_w, int n)"
SEED = f'#include <cuda_fp16.h>\n// current kernel of linear_f16 (fc0)\nvoid {ENTRY}(const __half *in_x, __half *out_y, __half *scratch_t, const __half *w_w, int n) {{ /* seed */ }}\n'


def make_problem(root: Path, extra_slot: bool = True) -> SlotProblem:
    root.mkdir(parents=True, exist_ok=True)
    (root / "slots").mkdir(exist_ok=True)
    (root / "driver.cpp").write_text("// driver")
    (root / "data.bin").write_bytes(b"\0" * 16)
    (root / "walker_head.cu").write_text("// head")
    (root / "walker_tail.cu").write_text("// tail")
    slots = [{"id": SID, "op": SID, "ops": [SID], "op_class": "gemm", "entry": ENTRY, "signature": SIG, "file": f"slots/{SID}.cu",
              "reference_file": f"slots/{SID}.ref.c", "semantics": "REGION slot: 2 ops (linear_f16, relu_f16) ... Checked outputs: out_y.",
              "seed_source": "region-seed:1/2", "dtype": "fp16", "tolerance": {"slot_rel": 2e-3, "slot_abs": 0.0},
              "instances": [{"op_name": "region:fc0..relu1", "op": SID, "index": 0, "shape": {"m0": {"M": 2, "K": 16, "N": 16}}}]}]
    (root / "slots" / f"{SID}.cu").write_text(SEED)
    (root / "slots" / f"{SID}.ref.c").write_text("/* reference */")
    if extra_slot:
        slots.append({"id": "relu", "op": "relu", "ops": ["relu"], "op_class": "elementwise", "entry": "kernel_relu",
                      "signature": "void kernel_relu(const float *input, float *output, int n)", "file": "slots/relu.cu",
                      "reference_file": None, "semantics": "relu", "seed_source": "hand", "dtype": "fp16",
                      "tolerance": {"slot_rel": 2e-3, "slot_abs": 0.0},
                      "instances": [{"op_name": "r", "op": "relu", "index": 1, "shape": {"n": 8}}]})
        (root / "slots" / "relu.cu").write_text('void kernel_relu(const float *input, float *output, int n) {}\n')
    man = {"format": "mb-kb-problem/1", "name": "regp", "source_ir": "x", "ir_name": "x", "partition": {"op_names": []}, "target": "cuda",
           "dtype": "fp16", "tolerance": {"slot_rel": 2e-3, "slot_abs": 0.0},
           "assembly": {"head": "walker_head.cu", "tail": "walker_tail.cu", "namespace_prefix": "mbslot_", "order": [s["id"] for s in slots]},
           "driver": "driver.cpp", "data": "data.bin", "data_sha1": "0", "slots": slots,
           "ops": [{"index": i, "name": s["id"], "op": s["op"], "slot": s["id"], "inputs": [0], "outputs": [1]} for i, s in enumerate(slots)],
           "buffers": [{"name": "x", "kind": "input", "dtype": "f16", "numel": 32, "alias": None, "ref": [0, 64]},
                       {"name": "y", "kind": "output", "dtype": "f16", "numel": 32, "alias": None, "ref": [64, 64]}],
           "inputs": [0], "outputs": [1]}
    (root / "manifest.json").write_text(json.dumps(man))
    return SlotProblem(root)


# the behaviour of a candidate is a function of its text
def behave(src: str) -> dict:
    d = {"ok": True, "stage": "ok", "us": 100.0, "ratio": 1.0, "rel": 1e-4}
    if "COMPILEFAIL" in src:
        d.update(ok=False, stage="compile")
    elif "BROKEN" in src:
        d.update(ok=False, stage="check")
    elif "XFAST" in src:
        d.update(us=60.0, ratio=0.6)
    elif "FAST" in src:
        d.update(us=80.0, ratio=0.8)
    elif "MEH" in src:
        d.update(us=92.0, ratio=0.92)
    elif "SLOW" in src:
        d.update(us=130.0, ratio=1.3)
    elif "NOCAPTURE" in src:
        d.update(capture=False)
    elif "FLAKY" in src:
        d.update(flaky=True)
    return d


class FakeBackend(rl.Backend):
    arch = "100"

    def __init__(self, paired=True, delay=0.0):
        self.calls, self.paired, self.delay, self.scratch_calls, self.repeat = [], paired, delay, [], {}

    def pairs_incumbent(self):
        return self.paired

    async def evaluate(self, sources, tag, *, incumbent=None, screen=False, only_slots=None):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.calls.append((tag, incumbent is not None))
        src = sources[SID]
        b = behave(src)
        if not b["ok"]:
            return EvalResult(ok=False, stage=b["stage"], errors=[f"{b['stage']} error in the candidate"],
                              slot_checks={SID: {"ok": False, "rel_err": 0.5}} if b["stage"] == "check" else {})
        r = EvalResult(ok=True, stage="ok", partition_us=b["us"], slot_us={SID: b["us"] / 2, "relu": 3.0}, e2e_us=b["us"], e2e_basis="graph",
                       slot_checks={SID: {"ok": True, "rel_err": b["rel"]}}, attr_slot_us={SID: b["us"] / 2, "relu": 3.0},
                       attr_slot_launches={SID: 1.0}, attr_op_us={SID: b["us"] / 2}, graph_nodes=3)
        if b.get("capture") is False:
            r.e2e_basis, r.graph_skip, r.graph_nodes = "eager", "cudaStreamSynchronize in the slot", 0
        if incumbent is not None and self.paired:
            cur = behave(incumbent[SID])
            r.ref_e2e_us = cur["us"]
            r.e2e_ratio = {"med": b["us"] / cur["us"], "n": 5, "mean": b["us"] / cur["us"], "min": 0.0, "max": 9.0}
        return r

    async def scratch(self, code, seconds_left):
        self.scratch_calls.append(code)
        return {"ok": True, "stage": "run", "exit": 0, "out": "clock64 per step: 412 cycles", "wall_s": 1.5}

    async def extra_checks(self, sources, tag, first):
        if "FLAKY" in sources[SID]:
            return [{"name": "repeat_run", "ok": False, "detail": "second run: output mismatch"}]
        return []


class Script:
    """A scripted LLM: replies in order (a str, or (str, usage)); records the messages of every call."""

    def __init__(self, replies, usd=0.0):
        self.replies, self.seen, self.usd = list(replies), [], usd

    async def __call__(self, messages, role):
        self.seen.append([dict(m) for m in messages])
        if not self.replies:
            raise RuntimeError("script exhausted")
        r = self.replies.pop(0)
        usage = {"prompt_tokens": 1000, "completion_tokens": 200, "cost_usd": self.usd}
        if isinstance(r, tuple):
            r, usage = r
        return r, usage


def act(tool, code=None, thought="", **args):
    s = (thought + "\n" if thought else "") + "```action\n" + json.dumps({"tool": tool, "args": args}) + "\n```"
    if code is not None:
        s += "\n```cuda\n" + code + "\n```"
    return s


def src(tag=""):
    return f'#include <cuda_fp16.h>\n// {tag}\nvoid {ENTRY}(const __half *in_x, __half *out_y, __half *scratch_t, const __half *w_w, int n) {{ /* {tag} */ }}\n'


def make_loop(tmp_path, replies, *, backend=None, cfg=None, usd=0.0, **kw):
    prob = make_problem(tmp_path / "prob")
    be = backend or FakeBackend()
    llm = Script(replies, usd=usd)
    sources = prob.seed_sources()
    loop = rl.RegionLoop(problem=prob, sid=SID, backend=be, llm=llm, model="stub:rl", sources=sources, baseline_us=100.0,
                         cfg=cfg or rl.LoopConfig(), out_dir=tmp_path / "out", knowledge_text="Lesson: persistent kernels win.", **kw)
    return loop, llm, be, prob


def run(loop):
    return asyncio.run(loop.run())


# ---------------------------------------------------------------------------------------------- the protocol
def test_relay_has_no_tool_passthrough():
    """The finding the protocol rests on: the relay forwards {role, content: str} messages and nothing else."""
    import kernelblaster.servers.llm_server as srv
    import inspect
    body = inspect.getsource(srv.LLMServer.handle) if hasattr(srv, "LLMServer") else inspect.getsource(srv)
    assert "tools" not in body and "tool_choice" not in body and "function_call" not in body
    from kernelblaster.agents.utils import query
    assert "tools" not in inspect.signature(query.generate_code).parameters


def test_parse_reply_forms():
    a = rl.parse_reply('thinking...\n```action\n{"tool": "show", "args": {"what": "outline"}}\n```')
    assert (a.tool, a.args, a.code, a.thought) == ("show", {"what": "outline"}, None, "thinking...")
    a = rl.parse_reply(act("verify", code=src("x")))
    assert a.tool == "verify" and "x" in a.code
    a = rl.parse_reply('```json\n{"tool": "status"}\n```')
    assert a.tool == "status" and a.args == {}
    a = rl.parse_reply('{"tool": "finish", "args": {"summary": "done"}}')
    assert a.tool == "finish"
    a = rl.parse_reply('```action\n{"tool": "verify", "args": {"source": "int x;"}}\n```')            # source inside the JSON args
    assert a.code == "int x;\n"
    two = '```action\n{"tool": "show"}\n```\n```action\n{"tool": "status"}\n```'
    assert rl.parse_reply(two).extra_actions == 1
    for bad, why in (("no block here", "no action"), ('```action\n{oops}\n```', "not valid JSON"),
                     ('```action\n{"tool": "rm"}\n```', "unknown tool"), ('```action\n{"tool": "show", "args": 3}\n```', "args must")):
        with pytest.raises(rl.ProtocolError, match=why):
            rl.parse_reply(bad)


def test_budget_parse_and_enable(monkeypatch):
    assert rl.parse_budget(None) == rl.DEFAULT_BUDGET
    assert rl.parse_budget("10,,2.5") == {"turns": 10, "tools": 120, "usd": 2.5, "wall_s": 3600.0}
    for bad in ("a", "1,2,3,4,5", "0"):
        with pytest.raises(ValueError):
            rl.parse_budget(bad)
    assert rl.enabled("on") and not rl.enabled("off") and not rl.enabled(None, env={})
    assert rl.enabled(None, env={rl.ENV: "1"})
    with pytest.raises(ValueError):
        rl.enabled("maybe")
    c = rl.LoopConfig.from_env({rl.ENV_BUDGET: "7,9,1.5,60"})
    assert (c.max_turns, c.max_tool_calls, c.max_usd, c.max_wall_s) == (7, 9, 1.5, 60.0)
    s = c.scaled(0.5)
    assert (s.max_turns, s.max_usd, s.max_wall_s) == (6, 0.75, 30.0)      # the turn floor


def test_lint():
    ok, warn = rl.lint_source(src(), ENTRY)
    assert ok == [] and warn == []
    errs, _ = rl.lint_source("void other() {}", ENTRY)
    assert "not defined" in errs[0]
    for bad in ("cudaDeviceSynchronize();", "cudaMemcpy(a, b, n, cudaMemcpyDeviceToHost);", "int main() {}", "#include <cublas_v2.h>",
                "launch_gpu_implementation();"):
        errs, _ = rl.lint_source(src() + bad, ENTRY)
        assert errs, bad
    assert rl.lint_source(src() + "cudaMemcpyAsync(a,b,n,cudaMemcpyDeviceToDevice);", ENTRY)[0] == []
    # streams / threads: fine in a region, refused elsewhere
    code = src() + "cudaStream_t s; cudaEventRecord(e, s); std::thread t;"
    assert rl.lint_source(code, ENTRY, region=True)[0] == []
    assert rl.lint_source(code, ENTRY, region=False)[0]
    assert rl.lint_source(src() + "cudaStreamSynchronize(s);", ENTRY)[1]          # a warning, not an error


def test_outline_of_a_composite_seed():
    o = rl.outline("namespace member0 {\n__global__ void k0(int a) {}\n}\n// current kernel of linear_f16 (fc0)\nnamespace member1 {}\nvoid kernel_region__x(int a) {}\n")
    assert "2 namespace block(s)" in o and "k0" in o and "current kernel of linear_f16" in o


# ---------------------------------------------------------------------------------------------- the loop
def test_happy_path_commit(tmp_path):
    fast = src("FAST v1")
    loop, llm, be, prob = make_loop(tmp_path, [
        act("show", what="region"),
        act("profile"),
        act("verify", code=fast, thought="a persistent kernel"),
        act("time"),
        act("commit"),
        act("finish", summary="done")])
    res = run(loop)
    assert res["stop"] == "finished" and res["summary"] == "done"
    assert len(res["commits"]) == 1 and res["commits"][0]["gain"] == pytest.approx(0.2, abs=1e-6)
    assert res["changed"] and loop.current[SID] == fast
    assert res["start_us"] == 100.0 and res["final_us"] == 80.0 and res["speedup"] == pytest.approx(1.25)
    assert res["final_check"]["ok"]
    led = res["ledger"]
    assert led["turns"] == 6 and led["tool_calls"] == 6
    assert set(led["per_tool"]) == {"show", "profile", "verify", "time", "commit", "finish"}
    assert led["per_tool"]["time"]["calls"] == 1 and led["prompt_tokens"] == 6000
    # the paired timing ran time_reps evaluations, each against the committed program
    paired = [c for c in be.calls if c[1]]
    assert len(paired) == 5 and all(t.startswith("rl_") for t, _ in be.calls)
    # transcript artefacts
    ev = [json.loads(l) for l in (tmp_path / "out" / "events.jsonl").read_text().splitlines()]
    assert {"start", "tool", "commit", "end", "verify", "timing"} <= {e["kind"] for e in ev}
    assert json.loads((tmp_path / "out" / "result.json").read_text())["speedup"] == pytest.approx(1.25)
    # the model was shown the region contract, the knowledge, and tool results alternate with replies
    first = llm.seen[0]
    assert first[0]["role"] == "system" and "REGION" in first[0]["content"] and "compute_100a" in first[0]["content"]
    assert "ncu" in first[0]["content"].lower() and "persistent" in first[0]["content"]
    assert "Lesson: persistent kernels win." in first[1]["content"]
    for msgs in llm.seen:
        roles = [m["role"] for m in msgs]
        assert roles[0] == "system" and roles[1] == "user"
        assert all(a != b for a, b in zip(roles[1:], roles[2:])), roles       # strict alternation (the relay's providers need it)


def test_verify_rejection_then_time_and_commit_refused(tmp_path):
    loop, llm, be, _ = make_loop(tmp_path, [
        act("verify", code=src("BROKEN")),
        act("time"),
        act("commit"),
        act("verify", code=src("COMPILEFAIL")),
        act("verify", code=src("FAST ok")),
        act("verify", code=src("NOCAPTURE")),
        act("verify", code=src("FLAKY")),
        act("verify", code="void nothing() {}"),
        act("finish")])
    res = run(loop)
    ev = [json.loads(l) for l in (tmp_path / "out" / "events.jsonl").read_text().splitlines() if '"kind": "tool"' in l]
    by = [(e["tool"], e["ok"]) for e in ev]
    assert by[:3] == [("verify", False), ("time", False), ("commit", False)]
    assert by[3:] == [("verify", False), ("verify", True), ("verify", False), ("verify", False), ("verify", False), ("finish", True)]
    assert "no passing `verify`" in ev[1]["result"] and "refused" in ev[2]["result"]
    assert not res["changed"] and res["commits"] == [] and loop.current[SID] == src_seed(loop)
    texts = " ".join(m["content"] for m in llm.seen[-1])
    assert "FAILED at stage `check`" in texts and "FAILED at stage `compile`" in texts
    assert "breaks CUDA-graph capture" in texts and "output mismatch" in texts and "static check failed" in texts


def src_seed(loop):
    return loop.start_sources[SID]


def test_commit_rule(tmp_path):
    # one 8% timing is not enough; the same source timed again (two consecutive gains) commits
    loop, llm, be, _ = make_loop(tmp_path, [act("verify", code=src("MEH")), act("time"), act("commit"), act("time"), act("commit"), act("finish")])
    res = run(loop)
    ev = [json.loads(l) for l in (tmp_path / "out" / "events.jsonl").read_text().splitlines() if '"kind": "tool"' in l]
    assert [(e["tool"], e["ok"]) for e in ev] == [("verify", True), ("time", True), ("commit", False), ("time", True), ("commit", True), ("finish", True)]
    assert "not yet" in ev[1]["result"] and "SATISFIED" in ev[3]["result"]
    assert len(res["commits"]) == 1 and res["commits"][0]["gain"] == pytest.approx(0.08)
    # a regression never commits, however often it is timed
    loop, llm, be, _ = make_loop(tmp_path / "b", [act("verify", code=src("SLOW")), act("time"), act("time"), act("commit"), act("finish")])
    res = run(loop)
    assert res["commits"] == [] and not res["changed"]
    # a big win commits on the first timing
    loop, llm, be, _ = make_loop(tmp_path / "c", [act("verify", code=src("XFAST")), act("time"), act("commit"), act("finish")])
    assert run(loop)["commits"][0]["gain"] == pytest.approx(0.4)


def test_new_source_resets_verification(tmp_path):
    loop, llm, be, _ = make_loop(tmp_path, [act("verify", code=src("FAST a")), act("time", code=src("FAST b")), act("finish")])
    run(loop)
    ev = [json.loads(l) for l in (tmp_path / "out" / "events.jsonl").read_text().splitlines() if '"kind": "tool"' in l]
    assert ev[1]["ok"] is False and "no passing `verify`" in ev[1]["result"]        # b was never verified


def test_unpaired_backend_alternates(tmp_path):
    be = FakeBackend(paired=False)
    loop, llm, be, _ = make_loop(tmp_path, [act("verify", code=src("FAST")), act("time"), act("commit"), act("finish")], backend=be)
    res = run(loop)
    assert len(res["commits"]) == 1
    tags = [t for t, _ in be.calls]
    assert sum("timeA" in t for t in tags) == 5 and sum("timeB" in t for t in tags) == 5
    assert [t.split("_")[-1][:5] for t in tags if "time" in t][:4] == ["timeA", "timeB", "timeA", "timeB"]        # alternating A B A B


def test_revert_pending_and_commits(tmp_path):
    loop, llm, be, _ = make_loop(tmp_path, [
        act("verify", code=src("FAST one")), act("time"), act("commit"),
        act("verify", code=src("XFAST two")), act("revert"),                       # drops the pending candidate
        act("time"),                                                                # nothing pending any more
        act("verify", code=src("XFAST three")), act("time"), act("commit"),
        act("revert", n=1),                                                         # undo "three": back to "one"
        act("status"), act("revert", n=1), act("revert", n=1), act("finish")])
    res = run(loop)
    ev = [json.loads(l) for l in (tmp_path / "out" / "events.jsonl").read_text().splitlines() if '"kind": "tool"' in l]
    by = [(e["tool"], e["ok"]) for e in ev]
    assert by[4] == ("revert", True) and by[5] == ("time", False)
    assert by[9] == ("revert", True)
    assert [e["kind"] for e in (json.loads(l) for l in (tmp_path / "out" / "events.jsonl").read_text().splitlines()) if e["kind"] == "revert"]
    assert by[11] == ("revert", True) and by[12] == ("revert", False)               # nothing left to revert
    # after the final undo the program is the seed again
    assert loop.current[SID] == src_seed(loop) and not res["changed"] and res["commits"] == []
    assert res["final_us"] == 100.0


def test_final_check_failure_reverts_to_the_start(tmp_path):
    class Flip(FakeBackend):
        async def evaluate(self, sources, tag, **kw):
            if tag.endswith("final") or "_final" in tag:
                return EvalResult(ok=False, stage="check", errors=["final composite broke"])
            return await super().evaluate(sources, tag, **kw)
    loop, llm, be, _ = make_loop(tmp_path, [act("verify", code=src("XFAST")), act("time"), act("commit"), act("finish")], backend=Flip())
    res = run(loop)
    assert res["final_check"]["ok"] is False and res["final_check"]["reverted_to_start"]
    assert loop.current[SID] == src_seed(loop) and res["final_us"] == 100.0 and not res["changed"]


# ---------------------------------------------------------------------------------------------- budgets
def test_budget_turns(tmp_path):
    loop, llm, be, _ = make_loop(tmp_path, [act("status")] * 10, cfg=rl.LoopConfig(max_turns=3))
    res = run(loop)
    assert res["stop"] == "turns" and res["ledger"]["turns"] == 3 and len(llm.replies) == 7


def test_budget_tool_calls(tmp_path):
    loop, llm, be, _ = make_loop(tmp_path, [act("status")] * 10, cfg=rl.LoopConfig(max_tool_calls=4))
    res = run(loop)
    assert res["stop"] == "tool_calls" and res["ledger"]["tool_calls"] == 4


def test_budget_usd_from_usage(tmp_path, monkeypatch):
    loop, llm, be, _ = make_loop(tmp_path, [act("status")] * 10, cfg=rl.LoopConfig(max_usd=0.25), usd=0.1)
    res = run(loop)
    assert res["stop"] == "usd" and res["ledger"]["turns"] == 3 and res["ledger"]["usd"] == pytest.approx(0.3)
    # without a reported cost: tokens x the price table
    assert rl.usage_usd({"input_tokens": 1_000_000, "output_tokens": 0}, "claude-opus-4-5") > 0          # the price table
    assert rl.usage_usd({"cost_usd": 1.5}, "x") == 1.5
    assert rl.usage_usd({}, "unknown-model") == 0.0                                                        # no tokens, no cost
    seen = []
    assert rl.usage_usd({"prompt_tokens": 1_000_000, "completion_tokens": 100_000}, "no-such-model", seen) == pytest.approx(5.0 + 2.5)
    assert seen == ["no-such-model"]                                                                       # never blind: the fallback price
    monkeypatch.setenv(rl.ENV_PRICE, "1,2")
    assert rl.usage_usd({"prompt_tokens": 1_000_000}, "no-such-model") == pytest.approx(1.0)


def test_budget_wall(tmp_path):
    loop, llm, be, _ = make_loop(tmp_path, [act("verify", code=src("FAST")), act("time"), act("commit")] + [act("status")] * 5,
                                 backend=FakeBackend(delay=0.06), cfg=rl.LoopConfig(max_wall_s=0.2))
    res = run(loop)
    assert res["stop"] == "wall"


def test_protocol_errors_stop_the_loop(tmp_path):
    loop, llm, be, _ = make_loop(tmp_path, ["I think so.", "Maybe?", act("status"), "nope", "no", "no", "no"], cfg=rl.LoopConfig(max_protocol_errors=3))
    res = run(loop)
    assert res["stop"] == "protocol_errors" and res["ledger"]["tool_calls"] == 1
    # the model is told what was wrong
    assert any("no action" in m["content"] for m in llm.seen[1] if m["role"] == "user")


def test_llm_error_ends_the_loop_not_the_program(tmp_path):
    loop, llm, be, _ = make_loop(tmp_path, [act("status")])                   # the script runs dry on turn 2: RuntimeError
    res = run(loop)
    assert res["stop"] == "llm_error" and res["ledger"]["turns"] == 1


def test_tool_crash_is_a_failed_call(tmp_path):
    class Boom(FakeBackend):
        async def evaluate(self, *a, **k):
            raise ValueError("gpu fell over")
    loop, llm, be, _ = make_loop(tmp_path, [act("profile"), act("finish")], backend=Boom())
    res = run(loop)
    ev = [json.loads(l) for l in (tmp_path / "out" / "events.jsonl").read_text().splitlines() if '"kind": "tool"' in l]
    assert ev[0]["ok"] is False and "gpu fell over" in ev[0]["result"] and res["stop"] == "finished"


def test_scratch_and_its_budget(tmp_path):
    prog = "#include <cstdio>\nint main(){ printf(\"hi\"); }"
    loop, llm, be, _ = make_loop(tmp_path, [act("scratch", code=prog), act("scratch"), act("scratch", code=prog), act("scratch", code=prog), act("finish")],
                                 cfg=rl.LoopConfig(scratch_runs=2))
    res = run(loop)
    assert len(be.scratch_calls) == 2 and res["ledger"]["scratch_runs"] == 2
    ev = [json.loads(l) for l in (tmp_path / "out" / "events.jsonl").read_text().splitlines() if '"kind": "tool"' in l]
    assert [e["ok"] for e in ev[:4]] == [True, False, True, False]
    assert "scratch budget used up" in ev[3]["result"] and "412 cycles" in ev[0]["result"]


def test_show_pages_and_outline(tmp_path):
    big = "\n".join(f"// line {i}" for i in range(500)) + "\n" + SEED
    prob = make_problem(tmp_path / "prob")
    loop = rl.RegionLoop(problem=prob, sid=SID, backend=FakeBackend(), llm=Script([]), model="m", sources={**prob.seed_sources(), SID: big},
                         baseline_us=100.0, cfg=rl.LoopConfig(), knowledge_text="")
    ok, t = asyncio.run(loop.t_show(rl.Action("show", {"what": "source", "limit": 5})))
    assert ok and "line 0" in t and "offset=5" in t and "line 5" not in t.split("\n[lines")[0]
    ok, t = asyncio.run(loop.t_show(rl.Action("show", {"what": "source", "offset": 498, "limit": 50})))
    assert "[end: 503 lines]" in t
    ok, t = asyncio.run(loop.t_show(rl.Action("show", {"what": "region"})))
    assert SIG in t and "REGION slot" in t and "region:fc0..relu1" in t
    assert not asyncio.run(loop.t_show(rl.Action("show", {"what": "candidate"})))[0]
    assert not asyncio.run(loop.t_show(rl.Action("show", {"what": "nope"})))[0]


def test_compaction_elides_old_code_and_results(tmp_path):
    loop, llm, be, _ = make_loop(tmp_path, [act("verify", code=src("FAST " + "x" * 50 + str(i))) for i in range(5)] + [act("finish")],
                                 cfg=rl.LoopConfig(keep_full_turns=1))
    run(loop)
    last = llm.seen[-1]
    assistant = [m["content"] for m in last if m["role"] == "assistant"]
    assert "elided" in assistant[0] and "elided" not in assistant[-1]
    assert sum(len(m["content"]) for m in last) < sum(len(m["content"]) for m in loop.messages)           # the stored history is longer
    # the full candidate is still retrievable
    ok, t = asyncio.run(loop.t_show(rl.Action("show", {"what": "candidate"})))
    assert ok and "FAST" in t


def test_one_action_per_turn_is_said(tmp_path):
    two = act("status") + "\n" + act("show")
    loop, llm, be, _ = make_loop(tmp_path, [two, act("finish")])
    run(loop)
    assert "only the first of 2 actions was run" in llm.seen[1][-1]["content"]


# ---------------------------------------------------------------------------------------------- the agent integration
class AgentEv(Evaluator):
    """The agent's own evaluator (baseline / composite runs): every run passes, the region slot's time follows its source."""

    def __init__(self):
        self.calls = []

    async def evaluate(self, problem, sources, tag, *, only_slots=None, screen=False, incumbent=None):
        self.calls.append(tag)
        b = behave(sources[SID])
        us = b["us"]
        return EvalResult(ok=True, stage="ok", partition_us=us + 3, slot_us={SID: us / 2, "relu": 3.0}, e2e_us=us, e2e_basis="graph",
                          attr_slot_us={SID: us / 2, "relu": 3.0}, slot_checks={SID: {"ok": True, "rel_err": 1e-4}, "relu": {"ok": True, "rel_err": 0}})


def _agent(tmp_path, region_loop, replies=None, **kw):
    prob = make_problem(tmp_path / "prob")
    store = ImplStore(tmp_path / "store")
    olog = OutcomeLog(tmp_path / "outcomes.jsonl")
    rb = FakeBackend() if region_loop is not None else None
    llm = Script(replies or [])
    ag = MultiSlotAgent(prob, AgentEv(), model="stub:x", out_dir=tmp_path / "run", arch="100", impl_store=store, outcome_log=olog,
                        rollouts=1, steps=0, propose=False, region_loop=region_loop, region_backend=rb, region_llm=llm if region_loop else None, **kw)
    return ag, olog, store, llm, rb


def test_agent_runs_the_region_loop_and_feeds_knowledge(tmp_path):
    cfg = rl.LoopConfig(max_turns=12)
    fast = src("XFAST agent")
    ag, olog, store, llm, rb = _agent(tmp_path, cfg, [
        act("verify", code=src("BROKEN")), act("verify", code=fast), act("time"), act("commit"), act("finish", summary="persistent kernel")])
    res = asyncio.run(ag.run())
    assert res["final_ok"] and res["slots"][SID]["best_us"] < 50
    assert res["e2e_speedup"] == pytest.approx(100 / 60, rel=1e-3)
    rlr = res["region_loop"]
    assert rlr["commits"] == 1 and rlr["slots"][SID]["stop"] == "finished" and rlr["usd"] == pytest.approx(0.05)   # no price for "stub:x": the fallback price
    assert ag.best[SID].source == fast and "region_loop" in ag.best[SID].techniques
    assert (tmp_path / "run" / "best" / "slots" / f"{SID}.cu").read_text() == fast
    # usage joins the problem's totals under its own role
    assert any(u["role"] == "region_loop" and u["calls"] == 5 for u in res["llm_usage_by_model"])
    # outcome rows: a failed verify (wrong output), a timing (improved) -- technique region_loop, e2e basis
    rows = [r for r in olog if r.technique == "region_loop"]
    assert [r.status for r in rows] == ["wrong_output", "improved"]
    imp = rows[-1]
    assert imp.slot == SID and imp.speedup == pytest.approx(1 / 0.6) and imp.timing_basis.startswith("e2e") and imp.diff_summary
    # the committed code is in the implementation store (verified)
    assert any(r.slot == SID for r in store.records())
    # the loop saw the flow's knowledge section (technique menu)
    assert "Techniques the flow knows" in llm.seen[0][1]["content"]
    # no generic rollout touched the region slot (steps=0 here, but it is marked handled)
    assert SID in ag._region_handled and ag._pick_slot() != SID


def test_agent_flag_off_is_unchanged(tmp_path):
    ag, olog, store, llm, rb = _agent(tmp_path, None)
    res = asyncio.run(ag.run())
    assert "region_loop" not in res and "region_loop" not in ag._snapshot()
    assert not (tmp_path / "run" / "region_loop").exists()
    assert ag._region_handled == set() and res["final_ok"]
    assert all(u["role"] != "region_loop" for u in res["llm_usage_by_model"])


def test_agent_without_region_slots_runs_no_loop(tmp_path):
    prob = make_problem(tmp_path / "prob", extra_slot=True)
    # a problem whose region slot is renamed away: no region slot, the loop never starts and the llm is never called
    m = json.loads((prob.root / "manifest.json").read_text())
    for s in m["slots"]:
        s["id"] = s["id"].replace("region__", "plain__")
    for o in m["ops"]:
        o["slot"] = o["slot"].replace("region__", "plain__")
    m["assembly"]["order"] = [x.replace("region__", "plain__") for x in m["assembly"]["order"]]
    (prob.root / "slots" / f"{SID.replace('region__', 'plain__')}.cu").write_text(SEED)
    (prob.root / "manifest.json").write_text(json.dumps(m))
    prob = SlotProblem(prob.root)
    llm = Script([])
    ag = MultiSlotAgent(prob, AgentEv(), model="stub:x", out_dir=tmp_path / "run", arch="100", rollouts=1, steps=0, propose=False,
                        region_loop=rl.LoopConfig(), region_backend=FakeBackend(), region_llm=llm)

    class Ev2(AgentEv):
        async def evaluate(self, problem, sources, tag, **kw):
            r = await super().evaluate(problem, {**sources, SID: SEED}, tag, **kw)
            r.slot_us = {"plain__ops__abc123_f16": 10.0, "relu": 3.0}
            r.attr_slot_us = dict(r.slot_us)
            r.slot_checks = {k: {"ok": True, "rel_err": 0} for k in r.slot_us}
            return r
    ag.ev = Ev2()
    asyncio.run(ag.run())
    assert llm.seen == [] and ag._region_done == {}


def test_agent_snapshot_roundtrip_skips_done_slots(tmp_path):
    ag, olog, store, llm, rb = _agent(tmp_path, rl.LoopConfig(max_turns=4), [act("finish")])
    asyncio.run(ag.run())
    snap = ag._snapshot()
    assert snap["region_loop"]["done"][SID]["stop"] == "finished"
    ag2, *_ = _agent(tmp_path / "again", rl.LoopConfig(max_turns=4), [])
    ag2._restore(json.loads(json.dumps(snap, default=str)))
    assert ag2._region_handled == {SID}


# ---------------------------------------------------------------------------------------------- run_multislot plumbing
def _load_run_multislot():
    spec = importlib.util.spec_from_file_location("run_multislot_rl", Path(__file__).resolve().parents[1] / "scripts" / "run_multislot.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_multislot_rl"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_run_multislot_flag(monkeypatch):
    rm = _load_run_multislot()
    monkeypatch.delenv(rl.ENV, raising=False)
    monkeypatch.delenv(rl.ENV_BUDGET, raising=False)
    off = SimpleNamespace(region_loop=None, region_loop_budget=None)
    assert rm.region_loop_config(off) is None
    on = SimpleNamespace(region_loop="on", region_loop_budget="9,30,2,120")
    c = rm.region_loop_config(on)
    assert (c.max_turns, c.max_tool_calls, c.max_usd, c.max_wall_s) == (9, 30, 2.0, 120.0)
    monkeypatch.setenv(rl.ENV, "1")
    assert rm.region_loop_config(off).max_turns == 40                     # env on, default budget
    monkeypatch.setenv(rl.ENV, "off")
    assert rm.region_loop_config(SimpleNamespace()) is None
    # the parser knows the flags; default off
    ap = rm.build_parser()
    a = ap.parse_args(["x", "--out", "o"])
    assert a.region_loop is None and a.region_loop_budget is None
    a = ap.parse_args(["x", "--out", "o", "--region-loop", "on", "--region-loop-budget", "5,5,1,60"])
    assert a.region_loop == "on"


# ---------------------------------------------------------------------------------------------- the flow's own LLM path
def test_loop_over_the_flows_llm_path_with_a_stub_model(tmp_path):
    """``default_llm`` = generate_code_retry -> query.generate_code -> the ``stub:`` provider (what the relay / the provider paths replace):
    plain chat messages in, text + usage out. The scripted model also exercises the usage -> budget path."""
    from kernelblaster.agents.utils import agentic_llm
    replies = [act("show"), act("verify", code=src("XFAST stub")), act("time"), act("commit"), act("finish")]
    seen = []

    def stub(messages):
        seen.append(messages)
        return {"text": replies[len(seen) - 1], "usage": {"prompt_tokens": 500, "completion_tokens": 100, "cost_usd": 0.5}}
    agentic_llm.register_stub("region_script", stub)
    try:
        prob = make_problem(tmp_path / "prob")
        loop = rl.RegionLoop(problem=prob, sid=SID, backend=FakeBackend(), llm=rl.default_llm("stub:region_script"), model="stub:region_script",
                             sources=prob.seed_sources(), baseline_us=100.0, cfg=rl.LoopConfig(max_usd=1.9), out_dir=tmp_path / "o")
        res = run(loop)
    finally:
        agentic_llm.unregister_stub("region_script")
    assert res["stop"] == "usd" and res["ledger"]["usd"] == pytest.approx(2.0) and len(res["commits"]) == 1     # $0.5 per call: stopped after 4
    assert all(isinstance(m["content"], str) and m["role"] in ("system", "user", "assistant") for m in seen[-1])
    assert seen[0][0]["role"] == "system"


# ---------------------------------------------------------------------------------------------- edit tool, truncation, ssh
def edit_block(old, new):
    return "```edit\n<<<<<<< SEARCH\n" + old + "\n=======\n" + new + "\n>>>>>>> REPLACE\n```"


def test_parse_edit_forms():
    a = rl.parse_reply('```action\n{"tool": "edit"}\n```\n' + edit_block("a b", "c d") + "\n" + edit_block("x\ny", "z"))
    assert a.edits == [("a b", "c d"), ("x\ny", "z")]
    a = rl.parse_reply('```action\n{"tool": "edit", "args": {"edits": [{"old": "p", "new": "q"}]}}\n```')
    assert a.edits == [("p", "q")]
    with pytest.raises(rl.ProtocolError, match="SEARCH"):
        rl.parse_reply('```action\n{"tool": "edit"}\n```')


def test_edit_tool(tmp_path):
    base = src("v1")
    loop, llm, be, _ = make_loop(tmp_path, [
        act("verify", code=src("FAST v1")),
        '```action\n{"tool": "edit"}\n```\n' + edit_block("// FAST v1", "// XFAST v2"),         # applied to the pending candidate
        act("time"),                                                                        # v2 is unverified: refused
        act("verify"),                                                                      # verify the edited pending
        act("time"), act("commit"),
        '```action\n{"tool": "edit"}\n```\n' + edit_block("not in the source", "x"),     # not found
        '```action\n{"tool": "edit"}\n```\n' + edit_block("const __half", "__half"),       # matches two places: ambiguous
        '```action\n{"tool": "edit"}\n```\n' + edit_block("XFAST v2", "XFAST v3"),        # on the committed source (nothing pending)
        act("finish")])
    res = run(loop)
    ev = [json.loads(l) for l in (tmp_path / "out" / "events.jsonl").read_text().splitlines() if '"kind": "tool"' in l]
    by = [(e["tool"], e["ok"]) for e in ev]
    assert by == [("verify", True), ("edit", True), ("time", False), ("verify", True), ("time", True), ("commit", True),
                  ("edit", False), ("edit", False), ("edit", True), ("finish", True)]
    assert "applied to the pending candidate" in ev[1]["result"] and "applied to the committed source" in ev[8]["result"]
    assert "not found" in ev[6]["result"] and "matches 2 places" in ev[7]["result"]
    assert len(res["commits"]) == 1 and "XFAST v2" in loop.current[SID]
    assert loop.pending is not None and "XFAST v3" in loop.pending.code and loop.pending.verified is None      # edit resets verification
    assert res["final_check"]["ok"]


def test_truncated_reply_is_not_executed(tmp_path):
    cut = ("```action\n{\"tool\": \"verify\", \"args\": {}}\n```\n```cuda\nvoid kernel_x() { /* cut here", {"prompt_tokens": 10, "completion_tokens": 8192,
                                                                                                             "telemetry": {"stop_reasons": ["max_tokens"]}})
    loop, llm, be, _ = make_loop(tmp_path, [cut, act("status"), act("finish")])
    res = run(loop)
    assert res["ledger"]["truncated_replies"] == 1 and res["ledger"]["tool_calls"] == 2 and be.calls == []
    assert "output-token limit" in llm.seen[1][-1]["content"] and "edit" in llm.seen[1][-1]["content"]
    assert rl.truncated({"telemetry": {"stop_reason": "length"}}) and not rl.truncated({"telemetry": {"stop_reasons": ["end_turn"]}})


def test_scratch_refused_on_an_ssh_evaluator(tmp_path):
    class Ssh(Evaluator):
        def _ssh_base(self):
            return []
    prob = make_problem(tmp_path / "p")
    be = rl.EvaluatorBackend(Ssh(), prob, "100", tmp_path / "w")
    r = asyncio.run(be.scratch("int main(){}", 30))
    assert not r["ok"] and "over ssh" in r["out"]
    assert asyncio.run(be.ncu({}, "t", SID))[0] is None


def test_clock_suspect_timing_is_not_evidence(tmp_path):
    class Hot(FakeBackend):
        async def evaluate(self, sources, tag, **kw):
            r = await super().evaluate(sources, tag, **kw)
            if "_time" in tag:
                r.gpu = {"clock": {"suspect": True}}
            return r
    loop, llm, be, _ = make_loop(tmp_path, [act("verify", code=src("XFAST")), act("time"), act("commit"), act("finish")], backend=Hot())
    res = run(loop)
    ev = [json.loads(l) for l in (tmp_path / "out" / "events.jsonl").read_text().splitlines() if '"kind": "tool"' in l]
    assert ev[1]["ok"] is False and "clock guard" in ev[1]["result"] and not res["commits"]
