"""Telemetry (docs/telemetry.md): spans, collectors, LLM token records,
truncation detection, evaluator instrumentation, the report CLI."""
import asyncio
import json
import time

import pytest

from kernelblaster import telemetry as tel
from kernelblaster.agents.utils import agentic_llm
from kernelblaster.agents.utils import query
from kernelblaster.knowledge.slots import EvalResult, Evaluator
from kernelblaster.telemetry import report


@pytest.fixture(autouse=True)
def _telemetry_on():
    tel.set_enabled(True)
    yield
    tel.set_enabled(True)
    tel.close()


def test_nesting_self_time_and_attribution_sum_to_wall():
    with tel.span("step", collect=True) as st:
        with tel.span("build"):
            time.sleep(0.03)
            with tel.span("build.nvcc"):
                time.sleep(0.05)
        with tel.span("run.exec"):
            time.sleep(0.02)
        time.sleep(0.01)                                   # untracked: the step's own time
    s = tel.summary_of(st)
    assert s["wall_s"] == pytest.approx(sum(s["time_s"].values()), abs=5e-3)
    assert s["time_s"]["build.nvcc"] == pytest.approx(0.05, abs=0.02)
    assert s["time_s"]["build"] == pytest.approx(0.03, abs=0.02)      # exclusive of build.nvcc
    assert s["busy_s"]["build"] == pytest.approx(s["time_s"]["build"], abs=1e-3)
    assert s["time_s"]["step"] == pytest.approx(0.01, abs=0.01)
    assert s["gpu_busy_s"]["default"] == pytest.approx(0.02, abs=0.01)


def test_concurrent_tasks_keep_their_own_spans():
    async def rollout(i):
        with tel.span("step", collect=True, laps=True, rollout=i) as st:
            tel.lap("prompt.build", slot=f"s{i}")
            await asyncio.sleep(0.02)
            tel.lap("candidate")
            with tel.span(f"work{i}"):
                await asyncio.sleep(0.05 * (i + 1))
            await asyncio.to_thread(time.sleep, 0.01)       # threads inherit the context too
            with tel.span("io.x"):
                pass
            tel.end_laps(st)
        return st

    async def main():
        with tel.span("problem", collect=True, problem="p") as ps:
            sts = await asyncio.gather(*[rollout(i) for i in range(3)])
        return ps, sts

    ps, sts = asyncio.run(main())
    for i, st in enumerate(sts):
        s = tel.summary_of(st)
        assert set(s["busy_s"]) == {"prompt.build", "candidate", f"work{i}", "io.x"}   # nothing from siblings
        assert s["time_s"][f"work{i}"] == pytest.approx(0.05 * (i + 1), abs=0.02)
        assert sum(s["time_s"].values()) == pytest.approx(s["wall_s"], abs=5e-3)
    p = tel.summary_of(ps)
    # concurrent leaves share the wall clock; busy sums every rollout's time
    assert sum(p["time_s"].values()) == pytest.approx(p["wall_s"], abs=5e-3)
    assert p["busy_s"]["prompt.build"] == pytest.approx(0.06, abs=0.03)
    assert p["time_s"]["prompt.build"] == pytest.approx(0.02, abs=0.01)
    assert p["wall_s"] == pytest.approx(0.02 + 0.15 + 0.01, abs=0.05)


def test_disabled_is_a_noop(tmp_path):
    tel.set_enabled(False)
    tel.configure(tmp_path / "t.jsonl")
    assert tel.sink() is None
    with tel.span("step", collect=True) as st:
        tel.lap("x")
        tel.count("win")
    assert st is tel.NOSPAN and tel.summary_of(st) is None and tel.full_summary(st) is None
    assert tel.ctx_summary(st) is None
    assert not (tmp_path / "t.jsonl").exists()


def test_normalize_usage_providers_and_truncation():
    oa = tel.normalize_usage({"prompt_tokens": 1000, "completion_tokens": 50,
                              "prompt_tokens_details": {"cached_tokens": 600},
                              "telemetry": {"stop_reasons": ["length"], "max_tokens": 50}}, "gpt-4o")
    assert oa["input_tokens"] == 1000 and oa["cache_read_input_tokens"] == 600 and oa["uncached_input_tokens"] == 400
    assert oa["truncated"] == 1 and oa["stop_reason"] == "length"
    br = tel.normalize_usage({"input_tokens": 100, "output_tokens": 16384, "cache_read_input_tokens": 900,
                              "cache_write_input_tokens": 50,
                              "telemetry": {"input_includes_cache": False, "stop_reasons": ["max_tokens"],
                                            "max_tokens": 16384, "server_latency_ms": 2500, "provider_retries": 1}},
                             "us.anthropic.claude-haiku-4-5-20251001-v1:0")
    assert br["input_tokens"] == 1050 and br["uncached_input_tokens"] == 100
    assert br["truncated"] == 1 and br["server_latency_s"] == 2.5 and br["provider_retries"] == 1
    # haiku 4.5: $1 in, $5 out, $0.10 cache read, $1.25 cache write per 1M
    assert br["est_usd"] == pytest.approx((100 * 1 + 900 * 0.1 + 50 * 1.25 + 16384 * 5) / 1e6)
    # no stop reason: output at the cap counts as truncated
    assert tel.normalize_usage({"input_tokens": 1, "output_tokens": 8192,
                                "telemetry": {"max_tokens": 8192}}, "m")["truncated"] == 1
    assert tel.normalize_usage({"input_tokens": 1, "output_tokens": 10,
                                "telemetry": {"stop_reasons": ["end_turn"], "max_tokens": 8192}}, "m")["truncated"] == 0
    cli = tel.normalize_usage({"prompt_tokens": 500, "completion_tokens": 9, "cost_usd": 0.01,
                               "cache_read_input_tokens": 300, "cache_creation_input_tokens": 100}, "claude-code:haiku")
    assert cli["input_tokens"] == 500 and cli["uncached_input_tokens"] == 100 and cli["cost_usd"] == 0.01


def test_stub_llm_calls_aggregate_by_role_and_model(tmp_path):
    replies = iter([
        {"text": "a", "usage": {"prompt_tokens": 100, "completion_tokens": 10, "stop_reason": "end_turn"}},
        {"text": "b", "usage": {"prompt_tokens": 200, "completion_tokens": 4096, "stop_reason": "max_tokens"}},
        "plain",
    ])
    agentic_llm.register_stub("tel", lambda m: next(replies))
    tel.configure(tmp_path / "telemetry.jsonl", run_id="r1")

    async def main():
        with tel.span("step", collect=True, step=3) as st:
            with tel.attrs(role="codegen_hard"):
                await query.generate_code_retry([{"role": "user", "content": "x" * 40}], "stub:tel")
            with tel.attrs(role="fix"):
                await query.generate_code_retry([{"role": "user", "content": "y"}], "stub:tel")
            r = await query.generate_code_retry([{"role": "user", "content": "z"}], "stub:tel")
        return st, r

    try:
        st, r = asyncio.run(main())
    finally:
        agentic_llm.unregister_stub("tel")
    assert r.generations == ["plain"]
    s = tel.summary_of(st)
    t = s["llm"]["total"]
    assert t["calls"] == 3 and t["input_tokens"] == 300 and t["output_tokens"] == 4106 and t["truncated"] == 1
    by = {b["role"]: b for b in s["llm"]["by_role_model"]}
    assert set(by) == {"codegen_hard", "fix", "other"} and by["fix"]["truncated"] == 1
    assert all(b["model"] == "stub:tel" for b in by.values())
    assert {"llm.codegen_hard", "llm.fix", "llm.other"} <= set(s["busy_s"])
    tel.close()
    evs = [json.loads(x) for x in (tmp_path / "telemetry.jsonl").read_text().splitlines()]
    llm = [e for e in evs if e["t"] == "llm"]
    assert [e["role"] for e in llm] == ["codegen_hard", "fix", "other"]
    assert llm[0]["prompt_chars"] == 40 and llm[0]["step"] == 3 and llm[1]["stop_reason"] == "max_tokens"
    assert all(e["run"] == "r1" for e in evs)


def test_backoff_and_failed_attempts_are_recorded(monkeypatch):
    calls = {"n": 0}

    def flaky(m):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("rate limit exceeded")
        return "ok"

    real_sleep = asyncio.sleep
    monkeypatch.setattr(query.random, "uniform", lambda a, b: 0.0)
    monkeypatch.setattr(asyncio, "sleep", lambda d, *a, **k: real_sleep(0))
    agentic_llm.register_stub("flaky", flaky)

    async def main():
        with tel.span("step", collect=True) as st:
            with tel.attrs(role="reflect"):
                await query.generate_code_retry([{"role": "user", "content": "q"}], "stub:flaky", max_retries=3)
        return st

    try:
        st = asyncio.run(main())
    finally:
        agentic_llm.unregister_stub("flaky")
    b = tel.summary_of(st)["llm"]["by_role_model"][0]
    assert b["role"] == "reflect" and b["calls"] == 2 and b["errors"] == 1
    assert b["backoffs"] == 1 and b["backoff_s"] == pytest.approx(1.0)


def test_bedrock_usage_and_truncation(monkeypatch):
    class FakeBedrock:
        def converse(self, **kw):
            assert kw["inferenceConfig"]["maxTokens"] == 64
            return {"output": {"message": {"content": [{"text": "```cuda\nint x;"}]}},
                    "usage": {"inputTokens": 20, "outputTokens": 64, "cacheReadInputTokens": 500,
                              "cacheWriteInputTokens": 0},
                    "stopReason": "max_tokens", "metrics": {"latencyMs": 1234},
                    "ResponseMetadata": {"RetryAttempts": 2}}

    monkeypatch.setattr(query, "bedrock_client", FakeBedrock())
    monkeypatch.setenv("BEDROCK_MAX_TOKENS", "64")
    model = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

    async def main():
        with tel.span("step", collect=True) as st:
            with tel.attrs(role="codegen_simple"):
                r = await query.generate_code_retry([{"role": "user", "content": "hi"}], model)
        return st, r

    st, r = asyncio.run(main())
    assert r.usage["input_tokens"] == 20 and r.usage["output_tokens"] == 64      # legacy keys unchanged
    t = tel.summary_of(st)["llm"]["total"]
    assert t["input_tokens"] == 520 and t["cache_read_input_tokens"] == 500 and t["truncated"] == 1
    assert t["server_latency_s"] == pytest.approx(1.234) and t["provider_retries"] == 2


class _FakeEval(Evaluator):
    def __init__(self):
        self.builds = 0

    async def _build(self, problem, sources, work):
        self.builds += 1
        await asyncio.sleep(0.01)
        return None

    async def _run(self, exe, env=None, prefix="", device=None, problem=None):
        await asyncio.sleep(0.05)
        return ("MB_SLOT_US a 3.0\nMB_PARTITION_US 3.0\nMB_PHASE_MS start 1.0\nMB_PHASE_MS cuda_init 20.0\n"
                "MB_PHASE_MS check_ops 10.0\nMB_PHASE_MS time_ops 5.0\nMB_PHASE_MS total 36.0\npassed\n")

    async def evaluate(self, problem, sources, tag, *, only_slots=None, screen=False):
        bad = await self._build(problem, sources, None)
        from kernelblaster.knowledge.slots import parse_driver_output
        return bad or parse_driver_output(await self._run("exe"))

    async def profile(self, problem, sources, tag, slots=None):
        await self._run("exe", prefix="ncu")
        return {}, "no ncu here"


class _SubEval(_FakeEval):
    async def _build(self, problem, sources, work):                    # super() call: still one span
        return await super()._build(problem, sources, work)


def test_evaluator_boundary_spans_purposes_and_driver_phases():
    ev = _SubEval()

    async def main():
        with tel.span("step", collect=True) as st:
            for tag in ("r0_s1_relu_tech_f0", "r0_s1_relu_tech_f1", "r0_s1_relu_confirm", "baseline",
                        "final_composite"):
                res = await ev.evaluate(None, {}, tag)
                assert isinstance(res, EvalResult) and res.ok
            await ev.profile(None, {}, "r0_s1_relu")
            with tel.attrs(purpose="remeasure"):
                await ev.evaluate(None, {}, "anything")
        return st

    st = asyncio.run(main())
    s = tel.summary_of(st)
    assert {k: v["n"] for k, v in s["evals"].items()} == {
        "candidate": 1, "fix": 1, "confirm": 1, "baseline": 1, "final": 1, "profile": 1, "remeasure": 1}
    assert s["evals"]["profile"]["failed"] == 1
    assert ev.builds == 6
    assert s["busy_s"]["build"] == pytest.approx(0.06, abs=0.03)                # one span per build
    for k in ("driver.cuda_init", "driver.check_ops", "driver.time_ops", "driver.start", "ncu.driver.cuda_init",
              "run.exec", "ncu.exec"):
        assert k in s["time_s"], k
    assert s["time_s"]["driver.cuda_init"] == pytest.approx(6 * 0.020, abs=0.01)
    assert s["counters"]["eval.candidate"] == 1
    cats = tel.by_category(s["time_s"])
    assert {"checks", "timing", "driver_startup", "process_overhead", "profile_ncu", "compile"} <= set(cats)
    assert sum(s["time_s"].values()) == pytest.approx(s["wall_s"], abs=5e-3)


def test_purpose_of_tags():
    assert tel.purpose_of("evaluate", "r1_s2_conv_implicit_gemm_f0") == "candidate"
    assert tel.purpose_of("evaluate", "r1_s2_conv_implicit_gemm_f2") == "fix"
    assert tel.purpose_of("evaluate", "r1_s2_conv_confirm") == "confirm"
    assert tel.purpose_of("evaluate", "store_seeds") == "seed"
    assert tel.purpose_of("evaluate", "final_repaired") == "final"
    assert tel.purpose_of("profile", "baseline") == "profile"


def test_category_mapping():
    assert tel.category("llm.codegen_hard") == "llm_wait"
    assert tel.category("llm.backoff") == "llm_backoff"
    assert tel.category("build.queue") == "build_queue"
    assert tel.category("build.nvcc") == "compile" and tel.category("build") == "compile"
    assert tel.category("buildx") == "agent"
    assert tel.category("driver.check_partition") == "checks"
    assert tel.category("ncu.driver.time_ops") == "profile_ncu"
    assert tel.category("upkeep.hypothesis") == "knowledge"
    assert tel.category("kb.run") == "agent" and tel.category("mb.kb_wait") == "kb_wait"


def test_report_cli(tmp_path, capsys):
    tel.configure(tmp_path / "a" / "telemetry.jsonl", run_id="ra")
    agentic_llm.register_stub("cut", lambda m: {"text": "x", "usage": {"prompt_tokens": 7, "completion_tokens": 99,
                                                                         "stop_reason": "max_tokens"}})

    async def main():
        with tel.span("kb.run", collect=True):
            with tel.span("problem", collect=True, problem="p1"):
                with tel.span("step", collect=True, step=0, slot="relu"):
                    tel.count("step")
                    with tel.attrs(role="codegen_simple"):
                        await query.generate_code_retry([{"role": "user", "content": "q"}], "stub:cut")
                    with tel.span("build.nvcc"):
                        await asyncio.sleep(0.03)

    try:
        asyncio.run(main())
    finally:
        agentic_llm.unregister_stub("cut")
    tel.close()
    # a second run in another directory, and a torn last line
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "telemetry.jsonl").write_text(
        json.dumps({"t": "span", "id": 1, "parent": None, "name": "gpu.queue", "ts": 10.0, "dur": 2.0,
                    "self": 2.0, "run": "rb"}) + "\n{\"t\": \"sp")
    assert report.main([str(tmp_path), "--by-problem", "--top", "5"]) == 0
    out = capsys.readouterr().out
    assert "Where the wall clock goes" in out and "compile" in out and "gpu_queue" in out
    assert "Truncated LLM outputs (1)" in out and "codegen_simple" in out
    assert "problem p1" in out and "combined over runs" in out
    js = json.loads(report.report([tmp_path / "a"], as_json=True))
    (key,) = js
    s = js[key]["run"]
    assert s["llm"]["total"]["truncated"] == 1 and s["derived"]["steps"] == 1
    assert s["wall_s"] == pytest.approx(sum(s["time_s"].values()), abs=5e-3)


def test_span_overhead_is_small():
    n = 2000
    t0 = time.perf_counter()
    with tel.span("step", collect=True) as st:
        for i in range(n):
            with tel.span("x"):
                pass
    per = (time.perf_counter() - t0) / n
    assert per < 2e-4, per                          # well under 0.2 ms per span (a step has ~60)
    assert tel.summary_of(st)["busy_s"]["x"] >= 0
