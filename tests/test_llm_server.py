"""LLM relay server (servers/llm_server.py) + client (agents/utils/llm_relay.py)
with a fake provider in the server: auth, the bind rules, concurrency, budget,
idempotent retries, a dropped connection, fallback, telemetry parity, and the
multi-slot agent end to end on the LeNet fixture (GPU)."""
import asyncio
import json
import secrets
import shutil
import socket
import threading
import time

import aiohttp
import pytest
import uvicorn

from kernelblaster import telemetry as tel
from kernelblaster.agents.utils import agentic_llm, llm_relay, query
from kernelblaster.servers import llm_server as ls

MODEL = "us.anthropic.claude-haiku-5-5"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeProvider:
    """Bedrock-shaped responses; counts calls and the peak concurrency."""

    def __init__(self, delay=0.0, usage=None, fail=None):
        self.calls = 0
        self.active = self.peak = 0
        self.delay, self.fail = delay, fail
        self.usage = usage or {"input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500,
                               "cache_read_input_tokens": 400, "cache_write_input_tokens": 100,
                               "telemetry": {"input_includes_cache": False, "stop_reasons": ["max_tokens"],
                                             "max_tokens": 300, "server_latency_ms": 850.0,
                                             "provider_retries": 1}}

    async def __call__(self, messages, n_tasks, model):
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.fail is not None:
                e = self.fail(self.calls)
                if e is not None:
                    raise e
            await asyncio.sleep(self.delay)
            return query.LLMResponse(messages, [f"echo {messages[-1]['content']}"] * n_tasks,
                                     json.loads(json.dumps(self.usage)), model, n_tasks, self.delay)
        finally:
            self.active -= 1


class Server:
    def __init__(self, settings: ls.Settings, generate):
        settings.port = settings.port if settings.port != 18081 else _free_port()
        self.settings = settings
        self.app = ls.create_app(settings, generate)
        self.svc = self.app.state.svc
        self.srv = uvicorn.Server(uvicorn.Config(self.app, host=settings.host, port=settings.port,
                                                 log_level="warning", access_log=False))
        self.th = threading.Thread(target=self.srv.run, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.settings.port}"

    def __enter__(self):
        self.th.start()
        t0 = time.time()
        while not self.srv.started:
            assert time.time() - t0 < 10, "server did not start"
            time.sleep(0.02)
        return self

    def __exit__(self, *a):
        self.srv.should_exit = True
        self.th.join(10)


@pytest.fixture
def token():
    return secrets.token_urlsafe(24)


@pytest.fixture
def relay_env(monkeypatch, token):
    for k in ("KB_LLM_SERVER_URL", "KB_LLM_SERVER_TOKEN", "KB_LLM_SERVER_TOKEN_FILE", "KB_LLM_SERVER_FALLBACK"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("KB_LLM_SERVER_TOKEN", token)
    monkeypatch.setenv("KB_LLM_SERVER_RETRIES", "4")
    llm_relay.reset_state()
    yield monkeypatch
    llm_relay.reset_state()


def _msgs(text="hi"):
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": text}]


async def _post(url, body, token=None, path="/v1/generate"):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    async with aiohttp.ClientSession() as s:
        if body is None:
            async with s.get(url + path, headers=h) as r:
                return r.status, await r.json()
        async with s.post(url + path, json=body, headers=h) as r:
            return r.status, await r.json()


# ----------------------------------------------------------------------------- config rules

def test_bind_and_token_rules(token):
    ls.validate(ls.Settings(token=token))
    ls.validate(ls.Settings(no_auth=True))                       # loopback, explicit opt-out
    with pytest.raises(ls.ConfigError, match="--no-auth"):
        ls.validate(ls.Settings())                                # loopback, no token, no opt-out
    for host in ("0.0.0.0", "10.1.2.3", "::"):
        with pytest.raises(ls.ConfigError, match="non-loopback"):
            ls.validate(ls.Settings(host=host, no_auth=True))
    ls.validate(ls.Settings(host="0.0.0.0", token=token))
    with pytest.raises(ls.ConfigError, match="too short"):
        ls.validate(ls.Settings(token="short"))
    with pytest.raises(ls.ConfigError, match="--ledger"):
        ls.validate(ls.Settings(token=token, budget_usd=1.0))
    assert ls.main(["--host", "0.0.0.0", "--port", "1"]) == 2


def test_write_token_file(tmp_path, capsys):
    p = tmp_path / "tok"
    assert ls.main(["--write-token", str(p)]) == 0
    tok = p.read_text().strip()
    assert len(tok) >= ls.MIN_TOKEN_LEN and (p.stat().st_mode & 0o077) == 0
    assert tok not in capsys.readouterr().out                    # never printed
    assert ls.read_token_file(p) == tok


# ----------------------------------------------------------------------------- auth / health

def test_auth_reject_and_accept(token):
    prov = FakeProvider()
    with Server(ls.Settings(token=token), prov) as srv:
        body = {"messages": _msgs(), "model": MODEL, "n_tasks": 1}
        for t in (None, "wrong-" + token, token[:-1]):
            st, b = asyncio.run(_post(srv.url, body, t))
            assert st == 401 and b["error"] == "unauthorized"
            st, _ = asyncio.run(_post(srv.url, None, t, "/v1/health"))
            assert st == 401
        assert prov.calls == 0
        st, b = asyncio.run(_post(srv.url, body, token))
        assert st == 200 and b["generations"] == ["echo hi"] and b["usage"]["input_tokens"] == 1200
        st, h = asyncio.run(_post(srv.url, None, token, "/v1/health"))
        assert st == 200 and h["auth"] and set(h["providers"]) >= {"bedrock", "openai"}
        assert token not in json.dumps(h)
        st, s = asyncio.run(_post(srv.url, None, token, "/v1/stats"))
        assert s["provider_calls"] == 1 and token not in json.dumps(s)


def test_model_allow_list_and_stub_refused(token):
    prov = FakeProvider()
    with Server(ls.Settings(token=token, allow_models=r"us\.anthropic\..*"), prov) as srv:
        for m in ("gpt-4o", "stub:x"):
            st, b = asyncio.run(_post(srv.url, {"messages": _msgs(), "model": m}, token))
            assert st == 403 and b["error"] == "model_not_allowed"
        assert prov.calls == 0


def test_client_errors_are_clear(relay_env, token):
    relay_env.setenv("KB_LLM_SERVER_URL", f"http://127.0.0.1:{_free_port()}")
    relay_env.setenv("KB_LLM_SERVER_RETRIES", "1")
    with pytest.raises(llm_relay.RelayUnavailable, match="unreachable.*llm_server.md"):
        asyncio.run(query.generate_code(_msgs(), 1, MODEL))
    with Server(ls.Settings(token=token), FakeProvider()) as srv:
        relay_env.setenv("KB_LLM_SERVER_URL", srv.url)
        relay_env.setenv("KB_LLM_SERVER_TOKEN", "x" * 20)
        llm_relay.reset_state()
        with pytest.raises(llm_relay.RelayAuthError, match="KB_LLM_SERVER_TOKEN"):
            asyncio.run(query.generate_code(_msgs(), 1, MODEL))
        # fatal relay errors are not retried by generate_code_retry
        t0 = time.time()
        with pytest.raises(llm_relay.RelayAuthError):
            asyncio.run(query.generate_code_retry(_msgs(), MODEL, max_retries=5))
        assert time.time() - t0 < 5


def test_token_file_and_stub_stays_local(relay_env, token, tmp_path):
    relay_env.delenv("KB_LLM_SERVER_TOKEN")
    (tmp_path / "t").write_text(token + "\n")
    relay_env.setenv("KB_LLM_SERVER_TOKEN_FILE", str(tmp_path / "t"))
    prov = FakeProvider()
    with Server(ls.Settings(token=token), prov) as srv:
        relay_env.setenv("KB_LLM_SERVER_URL", srv.url)
        r = asyncio.run(query.generate_code(_msgs("a"), 1, MODEL))
        assert r.generations == ["echo a"] and prov.calls == 1
        agentic_llm.register_stub("relay_local", lambda m: "local")
        try:
            r = asyncio.run(query.generate_code(_msgs(), 1, "stub:relay_local"))
        finally:
            agentic_llm.unregister_stub("relay_local")
        assert r.generations == ["local"] and prov.calls == 1


# ----------------------------------------------------------------------------- concurrency

def test_concurrency_limit(relay_env, token):
    prov = FakeProvider(delay=0.3)
    with Server(ls.Settings(token=token, max_concurrency=2), prov) as srv:
        relay_env.setenv("KB_LLM_SERVER_URL", srv.url)

        async def go():
            return await asyncio.gather(*[query.generate_code(_msgs(str(i)), 1, MODEL) for i in range(6)])
        t0 = time.time()
        rs = asyncio.run(go())
        assert prov.calls == 6 and prov.peak == 2 and time.time() - t0 >= 0.85
        assert sorted(r.generations[0] for r in rs) == [f"echo {i}" for i in range(6)]
        # the queue wait shows up client side
        assert max(r.usage["telemetry"]["relay_queue_s"] for r in rs) >= 0.5


def test_fair_limiter_round_robin():
    order = []

    async def go():
        lim = ls.FairLimiter(1, 100)
        await lim.acquire("hold")

        async def job(c, i):
            await lim.acquire(c)
            order.append(f"{c}{i}")
            lim.release()
        tasks = [asyncio.create_task(job("a", i)) for i in range(4)]
        await asyncio.sleep(0)
        tasks.append(asyncio.create_task(job("b", 0)))
        await asyncio.sleep(0)
        assert lim.queued == 5 and lim.queued_by_client() == {"a": 4, "b": 1}
        lim.release()
        await asyncio.gather(*tasks)
        assert lim.active == 0
    asyncio.run(go())
    assert order.index("b0") == 1                                # not starved behind a1..a3


def test_queue_full(token):
    prov = FakeProvider(delay=0.5)
    with Server(ls.Settings(token=token, max_concurrency=1, max_queue=1), prov) as srv:
        async def go():
            body = lambda i: {"messages": _msgs(str(i)), "model": MODEL}  # noqa: E731
            return await asyncio.gather(*[_post(srv.url, body(i), token) for i in range(3)])
        sts = sorted(st for st, _ in asyncio.run(go()))
        assert sts == [200, 200, 503]


# ----------------------------------------------------------------------------- budget / ledger

def test_budget_cap_rejects(relay_env, token, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"who": "someone-else", "est_usd": 50.0}) + "\n")
    prov = FakeProvider()
    with Server(ls.Settings(token=token, ledger=ledger, budget_usd=0.002), prov) as srv:
        relay_env.setenv("KB_LLM_SERVER_URL", srv.url)
        asyncio.run(query.generate_code(_msgs(), 1, MODEL))      # ~$0.0028 > budget
        with pytest.raises(llm_relay.RelayBudgetExceeded, match="budget"):
            asyncio.run(query.generate_code_retry(_msgs("again"), MODEL, max_retries=5))
        assert prov.calls == 1
        st, h = asyncio.run(_post(srv.url, None, token, "/v1/health"))
        assert h["budget"]["remaining_usd"] == 0 and h["budget"]["spent_usd"] > 0.002
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    mine = [r for r in rows if r["who"] == "llm-server"]
    assert len(mine) == 1
    r = mine[0]
    assert {"ts", "what", "model", "calls", "prompt_tokens", "completion_tokens", "est_usd"} <= set(r)
    assert r["prompt_tokens"] == 1700 and r["completion_tokens"] == 300 and r["calls"] == 1
    # the budget is persisted: a restarted server with the same ledger refuses at once
    svc = ls.RelayService(ls.Settings(token=token, ledger=ledger, budget_usd=0.002), prov)
    assert svc.ledger.over()


# ----------------------------------------------------------------------------- idempotency

def test_idempotent_retry_no_double_billing(token, tmp_path):
    ledger = tmp_path / "l.jsonl"
    prov = FakeProvider(delay=0.2)
    with Server(ls.Settings(token=token, ledger=ledger), prov) as srv:
        body = {"messages": _msgs(), "model": MODEL, "request_id": "rid-1"}

        async def go():                                          # concurrent duplicate + later replay
            a, b = await asyncio.gather(_post(srv.url, body, token), _post(srv.url, body, token))
            c = await _post(srv.url, body, token)
            return a, b, c
        (sa, a), (sb, b), (sc, c) = asyncio.run(go())
        assert sa == sb == sc == 200 and prov.calls == 1
        assert a["generations"] == b["generations"] == c["generations"]
        assert c["server"]["replayed"]
        st, e = asyncio.run(_post(srv.url, {**body, "messages": _msgs("other")}, token))
        assert st == 409 and e["error"] == "request_id_conflict"
    assert len(ledger.read_text().splitlines()) == 1


def test_failed_request_is_not_cached(token):
    prov = FakeProvider(fail=lambda n: RuntimeError("boom") if n == 1 else None)
    with Server(ls.Settings(token=token), prov) as srv:
        body = {"messages": _msgs(), "model": MODEL, "request_id": "rid-f"}
        st, e = asyncio.run(_post(srv.url, body, token))
        assert st == 502 and e["error"] == "provider_error" and "boom" in e["message"]
        st, b = asyncio.run(_post(srv.url, body, token))
        assert st == 200 and prov.calls == 2


def test_provider_errors_are_scrubbed(relay_env, token):
    secret = secrets.token_urlsafe(30)
    relay_env.setenv("AWS_BEARER_TOKEN_BEDROCK", secret)
    prov = FakeProvider(fail=lambda n: RuntimeError(f"bad key {secret} and {token}"))
    with Server(ls.Settings(token=token), prov) as srv:
        st, e = asyncio.run(_post(srv.url, {"messages": _msgs(), "model": MODEL}, token))
        assert st == 502 and secret not in json.dumps(e) and token not in json.dumps(e)
        assert "[REDACTED]" in e["message"]
        relay_env.setenv("KB_LLM_SERVER_URL", srv.url)
        with pytest.raises(llm_relay.RelayError) as ei:          # retryable, generate_code_retry decides
            asyncio.run(query.generate_code(_msgs(), 1, MODEL))
        assert not isinstance(ei.value, llm_relay.RelayFatalError) and secret not in str(ei.value)


def test_server_side_throttle_backoff(relay_env, token, tmp_path):
    class Throttle(Exception):
        pass
    prov = FakeProvider(fail=lambda n: Throttle("ThrottlingException: Too many requests") if n <= 2 else None)
    s = ls.Settings(token=token, throttle_max_backoff_s=0.05)
    tel.set_enabled(True)
    tel.configure(tmp_path / "t.jsonl", run_id="thr")
    try:
        with Server(s, prov) as srv:
            relay_env.setenv("KB_LLM_SERVER_URL", srv.url)
            asyncio.run(query.generate_code(_msgs(), 1, MODEL))
    finally:
        tel.close()
    assert prov.calls == 3
    ev = [e for e in map(json.loads, (tmp_path / "t.jsonl").read_text().splitlines()) if e.get("t") == "llm"]
    assert len(ev) == 1 and ev[0]["backoffs"] == 2 and 0 < ev[0]["backoff_s"] <= 0.1
    # throttling that outlasts the server's retries -> 429 -> the client's rate-limit path
    prov2 = FakeProvider(fail=lambda n: Throttle("ThrottlingException"))
    with Server(ls.Settings(token=token, throttle_retries=1, throttle_max_backoff_s=0.01), prov2) as srv:
        st, e = asyncio.run(_post(srv.url, {"messages": _msgs(), "model": MODEL}, token))
        assert st == 429 and e["error"] == "rate_limited"


# ----------------------------------------------------------------------------- dropped connection

async def _dropping_proxy(upstream_port: int, drops: dict):
    """TCP proxy that swallows the response to the first ``POST /v1/generate`` and
    closes the client connection (the server has already done and billed the work)."""
    async def handle(cr, cw):
        ur, uw = await asyncio.open_connection("127.0.0.1", upstream_port)
        state = {"post": False}

        async def c2u():
            try:
                while data := await cr.read(65536):
                    if b"POST /v1/generate" in data:
                        state["post"] = True
                    uw.write(data)
                    await uw.drain()
            except (ConnectionError, asyncio.CancelledError):
                pass
            finally:
                uw.close()

        async def u2c():
            try:
                while data := await ur.read(65536):
                    if state["post"] and drops["left"] > 0:
                        drops["left"] -= 1
                        drops["dropped"] += 1
                        cw.close()
                        return
                    cw.write(data)
                    await cw.drain()
            except (ConnectionError, asyncio.CancelledError):
                pass
            finally:
                cw.close()
        await asyncio.gather(c2u(), u2c())
    return await asyncio.start_server(handle, "127.0.0.1", 0)


def test_connection_drop_retry_same_request_id(relay_env, token, tmp_path):
    ledger = tmp_path / "l.jsonl"
    prov = FakeProvider(delay=0.05)
    tel.set_enabled(True)
    tel.configure(tmp_path / "t.jsonl", run_id="drop")
    drops = {"left": 1, "dropped": 0}
    try:
        with Server(ls.Settings(token=token, ledger=ledger), prov) as srv:
            async def go():
                px = await _dropping_proxy(srv.settings.port, drops)
                relay_env.setenv("KB_LLM_SERVER_URL", f"http://127.0.0.1:{px.sockets[0].getsockname()[1]}")
                try:
                    return await query.generate_code(_msgs(), 1, MODEL)
                finally:
                    px.close()
            r = asyncio.run(go())
            stats = srv.svc.stats()
    finally:
        tel.close()
    assert drops["dropped"] == 1 and prov.calls == 1 and r.generations == ["echo hi"]
    assert r.usage["telemetry"]["relay_attempts"] == 2 and r.usage["telemetry"]["relay_replayed"]
    assert len(ledger.read_text().splitlines()) == 1
    assert sum(c["replays"] for c in stats["by_client"].values()) == 1
    ev = [e for e in map(json.loads, (tmp_path / "t.jsonl").read_text().splitlines()) if e.get("t") == "llm"]
    assert len(ev) == 1 and ev[0]["input_tokens"] == 1700        # recorded once client side


# ----------------------------------------------------------------------------- fallback

def test_fallback_direct(relay_env, token):
    prov = FakeProvider()
    relay_env.setattr(query, "generate_code_direct", prov)
    relay_env.setenv("KB_LLM_SERVER_URL", f"http://127.0.0.1:{_free_port()}")
    relay_env.setenv("KB_LLM_SERVER_RETRIES", "1")
    with pytest.raises(llm_relay.RelayUnavailable):               # off by default
        asyncio.run(query.generate_code(_msgs(), 1, MODEL))
    assert prov.calls == 0
    relay_env.setenv("KB_LLM_SERVER_FALLBACK", "direct")
    r = asyncio.run(query.generate_code(_msgs("fb"), 1, MODEL))
    assert r.generations == ["echo fb"] and prov.calls == 1
    t0 = time.time()                                              # cooldown: straight to direct
    asyncio.run(query.generate_code(_msgs("fb2"), 1, MODEL))
    assert prov.calls == 2 and time.time() - t0 < 0.5
    # a bad token is not "unreachable": no fallback
    llm_relay.reset_state()
    with Server(ls.Settings(token=token), FakeProvider()) as srv:
        relay_env.setenv("KB_LLM_SERVER_URL", srv.url)
        relay_env.setenv("KB_LLM_SERVER_TOKEN", "y" * 20)
        with pytest.raises(llm_relay.RelayAuthError):
            asyncio.run(query.generate_code(_msgs(), 1, MODEL))
    assert prov.calls == 2


# ----------------------------------------------------------------------------- telemetry parity

PARITY_KEYS = ("calls", "n", "role", "model", "input_tokens", "uncached_input_tokens", "cache_read_input_tokens",
               "cache_write_input_tokens", "output_tokens", "stop_reason", "truncated", "max_tokens",
               "server_latency_s", "provider_retries", "est_usd", "cost_usd", "prompt_chars", "output_chars")


def test_telemetry_parity_with_direct_call(relay_env, token, tmp_path):
    prov = FakeProvider()

    def run(path):
        tel.set_enabled(True)
        tel.configure(path, run_id="parity")
        try:
            async def go():
                with tel.attrs(role="fix"):
                    return await query.generate_code(_msgs("parity"), 1, MODEL)
            r = asyncio.run(go())
        finally:
            tel.close()
        ev = [e for e in map(json.loads, path.read_text().splitlines()) if e.get("t") == "llm"]
        assert len(ev) == 1
        return r, ev[0]

    relay_env.setattr(query, "generate_code_direct", prov)
    r_direct, e_direct = run(tmp_path / "direct.jsonl")
    with Server(ls.Settings(token=token, ledger=tmp_path / "l.jsonl"), prov) as srv:
        relay_env.setenv("KB_LLM_SERVER_URL", srv.url)
        r_relay, e_relay = run(tmp_path / "relay.jsonl")
    assert prov.calls == 2
    assert {k: e_relay.get(k) for k in PARITY_KEYS} == {k: e_direct.get(k) for k in PARITY_KEYS}
    assert e_relay["truncated"] == 1 and e_relay["stop_reason"] == "max_tokens"
    assert e_relay["cache_read_input_tokens"] == 400 and e_relay["input_tokens"] == 1700
    assert "relay_queue_s" in e_relay and "relay_network_s" in e_relay and "relay_queue_s" not in e_direct
    assert e_relay["latency_s"] >= e_relay["relay_network_s"]
    # the usage the agent sums (llm_usage / llm_usage_by_model) is the provider's, unchanged
    num = lambda u: {k: v for k, v in u.items() if isinstance(v, (int, float))}  # noqa: E731
    assert num(r_relay.usage) == num(r_direct.usage)
    # and the server's ledger agrees with the client's record
    row = json.loads((tmp_path / "l.jsonl").read_text())
    assert row["prompt_tokens"] == e_relay["input_tokens"] and row["completion_tokens"] == e_relay["output_tokens"]
    assert row["est_usd"] == pytest.approx(e_relay["est_usd"], abs=1e-6) and row["what"].startswith("llm-server role=fix")


# ----------------------------------------------------------------------------- end to end (GPU)

from test_multislot_agent import ARCH, FIX, needs_gpu, scripted  # noqa: E402


@needs_gpu
def test_multislot_lenet_through_relay(relay_env, token, tmp_path):
    from kernelblaster.agents.opt_multislot import MultiSlotAgent
    from kernelblaster.knowledge.impl_store import ImplStore
    from kernelblaster.knowledge.outcomes import OutcomeLog
    from kernelblaster.knowledge.slots import LocalEvaluator, SlotProblem

    state = {"broke": False, "fix_calls": 0, "linear_ok": None}
    fn = scripted(state)
    usage = {"prompt_tokens": 0, "completion_tokens": 300, "stop_reason": "end_turn"}
    agentic_llm.register_stub("relay_e2e", lambda m: {"text": fn(m), "usage": {
        **usage, "prompt_tokens": 1000 + len(m[-1]["content"])}})

    async def server_side(messages, n, model):                   # the "provider" behind the relay
        assert model == "relay-e2e-model"
        return await query.generate_code_direct(messages, n, "stub:relay_e2e")

    prob = tmp_path / "p"
    shutil.copytree(FIX, prob)
    ledger = tmp_path / "ledger.jsonl"
    tel.set_enabled(True)
    tel.configure(tmp_path / "telemetry.jsonl", run_id="relay-e2e")
    try:
        with Server(ls.Settings(token=token, ledger=ledger, max_concurrency=2), server_side) as srv:
            relay_env.setenv("KB_LLM_SERVER_URL", srv.url)
            agent = MultiSlotAgent(SlotProblem(prob), LocalEvaluator(ARCH, work_root=tmp_path / "work"),
                                   model="relay-e2e-model", out_dir=tmp_path / "out", arch=ARCH,
                                   impl_store=ImplStore(tmp_path / "impl"),
                                   outcome_log=OutcomeLog(tmp_path / "o.jsonl"), min_share=0.0, seed=1,
                                   rollouts=2, steps=3)
            res = asyncio.run(agent.run())
            stats = srv.svc.stats()
    finally:
        agentic_llm.unregister_stub("relay_e2e")
        tel.close()
    assert res["final_ok"] and state["fix_calls"] >= 1
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert stats["provider_calls"] == len(rows) == res["llm_calls"]
    assert sum(r["prompt_tokens"] for r in rows) == res["llm_usage"]["prompt_tokens"]
    assert sum(r["completion_tokens"] for r in rows) == res["llm_usage"]["completion_tokens"]
    tm = json.loads((tmp_path / "out" / "result.json").read_text())["telemetry"]["llm"]["total"]
    assert tm["calls"] == res["llm_calls"] and tm["input_tokens"] == res["llm_usage"]["prompt_tokens"]
    assert tm["relay_network_s"] > 0
