# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM relay server: makes the provider API calls on behalf of remote KB / MB agents.

Run it on the machine that can reach the provider (VPN, network policy, or the
only place the API keys live); point the agents at it with ``KB_LLM_SERVER_URL``
(client: ``agents/utils/llm_relay.py``). Setup, the ssh tunnel recipe and the
security model are in ``docs/llm_server.md``.

    PYTHONPATH=src python -m kernelblaster.servers.llm_server --port 18081 \\
        --token-file ~/.kb_llm_token [--max-concurrency 8] \\
        [--ledger L.jsonl --budget-usd 5] [--allow-models 'us\\.anthropic\\..*']
    python -m kernelblaster.servers.llm_server --write-token ~/.kb_llm_token

Endpoints (all need ``Authorization: Bearer <token>`` when a token is set):

``POST /v1/generate``
    ``{messages, n_tasks, model, role?, client?, run_id?, request_id?}`` → runs
    the normal provider dispatch (``query.generate_code_direct``: Bedrock,
    OpenAI, perflab, local models, the Claude CLI) and returns the
    ``LLMResponse`` fields (generations, usage with the provider's telemetry
    extras, model, num_tasks, elapsed_time) plus ``server`` timing (queue,
    provider, back-off, retries, cost). A ``request_id`` seen before (in the
    last ``--idempotency-ttl`` seconds) returns the stored result, or joins the
    in-flight call, instead of calling the provider again.
``GET /v1/health``
    providers configured, model allow-list, concurrency / queue depth, budget.
``GET /v1/stats``
    per client / per model counts, tokens, cost, errors.

Errors are JSON ``{"error": <code>, "message": ...}``: 400 ``bad_request``, 401
``unauthorized``, 403 ``model_not_allowed``, 402 ``budget_exceeded``, 409
``request_id_conflict``, 413 ``too_large``, 429 ``rate_limited`` (provider
throttling after the server's own back-off), 502 ``provider_error``, 503
``queue_full``.

Nothing secret is ever returned or logged: the token is compared in constant
time and never echoed, provider error messages are scrubbed of the configured
key values, and the request log has no message contents unless
``--log-messages``.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import datetime as _dt
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import random
import re
import secrets
import shutil
import socket
import stat
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("kernelblaster.llm_server")

API_VERSION = 1
MIN_TOKEN_LEN = 16
# env vars whose values must never leave the server (error-message scrubbing)
SECRET_ENVS = ("AWS_BEARER_TOKEN_BEDROCK", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "OPENAI_API_KEY",
               "PERFLAB_KEY", "ANTHROPIC_API_KEY", "KB_LLM_SERVER_TOKEN")
THROTTLE_CODES = {"ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException",
                  "ModelNotReadyException", "RateLimitError"}
THROTTLE_TEXT = ("rate limit", "rate_limit", "ratelimit", "throttl", "too many requests")

GenerateFn = Callable[[list, int, str], Awaitable[Any]]


# --------------------------------------------------------------------------- settings / helpers

@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 18081
    token: Optional[str] = None
    no_auth: bool = False
    max_concurrency: int = 8
    max_queue: int = 256
    ledger: Optional[Path] = None
    ledger_who: str = "llm-server"
    budget_usd: Optional[float] = None
    allow_models: Optional[str] = None
    idempotency_ttl_s: float = 600.0
    throttle_retries: int = 4
    throttle_max_backoff_s: float = 60.0
    max_request_bytes: int = 8 << 20
    log_messages: bool = False
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None


class ConfigError(Exception):
    pass


def is_loopback(host: str) -> bool:
    if host in ("localhost", "ip6-localhost"):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def read_token_file(path: str | Path) -> str:
    p = Path(path).expanduser()
    tok = p.read_text().strip()
    try:
        if p.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            logger.warning(f"token file {p} is readable by group/others; chmod 600 it")
    except OSError:
        pass
    return tok


def write_token_file(path: str | Path) -> Path:
    """Create a fresh random token in ``path`` (mode 0600); never prints it."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(secrets.token_urlsafe(32) + "\n")
    os.chmod(p, 0o600)
    return p


def validate(s: Settings) -> None:
    """Refuse unsafe configurations (see docs/llm_server.md, Security model)."""
    if s.token is not None and len(s.token) < MIN_TOKEN_LEN:
        raise ConfigError(f"token too short (< {MIN_TOKEN_LEN} chars); use --write-token to make one")
    if not s.token:
        if not is_loopback(s.host):
            raise ConfigError(f"refusing to bind non-loopback host {s.host!r} without a token "
                              "(--token-file or KB_LLM_SERVER_TOKEN)")
        if not s.no_auth:
            raise ConfigError("no token configured: pass --token-file / KB_LLM_SERVER_TOKEN, or --no-auth to "
                              "accept unauthenticated requests on loopback (any local user could use it)")
    if s.allow_models:
        re.compile(s.allow_models)
    if s.budget_usd is not None and s.ledger is None:
        raise ConfigError("--budget-usd needs --ledger (the spend is persisted there)")
    if s.max_concurrency < 1:
        raise ConfigError("--max-concurrency must be >= 1")
    if not is_loopback(s.host) and not s.ssl_certfile:
        logger.warning("binding a non-loopback host without TLS: the token travels in clear text; "
                       "prefer an ssh tunnel or --ssl-certfile/--ssl-keyfile")


def _secret_values() -> list[str]:
    return [v for v in (os.getenv(k) for k in SECRET_ENVS) if v and len(v) >= 8]


def redact(text: str, extra: tuple = ()) -> str:
    for v in [*_secret_values(), *[e for e in extra if e]]:
        text = text.replace(v, "[REDACTED]")
    return text


def _jsonable(o):
    for attr in ("model_dump", "dict"):
        f = getattr(o, attr, None)
        if callable(f):
            try:
                return f()
            except Exception:  # noqa: BLE001
                pass
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    return str(o)


def to_json_obj(x):
    return json.loads(json.dumps(x, default=_jsonable))


def is_throttle(e: BaseException) -> bool:
    if type(e).__name__ in THROTTLE_CODES:
        return True
    resp = getattr(e, "response", None)
    if isinstance(resp, dict) and (resp.get("Error") or {}).get("Code") in THROTTLE_CODES:
        return True
    if getattr(e, "status_code", None) == 429:
        return True
    s = str(e).lower()
    return any(k in s for k in THROTTLE_TEXT)


def payload_digest(messages, n_tasks, model) -> str:
    h = hashlib.sha256(json.dumps([messages, n_tasks, model], sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def providers_configured() -> dict:
    from ..agents.utils import query as q
    return {"bedrock": q.bedrock_client is not None, "openai": q.oai_client is not None,
            "perflab": q.perflab_client is not None, "claude_cli": shutil.which(os.getenv("KB_CLAUDE_CLI", "claude"))
            is not None}


def call_cost(usage, model, n) -> tuple[dict, float]:
    """The telemetry record for this call and its USD (provider-reported, else the price table)."""
    from ..telemetry.llm import normalize_usage
    rec = normalize_usage(usage, model, n)
    return rec, float(rec.get("cost_usd") or rec.get("est_usd") or 0.0)


# --------------------------------------------------------------------------- fair limiter

class FairLimiter:
    """At most ``k`` concurrent holders; waiters are served round-robin per client,
    so one busy client cannot starve the others. Single event loop, no locks."""

    def __init__(self, k: int, max_queue: int):
        self.k, self.max_queue = k, max_queue
        self.active = 0
        self.waiting: "collections.OrderedDict[str, collections.deque]" = collections.OrderedDict()

    @property
    def queued(self) -> int:
        return sum(len(d) for d in self.waiting.values())

    def queued_by_client(self) -> dict:
        return {c: len(d) for c, d in self.waiting.items() if d}

    async def acquire(self, client: str) -> None:
        if self.active < self.k and not self.queued:
            self.active += 1
            return
        if self.queued >= self.max_queue:
            raise QueueFull()
        fut = asyncio.get_running_loop().create_future()
        self.waiting.setdefault(client, collections.deque()).append(fut)
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release()                       # granted and cancelled at once: hand it on
            else:
                d = self.waiting.get(client)
                if d is not None and fut in d:
                    d.remove(fut)
                    if not d:
                        del self.waiting[client]
            raise

    def release(self) -> None:
        while self.waiting:
            client, d = next(iter(self.waiting.items()))
            fut = d.popleft()
            if d:
                self.waiting.move_to_end(client)      # round robin
            else:
                del self.waiting[client]
            if not fut.done():
                fut.set_result(None)                  # the slot passes to the waiter
                return
        self.active -= 1


class QueueFull(Exception):
    pass


class HTTPError(Exception):
    def __init__(self, status: int, code: str, message: str, **extra):
        super().__init__(message)
        self.status, self.code, self.message, self.extra = status, code, message, extra


# --------------------------------------------------------------------------- ledger / budget

class Ledger:
    """Append-only JSONL spend ledger (the shared llm_budget format: ts, who, what,
    model, calls, prompt_tokens, completion_tokens, est_usd, plus the relay ids)."""

    def __init__(self, path: Optional[Path], who: str, budget: Optional[float]):
        self.path, self.who, self.budget = path, who, budget
        self.spent = 0.0
        if path is not None and path.exists():
            for line in path.read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("who") == who:
                    self.spent += float(r.get("est_usd") or 0.0)

    @property
    def remaining(self) -> Optional[float]:
        return None if self.budget is None else max(self.budget - self.spent, 0.0)

    def over(self) -> bool:
        return self.budget is not None and self.spent >= self.budget

    def append(self, row: dict, usd: float) -> None:
        self.spent += usd
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row) + "\n"
        with open(self.path, "a") as f:
            try:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX)
            except Exception:  # noqa: BLE001
                pass
            f.write(line)
            f.flush()


# --------------------------------------------------------------------------- core service

@dataclass
class _Entry:
    digest: str
    task: asyncio.Task
    t_done: Optional[float] = None


@dataclass
class _Stat:
    requests: int = 0
    ok: int = 0
    errors: int = 0
    replays: int = 0
    rejected: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    provider_s: float = 0.0
    queue_s: float = 0.0

    def to_json(self):
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in vars(self).items()}


class RelayService:
    def __init__(self, s: Settings, generate: Optional[GenerateFn] = None):
        self.s = s
        if generate is None:
            from ..agents.utils.query import generate_code_direct
            generate = generate_code_direct
        self.generate = generate
        self.limiter = FairLimiter(s.max_concurrency, s.max_queue)
        self.ledger = Ledger(s.ledger, s.ledger_who, s.budget_usd)
        self.allow = re.compile(s.allow_models) if s.allow_models else None
        self.cache: dict[str, _Entry] = {}
        self.by_client: dict[str, _Stat] = collections.defaultdict(_Stat)
        self.by_model: dict[str, _Stat] = collections.defaultdict(_Stat)
        self.t_start = time.time()
        self.provider_calls = 0

    # ---- auth
    def check_auth(self, header: Optional[str]) -> bool:
        if not self.s.token:
            return True
        if not header or not header.startswith("Bearer "):
            return False
        return hmac.compare_digest(header[7:].strip().encode(), self.s.token.encode())

    # ---- idempotency
    def _gc(self) -> None:
        now = time.time()
        for rid in [r for r, e in self.cache.items() if e.t_done and now - e.t_done > self.s.idempotency_ttl_s]:
            del self.cache[rid]

    async def handle(self, req: dict) -> dict:
        t_arrive = time.time()
        messages, model = req.get("messages"), req.get("model")
        n = req.get("n_tasks", 1)
        if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
            raise HTTPError(400, "bad_request", "messages must be a non-empty list of {role, content}")
        if not isinstance(model, str) or not model:
            raise HTTPError(400, "bad_request", "model is required")
        if not isinstance(n, int) or not 1 <= n <= 64:
            raise HTTPError(400, "bad_request", "n_tasks must be an int in [1, 64]")
        client = str(req.get("client") or "anonymous")[:128]
        rid = str(req.get("request_id") or "")[:128] or secrets.token_hex(12)
        meta = {"client": client, "role": str(req.get("role") or "other")[:64],
                "run_id": str(req.get("run_id") or "")[:128], "request_id": rid, "model": model, "n": n}
        cs, ms = self.by_client[client], self.by_model[model]
        cs.requests += 1
        ms.requests += 1

        def reject(err: HTTPError):
            cs.rejected += 1
            ms.rejected += 1
            self._log("reject", meta, error=err.code)
            return err

        if model.startswith("stub:"):
            raise reject(HTTPError(403, "model_not_allowed", "stub: models run in the client process"))
        if self.allow is not None and not self.allow.fullmatch(model):
            raise reject(HTTPError(403, "model_not_allowed", f"model {model!r} is not allowed by this server",
                                   allow_models=self.s.allow_models))
        digest = payload_digest(messages, n, model)
        self._gc()
        entry = self.cache.get(rid)
        replay = entry is not None
        if entry is not None:
            if entry.digest != digest:
                raise reject(HTTPError(409, "request_id_conflict", "request_id reused with a different payload"))
        else:
            if self.ledger.over():
                raise reject(HTTPError(402, "budget_exceeded",
                                       f"server budget exhausted: spent ${self.ledger.spent:.4f} of "
                                       f"${self.ledger.budget:.4f}", spent_usd=round(self.ledger.spent, 6),
                                       budget_usd=self.ledger.budget))
            if self.limiter.queued >= self.s.max_queue:
                raise reject(HTTPError(503, "queue_full", "server queue is full; retry later"))
            # the provider call runs as its own task: a dropped client connection does not
            # cancel it, and the retry (same request_id) picks up the result
            entry = _Entry(digest, asyncio.get_running_loop().create_task(self._run(messages, n, model, meta,
                                                                                     t_arrive)))
            self.cache[rid] = entry
            entry.task.add_done_callback(lambda t, e=entry, r=rid: self._done(r, e, t))
        try:
            out = await asyncio.shield(entry.task)
        except HTTPError as err:
            cs.errors += 1
            ms.errors += 1
            raise
        if replay:
            cs.replays += 1
            ms.replays += 1
            self._log("replay", meta)
            out = {**out, "server": {**out["server"], "replayed": True}}
        return out

    def _done(self, rid: str, entry: _Entry, task: asyncio.Task) -> None:
        if task.cancelled() or task.exception() is not None:
            if self.cache.get(rid) is entry:          # failures are not cached: a retry re-runs
                del self.cache[rid]
        else:
            entry.t_done = time.time()

    async def _run(self, messages, n, model, meta, t_arrive) -> dict:
        client = meta["client"]
        try:
            await self.limiter.acquire(client)
        except QueueFull:
            raise HTTPError(503, "queue_full", "server queue is full; retry later")
        t_start = time.time()
        backoff_s, retries = 0.0, 0
        try:
            if self.ledger.over():                   # the budget may have run out while queued
                raise HTTPError(402, "budget_exceeded", f"server budget exhausted: spent ${self.ledger.spent:.4f}",
                                spent_usd=round(self.ledger.spent, 6), budget_usd=self.ledger.budget)
            while True:
                t_call = time.time()
                try:
                    self.provider_calls += 1
                    resp = await self.generate(messages, n, model)
                    break
                except Exception as e:  # noqa: BLE001
                    if is_throttle(e) and retries < self.s.throttle_retries:
                        delay = min(2 ** retries + random.uniform(0, 1), self.s.throttle_max_backoff_s)
                        self._log("throttle", meta, delay_s=round(delay, 3), error=type(e).__name__)
                        retries += 1
                        backoff_s += delay
                        await asyncio.sleep(delay)
                        continue
                    msg = redact(str(e), (self.s.token,))[:1000]
                    code, status = ("rate_limited", 429) if is_throttle(e) else ("provider_error", 502)
                    self._log("error", meta, error=code, type=type(e).__name__,
                              provider_s=round(time.time() - t_call, 3))
                    raise HTTPError(status, code, f"{type(e).__name__}: {msg}", type=type(e).__name__)
        finally:
            self.limiter.release()
        t_end = time.time()
        usage = getattr(resp, "usage", None)
        usage = to_json_obj(usage if usage is not None else {})
        rec, usd = call_cost(usage, model, n)
        provider_s = t_end - t_call
        self.ledger.append({"ts": _dt.datetime.now().isoformat(timespec="seconds"), "who": self.s.ledger_who,
                            "what": f"llm-server role={meta['role']} client={meta['client']} run={meta['run_id']}",
                            "model": model, "calls": n, "prompt_tokens": rec["input_tokens"],
                            "completion_tokens": rec["output_tokens"],
                            "cache_read_input_tokens": rec["cache_read_input_tokens"],
                            "cache_write_input_tokens": rec["cache_write_input_tokens"],
                            "est_usd": round(usd, 6), "request_id": meta["request_id"]}, usd)
        for st in (self.by_client[client], self.by_model[model]):
            st.ok += 1
            st.input_tokens += rec["input_tokens"]
            st.output_tokens += rec["output_tokens"]
            st.usd += usd
            st.provider_s += provider_s
            st.queue_s += t_start - t_arrive
        server = {"queue_s": t_start - t_arrive, "provider_s": provider_s, "backoff_s": backoff_s,
                  "throttle_retries": retries, "total_s": t_end - t_arrive, "usd": usd, "replayed": False}
        gens = list(getattr(resp, "generations", None) or [])
        extra = {}
        if self.s.log_messages:
            extra = {"messages": messages, "generations": gens}
        self._log("ok", meta, input_tokens=rec["input_tokens"], output_tokens=rec["output_tokens"],
                  cache_read=rec["cache_read_input_tokens"], cache_write=rec["cache_write_input_tokens"],
                  stop_reason=rec.get("stop_reason"), truncated=rec["truncated"], usd=round(usd, 6),
                  queue_s=round(server["queue_s"], 3), provider_s=round(provider_s, 3),
                  backoff_s=round(backoff_s, 3), prompt_chars=sum(len(str(m.get("content") or ""))
                                                                    for m in messages), **extra)
        return {"api": API_VERSION, "request_id": meta["request_id"], "generations": gens, "usage": usage,
                "model": getattr(resp, "model", model), "num_tasks": getattr(resp, "num_tasks", n),
                "elapsed_time": getattr(resp, "elapsed_time", provider_s), "server": server}

    # ---- reporting
    def _log(self, event: str, meta: dict, **kw) -> None:
        logger.info(json.dumps({"ts": round(time.time(), 3), "event": event, **meta, **kw}, default=str))

    def health(self) -> dict:
        return {"ok": True, "api": API_VERSION, "uptime_s": round(time.time() - self.t_start, 1),
                "host": socket.gethostname(), "auth": bool(self.s.token),
                "providers": providers_configured(), "allow_models": self.s.allow_models,
                "concurrency": {"max": self.s.max_concurrency, "active": self.limiter.active,
                                "queued": self.limiter.queued, "queued_by_client": self.limiter.queued_by_client(),
                                "max_queue": self.s.max_queue},
                "budget": {"budget_usd": self.ledger.budget, "spent_usd": round(self.ledger.spent, 6),
                           "remaining_usd": None if self.ledger.remaining is None else round(self.ledger.remaining, 6),
                           "ledger_who": self.s.ledger_who, "ledger": self.s.ledger.name if self.s.ledger else None},
                "idempotency": {"ttl_s": self.s.idempotency_ttl_s, "cached": len(self.cache)}}

    def stats(self) -> dict:
        return {"provider_calls": self.provider_calls,
                "by_client": {k: v.to_json() for k, v in self.by_client.items()},
                "by_model": {k: v.to_json() for k, v in self.by_model.items()},
                "spent_usd": round(self.ledger.spent, 6)}


# --------------------------------------------------------------------------- HTTP app

def create_app(s: Settings, generate: Optional[GenerateFn] = None):
    svc = RelayService(s, generate)
    app = FastAPI(title="KernelBlaster LLM relay", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.svc = svc

    def err(e: HTTPError) -> JSONResponse:
        return JSONResponse({"error": e.code, "message": e.message, **e.extra}, status_code=e.status)

    def authed(request: Request) -> bool:
        ok = svc.check_auth(request.headers.get("authorization"))
        if not ok:
            peer = request.client.host if request.client else "?"
            logger.warning(json.dumps({"ts": round(time.time(), 3), "event": "unauthorized", "peer": peer,
                                       "path": request.url.path}))
        return ok

    unauthorized = HTTPError(401, "unauthorized", "missing or invalid bearer token")

    @app.get("/v1/health")
    async def health(request: Request):
        if not authed(request):
            return err(unauthorized)
        return svc.health()

    @app.get("/v1/stats")
    async def stats(request: Request):
        if not authed(request):
            return err(unauthorized)
        return svc.stats()

    @app.post("/v1/generate")
    async def generate(request: Request):
        if not authed(request):                       # before reading the body
            return err(unauthorized)
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > s.max_request_bytes:
            return err(HTTPError(413, "too_large", f"request body over {s.max_request_bytes} bytes"))
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > s.max_request_bytes:
                return err(HTTPError(413, "too_large", f"request body over {s.max_request_bytes} bytes"))
        try:
            req = json.loads(bytes(body))
            if not isinstance(req, dict):
                raise ValueError("not an object")
        except ValueError as e:
            return err(HTTPError(400, "bad_request", f"invalid JSON: {e}"))
        try:
            return JSONResponse(await svc.handle(req))
        except HTTPError as e:
            return err(e)

    return app


def settings_from_args(argv=None) -> tuple[Settings, argparse.Namespace]:
    ap = argparse.ArgumentParser(description="KernelBlaster LLM relay server (docs/llm_server.md)")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default loopback: use an ssh tunnel)")
    ap.add_argument("--port", type=int, default=18081)
    ap.add_argument("--token-file", help="shared secret file (else env KB_LLM_SERVER_TOKEN)")
    ap.add_argument("--no-auth", action="store_true", help="allow no token (loopback only)")
    ap.add_argument("--write-token", metavar="FILE", help="write a fresh random token to FILE (0600) and exit")
    ap.add_argument("--max-concurrency", type=int, default=8, help="concurrent provider calls")
    ap.add_argument("--max-queue", type=int, default=256, help="queued requests before 503 queue_full")
    ap.add_argument("--ledger", type=Path, help="append one JSONL spend row per provider call")
    ap.add_argument("--ledger-who", default="llm-server", help="'who' field of the ledger rows / budget scope")
    ap.add_argument("--budget-usd", type=float, help="reject (402) once the ledger's rows for --ledger-who reach this")
    ap.add_argument("--allow-models", help="regex (full match) of model ids the server will call")
    ap.add_argument("--idempotency-ttl", type=float, default=600.0, help="seconds a result is kept per request_id")
    ap.add_argument("--throttle-retries", type=int, default=4, help="server-side back-off retries on throttling")
    ap.add_argument("--max-request-mb", type=float, default=8.0)
    ap.add_argument("--log-messages", action="store_true", help="log prompts and generations (off by default)")
    ap.add_argument("--log-file", help="also write the structured request log here")
    ap.add_argument("--ssl-certfile")
    ap.add_argument("--ssl-keyfile")
    a = ap.parse_args(argv)
    token = None
    if a.token_file:
        token = read_token_file(a.token_file)
    elif os.getenv("KB_LLM_SERVER_TOKEN"):
        token = os.environ["KB_LLM_SERVER_TOKEN"].strip()
    s = Settings(host=a.host, port=a.port, token=token, no_auth=a.no_auth, max_concurrency=a.max_concurrency,
                 max_queue=a.max_queue, ledger=a.ledger, ledger_who=a.ledger_who, budget_usd=a.budget_usd,
                 allow_models=a.allow_models, idempotency_ttl_s=a.idempotency_ttl,
                 throttle_retries=a.throttle_retries, max_request_bytes=int(a.max_request_mb * (1 << 20)),
                 log_messages=a.log_messages, ssl_certfile=a.ssl_certfile, ssl_keyfile=a.ssl_keyfile)
    return s, a


def _setup_logging(log_file: Optional[str]) -> None:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(message)s")
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(fmt)
    logger.addHandler(h)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)


def main(argv=None) -> int:
    s, a = settings_from_args(argv)
    if a.write_token:
        p = write_token_file(a.write_token)
        print(f"wrote a new token to {p} (mode 0600)")
        return 0
    _setup_logging(a.log_file)
    try:
        validate(s)
    except (ConfigError, re.error) as e:
        print(f"llm_server: {e}", file=sys.stderr)
        return 2
    import uvicorn
    app = create_app(s)
    h = app.state.svc.health()
    logger.info(json.dumps({"event": "start", "host": s.host, "port": s.port, "auth": h["auth"],
                            "providers": h["providers"], "allow_models": s.allow_models,
                            "max_concurrency": s.max_concurrency, "budget": h["budget"]}))
    uvicorn.run(app, host=s.host, port=s.port, access_log=False, log_level="warning",
                ssl_certfile=s.ssl_certfile, ssl_keyfile=s.ssl_keyfile, timeout_keep_alive=75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
