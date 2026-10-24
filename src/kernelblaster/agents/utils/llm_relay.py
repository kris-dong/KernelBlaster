# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Client side of the LLM relay (``servers/llm_server.py``, ``docs/llm_server.md``).

``query.generate_code`` forwards every request here when ``KB_LLM_SERVER_URL``
is set (``stub:`` models stay in-process). The server makes the provider call
and returns the ``LLMResponse`` fields with the provider's usage and telemetry
extras, so ``llm_usage``, ``llm_usage_by_model`` and telemetry record a relayed
call exactly like a local one. The relay adds ``relay_queue_s`` (waiting for a
server slot), ``relay_network_s`` (round trip minus server time), the server's
throttling back-off (as ``backoff_s``) and its retries (as ``provider_retries``).

Env:
    KB_LLM_SERVER_URL          e.g. http://127.0.0.1:18080 (an ssh -L tunnel) or https://host:port
    KB_LLM_SERVER_TOKEN        shared secret, or
    KB_LLM_SERVER_TOKEN_FILE   a file holding it
    KB_LLM_SERVER_FALLBACK     ``direct``: call the providers in this process when the
                               server is unreachable (off by default)
    KB_LLM_SERVER_TIMEOUT_S    per-attempt timeout (default query.TIMEOUT, 1800)
    KB_LLM_SERVER_RETRIES      attempts on connection errors / 5xx, same request_id (default 6)
    KB_LLM_SERVER_CLIENT       client name for the server's fairness / logs (default host:pid)
    KB_LLM_SERVER_CA_FILE      CA bundle for an https server with a private certificate
"""
from __future__ import annotations

import asyncio
import os
import random
import socket
import time
import uuid
from copy import deepcopy
from typing import Optional

STUB_PREFIX = "stub:"
DEFAULT_RETRIES = 6
FALLBACK_COOLDOWN_S = 60.0


class RelayError(RuntimeError):
    """A relay failure ``generate_code_retry`` may retry (server or provider side)."""


class RelayFatalError(RelayError):
    """Not worth retrying: bad token, model not allowed, budget exhausted, bad request."""


class RelayUnavailable(RelayError):
    """The server could not be reached (after the relay's own retries)."""


class RelayAuthError(RelayFatalError):
    pass


class RelayBudgetExceeded(RelayFatalError):
    pass


class RelayModelNotAllowed(RelayFatalError):
    pass


class RelayRateLimited(RelayError):
    pass


_state = {"healthy_url": None, "down_until": 0.0, "fallback_warned": 0.0}


def server_url() -> Optional[str]:
    u = (os.getenv("KB_LLM_SERVER_URL") or "").strip()
    return u.rstrip("/") or None


def relay_enabled(model: Optional[str]) -> bool:
    return server_url() is not None and not (model or "").startswith(STUB_PREFIX)


def _token() -> Optional[str]:
    t = os.getenv("KB_LLM_SERVER_TOKEN")
    if t:
        return t.strip()
    f = os.getenv("KB_LLM_SERVER_TOKEN_FILE")
    if f:
        try:
            return open(os.path.expanduser(f)).read().strip()
        except OSError as e:
            raise RelayFatalError(f"KB_LLM_SERVER_TOKEN_FILE unreadable: {e}") from None
    return None


def _headers() -> dict:
    t = _token()
    return {"Authorization": f"Bearer {t}"} if t else {}


def _ssl():
    ca = os.getenv("KB_LLM_SERVER_CA_FILE")
    if not ca:
        return None
    import ssl
    return ssl.create_default_context(cafile=ca)


def _client_name() -> str:
    return os.getenv("KB_LLM_SERVER_CLIENT") or f"{socket.gethostname()}:{os.getpid()}"


def _fallback() -> bool:
    return (os.getenv("KB_LLM_SERVER_FALLBACK") or "").strip().lower() == "direct"


def _log():
    from loguru import logger
    return logger


def _context() -> tuple[str, str]:
    """(role, run id) from the telemetry context, for the server's log / ledger."""
    try:
        from ...telemetry import core
        cur = core.current()
        role = (cur.attrs.get("role") if cur is not None else None) or "other"
        return role, core.run_id()
    except Exception:  # noqa: BLE001
        return "other", ""


def _raise_for(status: int, body: dict, url: str):
    code = body.get("error") if isinstance(body, dict) else None
    msg = (body.get("message") if isinstance(body, dict) else None) or str(body)[:300]
    text = f"LLM relay {url}: HTTP {status} {code or ''}: {msg}"
    if status == 401:
        raise RelayAuthError(text + " (check KB_LLM_SERVER_TOKEN / KB_LLM_SERVER_TOKEN_FILE)")
    if status == 402:
        raise RelayBudgetExceeded(text)
    if status == 403:
        raise RelayModelNotAllowed(text)
    if status in (400, 409, 413, 422):
        raise RelayFatalError(text)
    if status == 429:
        raise RelayRateLimited(text + " (rate limit)")    # generate_code_retry backs off on "rate limit"
    raise RelayError(text)


async def _json(resp) -> dict:
    try:
        return await resp.json(content_type=None)
    except Exception:  # noqa: BLE001
        return {"message": (await resp.text())[:300]}


async def check_health(session=None, timeout: float = 15.0) -> dict:
    """``GET /v1/health``; raises :class:`RelayUnavailable` / :class:`RelayAuthError`
    with a message saying what to check."""
    import aiohttp
    url = server_url()
    own = session is None
    if own:
        session = aiohttp.ClientSession()
    try:
        async with session.get(f"{url}/v1/health", headers=_headers(), ssl=_ssl(),
                               timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            body = await _json(r)
            if r.status != 200:
                _raise_for(r.status, body, url)
            return body
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        raise RelayUnavailable(f"LLM relay {url} unreachable ({type(e).__name__}: {e}); is llm_server running "
                               "and the ssh tunnel up? see docs/llm_server.md") from None
    finally:
        if own:
            await session.close()


async def _post(messages, n_tasks, model) -> tuple[dict, float]:
    import aiohttp
    url = server_url()
    role, run_id = _context()
    rid = uuid.uuid4().hex
    payload = {"messages": messages, "n_tasks": n_tasks, "model": model, "role": role,
               "client": _client_name(), "run_id": run_id, "request_id": rid}
    from .query import TIMEOUT
    timeout = aiohttp.ClientTimeout(total=float(os.getenv("KB_LLM_SERVER_TIMEOUT_S") or TIMEOUT), sock_connect=15)
    attempts = max(int(os.getenv("KB_LLM_SERVER_RETRIES") or DEFAULT_RETRIES), 1)
    last: Exception | None = None
    async with aiohttp.ClientSession() as session:
        if _state["healthy_url"] != url:
            await check_health(session)
            _state["healthy_url"] = url
        for i in range(attempts):
            t0 = time.time()
            try:
                async with session.post(f"{url}/v1/generate", json=payload, headers=_headers(), ssl=_ssl(),
                                        timeout=timeout) as r:
                    body = await _json(r)
                    if r.status == 200:
                        body.setdefault("server", {})["attempts"] = i + 1
                        return body, time.time() - t0
                    if r.status < 500 or r.status == 502:    # 502: the provider failed; not a relay fault
                        _raise_for(r.status, body, url)
                    last = RelayError(f"LLM relay {url}: HTTP {r.status}: {body.get('message', '')}"[:500])
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                last = RelayUnavailable(f"LLM relay {url}: {type(e).__name__}: {e}")
            delay = min(2 ** i, 30) * (0.5 + random.random() / 2)
            _log().warning(f"[llm_relay] attempt {i + 1}/{attempts} failed ({last}); retrying in {delay:.1f}s "
                           f"(same request_id)")
            await asyncio.sleep(delay)
    if isinstance(last, RelayUnavailable):
        raise RelayUnavailable(f"{last} (after {attempts} attempts); see docs/llm_server.md")
    raise last or RelayError("LLM relay: no attempts")


def _to_response(body: dict, messages, rt_s: float, model):
    from .query import LLMResponse
    usage = body.get("usage")
    usage = dict(usage) if isinstance(usage, dict) else {}
    srv = body.get("server") or {}
    tel = dict(usage.get("telemetry") or {})
    # the relay's share, folded into the same telemetry record a local call produces
    tel["relay_queue_s"] = float(srv.get("queue_s") or 0.0)
    tel["relay_network_s"] = max(rt_s - float(srv.get("total_s") or 0.0), 0.0)
    tel["relay_attempts"] = int(srv.get("attempts") or 1)
    if srv.get("replayed"):
        tel["relay_replayed"] = True
    if srv.get("backoff_s"):
        tel["relay_backoff_s"] = float(srv["backoff_s"])
        tel["relay_backoffs"] = int(srv.get("throttle_retries") or 0)
    usage["telemetry"] = tel
    return LLMResponse(deepcopy(messages), list(body.get("generations") or []), usage,
                       body.get("model") or model, int(body.get("num_tasks") or 1),
                       body.get("elapsed_time") or 0)


async def generate_via_relay(messages, n_tasks, model):
    """The relayed ``generate_code`` (see the module doc)."""
    fb = _fallback()
    if fb and time.time() < _state["down_until"]:
        return await _direct(messages, n_tasks, model, None)
    try:
        body, rt = await _post(messages, n_tasks, model)
    except RelayUnavailable as e:
        if not fb:
            raise
        _state["down_until"] = time.time() + FALLBACK_COOLDOWN_S
        _state["healthy_url"] = None
        return await _direct(messages, n_tasks, model, e)
    _state["down_until"] = 0.0
    return _to_response(body, messages, rt, model)


async def _direct(messages, n_tasks, model, err):
    now = time.time()
    if now - _state["fallback_warned"] > FALLBACK_COOLDOWN_S:
        _state["fallback_warned"] = now
        _log().warning(f"[llm_relay] server unreachable{f' ({err})' if err else ''}; "
                       "KB_LLM_SERVER_FALLBACK=direct: calling the provider from this process")
    from .query import generate_code_direct
    return await generate_code_direct(messages, n_tasks, model)


def reset_state() -> None:
    """Forget the health check / fallback cooldown (tests, URL changes)."""
    _state.update(healthy_url=None, down_until=0.0, fallback_warned=0.0)
