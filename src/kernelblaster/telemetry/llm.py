# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM-call telemetry: one record per provider call (``query.generate_code``).

``llm_traced`` wraps the provider dispatcher, so every caller (the agent's
codegen / fix calls, reflection, knowledge-v2 upkeep, legacy agents) is
covered by one hook. The *role* comes from the enclosing spans' attributes
(``attrs(role="fix")``, ``traced("reflect", role="reflect")``), the step /
slot / problem likewise.

Per call: input tokens (all prompt tokens, cached or not), uncached input,
cache-read and cache-write input tokens where the provider reports them,
output tokens, stop reason and truncation (stop reason ``max_tokens`` /
``length``, or output tokens at the request's ``max_tokens``), latency (time
to the full response; ``ttft_s`` when the provider streams), the provider's
own latency (Bedrock ``metrics.latencyMs``), provider-side retries (boto3
``RetryAttempts``), the provider-reported cost and a price-table estimate.
A call relayed through the LLM server also gets ``relay_queue_s`` and
``relay_network_s``.
``generate_code_retry``'s back-off sleeps are recorded by :func:`backoff`.
"""
from __future__ import annotations

import asyncio
import functools
from typing import Any, Optional

from . import core

TRUNCATION_STOPS = {"max_tokens", "length", "max_output_tokens", "model_length"}

_log = None


def _logger():
    global _log
    if _log is None:
        try:
            from loguru import logger as _l
        except Exception:  # noqa: BLE001
            import logging
            _l = logging.getLogger("kernelblaster.telemetry")
        _log = _l
    return _log


def _as_dict(u: Any) -> dict:
    if u is None:
        return {}
    if isinstance(u, dict):
        return u
    out = {}
    for k in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens", "total_tokens",
              "cache_read_input_tokens", "cache_creation_input_tokens", "prompt_tokens_details"):
        v = getattr(u, k, None)
        if v is not None:
            out[k] = v
    return out


def _int(*vals) -> int:
    for v in vals:
        if v:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    return 0


@functools.lru_cache(maxsize=256)
def _prices(model: str) -> tuple[float, float, float, float]:
    """(input, output, cache read, cache write) USD per 1M tokens."""
    try:
        from ..agents.cost_tracker import lookup_cache_pricing, lookup_pricing
        pin, pout = lookup_pricing(model or "")
        cw, _cw1h, cr = lookup_cache_pricing(model or "")
    except Exception:  # noqa: BLE001
        return 0.0, 0.0, 0.0, 0.0
    # Anthropic's standard multipliers when the cache table has no row
    return pin, pout, cr or 0.1 * pin, cw or 1.25 * pin


def normalize_usage(usage: Any, model: Optional[str], n: int = 1) -> dict:
    """Provider usage dict (OpenAI, Bedrock Converse, Anthropic, Claude CLI,
    stub) -> the telemetry record fields."""
    u = _as_dict(usage)
    if isinstance(u.get("telemetry"), dict):          # provider extras (stop reasons, max_tokens, ...)
        u = {**u, **u["telemetry"]}
    details = u.get("prompt_tokens_details") or {}
    if not isinstance(details, dict):
        details = {"cached_tokens": getattr(details, "cached_tokens", 0)}
    inp = _int(u.get("prompt_tokens"), u.get("input_tokens"), u.get("inputTokens"))
    out = _int(u.get("completion_tokens"), u.get("output_tokens"), u.get("outputTokens"))
    cr = _int(u.get("cache_read_input_tokens"), u.get("cacheReadInputTokens"), details.get("cached_tokens"))
    cw = _int(u.get("cache_creation_input_tokens"), u.get("cache_write_input_tokens"),
              u.get("cacheWriteInputTokens"))
    # OpenAI / the Claude CLI wrapper count cached tokens inside prompt_tokens;
    # Anthropic / Bedrock report input_tokens next to the cache fields
    includes = u.get("input_includes_cache")
    if includes is None:
        includes = "prompt_tokens" in u
    total_in = inp if includes else inp + cr + cw
    uncached = max(total_in - cr - cw, 0)
    stops = u.get("stop_reasons") or ([u["stop_reason"]] if u.get("stop_reason") else [])
    stops = [str(s) for s in stops if s]
    max_tokens = _int(u.get("max_tokens"))
    truncated = sum(1 for s in stops if s.lower() in TRUNCATION_STOPS)
    if not stops and max_tokens and out >= max_tokens * max(n, 1):
        truncated = max(n, 1)
    reported = float(u.get("cost_usd") or 0.0)
    pin, pout, pcr, pcw = _prices(model or "")
    est = (uncached * pin + cr * pcr + cw * pcw + out * pout) / 1e6
    rec = {"input_tokens": total_in, "uncached_input_tokens": uncached, "cache_read_input_tokens": cr,
           "cache_write_input_tokens": cw, "output_tokens": out, "cost_usd": reported, "est_usd": est,
           "truncated": truncated, "provider_retries": _int(u.get("provider_retries"))}
    if stops:
        rec["stop_reason"] = stops[0] if len(set(stops)) == 1 else ",".join(stops)
    if max_tokens:
        rec["max_tokens"] = max_tokens
    if u.get("server_latency_ms"):
        rec["server_latency_s"] = float(u["server_latency_ms"]) / 1000.0
    if u.get("ttft_s"):
        rec["ttft_s"] = float(u["ttft_s"])
    # relayed call (agents/utils/llm_relay.py): the server's queue wait, the network round trip,
    # and the server's throttling back-off (counted like generate_code_retry's back-off)
    for k in ("relay_queue_s", "relay_network_s"):
        if u.get(k) is not None:
            rec[k] = float(u[k])
    if u.get("relay_backoff_s"):
        rec["backoff_s"] = float(u["relay_backoff_s"])
        rec["backoffs"] = _int(u.get("relay_backoffs"))
    if u.get("relay_replayed"):
        rec["relay_replayed"] = True
    return rec


def _chars(messages) -> int:
    try:
        return sum(len(m.get("content") or "") for m in messages)
    except Exception:  # noqa: BLE001
        return 0


def _add(sp, rec: dict) -> None:
    for c in getattr(sp, "collectors", ()):
        c.add_llm(rec)


def _ctx_fields(attrs: dict) -> dict:
    return {k: attrs[k] for k in ("problem", "rollout", "step", "slot", "purpose") if attrs.get(k) is not None}


def llm_traced(fn):
    """Decorator for ``generate_code(messages, n_tasks=1, model=None)``."""
    @functools.wraps(fn)
    async def wrapper(*a, **kw):
        if not core.enabled():
            return await fn(*a, **kw)
        messages = kw.get("messages", a[0] if a else None)
        n = kw.get("n_tasks", a[1] if len(a) > 1 else 1) or 1
        model = kw.get("model", a[2] if len(a) > 2 else None)
        cur = core.current()
        role = (cur.attrs.get("role") if cur is not None else None) or "other"
        with core.span(f"llm.{role}", role=role, model=model) as sp:
            t0 = core._now()
            base = {"t": "llm", "span": sp.id, "role": role, "model": model, "n": n,
                    "prompt_chars": _chars(messages or []), **_ctx_fields(sp.attrs)}
            try:
                resp = await fn(*a, **kw)
            except BaseException as e:
                lat = core._now() - t0
                rec = {**base, "calls": 1, "errors": 1, "latency_s": lat, "error": type(e).__name__,
                       "error_text": str(e)[:300]}
                _add(sp, rec)
                core.emit({**rec, "ts": round(core.epoch(t0), 6)})
                raise
            lat = core._now() - t0
            gens = getattr(resp, "generations", None) or []
            rec = {**base, "calls": 1, "latency_s": lat,
                   **normalize_usage(getattr(resp, "usage", None), model, n),
                   "output_chars": sum(len(g or "") for g in gens)}
            sp.set(input_tokens=rec["input_tokens"], output_tokens=rec["output_tokens"],
                   stop_reason=rec.get("stop_reason"), truncated=rec["truncated"] or None)
            _add(sp, rec)
            core.emit({**rec, "ts": round(core.epoch(t0), 6)})
            if rec["truncated"]:
                _logger().warning(f"[telemetry] LLM output truncated ({rec.get('stop_reason') or 'max_tokens'}): "
                                  f"role={role} model={model} output_tokens={rec['output_tokens']}"
                                  + (f" max_tokens={rec['max_tokens']}" if rec.get("max_tokens") else ""))
            return resp
    wrapper.__kb_traced__ = "llm"
    return wrapper


async def backoff(delay: float, model: Optional[str] = None, reason: str = "") -> None:
    """``await asyncio.sleep(delay)`` recorded as an ``llm.backoff`` span."""
    if not core.enabled():
        await asyncio.sleep(delay)
        return
    cur = core.current()
    role = (cur.attrs.get("role") if cur is not None else None) or "other"
    with core.span("llm.backoff", role=role, model=model, reason=reason or None) as sp:
        await asyncio.sleep(delay)
    rec = {"role": role, "model": model, "backoff_s": float(delay), "backoffs": 1}
    _add(sp, rec)
    core.emit({"t": "llm_backoff", "span": sp.id, "ts": round(core.epoch(sp.t0), 6), **rec, "reason": reason})
