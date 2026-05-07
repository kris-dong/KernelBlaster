"""Live token + USD cost tracking for the optimised RL flow.

A single ``CostTracker`` instance is shared across the agent, the database, and
the runner script. Every LLM call wraps ``generate_code_retry`` (or any
``LLMResponse``-returning helper) and feeds the resulting ``usage`` dict back
through ``CostTracker.record(...)``.

Pricing
-------
The default pricing table is hard-coded for common models (per 1M tokens, USD).
Real prices change; override via env vars:

    COST_PRICE_<MODEL_SLUG>_INPUT   = float (USD / 1M input tokens)
    COST_PRICE_<MODEL_SLUG>_OUTPUT  = float (USD / 1M output tokens)

where ``MODEL_SLUG`` is the lowercased / non-alphanumeric-stripped model name.
For example, ``claude-sonnet-4-6`` becomes ``CLAUDESONNET46``.

Unknown models log a warning once and are tracked at $0 — token counts are
still accurate, just dollars are missing for that row.

Live output
-----------
``log_summary()`` prints a compact table by role and by model. The agent /
runner can call this on a timer (default every 30 s) via
``CostTracker.start_live_logging(logger, interval_s=30.0)``. The background
task is cancelled by ``stop_live_logging()`` or when the tracker is GC'd.

Final dump
----------
``write_summary_json(path)`` writes a JSON file with totals + per-(model, role)
breakdowns + per-problem aggregates if problem ids were attached to records.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Default pricing (USD per 1M tokens). All numbers are approximate — override
# any of them with COST_PRICE_<MODEL_SLUG>_{INPUT,OUTPUT} env vars.
# ---------------------------------------------------------------------------


# Standard input / output pricing (USD per 1M tokens).
# Anthropic numbers below match the published Anthropic API rate card as of
# 2026-04. Long-context variants share the base price for these models
# except where noted in ``_LONG_CONTEXT_MULTIPLIER`` (currently unused —
# requires explicit per-call signalling that the existing usage dict does
# not carry).
_DEFAULT_PRICING: Dict[str, Tuple[float, float]] = {
    # Anthropic — current generation
    "claude-opus-4-7":       (5.00, 25.00),
    "claude-opus-4-6":       (5.00, 25.00),
    "claude-opus-4-5":       (5.00, 25.00),
    "claude-sonnet-4-6":     (3.00, 15.00),
    "claude-sonnet-4-5":     (3.00, 15.00),
    "claude-sonnet-4":       (3.00, 15.00),
    "claude-haiku-4-5":      (1.00, 5.00),
    # Older proven IDs that may still show up in the repo
    "claude-3-5-sonnet":     (3.00, 15.00),
    "claude-3-5-haiku":      (0.80, 4.00),
    "claude-3-opus":         (15.00, 75.00),
    # OpenAI (best-effort approximations)
    "gpt-5":                 (5.00, 20.00),
    "gpt-5-mini":            (0.25, 2.00),
    "gpt-4.1":               (3.00, 12.00),
    "gpt-4.1-mini":          (0.40, 1.60),
    "gpt-4o":                (2.50, 10.00),
    "gpt-4o-mini":           (0.15, 0.60),
}

# Batch-API pricing (USD per 1M tokens). Half the interactive rate for
# Anthropic. Currently unused — the runner does not submit through the
# Batch API. Kept here so a future ``role="batch_*"`` dispatch can call
# ``lookup_batch_pricing()`` without re-discovering the numbers.
_DEFAULT_BATCH_PRICING: Dict[str, Tuple[float, float]] = {
    "claude-opus-4-6":       (2.50, 12.50),
    "claude-opus-4-5":       (2.50, 12.50),
    "claude-sonnet-4-6":     (1.50, 7.50),
    "claude-sonnet-4-5":     (1.50, 7.50),
    "claude-sonnet-4":       (1.50, 7.50),
    "claude-haiku-4-5":      (0.50, 2.50),
    # opus-4-7 batch: not yet published.
}

# Cache pricing (USD per 1M tokens), as returned by Anthropic's API in the
# ``cache_creation_input_tokens`` / ``cache_read_input_tokens`` usage fields.
# Tuple is ``(write_5min_ttl, write_1hr_ttl, read)``. ``None`` for write_1hr
# means the model does not support 1-hour caches.
#
# Charged in addition to (or in place of) the base input price depending on
# whether the request is cache-creation or cache-read. Currently captured
# here for cost-projection use; live tracking still uses the standard rate
# from ``_DEFAULT_PRICING`` because ``LLMResponse.usage`` only carries
# aggregate ``input_tokens`` / ``output_tokens`` — wiring through the cache
# fields requires patching ``generate_code_anthropic`` to forward them.
_DEFAULT_CACHE_PRICING: Dict[str, Tuple[float, Optional[float], float]] = {
    "claude-opus-4-7":       (6.25, 10.00, 0.50),
    "claude-opus-4-6":       (6.25, 10.00, 0.50),
    "claude-opus-4-5":       (6.25, 10.00, 0.50),
    "claude-sonnet-4-6":     (3.75, 6.00, 0.30),
    "claude-sonnet-4-5":     (3.75, 6.00, 0.30),
    "claude-sonnet-4":       (3.75, None, 0.30),
    "claude-haiku-4-5":      (1.25, 2.00, 0.10),
}


def _slug_for_env(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", model).upper()


def _normalise_model(model: str) -> str:
    """Strip provider prefixes / version suffixes for pricing-table matching.

    Handles the spectrum of name shapes seen in this repo:
      anthropic.claude-opus-4-6-v1                    → claude-opus-4-6
      anthropic.claude-sonnet-4-5-20250929-v1:0       → claude-sonnet-4-5
      us.anthropic.claude-haiku-3-5-20241022-v1:0     → claude-haiku-3-5
      llmgateway/openai/gpt-5-mini-2025-08-07         → gpt-5-mini
      claude-haiku-4-5                                → claude-haiku-4-5
    """
    s = (model or "").strip().lower()
    # Strip path-style provider prefixes ("llmgateway/openai/...").
    for pfx in ("llmgateway/", "eos/", "chipnemo/", "azure/", "anthropic/", "openai/"):
        s = s.replace(pfx, "")
    # Strip Bedrock-style provider prefix ("anthropic." / "us.anthropic.").
    s = re.sub(r"^([a-z]{2,5}\.)?anthropic\.", "", s)
    s = re.sub(r"^([a-z]{2,5}\.)?meta\.", "", s)
    # Strip Bedrock version suffix ("-v1", "-v1:0", "-v2:1", etc.).
    s = re.sub(r"-v\d+(:\d+)?$", "", s)
    # Strip trailing date stamp ("-2025-08-07", "-20250929", etc.).
    s = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", s)
    s = re.sub(r"-\d{8}$", "", s)
    return s


def _lookup_in(table: Dict[str, Any], model: str, default: Any) -> Any:
    """Best-effort lookup keyed by normalised model id with prefix fallback."""
    norm = _normalise_model(model)
    if norm in table:
        return table[norm]
    for key, val in table.items():
        if norm.startswith(key) or key in norm:
            return val
    return default


def lookup_pricing(model: str) -> Tuple[float, float]:
    """Return (input_per_M, output_per_M) USD prices for ``model`` (best effort).

    Env-var overrides take precedence:
      ``COST_PRICE_<SLUG>_INPUT`` / ``COST_PRICE_<SLUG>_OUTPUT``
    """
    slug = _slug_for_env(model)
    env_in = os.getenv(f"COST_PRICE_{slug}_INPUT")
    env_out = os.getenv(f"COST_PRICE_{slug}_OUTPUT")
    if env_in and env_out:
        try:
            return float(env_in), float(env_out)
        except ValueError:
            pass
    return _lookup_in(_DEFAULT_PRICING, model, (0.0, 0.0))


def lookup_batch_pricing(model: str) -> Tuple[float, float]:
    """Return Anthropic Batch-API (input, output) USD/M for ``model``.

    Returns ``(0.0, 0.0)`` for models without published batch pricing
    (e.g. opus-4-7 at the time of writing). Currently unused by the live
    tracker — provided for future cost projection.
    """
    return _lookup_in(_DEFAULT_BATCH_PRICING, model, (0.0, 0.0))


def lookup_cache_pricing(model: str) -> Tuple[float, Optional[float], float]:
    """Return cache pricing ``(write_5min, write_1hr, read)`` USD/M.

    ``write_1hr`` is ``None`` for models without 1-hour TTL support.
    Returns ``(0.0, 0.0, 0.0)`` when no entry exists.

    Live cost tracking does not currently use this — wiring through
    requires forwarding ``cache_creation_input_tokens`` /
    ``cache_read_input_tokens`` from the Anthropic ``usage`` block.
    """
    return _lookup_in(_DEFAULT_CACHE_PRICING, model, (0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class _CallRecord:
    timestamp: float
    model: str
    role: str
    problem_id: Optional[str]
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class _Aggregate:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, rec: _CallRecord) -> None:
        self.calls += 1
        self.input_tokens += rec.input_tokens
        self.output_tokens += rec.output_tokens
        self.cost_usd += rec.cost_usd


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Thread-safe live LLM cost tracker. All record/read paths take a lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: List[_CallRecord] = []
        self._by_model: Dict[str, _Aggregate] = defaultdict(_Aggregate)
        self._by_role: Dict[str, _Aggregate] = defaultdict(_Aggregate)
        self._by_problem: Dict[str, _Aggregate] = defaultdict(_Aggregate)
        self._unknown_models_warned: set[str] = set()
        self._live_task: Optional[asyncio.Task] = None
        self._t0 = time.time()

    # ------------------------------------------------------------------
    # Resume — re-hydrate aggregates from a prior snapshot
    # ------------------------------------------------------------------

    def load_snapshot(self, snapshot_path: Path, *, logger=None) -> bool:
        """Re-hydrate aggregates from a prior ``cost_live.json`` snapshot.

        Per-call ``_records`` are NOT restored (the snapshot doesn't carry
        them) — only the rolled-up by_model / by_role / by_problem aggregates
        and the running totals. ``_t0`` is back-dated so ``elapsed_s``
        continues from where the prior run left off.

        Returns True on a successful merge, False if the snapshot is missing
        or unparseable.
        """
        snapshot_path = Path(snapshot_path)
        if not snapshot_path.exists():
            return False
        try:
            payload = json.loads(snapshot_path.read_text())
        except Exception as e:
            if logger:
                logger.warning(f"CostTracker: could not parse {snapshot_path}: {e}")
            return False

        def _agg_from(d: Dict[str, Any]) -> _Aggregate:
            return _Aggregate(
                calls=int(d.get("calls", 0)),
                input_tokens=int(d.get("input_tokens", 0)),
                output_tokens=int(d.get("output_tokens", 0)),
                cost_usd=float(d.get("cost_usd", 0.0)),
            )

        with self._lock:
            for k, d in (payload.get("by_model") or {}).items():
                self._by_model[k] = _agg_from(d)
            for k, d in (payload.get("by_role") or {}).items():
                self._by_role[k] = _agg_from(d)
            for k, d in (payload.get("by_problem") or {}).items():
                self._by_problem[k] = _agg_from(d)
            prior_elapsed = float((payload.get("totals") or {}).get("elapsed_s", 0.0))
            # Back-date t0 so the next snapshot reports cumulative elapsed.
            self._t0 = time.time() - prior_elapsed
        if logger:
            tot = self.totals()
            logger.info(
                f"CostTracker: resumed from {snapshot_path} — "
                f"calls={tot['calls']}, in={tot['input_tokens']:,}, "
                f"out={tot['output_tokens']:,}, cost=${tot['cost_usd']:.3f}, "
                f"prior elapsed={prior_elapsed:.0f}s"
            )
        return True

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        model: str,
        usage: Optional[Dict[str, Any]],
        role: str,
        problem_id: Optional[str] = None,
        logger=None,
    ) -> _CallRecord:
        """Persist one LLM call's usage. Safe to call from any thread/coroutine.

        ``usage`` is the dict attached to ``LLMResponse.usage``; we read
        ``input_tokens`` and ``output_tokens`` (defaulting to 0). When the
        model isn't in the pricing table we still record token counts and emit
        a one-time warning per unknown model.
        """
        usage = usage or {}
        in_tok = int(usage.get("input_tokens", 0) or 0)
        out_tok = int(usage.get("output_tokens", 0) or 0)
        in_p_M, out_p_M = lookup_pricing(model)
        cost = (in_tok / 1_000_000.0) * in_p_M + (out_tok / 1_000_000.0) * out_p_M

        rec = _CallRecord(
            timestamp=time.time(),
            model=model,
            role=role,
            problem_id=problem_id,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )

        with self._lock:
            self._records.append(rec)
            self._by_model[model].add(rec)
            self._by_role[role].add(rec)
            if problem_id:
                self._by_problem[problem_id].add(rec)

            if (in_p_M, out_p_M) == (0.0, 0.0):
                if model not in self._unknown_models_warned:
                    self._unknown_models_warned.add(model)
                    if logger:
                        logger.warning(
                            f"CostTracker: no pricing for model {model!r}; "
                            f"tokens recorded but cost will be $0. "
                            f"Set COST_PRICE_{_slug_for_env(model)}_INPUT / _OUTPUT to override."
                        )
        return rec

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def totals(self) -> Dict[str, float]:
        with self._lock:
            calls = sum(a.calls for a in self._by_role.values())
            in_tok = sum(a.input_tokens for a in self._by_role.values())
            out_tok = sum(a.output_tokens for a in self._by_role.values())
            cost = sum(a.cost_usd for a in self._by_role.values())
        return {
            "calls": calls,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
            "cost_usd": cost,
            "elapsed_s": time.time() - self._t0,
        }

    def snapshot(self) -> Dict[str, Any]:
        # NOTE: do not call self.totals() while holding self._lock —
        # totals() takes the same lock, which deadlocks a non-reentrant
        # threading.Lock. Compute totals first (it locks internally), then
        # acquire once for the per-aggregate dicts.
        totals = self.totals()
        with self._lock:
            return {
                "totals": totals,
                "by_model": {m: asdict(a) for m, a in self._by_model.items()},
                "by_role": {r: asdict(a) for r, a in self._by_role.items()},
                "by_problem": {p: asdict(a) for p, a in self._by_problem.items()},
            }

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_summary(self, logger, *, header: str = "Cost so far") -> None:
        snap = self.snapshot()
        tot = snap["totals"]
        bar = "─" * 78
        logger.info(bar)
        logger.info(
            f"{header}  ·  calls={tot['calls']}  "
            f"in={tot['input_tokens']:,}  out={tot['output_tokens']:,}  "
            f"cost=${tot['cost_usd']:.3f}  "
            f"({tot['elapsed_s']:.0f}s elapsed)"
        )
        if snap["by_role"]:
            logger.info(f"  by role:")
            for role, agg in sorted(snap["by_role"].items()):
                logger.info(
                    f"    {role:<22}  calls={agg['calls']:>4}  "
                    f"in={agg['input_tokens']:>10,}  out={agg['output_tokens']:>9,}  "
                    f"${agg['cost_usd']:>7.3f}"
                )
        if snap["by_model"]:
            logger.info(f"  by model:")
            for model, agg in sorted(snap["by_model"].items()):
                logger.info(
                    f"    {model:<32}  calls={agg['calls']:>4}  "
                    f"in={agg['input_tokens']:>10,}  out={agg['output_tokens']:>9,}  "
                    f"${agg['cost_usd']:>7.3f}"
                )
        logger.info(bar)

    # ------------------------------------------------------------------
    # Live periodic logging (background task)
    # ------------------------------------------------------------------

    def start_live_logging(self, logger, *, interval_s: float = 30.0) -> None:
        """Start a background task that prints the running summary every ``interval_s``."""
        if self._live_task is not None and not self._live_task.done():
            return  # already running
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return

        async def _periodic():
            try:
                while True:
                    await asyncio.sleep(interval_s)
                    self.log_summary(logger, header=f"Cost (live, every {int(interval_s)}s)")
            except asyncio.CancelledError:
                return

        self._live_task = loop.create_task(_periodic())

    def stop_live_logging(self) -> None:
        if self._live_task is not None and not self._live_task.done():
            self._live_task.cancel()

    # ------------------------------------------------------------------
    # Live file dump (no logger spam)
    # ------------------------------------------------------------------

    def start_file_dump(
        self,
        *,
        snapshot_path: Path,
        history_path: Optional[Path] = None,
        interval_s: float = 5.0,
    ) -> None:
        """Periodically write a snapshot JSON (overwritten) + append to a JSONL history.

        ``snapshot_path``  — overwritten on each tick with the current totals
                            and per-model/per-role/per-problem aggregates.
                            ``cat <path>`` shows the live state.
        ``history_path``   — JSONL file, one line per tick with the same payload.
                            ``tail -f <path>`` watches the timeseries.
        ``interval_s``     — seconds between writes.

        No logger output. Cancel via :meth:`stop_live_logging`.
        """
        if self._live_task is not None and not self._live_task.done():
            return
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return

        snapshot_path = Path(snapshot_path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        if history_path is not None:
            history_path = Path(history_path)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            # Ensure file exists so tail -f works immediately.
            if not history_path.exists():
                history_path.write_text("")

        def _write_one() -> None:
            snap = self.snapshot()
            payload = {"ts": time.time(), **snap}
            try:
                snapshot_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            except Exception:
                pass
            if history_path is not None:
                try:
                    with history_path.open("a") as f:
                        f.write(json.dumps(payload, sort_keys=True) + "\n")
                except Exception:
                    pass

        async def _periodic():
            try:
                _write_one()  # immediate first write so the file exists with content
                while True:
                    await asyncio.sleep(interval_s)
                    _write_one()
            except asyncio.CancelledError:
                _write_one()  # final write on shutdown
                return

        self._live_task = loop.create_task(_periodic())

    # ------------------------------------------------------------------
    # Dump to disk
    # ------------------------------------------------------------------

    def write_summary_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Compute snapshot BEFORE acquiring the lock — snapshot() locks internally.
        snap = self.snapshot()
        with self._lock:
            payload = {
                "snapshot": snap,
                "n_records": len(self._records),
                "records": [asdict(r) for r in self._records],
                "pricing_overrides_in_env": [
                    k for k in os.environ
                    if k.startswith("COST_PRICE_") and k.endswith(("_INPUT", "_OUTPUT"))
                ],
            }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Convenience: wrap an LLM-call coroutine and auto-record its usage
# ---------------------------------------------------------------------------


async def record_call(
    coro,
    *,
    tracker: Optional[CostTracker],
    model: str,
    role: str,
    problem_id: Optional[str] = None,
    logger=None,
):
    """Await ``coro`` (an LLMResponse-returning coroutine), record its usage.

    Returns the awaited result unchanged so call sites are minimally invasive::

        response = await record_call(
            generate_code_retry(...),
            tracker=tracker, model=model, role="codegen_simple",
            problem_id=problem_id, logger=logger,
        )
    """
    response = await coro
    if tracker is not None:
        usage = getattr(response, "usage", None)
        tracker.record(
            model=model, usage=usage, role=role,
            problem_id=problem_id, logger=logger,
        )
    return response
