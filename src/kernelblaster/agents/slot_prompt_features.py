"""Two opt-in (default OFF) prompt features for the multislot codegen prompt.

* ``MB_SLOT_HISTORY=K`` (int, 0 / unset / invalid = off): an "Attempt history on this slot" section built from
  the slot's own outcome rows (same problem + slot + run) in the knowledge DB.
* ``MB_STEP_FREEDOM=none|combo|free`` (unset / unknown = none): replaces the final "Technique to apply" block
  with one that lets the model combine / restructure; the model declares what it applied on a
  ``// techniques: a, b`` line, which is parsed into the outcome's v2 extras (``declared_techniques``).

Both are pure functions of their inputs (no I/O besides reading the env), so they are deterministic and
unit-testable offline. Credit assignment is not touched: the step is still booked on the primary technique.
"""
from __future__ import annotations

import os
import re
from typing import Optional

IMPROVED = "improved"                    # == knowledge.outcomes.IMPROVED (kept local: no import cycle, easy tests)

FREEDOM_MODES = ("none", "combo", "free")
HISTORY_BUDGET_CHARS = 2500
HISTORY_MAX_K = 50


def history_k(env: Optional[dict] = None) -> int:
    """K from ``MB_SLOT_HISTORY``: a positive int, else 0 (off). Capped at HISTORY_MAX_K."""
    raw = (env if env is not None else os.environ).get("MB_SLOT_HISTORY", "")
    try:
        k = int(str(raw).strip())
    except (TypeError, ValueError):
        return 0
    return max(0, min(k, HISTORY_MAX_K))


def step_freedom(env: Optional[dict] = None) -> str:
    """Mode from ``MB_STEP_FREEDOM``: combo | free, anything else (incl. unset / unknown) -> none."""
    raw = str((env if env is not None else os.environ).get("MB_STEP_FREEDOM", "")).strip().lower()
    return raw if raw in ("combo", "free") else "none"


def features_tag(env: Optional[dict] = None) -> dict:
    """Which prompt features are on (logged per step so a run's mode can be verified)."""
    return {"slot_history": history_k(env), "step_freedom": step_freedom(env)}


# ---------------------------------------------------------------- feature 2: step freedom

def technique_block(mode: str, technique: str, technique_desc: str) -> str:
    """The final technique block. ``none`` is exactly the historical text."""
    if mode == "combo":
        return (f"## Primary technique: {technique}\n{technique_desc}\n\n"
                "You may ALSO apply up to two other compatible optimizations in the same kernel if you judge they "
                "help this slot (they must be consistent with the signature, semantics and the target). List every "
                "optimization you applied on one line at the top of the kernel as `// techniques: a, b, c` "
                "(primary technique first).")
    if mode == "free":
        return (f"## Suggested technique: {technique} (a hint, not a constraint)\n{technique_desc}\n\n"
                "Make the changes you judge most valuable for this slot, including restructuring the kernel "
                "completely, as long as the signature and semantics are preserved. List the optimizations you "
                "applied on one line at the top of the kernel as `// techniques: a, b, c`.")
    return f"## Technique to apply: {technique}\n{technique_desc}"


_TECH_LINE = re.compile(r"^[ \t]*//[ \t]*techniques?[ \t]*:[ \t]*(.*)$", re.I | re.M)


def parse_declared_techniques(code: Optional[str], *, max_items: int = 8) -> list[str]:
    """Names from the first ``// techniques: a, b, c`` line of ``code`` ([] when absent / empty / garbled)."""
    if not code:
        return []
    m = _TECH_LINE.search(code)
    if not m:
        return []
    out: list[str] = []
    for tok in re.split(r"[,;]", m.group(1)):
        tok = tok.strip().strip("`'\"*").strip().rstrip(".").strip()
        tok = re.sub(r"\s+", " ", tok)
        if not tok or len(tok) > 60 or tok.lower() in ("none", "n/a") or tok in out:
            continue
        out.append(tok)
        if len(out) >= max_items:
            break
    return out


# ---------------------------------------------------------------- feature 1: attempt history

def _short(x: Optional[str]) -> str:
    return (x or "")[:8]


def _first_line(s: Optional[str], n: int = 100) -> str:
    s = (s or "").strip()
    return s.splitlines()[0].strip()[:n] if s else ""


def _top_metrics(row: dict, n: int = 3) -> str:
    """The metrics that moved most: relative to the 'before' value when the v2 vectors have it, else absolute."""
    delta = row.get("metrics_delta") or {}
    before = (row.get("v2") or {}).get("metrics_before") or {}
    scored = []
    for k, d in delta.items():
        try:
            d = float(d)
        except (TypeError, ValueError):
            continue
        b = before.get(k)
        bv = b.get("v") if isinstance(b, dict) else b
        try:
            score = abs(d) / max(abs(float(bv)), 1e-9) if bv is not None else abs(d)
        except (TypeError, ValueError):
            score = abs(d)
        scored.append((-score, k, d))
    scored.sort()
    return ", ".join(f"{k} {d:+.3g}" for _, k, d in scored[:n] if d)


def _name(row: dict) -> str:
    v2 = row.get("v2") or {}
    if v2.get("kind") == "hypothesis" and v2.get("item_id"):
        return f"hypothesis {v2['item_id']} ({row.get('technique')})"
    return str(row.get("technique"))


def _attempt_line(i: int, row: dict) -> str:
    sp = row.get("speedup")
    parts = [f"#{i} {_name(row)} [{row.get('status')}]",
             f"{sp:.2f}x vs parent" if sp else "no timing"]
    mv = _top_metrics(row)
    if mv:
        parts.append(f"moved: {mv}")
    if row.get("diff_summary"):
        parts.append(f"diff: {row['diff_summary'][:100]}")
    err = _first_line(row.get("error"))
    if err:
        parts.append(f'error: "{err}"')
    parts.append(f"impl {_short(row.get('parent_impl_id')) or 'seed'} -> {_short(row.get('impl_id')) or 'none'}")
    return "- " + "; ".join(parts)


def slot_history_text(rows: list[dict], k: int, *, budget: int = HISTORY_BUDGET_CHARS) -> str:
    """The "Attempt history" section for one slot. ``rows``: that slot's outcome dicts (OutcomeRecord fields +
    ``v2`` + ``seq``) of one problem. "" when off (k <= 0) or no rows. At most ``budget`` characters: oldest
    attempts are dropped first (the best-kernel and dead-end lines always stay)."""
    if k <= 0 or not rows:
        return ""
    rows = sorted(rows, key=lambda r: (r.get("seq", 0), str(r.get("id"))))
    verified = [r for r in rows if r.get("speedup") and r.get("metric_after")]
    seed_us = next((r["metric_before"] for r in rows if r.get("metric_before")), None)
    best = min(verified, key=lambda r: (r["metric_after"], r.get("seq", 0)), default=None)
    if best is not None and seed_us and best["metric_after"] < seed_us:
        best_line = (f"Current best kernel for this slot: impl {_short(best.get('impl_id')) or '?'}, "
                     f"{seed_us / best['metric_after']:.2f}x over the seed.")
    else:
        best_line = "Current best kernel for this slot: the seed (no verified improvement yet), 1.00x over the seed."
    tally: dict[str, list[int]] = {}
    for r in rows:
        t = tally.setdefault(str(r.get("technique")), [0, 0])
        t[0] += 1
        t[1] += r.get("status") == IMPROVED
    dead = [f"{t} ({n} tries)" for t, (n, wins) in sorted(tally.items()) if n >= 2 and not wins]
    dead_line = ("Dead ends (tried >=2 times, never improved): " + ", ".join(dead))[:400] if dead else ""

    shown = rows[-k:]
    first_idx = len(rows) - len(shown) + 1
    lines = [_attempt_line(first_idx + j, r) for j, r in enumerate(shown)]
    head = f"Attempt history on this slot (most recent {k} attempts, oldest first):"

    def render(ls: list[str], omitted: int) -> str:
        body = ([f"({omitted} older attempts omitted)"] if omitted else []) + ls
        return "\n".join([head, *body, best_line, *([dead_line] if dead_line else [])])

    omitted = first_idx - 1
    text = render(lines, omitted)
    while len(text) > budget and lines:
        lines.pop(0)
        omitted += 1
        text = render(lines, omitted)
    return text[:budget]


def history_rows(kv, problem_id: str, slot: str, run_id: Optional[str] = None) -> list[dict]:
    """This problem + slot (+ run) outcome rows from the v2 knowledge DB, in commit order. Reads the same
    table, at the same point of the step, as ``already_tried`` does, so parallel rollouts see exactly the
    attempts that are already committed and nothing else."""
    rows = kv.outcomes.rows("problem_id=? AND COALESCE(slot,'')=?", (problem_id, slot or ""))
    if run_id:
        rows = [r for r in rows if r.get("run_id") in (run_id, None)]
    return rows
