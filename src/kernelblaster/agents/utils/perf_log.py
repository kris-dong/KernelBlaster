"""SQLite-backed performance logging for the KernelBlaster RL flow.

One row per timed *span* (e.g. one LLM call, one compile, one NCU profile).
Grouping by ``phase`` answers "where is time being spent" without needing
distributed tracing. Safe to call from many tasks/threads concurrently — the
writer holds a process-wide lock around each commit.

Default DB path: ``out/perf_log.sqlite``. Override via
``KERNELBLASTER_PERF_LOG_DB`` (file path) or set
``KERNELBLASTER_PERF_LOG_DISABLE=1`` to make all helpers no-op.

Schema is migration-free: the ``perf_spans`` table is created on first write
and additional fields go into the ``extra`` JSON blob rather than schema
changes.

Usage::

    from kernelblaster.agents.utils.perf_log import perf_span, perf_record

    with perf_span(phase="llm_codegen", problem_id=pid, model=m,
                   step=step, trajectory=traj_idx) as span:
        response = await client.call(...)
        span.set_extra(input_tokens=usage.input_tokens)

Use ``perf_record`` for already-measured durations (e.g. when integrating
with ``NamedTimer``)::

    perf_record(phase="ncu_profile", duration_s=elapsed,
                problem_id=pid, success=True)
"""
from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "current_run_id",
    "set_run_id",
    "perf_span",
    "perf_record",
    "summarize_db",
    "is_disabled",
]

_DB_LOCK = threading.Lock()
_DB_CONN: Optional[sqlite3.Connection] = None
_RUN_ID = os.getenv("KERNELBLASTER_PERF_LOG_RUN_ID") or uuid.uuid4().hex[:12]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS perf_spans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    problem_id  TEXT,
    agent       TEXT,
    phase       TEXT NOT NULL,
    step        INTEGER,
    trajectory  INTEGER,
    attempt     INTEGER,
    model       TEXT,
    started_at  REAL NOT NULL,
    duration_s  REAL NOT NULL,
    success     INTEGER NOT NULL DEFAULT 1,
    extra       TEXT
);
CREATE INDEX IF NOT EXISTS idx_run_phase ON perf_spans (run_id, phase);
CREATE INDEX IF NOT EXISTS idx_problem   ON perf_spans (problem_id);
CREATE INDEX IF NOT EXISTS idx_started   ON perf_spans (started_at);
"""


def is_disabled() -> bool:
    return os.getenv("KERNELBLASTER_PERF_LOG_DISABLE", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _db_path() -> Path:
    env = os.getenv("KERNELBLASTER_PERF_LOG_DB")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[4] / "out" / "perf_log.sqlite"


def _get_conn() -> Optional[sqlite3.Connection]:
    """Lazily open the SQLite connection. Returns None when disabled."""
    global _DB_CONN
    if is_disabled():
        return None
    if _DB_CONN is not None:
        return _DB_CONN
    with _DB_LOCK:
        if _DB_CONN is not None:
            return _DB_CONN
        path = _db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: we serialize via _DB_LOCK ourselves so the
        # same connection is reused across threads/asyncio tasks safely.
        conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30.0)
        conn.executescript(_SCHEMA)
        # WAL gives us concurrent readers (summarize CLI) while the run writes.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.commit()
        _DB_CONN = conn
        return _DB_CONN


def current_run_id() -> str:
    return _RUN_ID


def set_run_id(run_id: str) -> None:
    """Override the run id for this process (call once early, before spans)."""
    global _RUN_ID
    _RUN_ID = run_id


def _insert(
    *,
    phase: str,
    started_at: float,
    duration_s: float,
    success: bool,
    problem_id: Optional[str],
    agent: Optional[str],
    step: Optional[int],
    trajectory: Optional[int],
    attempt: Optional[int],
    model: Optional[str],
    extra: Optional[dict],
) -> None:
    conn = _get_conn()
    if conn is None:
        return
    payload = json.dumps(extra) if extra else None
    with _DB_LOCK:
        conn.execute(
            "INSERT INTO perf_spans "
            "(run_id, problem_id, agent, phase, step, trajectory, attempt, "
            " model, started_at, duration_s, success, extra) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _RUN_ID,
                problem_id,
                agent,
                phase,
                step,
                trajectory,
                attempt,
                model,
                started_at,
                duration_s,
                1 if success else 0,
                payload,
            ),
        )
        conn.commit()


class _Span:
    """Handle returned by ``perf_span``; lets callers attach extra fields."""

    __slots__ = ("phase", "started_at", "_ctx", "_extra")

    def __init__(self, phase: str, started_at: float, ctx: dict):
        self.phase = phase
        self.started_at = started_at
        self._ctx = ctx
        self._extra: dict[str, Any] = {}

    def set_extra(self, **fields: Any) -> None:
        """Merge fields into the JSON ``extra`` blob persisted at span close."""
        self._extra.update(fields)


@contextlib.contextmanager
def perf_span(
    *,
    phase: str,
    problem_id: Optional[str] = None,
    agent: Optional[str] = None,
    step: Optional[int] = None,
    trajectory: Optional[int] = None,
    attempt: Optional[int] = None,
    model: Optional[str] = None,
):
    """Context manager that records a span on exit.

    On exception the span is still recorded with ``success=0`` and the
    exception type stored in ``extra``.
    """
    started = time.time()
    ctx = dict(
        problem_id=problem_id,
        agent=agent,
        step=step,
        trajectory=trajectory,
        attempt=attempt,
        model=model,
    )
    span = _Span(phase, started, ctx)
    success = True
    err_type: Optional[str] = None
    try:
        yield span
    except BaseException as e:
        success = False
        err_type = type(e).__name__
        raise
    finally:
        duration = time.time() - started
        extra = dict(span._extra)
        if err_type:
            extra["error_type"] = err_type
        _insert(
            phase=phase,
            started_at=started,
            duration_s=duration,
            success=success,
            extra=extra or None,
            **ctx,
        )


def perf_record(
    *,
    phase: str,
    duration_s: float,
    started_at: Optional[float] = None,
    problem_id: Optional[str] = None,
    agent: Optional[str] = None,
    step: Optional[int] = None,
    trajectory: Optional[int] = None,
    attempt: Optional[int] = None,
    model: Optional[str] = None,
    success: bool = True,
    extra: Optional[dict] = None,
) -> None:
    """Record an already-measured span. Useful for integrating with NamedTimer."""
    if started_at is None:
        started_at = time.time() - duration_s
    _insert(
        phase=phase,
        started_at=started_at,
        duration_s=float(duration_s),
        success=success,
        problem_id=problem_id,
        agent=agent,
        step=step,
        trajectory=trajectory,
        attempt=attempt,
        model=model,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Summarization helpers (used by scripts/summarize_perf_log.py)
# ---------------------------------------------------------------------------


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = (len(s) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def summarize_db(
    db_path: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> dict:
    """Return aggregates suitable for printing or feeding into a notebook.

    Output structure::

        {
            "by_phase":   [{phase, count, total_s, mean_s, p50_s, p95_s, fail_count}, ...],
            "by_problem": [{problem_id, total_s, top_phase, top_phase_s}, ...],
            "by_phase_problem": [{problem_id, phase, total_s, count}, ...],
            "wall_s": float,
            "run_id": str,
        }
    """
    path = Path(db_path) if db_path else _db_path()
    if not path.exists():
        raise FileNotFoundError(f"perf-log DB not found at {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        if run_id is None:
            row = conn.execute(
                "SELECT run_id FROM perf_spans ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return {
                    "by_phase": [],
                    "by_problem": [],
                    "by_phase_problem": [],
                    "wall_s": 0.0,
                    "run_id": "",
                }
            run_id = row["run_id"]

        rows = conn.execute(
            "SELECT phase, problem_id, started_at, duration_s, success "
            "FROM perf_spans WHERE run_id = ?",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()

    by_phase: dict[str, list[float]] = {}
    fail_by_phase: dict[str, int] = {}
    by_problem: dict[str, float] = {}
    top_phase_by_problem: dict[str, dict[str, float]] = {}
    wall_min, wall_max = float("inf"), 0.0
    for r in rows:
        ph = r["phase"]
        d = float(r["duration_s"])
        by_phase.setdefault(ph, []).append(d)
        if not r["success"]:
            fail_by_phase[ph] = fail_by_phase.get(ph, 0) + 1
        pid = r["problem_id"]
        if pid:
            by_problem[pid] = by_problem.get(pid, 0.0) + d
            top_phase_by_problem.setdefault(pid, {})
            top_phase_by_problem[pid][ph] = top_phase_by_problem[pid].get(ph, 0.0) + d
        st = float(r["started_at"])
        wall_min = min(wall_min, st)
        wall_max = max(wall_max, st + d)

    by_phase_out = []
    for ph, durs in sorted(by_phase.items(), key=lambda x: -sum(x[1])):
        by_phase_out.append(
            {
                "phase": ph,
                "count": len(durs),
                "total_s": sum(durs),
                "mean_s": sum(durs) / len(durs),
                "p50_s": _percentile(durs, 0.50),
                "p95_s": _percentile(durs, 0.95),
                "fail_count": fail_by_phase.get(ph, 0),
            }
        )

    by_problem_out = []
    for pid, total in sorted(by_problem.items(), key=lambda x: -x[1]):
        breakdown = top_phase_by_problem.get(pid, {})
        if breakdown:
            top_phase = max(breakdown.items(), key=lambda x: x[1])
        else:
            top_phase = ("", 0.0)
        by_problem_out.append(
            {
                "problem_id": pid,
                "total_s": total,
                "top_phase": top_phase[0],
                "top_phase_s": top_phase[1],
            }
        )

    by_phase_problem_out = []
    for pid, breakdown in top_phase_by_problem.items():
        for ph, total in breakdown.items():
            by_phase_problem_out.append(
                {
                    "problem_id": pid,
                    "phase": ph,
                    "total_s": total,
                    "count": sum(1 for r in rows if r["problem_id"] == pid and r["phase"] == ph),
                }
            )
    by_phase_problem_out.sort(key=lambda x: -x["total_s"])

    wall_s = (wall_max - wall_min) if wall_min != float("inf") else 0.0
    return {
        "by_phase": by_phase_out,
        "by_problem": by_problem_out,
        "by_phase_problem": by_phase_problem_out,
        "wall_s": wall_s,
        "run_id": run_id,
    }
