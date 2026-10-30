"""Live progress tracking for the optimised RL flow — file-only output.

Two artefacts under ``<out_root>/``:

  ``progress.json``    — current state snapshot, overwritten on every event.
                         Contents:
                           {
                             "ts": <unix>,
                             "totals": {
                               "problems_total": int,
                               "problems_running": int,
                               "problems_succeeded": int,
                               "problems_failed": int,
                             },
                             "problems": {
                               "<problem_id>": {
                                 "status": "running"|"success"|"failed"|"timeout"|"no_baseline",
                                 "init_cycles": int|null,
                                 "best_cycles": int|null,
                                 "improvement_pct": float|null,
                                 "step_count": int,
                                 "trajectories": {
                                   "<traj_idx>": {
                                     "last_step": int,
                                     "last_technique": str,
                                     "last_cycles": int,
                                     "last_improvement_pct": float
                                   },
                                   ...
                                 },
                                 "started_ts": float,
                                 "finished_ts": float|null,
                                 "final_cycles": int|null
                               },
                               ...
                             }
                           }

  ``progress.jsonl``   — append-only event stream, one event per line.
                         Event kinds:
                           {"type": "problem_started", "problem_id": ..., "init_cycles": ..., "ts": ...}
                           {"type": "step_done",       "problem_id": ..., "traj_idx": ..., "step_idx": ..., "technique": ..., "cycles": ..., "improvement_pct": ..., "ts": ...}
                           {"type": "problem_finished","problem_id": ..., "success": ..., "final_cycles": ..., "init_cycles": ..., "ts": ...}

This module emits **no logger output** — by design. Watch it from another
shell with::

    cat <out_root>/progress.json | jq '.totals'
    tail -f <out_root>/progress.jsonl
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class ProgressWriter:
    """Thread-safe progress tracker that mirrors live state to two files.

    With ``resume=True`` the writer:
      - Appends to ``progress.jsonl`` instead of truncating it.
      - Re-loads ``progress.json`` (if present) so the running state /
        per-problem counters survive a kill.
    """

    def __init__(self, out_root: Path, *, resume: bool = False) -> None:
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.out_root / "progress.jsonl"
        self.snapshot_path = self.out_root / "progress.json"
        self._lock = threading.Lock()
        self._state: Dict[str, Dict[str, Any]] = {}

        if resume:
            # Preserve the events history; only ensure the file exists.
            if not self.events_path.exists():
                self.events_path.write_text("")
            # Re-hydrate per-problem state from the prior snapshot if any.
            if self.snapshot_path.exists():
                try:
                    payload = json.loads(self.snapshot_path.read_text())
                    self._state = dict(payload.get("problems", {}))
                except Exception:
                    self._state = {}
            # Mark previously-running problems as interrupted — they'll be
            # re-driven by the runner if they aren't on the skip list.
            for pid, pstate in self._state.items():
                if pstate.get("status") == "running":
                    pstate["status"] = "interrupted"
        else:
            # Fresh run: truncate / create both files so ``tail -f`` works
            # from an empty start and the snapshot reflects this run only.
            self.events_path.write_text("")

        # Initial snapshot so the file always has something parseable.
        self._snapshot_unlocked()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, **fields: Any) -> None:
        rec = {"type": event_type, "ts": time.time(), **fields}
        # Append-only — a separate file handle per write avoids interleaving
        # issues across threads at the cost of a small open() syscall.
        try:
            with self.events_path.open("a") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        except Exception:
            pass

    def _snapshot_unlocked(self) -> None:
        # Caller must hold ``self._lock``.
        totals = {
            "problems_total": len(self._state),
            "problems_running": sum(1 for s in self._state.values() if s.get("status") == "running"),
            "problems_succeeded": sum(1 for s in self._state.values() if s.get("status") == "success"),
            # 'no_baseline' counts as a kept artifact (not a failure) — the kernel
            # works, we just couldn't measure speedup vs. a broken init.cu.
            "problems_no_baseline": sum(
                1 for s in self._state.values() if s.get("status") == "no_baseline"
            ),
            "problems_failed": sum(
                1 for s in self._state.values()
                if s.get("status") in ("failed", "timeout")
            ),
        }
        payload = {
            "ts": time.time(),
            "totals": totals,
            "problems": dict(self._state),
        }
        try:
            tmp = self.snapshot_path.with_suffix(self.snapshot_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
            tmp.replace(self.snapshot_path)  # atomic on POSIX
        except Exception:
            pass

    def _snapshot(self) -> None:
        with self._lock:
            self._snapshot_unlocked()

    # ------------------------------------------------------------------
    # Public hooks
    # ------------------------------------------------------------------

    def problem_started(
        self, problem_id: str, *, init_cycles: Optional[int] = None
    ) -> None:
        with self._lock:
            self._state[problem_id] = {
                "status": "running",
                "init_cycles": int(init_cycles) if init_cycles else None,
                "best_cycles": int(init_cycles) if init_cycles else None,
                "improvement_pct": 0.0 if init_cycles else None,
                "step_count": 0,
                "trajectories": {},
                "started_ts": time.time(),
                "finished_ts": None,
                "final_cycles": None,
            }
            self._snapshot_unlocked()
        self._emit(
            "problem_started",
            problem_id=problem_id,
            init_cycles=int(init_cycles) if init_cycles else None,
        )

    def step_done(
        self,
        problem_id: str,
        *,
        traj_idx: int,
        step_idx: int,
        technique: str,
        cycles: Optional[int],
        improvement_pct: Optional[float],
    ) -> None:
        with self._lock:
            s = self._state.setdefault(
                problem_id,
                {
                    "status": "running",
                    "init_cycles": None,
                    "best_cycles": None,
                    "improvement_pct": None,
                    "step_count": 0,
                    "trajectories": {},
                    "started_ts": time.time(),
                    "finished_ts": None,
                    "final_cycles": None,
                },
            )
            s["step_count"] = int(s.get("step_count", 0)) + 1
            s.setdefault("trajectories", {})[str(traj_idx)] = {
                "last_step": int(step_idx),
                "last_technique": technique,
                "last_cycles": int(cycles) if cycles else None,
                "last_improvement_pct": float(improvement_pct) if improvement_pct is not None else None,
            }
            best = s.get("best_cycles")
            if cycles and (best is None or cycles < best):
                s["best_cycles"] = int(cycles)
                init = s.get("init_cycles")
                if init and cycles:
                    s["improvement_pct"] = ((init - cycles) / init) * 100.0
            self._snapshot_unlocked()
        self._emit(
            "step_done",
            problem_id=problem_id,
            traj_idx=int(traj_idx),
            step_idx=int(step_idx),
            technique=technique,
            cycles=int(cycles) if cycles else None,
            improvement_pct=float(improvement_pct) if improvement_pct is not None else None,
        )

    def problem_finished(
        self,
        problem_id: str,
        *,
        success: bool,
        final_cycles: Optional[int] = None,
        init_cycles: Optional[int] = None,
        status_override: Optional[str] = None,
    ) -> None:
        with self._lock:
            s = self._state.setdefault(
                problem_id,
                {
                    "step_count": 0,
                    "trajectories": {},
                    "started_ts": time.time(),
                },
            )
            s["status"] = status_override or ("success" if success else "failed")
            s["finished_ts"] = time.time()
            if final_cycles is not None:
                s["final_cycles"] = int(final_cycles)
            if init_cycles is not None:
                s.setdefault("init_cycles", int(init_cycles))
            # Recompute improvement_pct one last time from the canonical numbers.
            init = s.get("init_cycles")
            best = s.get("best_cycles") or s.get("final_cycles")
            if init and best:
                s["improvement_pct"] = ((init - best) / init) * 100.0
            self._snapshot_unlocked()
        self._emit(
            "problem_finished",
            problem_id=problem_id,
            success=bool(success),
            status=status_override or ("success" if success else "failed"),
            final_cycles=int(final_cycles) if final_cycles else None,
            init_cycles=int(init_cycles) if init_cycles else None,
        )
