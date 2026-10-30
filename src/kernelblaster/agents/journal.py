# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Crash-safe journals that make multi-slot runs resumable (docs/multislot_optimization.md, "Resume").

Per problem, ``OUT/<problem>/journal/``:

* ``state.json``: a snapshot of the agent, written atomically (temp file,
  fsync, rename, fsync of the directory) after the baseline and after every
  completed step. It holds the rollout states, the slot bests and the composite,
  the bandit, the RNG state, LLM usage, the steps log, the steps that were in
  flight with the slot and technique they had picked, and the ``phase``
  (``init`` -> ``rollouts`` -> ``final`` -> ``done``).
* ``llm.jsonl``: every LLM response, appended and fsynced before the candidate
  is evaluated, keyed ``r<rollout>s<step>c<call>`` (call 0 = codegen, 1.. = fix
  attempts). A step that was killed during compile / eval / confirm / fix gets
  its responses from here instead of calling the model again.
* ``remote.jsonl`` (ssh evaluator): the remote files this run staged and the
  persistent hosts it started, so a resumed run removes exactly its own leftovers.

Run level, ``OUT/run.json``: the run configuration and its hash, each
problem's hash, the segments (one per process) and the run-level LLM usage
(reflection, knowledge upkeep) carried across segments.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

FORMAT = "kb-journal/1"
RUN_FORMAT = "kb-run/1"

# exit codes of scripts/run_multislot.py
EXIT_INTERRUPTED = 75      # stopped on SIGTERM / SIGINT after flushing the journals: resumable
EXIT_ABORTED = 76          # a second signal: exited at once (the journals are still consistent)
EXIT_CONFIG_MISMATCH = 3   # an existing out dir was made with another configuration (--force-restart)
EXIT_LOCKED = 4            # another process holds the out dir
EXIT_PAUSED = 77           # halfway sync (--pause-spec): every pausing problem parked at its pause step, journals flushed


def fsync_dir(d: Path) -> None:
    try:
        fd = os.open(str(d), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so a crash leaves either the old or the new file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=1, default=str))


def read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def append_jsonl(path: Path, rec: dict) -> None:
    """One fsynced JSON line (a torn last line of a killed writer is skipped by ``read_jsonl``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def sha(*parts: str, n: int = 12) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode())
        h.update(b"\0")
    return h.hexdigest()[:n]


def rng_state(rng: random.Random) -> list:
    v, st, g = rng.getstate()
    return [v, list(st), g]


def set_rng_state(rng: random.Random, s: list) -> None:
    rng.setstate((s[0], tuple(s[1]), s[2]))


def problem_hash(problem) -> str:
    """What the problem is: its manifest, driver and seed sources (a re-exported,
    changed problem must not resume an old journal)."""
    man = json.dumps(problem.manifest, sort_keys=True, default=str)
    seeds = problem.seed_sources()
    try:
        drv = Path(problem.root / problem.manifest["driver"]).read_text()
    except (OSError, KeyError, TypeError):
        drv = ""
    return sha(man, drv, *(f"{k}\n{seeds[k]}" for k in sorted(seeds)), n=16)


class ProblemJournal:
    """``OUT/<problem>/journal``: the agent snapshot and the LLM responses."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.state_path = self.root / "state.json"
        self.llm_path = self.root / "llm.jsonl"
        self._llm: dict[str, dict] = {}
        for r in read_jsonl(self.llm_path):
            if r.get("key"):
                self._llm[r["key"]] = r
        self.state: Optional[dict] = read_json(self.state_path)
        self.writes = 0

    @property
    def exists(self) -> bool:
        return self.state is not None

    @property
    def phase(self) -> Optional[str]:
        return (self.state or {}).get("phase")

    def save(self, state: dict) -> None:
        state = {"format": FORMAT, "saved_at": time.time(), **state}
        atomic_write_json(self.state_path, state)
        self.state = state
        self.writes += 1

    def llm_get(self, key: str) -> Optional[dict]:
        return self._llm.get(key)

    def llm_put(self, key: str, rec: dict) -> None:
        rec = {"key": key, "ts": time.time(), **rec}
        append_jsonl(self.llm_path, rec)
        self._llm[key] = rec

    def llm_keys(self) -> Iterable[str]:
        return self._llm.keys()

    def clear(self) -> None:
        for p in (self.state_path, self.llm_path, self.root / "remote.jsonl"):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        self.state, self._llm = None, {}


class RunJournal:
    """``OUT/run.json``: configuration hash, problem hashes, segments, run-level usage."""

    def __init__(self, out: Path):
        self.out = Path(out)
        self.path = self.out / "run.json"
        self.data: Optional[dict] = read_json(self.path)
        self._lock = threading.Lock()

    @property
    def exists(self) -> bool:
        return self.data is not None

    def check(self, config: dict, problems: dict[str, str]) -> list[str]:
        """Why this out dir cannot be resumed with ``config`` / ``problems`` ([] = it can)."""
        if self.data is None:
            return []
        why = []
        old = self.data.get("config", {})
        if self.data.get("config_hash") != config_hash(config):
            diff = sorted(k for k in set(old) | set(config) if old.get(k) != config.get(k))
            why.append("configuration differs: " + ", ".join(f"{k} {old.get(k)!r} -> {config.get(k)!r}"
                                                             for k in diff))
        for name, h in problems.items():
            o = self.data.get("problems", {}).get(name)
            if o is not None and o != h:
                why.append(f"problem {name} changed since the journal was written (hash {o} -> {h})")
        return why

    def begin(self, config: dict, problems: dict[str, str], argv: list[str]) -> int:
        """Start a segment; -> its number (0 for a fresh run)."""
        with self._lock:
            d = self.data or {"format": RUN_FORMAT, "created": time.time(), "segments": [], "usage": {}}
            d["config"], d["config_hash"] = config, config_hash(config)
            d.setdefault("problems", {}).update(problems)
            seg = len(d["segments"])
            d["segments"].append({"n": seg, "pid": os.getpid(), "start": time.time(), "status": "running",
                                  "argv": argv})
            d["status"] = "running"
            self.data = d
            atomic_write_json(self.path, d)
            return seg

    def set_usage(self, name: str, usage: dict, segment: int) -> None:
        """Run-level LLM usage of ``segment`` (reflection, upkeep): earlier segments are kept."""
        with self._lock:
            u = self.data.setdefault("usage", {}).setdefault(name, {})
            u[str(segment)] = {k: v for k, v in usage.items() if isinstance(v, (int, float))}
            atomic_write_json(self.path, self.data)

    def usage_before(self, name: str, segment: int) -> dict:
        tot: dict = {}
        for seg, u in (self.data or {}).get("usage", {}).get(name, {}).items():
            if int(seg) < segment:
                for k, v in u.items():
                    tot[k] = tot.get(k, 0) + v
        return tot

    def end(self, segment: int, status: str, **info) -> None:
        with self._lock:
            s = self.data["segments"][segment]
            s.update(end=time.time(), status=status, **info)
            self.data["status"] = status
            atomic_write_json(self.path, self.data)

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self.data = None


def config_hash(config: dict) -> str:
    return sha(json.dumps(config, sort_keys=True, default=str), n=16)


class RunInterrupted(Exception):
    """A problem stopped on request (SIGTERM / SIGINT) with its journal flushed: resumable."""


class RunPaused(Exception):
    """A problem parked at its pause step (``pause_at``: ModelBlaster's halfway sync) with its journal flushed:
    every rollout has completed ``pause_at`` steps and keeps its own current kernel; rerun without the pause to go on."""
