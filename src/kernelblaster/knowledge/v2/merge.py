# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A global knowledge store across problems and sweeps: versioned snapshots, seeds, delta merges.

Every KernelBlaster run keeps its knowledge in ``<knowledge dir>/knowledge.db`` (+ ``impl/``).
This module keeps ONE canonical store next to the controller and moves knowledge in both
directions:

    seed   (read path)   current snapshot --materialize(partition, problem)--> a run's knowledge.db
    merge  (write path)  finished run's knowledge.db - the db it started from --delta--> new snapshot

Layout of a store root (default ``~/Documents/ModelBlaster-clone/knowledge_global``)::

    current                 the current snapshot id ("v0007"); replaced atomically
    snapshots/vNNNN.db      immutable (read-only) SQLite snapshots, one per merge / import / rollback
    snapshots/vNNNN.json    manifest: parent, label, sha256, counts, merged run
    impl_code/<sha>.cu      content-addressed kernel sources referenced by gk_impl rows (write once)
    base/<hash>.db          cached repo seed (targets / states / techniques / mined.json) per repo seed version
    backups/                the ``current`` pointer + ledger dump saved before every merge / rollback
    .lock                   fcntl lock: merges, rollbacks and prunes serialise on it

A snapshot holds only the *learned layer* (``gk_*`` tables) -- never the repo seed, which is
rebuilt from the KernelBlaster tree when a seed is materialized. A seed therefore is
``repo seed + learned rows of one partition``, and a merge is ``run db - seed db``: seed rows
are never counted twice, statistics are additive on the delta only, outcomes are appended once
(by id), claims / hypotheses / states / techniques are matched by id (rows that came from the
snapshot keep their id) or by content (new rows), with evidence merged as unions of outcome ids.

Scoping. Every learned row lives in a *partition* ``<target key>|<dtype>`` (``cuda:sm_89|fp16``):
the arch and storage dtype the run optimised for. A seed for partition P carries only P's rows
(``share="strict"``, the default), so knowledge learned on sm_89 fp16 never reaches an sm_100 or
fp32 run. ``share="lifted"`` also ships claims/techniques other partitions validated above target
level (lifts need support on >= 2 targets); ``share="hierarchy"`` additionally ships other
partitions' node/universal statistics as shrinkage priors. Problem-local items (impl notes,
"already tried" outcomes, hypothesis attempts) are namespaced ``<problem>::<kb problem id>``
and only un-namespaced in a seed for the same problem, so they never leak into another one.

Bounded growth. Seeds are capped (claims ranked by support - contradict with a per-merge half
life, open / closed hypotheses, probationary states, outcomes); the store retires contradicted
claims (contradict >= support, >= 2), stale ineligible claims (no new evidence for
``stale_merges`` merges) and keeps at most ``max_active_claims`` active claims per partition.
Prompt size itself is bounded by the retrieval (``KnowledgeV2.retrieve``: k claims, max_chars).

Idempotency. A ledger in every snapshot records (run uid, fingerprint) of every merge; merging
the same run state again is a no-op, a run already merged must be merged against the copy merged
last (incremental merge), never against its seed again. Merges build the new snapshot in a temp
file and rename it into place under the lock; a failed merge leaves the store untouched.

CLI: ``python -m kernelblaster.knowledge.v2.merge {status,seed,merge,rollback,prune,verify,show}``.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_ROOT = Path.home() / "Documents" / "ModelBlaster-clone" / "knowledge_global"
FORMAT = "kb-global-knowledge/1"
NS = "::"                                   # problem namespace separator in problem ids

GK_SCHEMA = """
CREATE TABLE IF NOT EXISTS gk_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS gk_ledger (run_uid TEXT, seq INTEGER, fingerprint TEXT, part TEXT, problem TEXT,
    attempt TEXT, base TEXT, version TEXT, ts TEXT, report TEXT, PRIMARY KEY (run_uid, seq));
CREATE TABLE IF NOT EXISTS gk_claims (part TEXT, id TEXT, status TEXT, base_ck TEXT, data TEXT, prov TEXT,
    PRIMARY KEY (part, id));
CREATE TABLE IF NOT EXISTS gk_hypotheses (part TEXT, id TEXT, status TEXT, data TEXT, prov TEXT, PRIMARY KEY (part, id));
CREATE TABLE IF NOT EXISTS gk_states (part TEXT, id TEXT, status TEXT, data TEXT, prov TEXT, PRIMARY KEY (part, id));
CREATE TABLE IF NOT EXISTS gk_techniques (part TEXT, id TEXT, status TEXT, data TEXT, prov TEXT, PRIMARY KEY (part, id));
CREATE TABLE IF NOT EXISTS gk_stats (part TEXT, item TEXT, level TEXT, scope TEXT, op TEXT, state TEXT,
    n REAL, s REAL, ss REAL, PRIMARY KEY (part, item, level, scope, op, state));
CREATE TABLE IF NOT EXISTS gk_outcomes (id TEXT PRIMARY KEY, part TEXT, problem TEXT, run_uid TEXT, ts TEXT,
    rec TEXT, v2 TEXT);
CREATE INDEX IF NOT EXISTS gk_outcomes_pp ON gk_outcomes (part, problem);
CREATE TABLE IF NOT EXISTS gk_overrides (part TEXT, tech TEXT, data TEXT, PRIMARY KEY (part, tech));
CREATE TABLE IF NOT EXISTS gk_proposals (part TEXT, id TEXT, key TEXT, status TEXT, data TEXT, prov TEXT,
    PRIMARY KEY (part, id));
CREATE TABLE IF NOT EXISTS gk_triggers (part TEXT, key TEXT, PRIMARY KEY (part, key));
CREATE TABLE IF NOT EXISTS gk_impl (id TEXT PRIMARY KEY, part TEXT, kkey TEXT, slot_us REAL, data TEXT, prov TEXT);
CREATE INDEX IF NOT EXISTS gk_impl_k ON gk_impl (part, kkey);
CREATE TABLE IF NOT EXISTS gk_scopes (id TEXT PRIMARY KEY, parent TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS gk_targets (key TEXT PRIMARY KEY, node TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS gk_log (seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, ref TEXT, part TEXT,
    data TEXT);
CREATE TABLE IF NOT EXISTS gk_contrib (seq INTEGER, run_uid TEXT, part TEXT, kind TEXT, key TEXT, data TEXT);
CREATE INDEX IF NOT EXISTS gk_contrib_pks ON gk_contrib (part, kind, seq);
"""

GK_TABLES = ("gk_claims", "gk_hypotheses", "gk_states", "gk_techniques", "gk_stats", "gk_outcomes",
             "gk_overrides", "gk_proposals", "gk_triggers", "gk_impl", "gk_ledger")


class MergeError(RuntimeError):
    """The merge was refused; the store is unchanged."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _j(x) -> str:
    return json.dumps(x, sort_keys=True, default=str)


# ---------------------------------------------------------------- policy

@dataclass
class Policy:
    """Knobs of the read path (seed caps, cross-partition sharing) and the store's upkeep."""
    share: str = "strict"                    # strict | lifted | hierarchy (see module doc)
    seed_max_claims: int = 1500              # active claims shipped in a seed (in-run cap is 2000), see _select_claims
    seed_fresh_merges: int = 10              # claims with evidence from this many latest merges are shipped first ...
    seed_fresh_max: int = 400                # ... up to this many
    seed_max_open_hyp: int = 150             # proposed / testing hypotheses shipped
    seed_max_closed_hyp: int = 150           # retired / promoted ones (the generator's "do not duplicate" list)
    seed_max_prob_states: int = 60           # probationary discovered states shipped (core ones always)
    seed_max_problem_outcomes: int = 1500    # outcomes of the same problem (its "already tried")
    seed_max_cited_outcomes: int = 3000      # outcomes cited by the shipped claims (their evidence)
    seed_rejected_per_key: int = 3           # rejected state proposals per trigger key
    epochs_per_merge: float = 0.5            # a claim's age in a seed (reflection epochs): merges since its evidence x this
    half_life_merges: float = 100.0          # global ranking decay (a sweep is ~50 merges)
    stale_merges: int = 100                  # retire an ineligible claim without new evidence this long
    stale_hyp_merges: int = 100              # retire an open hypothesis nobody tested this long
    max_active_claims: int = 4000            # general (not problem-local) active claims per partition
    max_local_claims_per_problem: int = 400  # problem-local active claims (impl notes ...) per problem
    impl_min_speedup: float = 1.02           # impl records kept: speedup over their problem's seed
    impl_per_key: int = 1                    # kept per (op, signature, arch, dtype, shape set)
    impl_max_per_op: int = 8                 # kept per (op, signature, arch, dtype) over all shape sets
    proposals_per_key: int = 5               # rejected proposals kept per trigger key in the store
    refresh_max_claims: int = 300            # new claims in one mid-run refresh package (refresh.py)
    # A5 (impl database / knowledge store), all off by default -- the defaults reproduce the behaviour above
    impl_pool: bool = False                  # merge: keep per (op, signature, arch, dtype) a pool + cross-shape matrix
    pool_max_kernels: int = 16               # pool retention cap (usefulness + diversity, impl_pool.Pool.retain)
    pool_min_speedup: float = 1.0            # pool input: verified records at least this fast vs their seed
    seed_pool: bool = False                  # materialize: ship the partition's pools into impl/pool/
    struct_outcomes: bool = False            # merge: structural decisions (fusion / layout / attention) -> gk_struct
    seed_struct: bool = False                # materialize: write struct_priors.json (priors.StructPriors)

    @classmethod
    def from_json(cls, s) -> "Policy":
        if not s:
            return cls()
        d = json.loads(s) if isinstance(s, str) else dict(s)
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# ---------------------------------------------------------------- partitions / namespaces

def part_key(target_key: str, dtype: Optional[str]) -> str:
    """``cuda:sm_89`` + ``fp16`` -> ``cuda:sm_89|fp16``; arch numbers become target keys."""
    tk = target_key
    if ":" not in tk:
        tk = f"cuda:sm_{tk}" if tk.isdigit() else f"cuda:{tk}"
    return f"{tk}|{dtype or 'fp32'}"


def parse_part(part: str) -> tuple[str, str]:
    tk, _, dt = part.partition("|")
    return tk, dt or "fp32"


def ns(problem: Optional[str], pid: Optional[str]) -> Optional[str]:
    if not pid or not problem or NS in pid:
        return pid
    return f"{problem}{NS}{pid}"


def strip_ns(problem: Optional[str], pid: Optional[str]) -> Optional[str]:
    if pid and problem and pid.startswith(problem + NS):
        return pid[len(problem) + len(NS):]
    return pid


# ---------------------------------------------------------------- sqlite helpers

def _connect(path: Path, *, ro: bool = False) -> sqlite3.Connection:
    if ro:
        c = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro&immutable=1", uri=True)
    else:
        c = sqlite3.connect(str(path), isolation_level=None)
        c.execute("PRAGMA busy_timeout=30000")
    return c


def _tables(c: sqlite3.Connection) -> set:
    return {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _meta(c: sqlite3.Connection, key: str, default=None, table: str = "meta"):
    try:
        r = c.execute(f"SELECT value FROM {table} WHERE key=?", (key,)).fetchone()
    except sqlite3.Error:
        return default
    return json.loads(r[0]) if r else default


def _set_meta(c: sqlite3.Connection, key: str, value, table: str = "meta") -> None:
    c.execute(f"INSERT OR REPLACE INTO {table} VALUES (?, ?)", (key, json.dumps(value, sort_keys=True)))


def _check(c: sqlite3.Connection, what: str) -> None:
    r = c.execute("PRAGMA quick_check").fetchone()
    if not r or r[0] != "ok":
        raise MergeError(f"{what}: sqlite quick_check failed: {r}")


@contextlib.contextmanager
def _db_copy(path: Path):
    """A private, consistent single-file copy of a (possibly WAL-mode, possibly live) knowledge.db."""
    path = Path(path)
    if not path.exists():
        raise MergeError(f"{path}: no such database")
    with tempfile.TemporaryDirectory(prefix="gk_") as td:
        dst = Path(td) / "k.db"
        shutil.copy2(path, dst)
        for sfx in ("-wal",):
            if Path(str(path) + sfx).exists():
                shutil.copy2(str(path) + sfx, str(dst) + sfx)
        c = sqlite3.connect(str(dst), isolation_level=None)
        try:
            c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
        try:
            yield c
        finally:
            c.close()


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- reading a knowledge.db

def _claim_ck(d: dict) -> str:
    """Content key of a claim (kind, scope, polarity, normalised text): stable across fresh seeds."""
    from ..claims import Claim
    try:
        c = Claim.from_dict(d)
    except (TypeError, ValueError):
        return ""
    return hashlib.sha1(_j([list(c.dedupe_key()), c.norm]).encode()).hexdigest()[:20]


class KDB:
    """Read-only view of one knowledge.db (run, seed or base)."""

    def __init__(self, c: sqlite3.Connection):
        self.c = c
        t = _tables(c)
        if "meta" not in t or "outcomes" not in t:
            raise MergeError("not a knowledge v2 database (no meta / outcomes table)")
        self.t = t

    def meta(self, key, default=None):
        return _meta(self.c, key, default)

    def rows(self, table: str) -> dict:
        if table not in self.t:
            return {}
        out = {}
        for rid, data in self.c.execute(f"SELECT id, data FROM {table}"):
            try:
                out[rid] = json.loads(data)
            except (TypeError, ValueError):
                continue
        return out

    def techniques(self) -> dict:
        out = self.rows("techniques")
        if "realizations" in self.t:
            real: dict = {}
            for tid, data in self.c.execute("SELECT technique_id, data FROM realizations"):
                real.setdefault(tid, []).append(json.loads(data))
            for tid, d in out.items():
                d["realizations"] = real.get(tid, d.get("realizations", []))
        return out

    def stats(self) -> dict:
        if "stats" not in self.t:
            return {}
        return {tuple(r[:5]): (r[5], r[6], r[7]) for r in self.c.execute("SELECT * FROM stats")}

    def outcomes(self) -> dict:
        out = {}
        for oid, rec, v2 in self.c.execute("SELECT id, rec, v2 FROM outcomes ORDER BY seq"):
            try:
                out[oid] = (json.loads(rec), json.loads(v2) if v2 else {})
            except (TypeError, ValueError):
                continue
        return out

    def outcome_ids(self) -> set:
        return {r[0] for r in self.c.execute("SELECT id FROM outcomes")}


def fingerprint(c: sqlite3.Connection) -> str:
    """What a run has learned so far, as one hash (merging the same state twice is a no-op)."""
    h = hashlib.sha1()
    ids = sorted(r[0] for r in c.execute("SELECT id FROM outcomes"))
    h.update(_j(ids).encode())
    t = _tables(c)
    if "stats" in t:
        h.update(_j(c.execute("SELECT COUNT(*), ROUND(SUM(n), 6), ROUND(SUM(s), 6) FROM stats").fetchone()).encode())
    for tb in ("claims", "hypotheses", "states", "techniques", "proposals"):
        if tb in t:
            rows = sorted(c.execute(f"SELECT id, data FROM {tb}").fetchall())
            h.update(hashlib.sha1(_j(rows).encode()).hexdigest().encode())
    h.update(_j(_meta(c, "prediction_overrides", {})).encode())
    return h.hexdigest()[:20]


# ---------------------------------------------------------------- the store

def _lock_file(path: Path, timeout: float):
    import fcntl
    f = open(path, "a+")
    t0 = time.time()
    while True:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f
        except OSError:
            if time.time() - t0 > timeout:
                f.close()
                raise MergeError(f"store lock {path} busy for {timeout:.0f}s")
            time.sleep(0.2)


class GlobalStore:
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("KB_GLOBAL_KNOWLEDGE") or DEFAULT_ROOT).expanduser()
        self.snapdir = self.root / "snapshots"
        self.codedir = self.root / "impl_code"
        self.basedir = self.root / "base"
        self.backupdir = self.root / "backups"

    # -- layout --------------------------------------------------------------
    def ensure(self) -> "GlobalStore":
        for d in (self.root, self.snapdir, self.codedir, self.basedir, self.backupdir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    @contextlib.contextmanager
    def lock(self, timeout: float = 1800.0):
        self.ensure()
        f = _lock_file(self.root / ".lock", timeout)
        try:
            yield
        finally:
            f.close()

    def current_id(self) -> Optional[str]:
        p = self.root / "current"
        if not p.exists():
            return None
        v = p.read_text().strip()
        return v or None

    def path(self, vid: str) -> Path:
        return self.snapdir / f"{vid}.db"

    def versions(self) -> list[str]:
        if not self.snapdir.exists():
            return []
        return sorted(p.stem for p in self.snapdir.glob("v[0-9]*.db"))

    def manifest(self, vid: str) -> dict:
        p = self.snapdir / f"{vid}.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _next_id(self) -> str:
        vs = [int(v[1:]) for v in self.versions()]
        return f"v{(max(vs) if vs else 0) + 1:04d}"

    def _set_current(self, vid: str) -> None:
        tmp = self.root / f".current.{os.getpid()}"
        tmp.write_text(vid + "\n")
        os.replace(tmp, self.root / "current")

    def _backup(self, why: str) -> None:
        """Before every change: the pointer and the ledger as they are (snapshots never change)."""
        cur = self.current_id()
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{uuid.uuid4().hex[:4]}"
        d = self.backupdir / stamp
        d.mkdir(parents=True, exist_ok=True)
        (d / "current").write_text((cur or "") + "\n")
        info = {"why": why, "current": cur, "ts": _now()}
        if cur and self.path(cur).exists():
            c = _connect(self.path(cur), ro=True)
            try:
                info["ledger"] = [dict(zip(("run_uid", "seq", "fingerprint", "problem", "attempt", "version"), r))
                                  for r in c.execute("SELECT run_uid, seq, fingerprint, problem, attempt, version "
                                                     "FROM gk_ledger ORDER BY version, run_uid")]
                info["sha256"] = _file_sha(self.path(cur))
            finally:
                c.close()
        (d / "backup.json").write_text(json.dumps(info, indent=1))
        olds = sorted(p for p in self.backupdir.iterdir() if p.is_dir())
        for p in olds[:-200]:                              # bounded
            shutil.rmtree(p, ignore_errors=True)

    def _commit(self, tmp: Path, *, parent: Optional[str], label: str, extra: dict) -> str:
        """Seal a built snapshot: checks, read-only, manifest, pointer (caller holds the lock)."""
        c = _connect(tmp)
        try:
            _check(c, "new snapshot")
            c.execute("PRAGMA journal_mode=DELETE")
            c.execute("VACUUM")
            counts = _counts(c)
        finally:
            c.close()
        vid = self._next_id()
        dst = self.path(vid)
        os.replace(tmp, dst)
        os.chmod(dst, 0o444)
        man = {"format": FORMAT, "id": vid, "parent": parent, "label": label, "created": _now(),
               "sha256": _file_sha(dst), "bytes": dst.stat().st_size, "counts": counts, **extra}
        (self.snapdir / f"{vid}.json").write_text(json.dumps(man, indent=1, default=str))
        self._set_current(vid)
        return vid

    def _scratch(self) -> Path:
        d = self.root / "tmp"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"build-{os.getpid()}-{uuid.uuid4().hex[:8]}.db"

    def _start_from(self, vid: Optional[str]) -> Path:
        tmp = self._scratch()
        if vid:
            shutil.copyfile(self.path(vid), tmp)
            os.chmod(tmp, 0o644)
        c = _connect(tmp)
        try:
            c.executescript(GK_SCHEMA)
            if vid is None:
                _set_meta(c, "format", FORMAT, "gk_meta")
                _set_meta(c, "created", _now(), "gk_meta")
                _set_meta(c, "merge_seq", 0, "gk_meta")
        finally:
            c.close()
        return tmp

    # -- operations ----------------------------------------------------------------
    def rollback(self, vid: str, *, reason: str = "") -> dict:
        """Point ``current`` at an older snapshot (nothing is deleted; the next merge builds on it)."""
        with self.lock():
            if not self.path(vid).exists():
                raise MergeError(f"no snapshot {vid}")
            m = self.manifest(vid)
            if m.get("sha256") and _file_sha(self.path(vid)) != m["sha256"]:
                raise MergeError(f"snapshot {vid} does not match its manifest sha256")
            self._backup(f"rollback to {vid}: {reason}")
            old = self.current_id()
            self._set_current(vid)
            with open(self.root / "history.jsonl", "a") as f:
                f.write(_j({"ts": _now(), "op": "rollback", "from": old, "to": vid, "reason": reason}) + "\n")
            return {"from": old, "to": vid}

    def adopt(self, db_path, *, label: str, extra: Optional[dict] = None, code_dir=None) -> str:
        """Add a snapshot built elsewhere (e.g. a bootstrap import into a scratch store) as the next version."""
        with self.lock():
            self._backup(f"adopt {label}")
            tmp = self._scratch()
            shutil.copyfile(db_path, tmp)
            os.chmod(tmp, 0o644)
            if code_dir and Path(code_dir).exists():
                self.codedir.mkdir(parents=True, exist_ok=True)
                c = _connect(tmp)
                try:
                    from .pool_store import code_refs
                    ref = {r[0] for r in c.execute("SELECT json_extract(data, '$.code_sha') FROM gk_impl")}
                    ref |= code_refs(c)
                finally:
                    c.close()
                for sha in ref:
                    f = Path(code_dir) / f"{sha}.cu"
                    if f.exists() and not (self.codedir / f.name).exists():
                        shutil.copyfile(f, self.codedir / f.name)
            vid = self._commit(tmp, parent=self.current_id(), label=label, extra=extra or {})
            with open(self.root / "history.jsonl", "a") as f:
                f.write(_j({"ts": _now(), "op": "adopt", "version": vid, "label": label}) + "\n")
            return vid

    def gc_code(self) -> int:
        """Delete impl sources no snapshot references (caller holds the lock)."""
        ref = set()
        for v in self.versions():
            c = _connect(self.path(v), ro=True)
            try:
                from .pool_store import code_refs
                ref |= {r[0] for r in c.execute("SELECT json_extract(data, '$.code_sha') FROM gk_impl")}
                ref |= code_refs(c)                      # pool kernels (A5) are referenced too
            finally:
                c.close()
        n = 0
        for f in self.codedir.glob("*.cu"):
            if f.stem not in ref:
                f.unlink()
                n += 1
        return n

    def prune(self, keep: int) -> list[str]:
        """Delete all but the newest ``keep`` snapshots (the current one and v0001 are always kept) and the
        impl sources none of the rest references."""
        with self.lock():
            vs = self.versions()
            cur = self.current_id()
            protect = set(vs[-max(1, keep):]) | {cur, "v0001"}
            gone = []
            for v in vs:
                if v not in protect:
                    for p in (self.path(v), self.snapdir / f"{v}.json"):
                        if p.exists():
                            os.chmod(p, 0o644)
                            p.unlink()
                    gone.append(v)
            self.gc_code()
            return gone

    def verify(self) -> dict:
        out = {}
        for v in self.versions():
            m = self.manifest(v)
            ok = bool(m) and _file_sha(self.path(v)) == m.get("sha256")
            out[v] = "ok" if ok else "MISMATCH"
        return out

    def status(self) -> dict:
        cur = self.current_id()
        st = {"root": str(self.root), "current": cur, "versions": self.versions()}
        if cur and self.path(cur).exists():
            c = _connect(self.path(cur), ro=True)
            try:
                st["counts"] = _counts(c)
                st["parts"] = {}
                for part, status, n in c.execute("SELECT part, status, COUNT(*) FROM gk_claims GROUP BY 1, 2"):
                    st["parts"].setdefault(part, {}).setdefault("claims", {})[status] = n
                for tb in ("gk_hypotheses", "gk_states", "gk_techniques"):
                    for part, status, n in c.execute(f"SELECT part, status, COUNT(*) FROM {tb} GROUP BY 1, 2"):
                        st["parts"].setdefault(part, {}).setdefault(tb[3:], {})[status] = n
                for part, n in c.execute("SELECT part, COUNT(*) FROM gk_outcomes GROUP BY 1"):
                    st["parts"].setdefault(part, {})["outcomes"] = n
                for part, n in c.execute("SELECT part, COUNT(*) FROM gk_impl GROUP BY 1"):
                    st["parts"].setdefault(part, {})["impl"] = n
                from .pool_store import has_pool
                if has_pool(c):
                    for part, npk, nk in c.execute("SELECT part, COUNT(DISTINCT pkey), COUNT(*) FROM gk_pool GROUP BY 1"):
                        st["parts"].setdefault(part, {})["pool"] = {"pools": npk, "kernels": nk}
                if "gk_struct" in _tables(c):
                    for part, n in c.execute("SELECT part, COUNT(*) FROM gk_struct GROUP BY 1"):
                        st["parts"].setdefault(part, {})["struct_outcomes"] = n
                st["ledger"] = c.execute("SELECT COUNT(DISTINCT run_uid), COUNT(*) FROM gk_ledger").fetchone()
                st["merge_seq"] = _meta(c, "merge_seq", 0, "gk_meta")
            finally:
                c.close()
            st["manifest"] = {k: v for k, v in self.manifest(cur).items() if k != "counts"}
            st["bytes"] = self.path(cur).stat().st_size
        return st

    # -- base (repo seed) ---------------------------------------------------------------
    def base_db(self) -> Path:
        """A fresh repo-seeded knowledge.db (targets, states, techniques, mined.json), cached per seed version."""
        from .knowledge import SEED_VERSION
        from .targets import DATA_DIR
        h = hashlib.sha1(str(SEED_VERSION).encode())
        for f in sorted(DATA_DIR.rglob("*")):
            if f.is_file():
                h.update(f.name.encode())
                h.update(f.read_bytes())
        key = h.hexdigest()[:16]
        self.ensure()
        p = self.basedir / f"{key}.db"
        if not p.exists():
            p2 = build_base(self.basedir / f".build-{os.getpid()}-{uuid.uuid4().hex[:6]}")
            os.replace(p2, p)
        return p


def build_base(workdir: Path) -> Path:
    """Seed a knowledge.db from the repo data only (what every run starts from without a global store)."""
    from .knowledge import KnowledgeV2
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    kv = KnowledgeV2(workdir)
    kv.db.conn.close()
    del kv
    c = sqlite3.connect(str(workdir / "knowledge.db"), isolation_level=None)
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    c.execute("PRAGMA journal_mode=DELETE")
    c.close()
    out = workdir.with_suffix(".db")
    shutil.move(str(workdir / "knowledge.db"), out)
    shutil.rmtree(workdir, ignore_errors=True)
    return out


def _counts(c: sqlite3.Connection) -> dict:
    t = _tables(c)
    return {tb: c.execute(f"SELECT COUNT(*) FROM {tb}").fetchone()[0] for tb in GK_TABLES if tb in t}


# ---------------------------------------------------------------- provenance

def _prov_add(prov: Optional[dict], p: dict, seq: int) -> dict:
    prov = dict(prov or {})
    if "first" not in prov:
        prov["first"] = p
    prov["last"] = p
    runs = prov.get("runs", [])
    if p.get("run_uid") not in runs:
        runs = (runs + [p.get("run_uid")])[-20:]
        prov["n_runs"] = int(prov.get("n_runs", 0)) + 1
    prov["runs"] = runs
    probs = prov.get("problems", [])
    if p.get("problem") and p["problem"] not in probs:
        probs = (probs + [p["problem"]])[-20:]
    prov["problems"] = probs
    prov["last_merge"] = seq
    prov.setdefault("first_merge", seq)
    return prov


def _rank(d: dict, prov: dict, seq: int, pol: Policy) -> float:
    age = max(0, seq - int((prov or {}).get("last_evidence", (prov or {}).get("last_merge", seq))))
    return (int(d.get("support", 0)) - int(d.get("contradict", 0))) * 0.5 ** (age / max(pol.half_life_merges, 1e-9))


# ---------------------------------------------------------------- materialize (read path)

def materialize(store: GlobalStore, part: str, out_dir, *, problem: Optional[str] = None,
                snapshot: Optional[str] = None, policy: Optional[Policy] = None,
                run_uid: Optional[str] = None) -> dict:
    """Write ``out_dir/knowledge.db`` (+ ``impl/``, ``seed.json``): repo seed + the snapshot's rows of
    ``part`` (capped per :class:`Policy`). Returns the manifest (also in ``seed.json`` and the
    db's ``meta.gk_seed``)."""
    pol = policy or Policy()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "knowledge.db"
    if dst.exists():
        raise MergeError(f"{dst} exists; a seed is only written into an empty directory")
    vid = snapshot or store.current_id()
    tmp = out / ".knowledge.db.tmp"
    shutil.copyfile(store.base_db(), tmp)
    k = _connect(tmp)
    k.execute("PRAGMA journal_mode=DELETE")
    g = _connect(store.path(vid), ro=True) if vid else None
    counts: dict = {}
    try:
        if g is not None:
            counts = _fill_seed(k, g, part, problem, pol)
        n_out = k.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
        cm = _meta(k, "claims_meta", {}) or {}
        cm.update({"epoch": 0, "reflect_cursor": n_out, "reflections": 0})
        k.execute("BEGIN")
        _set_meta(k, "claims_meta", cm)
        _set_meta(k, "upkeep_cursor", n_out)
        man = {"format": FORMAT, "snapshot": vid, "part": part, "problem": problem,
               "run_uid": run_uid or uuid.uuid4().hex, "created": _now(), "store": str(store.root),
               "policy": asdict(pol), "counts": counts, "seed_outcomes": n_out,
               "snapshot_sha256": store.manifest(vid).get("sha256") if vid else None,
               "merge_seq": int(_meta(g, "merge_seq", 0, "gk_meta") or 0) if g is not None else 0}
        _set_meta(k, "gk_seed", man)
        k.execute("COMMIT")
        _check(k, "seed")
    finally:
        k.close()
        if g is not None:
            g.close()
    os.replace(tmp, dst)
    if g is not None:
        man["impl"] = _seed_impl(store, vid, part, out / "impl")
        if pol.seed_pool or pol.seed_struct:
            g2 = _connect(store.path(vid), ro=True)
            try:
                if pol.seed_pool:
                    from .pool_store import seed_pool
                    man["pool"] = seed_pool(g2, store, part, out / "impl")
                if pol.seed_struct:
                    from .priors import seed_struct
                    man["struct"] = seed_struct(g2, part, out / "struct_priors.json")
            finally:
                g2.close()
    (out / "seed.json").write_text(json.dumps(man, indent=1))
    return man


def _fill_seed(k: sqlite3.Connection, g: sqlite3.Connection, part: str, problem: Optional[str], pol: Policy) -> dict:
    from ..claims import ACTIVE
    seq = int(_meta(g, "merge_seq", 0, "gk_meta") or 0)
    parts_other = pol.share in ("lifted", "hierarchy")
    counts: dict = {}
    k.execute("BEGIN")
    # scope tree additions
    for nid, parent, data in g.execute("SELECT id, parent, data FROM gk_scopes"):
        k.execute("INSERT OR IGNORE INTO scopes VALUES (?, ?, ?)", (nid, parent, data))
    for key, node, data in g.execute("SELECT key, node, data FROM gk_targets"):
        k.execute("INSERT OR IGNORE INTO targets VALUES (?, ?, ?)", (key, node, data))
    # techniques (promoted hypotheses)
    tq = "SELECT part, id, data FROM gk_techniques WHERE part=?"
    rows = list(g.execute(tq, (part,)))
    if parts_other:
        for p2, tid, data in g.execute("SELECT part, id, data FROM gk_techniques WHERE part<>?", (part,)):
            d = json.loads(data)
            if d.get("level") in ("node", "family", "universal") and d.get("status") == "active":
                rows.append((p2, tid, data))
    have_t = {r[0] for r in k.execute("SELECT id FROM techniques")}
    nt = 0
    for _, tid, data in rows:
        if tid in have_t:
            continue
        d = json.loads(data)
        k.execute("INSERT OR REPLACE INTO techniques (id, source, data, updated) VALUES (?, ?, ?, ?)",
                  (tid, d.get("source"), _j(d), _now()))
        for r in d.get("realizations", []):
            k.execute("INSERT OR IGNORE INTO realizations VALUES (?, ?, ?)", (tid, r.get("scope", "universal"), _j(r)))
        have_t.add(tid)
        nt += 1
    counts["techniques"] = nt
    # states: core always, probationary newest first (capped), parents first
    have_s = {r[0] for r in k.execute("SELECT id FROM states")}
    srows = [(json.loads(d), json.loads(p or "{}")) for d, p in
             g.execute("SELECT data, prov FROM gk_states WHERE part=? AND status IN ('core','probationary')", (part,))]
    core = [d for d, _ in srows if d.get("status") == "core"]
    prob = sorted(((d, p) for d, p in srows if d.get("status") == "probationary"),
                  key=lambda x: -int(x[1].get("last_merge", 0)))[: pol.seed_max_prob_states]
    pending = core + [d for d, _ in prob]
    ns_ = 0
    while pending:
        rest = []
        for d in pending:
            if d["id"] in have_s:
                continue
            if d.get("parent") is not None and d["parent"] not in have_s:
                rest.append(d)
                continue
            ev = dict(d.get("evidence") or {})
            ev["admitted_seq"] = 0                      # rebased below once the outcome count is known
            d = dict(d, evidence=ev)
            k.execute("INSERT OR REPLACE INTO states (id, status, data, updated) VALUES (?, ?, ?, ?)",
                      (d["id"], d.get("status"), _j(d), _now()))
            have_s.add(d["id"])
            ns_ += 1
        if len(rest) == len(pending):
            break
        pending = rest
    counts["states"] = ns_
    # statistics: additive on top of the repo seed's
    q = "SELECT item, level, scope, op, state, n, s, ss FROM gk_stats WHERE part=?"
    srcs = [list(g.execute(q, (part,)))]
    if pol.share == "hierarchy":
        srcs.append([r for r in g.execute("SELECT item, level, scope, op, state, n, s, ss FROM gk_stats "
                                          "WHERE part<>? AND level IN ('node','family','universal')", (part,))])
    nst = 0
    for rows_ in srcs:
        for item, level, scope, op, state, n, s, ss in rows_:
            k.execute("INSERT INTO stats VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (item, level, scope, op, state) "
                      "DO UPDATE SET n=n+excluded.n, s=s+excluded.s, ss=ss+excluded.ss",
                      (item, level, scope, op, state, n, s, ss))
            nst += 1
    counts["stats_rows"] = nst
    # claims: overlays replace the repo claims they were made from; then learned ones, ranked + capped
    base = {}
    for cid, data in k.execute("SELECT id, data FROM claims"):
        base[_claim_ck(json.loads(data))] = cid
    crow = list(g.execute("SELECT part, id, status, base_ck, data, prov FROM gk_claims WHERE part=?", (part,)))
    if parts_other:
        for r in g.execute("SELECT part, id, status, base_ck, data, prov FROM gk_claims WHERE part<>? AND "
                           "status='active'", (part,)):
            d = json.loads(r[4])
            if d.get("level") in ("node", "family", "universal") and not d.get("problem_id") \
                    and "lift" in json.dumps(json.loads(r[5] or "{}").get("events", [])):
                crow.append(r)
    picked = []
    for p2, cid, status, bck, data, prov in crow:
        d = json.loads(data)
        pv = json.loads(prov or "{}")
        if d.get("problem_id"):                          # problem-local: only for the same problem
            if not problem or not str(d["problem_id"]).startswith(problem + NS):
                continue
            d["problem_id"] = strip_ns(problem, d["problem_id"])
        if bck and bck in base:                          # a repo claim with merged evidence / status
            k.execute("DELETE FROM claims WHERE id=?", (base[bck],))
            if status != ACTIVE:
                picked.append((float("-inf"), cid, d, pv, True))
                continue
        elif status != ACTIVE:
            continue
        picked.append((_rank(d, pv, seq, pol), cid, d, pv, False))
    active = _select_claims([x for x in picked if not x[4]], seq, pol)
    kept = active + [x for x in picked if x[4]]
    cited: list = []
    for _, cid, d, pv, _r in kept:
        age = max(0, seq - int(pv.get("last_evidence", pv.get("last_merge", seq))))
        d["last_epoch"] = d["created_epoch"] = -int(min(age * pol.epochs_per_merge, 600))
        d["id"] = cid
        k.execute("INSERT OR REPLACE INTO claims VALUES (?, ?, ?, ?)", (cid, d.get("status"), _j(d), d.get("updated_ts")))
        cited += list(d.get("support_ids") or []) + list(d.get("contradict_ids") or [])
    counts["claims"] = len(active)
    counts["claims_retired_overlays"] = len(kept) - len(active)
    # hypotheses (slot kind): open + closed, capped
    hrows = [(json.loads(d), json.loads(p or "{}")) for d, p in
             g.execute("SELECT data, prov FROM gk_hypotheses WHERE part=?", (part,))]
    opn = sorted([x for x in hrows if x[0].get("status") in ("proposed", "testing")],
                 key=lambda x: -int(x[1].get("last_merge", 0)))[: pol.seed_max_open_hyp]
    cls = sorted([x for x in hrows if x[0].get("status") not in ("proposed", "testing")],
                 key=lambda x: -int(x[1].get("last_merge", 0)))[: pol.seed_max_closed_hyp]
    for d, _ in opn + cls:
        d = dict(d)
        d["attempts"] = [dict(a, problem_id=strip_ns(problem, a.get("problem_id"))) for a in d.get("attempts", [])]
        k.execute("INSERT OR REPLACE INTO hypotheses (id, status, data, updated) VALUES (?, ?, ?, ?)",
                  (d["id"], d.get("status"), _j(d), _now()))
    counts["hypotheses_open"], counts["hypotheses_closed"] = len(opn), len(cls)
    # outcomes: this problem's (its "already tried") + the shipped claims' evidence
    want: list = []
    if problem:
        want = [r[0] for r in g.execute("SELECT id FROM gk_outcomes WHERE part=? AND problem=? ORDER BY ts DESC, id "
                                        "LIMIT ?", (part, problem, pol.seed_max_problem_outcomes))]
    seen = set(want)
    extra = [i for i in dict.fromkeys(cited) if i not in seen][: pol.seed_max_cited_outcomes]
    ids = want + extra
    rows_o = []
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        q = ",".join("?" * len(chunk))
        rows_o += list(g.execute(f"SELECT id, ts, rec, v2 FROM gk_outcomes WHERE id IN ({q})", chunk))
    rows_o.sort(key=lambda r: (r[1] or "", r[0]))
    for oid, ts, rec, v2 in rows_o:
        r = json.loads(rec)
        r["problem_id"] = strip_ns(problem, r.get("problem_id"))
        v = json.loads(v2) if v2 else {}
        k.execute("INSERT OR IGNORE INTO outcomes (id, ts, target_key, problem_id, slot, technique, item_id,"
                  " hypothesis_id, op_class, status, reward, state, mechanism, rec, v2)"
                  " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (oid, r.get("ts"), f"{r.get('target')}:{r.get('arch')}" if r.get("arch") else r.get("target"),
                   r.get("problem_id"), r.get("slot"), r.get("technique"), v.get("item_id") or r.get("technique"),
                   v.get("hypothesis_id"), r.get("op_class"), r.get("status"), v.get("reward"), r.get("state_before"),
                   (v.get("mechanism") or {}).get("label"), _j(r), _j(v)))
    n_out = k.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
    counts["outcomes"] = len(rows_o)
    # rebase sequence-relative fields: only this run's outcomes count as "later evidence"
    for sid, data in list(k.execute("SELECT id, data FROM states WHERE status='probationary'")):
        d = json.loads(data)
        d.setdefault("evidence", {})["admitted_seq"] = n_out
        k.execute("UPDATE states SET data=? WHERE id=?", (_j(d), sid))
    ov = {}
    for tech, data in g.execute("SELECT tech, data FROM gk_overrides WHERE part=?", (part,)):
        ov[tech] = [dict(o, since_seq=n_out) for o in json.loads(data)]
    if ov:
        _set_meta(k, "prediction_overrides", ov)
    counts["prediction_overrides"] = sum(len(v) for v in ov.values())
    trig = set(_meta(k, "hyp_triggers_seen", []) or [])
    trig |= {r[0] for r in g.execute("SELECT key FROM gk_triggers WHERE part=?", (part,))}
    if trig:
        _set_meta(k, "hyp_triggers_seen", sorted(trig))
    counts["triggers_seen"] = len(trig)
    # rejected state proposals (the discovery prompt shows the last ones per trigger key)
    per: dict = {}
    for pid, key, status, data in g.execute("SELECT id, key, status, data FROM gk_proposals WHERE part=? "
                                            "ORDER BY id", (part,)):
        if status == "rejected":
            per.setdefault(key, []).append((pid, data))
    npr = 0
    for key, lst in per.items():
        for pid, data in lst[-pol.seed_rejected_per_key:]:
            k.execute("INSERT OR IGNORE INTO proposals (id, status, data, updated) VALUES (?, ?, ?, ?)",
                      (pid, "rejected", data, _now()))
            npr += 1
    counts["proposals"] = npr
    k.execute("COMMIT")
    return counts


def _select_claims(cands: list, seq: int, pol: Policy) -> list:
    """The active claims a seed ships, at most ``seed_max_claims``: (1) claims with evidence from the last
    ``seed_fresh_merges`` merges, newest first (up to ``seed_fresh_max``) -- otherwise what the previous runs just learned
    (support 2-3) never beats the old, well-supported claims; (2) the rest round-robin over op classes, eligible
    claims first, best global rank first within each class -- so every kind of kernel keeps its lessons.
    ``cands``: [(rank, id, data, prov, is_retired_overlay)]."""
    from ..claims import Claim

    def eligible(d):
        try:
            return Claim.from_dict(d).eligible()
        except (TypeError, ValueError):
            return False

    def key(x):
        return (not eligible(x[2]), -x[0], x[1])

    def age(x):
        return seq - int(x[3].get("last_evidence", x[3].get("last_merge", -10**9)))

    fresh = sorted((x for x in cands if age(x) < pol.seed_fresh_merges),
                   key=lambda x: (age(x), *key(x)))[: min(pol.seed_fresh_max, pol.seed_max_claims)]
    out = list(fresh)
    taken = {x[1] for x in out}
    groups: dict = {}
    for x in sorted(cands, key=key):
        if x[1] not in taken:
            groups.setdefault(x[2].get("op_class") or "*", []).append(x)
    lanes = [groups[k] for k in sorted(groups)]
    i = 0
    while len(out) < pol.seed_max_claims and any(lanes):
        lane = lanes[i % len(lanes)]
        if lane:
            out.append(lane.pop(0))
        i += 1
    return out


def _seed_impl(store: GlobalStore, vid: str, part: str, out: Path) -> dict:
    g = _connect(store.path(vid), ro=True)
    try:
        rows = list(g.execute("SELECT data FROM gk_impl WHERE part=? ORDER BY kkey, slot_us", (part,)))
    finally:
        g.close()
    (out / "code").mkdir(parents=True, exist_ok=True)
    n = missing = 0
    with open(out / "index.jsonl", "a") as f:
        for (data,) in rows:
            d = json.loads(data)
            src = store.codedir / f"{d['code_sha']}.cu"
            if not src.exists():
                missing += 1
                continue
            dst = out / "code" / f"{d['code_sha']}.cu"
            if not dst.exists():
                shutil.copyfile(src, dst)
            f.write(json.dumps(d, sort_keys=True) + "\n")
            n += 1
    return {"records": n, "missing_code": missing}


# ---------------------------------------------------------------- merge (write path)

@dataclass
class RunInfo:
    """Provenance of one merged run."""
    run_uid: str
    problem: Optional[str] = None
    attempt: Optional[str] = None
    part: Optional[str] = None
    model: Optional[str] = None
    run_id: Optional[str] = None
    source: str = "run"                      # run | salvaged_db | salvaged_text
    date: str = field(default_factory=_now)

    def prov(self) -> dict:
        tk, dt = parse_part(self.part or "unknown|fp32")
        return {"run_uid": self.run_uid, "problem": self.problem, "attempt": self.attempt, "arch": tk, "dtype": dt,
                "model": self.model, "run_id": self.run_id, "source": self.source, "date": self.date}


def _seed_info(c: sqlite3.Connection) -> dict:
    return _meta(c, "gk_seed", {}) or {}


def merge_run(store: GlobalStore, run_db, base_db=None, *, info: RunInfo, impl_dir=None, base_impl_dir=None,
              policy: Optional[Policy] = None, review_path=None, label: str = "", struct_path=None) -> dict:
    """Merge ``run_db`` - ``base_db`` (the db the run started from, or the copy merged last; default:
    the repo seed) into a new snapshot. Returns a report; ``status`` is merged | duplicate | empty.
    ``struct_path`` (a ModelBlaster ``state.json``) is read only with ``Policy.struct_outcomes``."""
    pol = policy or Policy()
    run_db = Path(run_db)
    with store.lock():
        cur = store.current_id()
        with _db_copy(run_db) as rc:
            _check(rc, f"run db {run_db}")
            R = KDB(rc)
            rseed = _seed_info(rc)
            part = info.part or rseed.get("part")
            if not part:
                raise MergeError("no partition: pass part= (the run db has no gk_seed meta)")
            info.part = part
            fp = fingerprint(rc)
            if base_db is None:
                base_db = store.base_db()
            with _db_copy(Path(base_db)) as bc:
                _check(bc, f"base db {base_db}")
                B = KDB(bc)
                bfp = fingerprint(bc)
                # ledger checks against the current snapshot
                led = []
                if cur:
                    g0 = _connect(store.path(cur), ro=True)
                    try:
                        led = list(g0.execute("SELECT seq, fingerprint FROM gk_ledger WHERE run_uid=? ORDER BY seq",
                                              (info.run_uid,)))
                    finally:
                        g0.close()
                if any(f == fp for _, f in led):
                    return {"status": "duplicate", "run_uid": info.run_uid, "fingerprint": fp, "version": cur}
                if led and led[-1][1] != bfp:
                    raise MergeError(f"run {info.run_uid} was merged before (fingerprint {led[-1][1]}); merge it "
                                     f"against the copy merged last, not {base_db} ({bfp})")
                # rows the run imported mid-run (refresh.py) after this base was taken: into the base copy, so
                # the delta below is the run's own learning only
                from .refresh import adjust_base, imported_impl_ids
                adj = adjust_base(bc, rc)
                B = KDB(bc)
                if not B.outcome_ids() <= R.outcome_ids():
                    raise MergeError("the run db is not a continuation of the base db (outcomes missing)")
                store._backup(f"merge {info.problem} {info.run_uid}")
                tmp = store._start_from(cur)
                g = _connect(tmp)
                try:
                    g.execute("BEGIN IMMEDIATE")
                    seq = int(_meta(g, "merge_seq", 0, "gk_meta") or 0) + 1
                    _set_meta(g, "merge_seq", seq, "gk_meta")
                    rep = _apply_delta(g, R, B, info=info, part=part, seq=seq, pol=pol, review_path=review_path)
                    rep["impl"] = _merge_impl(g, store, impl_dir, base_impl_dir, info=info, part=part, seq=seq, pol=pol,
                                              exclude=imported_impl_ids(rc))
                    if pol.impl_pool and impl_dir:
                        from .pool_store import merge_pool
                        rep["pool"] = merge_pool(g, store, [impl_dir], base_impl_dir, part=part, pol=pol,
                                                 prov=info.prov(), exclude=imported_impl_ids(rc), problem=info.problem)
                    if pol.struct_outcomes and struct_path and Path(struct_path).exists():
                        from .priors import merge_struct
                        rep["struct"] = merge_struct(g, struct_path, part=part, run_uid=info.run_uid,
                                                     problem=info.problem)
                    rep["upkeep"] = upkeep(g, part, seq, pol)
                    if adj["rids"]:
                        rep["refresh_base_adjust"] = adj
                    vid_next = store._next_id()
                    lin = _meta(g, "lineage", None, "gk_meta")
                    if lin is None:
                        from .refresh import lineage
                        lin = lineage(store, cur)
                    _set_meta(g, "lineage", (list(lin) + [vid_next])[-2000:], "gk_meta")
                    g.execute("INSERT INTO gk_ledger VALUES (?,?,?,?,?,?,?,?,?,?)",
                              (info.run_uid, len(led) + 1, fp, part, info.problem, info.attempt,
                               "seed" if not led else "incremental", vid_next, _now(), _j(rep)))
                    g.execute("INSERT INTO gk_log (ts, kind, ref, part, data) VALUES (?,?,?,?,?)",
                              (_now(), "merge", info.run_uid, part, _j({"info": info.prov(), "report": rep})))
                    g.execute("COMMIT")
                except BaseException:
                    with contextlib.suppress(Exception):
                        g.execute("ROLLBACK")
                    g.close()
                    tmp.unlink(missing_ok=True)
                    raise
                g.close()
        vid = store._commit(tmp, parent=cur, label=label or f"merge {info.problem or info.run_uid}",
                            extra={"merged": info.prov(), "fingerprint": fp, "report": rep})
        with open(store.root / "history.jsonl", "a") as f:
            f.write(_j({"ts": _now(), "op": "merge", "version": vid, "parent": cur, "run": info.prov()}) + "\n")
        rep.update(status="merged", version=vid, parent=cur, fingerprint=fp, part=part)
        return rep


def _review_events(review_path) -> dict:
    """claim id -> set of audit kinds (claim_retired / claim_demoted / claim_lifted) from review.jsonl."""
    out: dict = {}
    if not review_path or not Path(review_path).exists():
        return out
    for line in Path(review_path).read_text(errors="replace").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("kind", "").startswith("claim_"):
            out.setdefault(d.get("ref") or d.get("claim_id"), set()).add(d["kind"])
    return out


def _apply_delta(g, R: KDB, B: KDB, *, info: RunInfo, part: str, seq: int, pol: Policy, review_path=None) -> dict:
    from ..claims import ACTIVE, RETIRED, Claim, MAX_CITED_IDS
    rep: dict = {"skipped": {}}
    prov = info.prov()
    prob = info.problem

    def skip(what, why):
        rep["skipped"].setdefault(what, {})
        rep["skipped"][what][why] = rep["skipped"][what].get(why, 0) + 1

    # -- states first (ids may be renamed / mapped, and everything below refers to them) -----------
    from .predicates import canonical
    smap: dict = {}
    gstates = {sid: json.loads(d) for sid, d in g.execute("SELECT id, data FROM gk_states WHERE part=?", (part,))}
    bystruct = {(d.get("parent"), canonical(d.get("predicate"))): sid for sid, d in gstates.items()}
    rs, bs = R.rows("states"), B.rows("states")
    pending = [sid for sid in rs if sid not in bs or rs[sid].get("status") != bs[sid].get("status")]
    added = changed = 0
    for _ in range(len(pending) + 1):
        rest = []
        for sid in pending:
            d = dict(rs[sid])
            par = d.get("parent")
            if par is not None and par in rs and par not in bs and par not in smap:
                rest.append(sid)
                continue
            d["parent"] = smap.get(par, par)
            if sid in bs:                                    # a status change of a state the run started with
                gid = smap.get(sid, sid)
                if gid in gstates:
                    gd = dict(gstates[gid], status=d.get("status"))
                    g.execute("UPDATE gk_states SET status=?, data=? WHERE part=? AND id=?",
                              (gd["status"], _j(gd), part, gid))
                    changed += 1
                continue
            key = (d["parent"], canonical(d.get("predicate")))
            if key in bystruct:
                smap[sid] = bystruct[key]
                gid = bystruct[key]
                if d.get("status") != gstates[gid].get("status") and d.get("status") in ("core", "retired"):
                    gd = dict(gstates[gid], status=d["status"])
                    g.execute("UPDATE gk_states SET status=?, data=? WHERE part=? AND id=?", (gd["status"], _j(gd), part, gid))
                continue
            gid = sid
            if gid in gstates or gid in bs:
                gid = f"{sid}_{hashlib.sha1((info.run_uid + sid).encode()).hexdigest()[:4]}"[:64]
            smap[sid] = gid
            d["id"] = gid
            g.execute("INSERT OR REPLACE INTO gk_states VALUES (?,?,?,?,?)",
                      (part, gid, d.get("status"), _j(d), _j(_prov_add(None, prov, seq))))
            gstates[gid] = d
            bystruct[key] = gid
            added += 1
        if not rest or len(rest) == len(pending):
            for sid in rest:
                skip("states", "unknown parent")
            break
        pending = rest
    rep["states"] = {"added": added, "status_changed": changed, "mapped": {k: v for k, v in smap.items() if k != v}}

    def mstate(s):
        return smap.get(s, s) if s else s

    # -- hypotheses (slot kind only; graph hypotheses are per-model hints) --------------------------------
    from .hypotheses import similar_mechanism
    hmap: dict = {}
    gh = {hid: (json.loads(d), json.loads(p or "{}")) for hid, d, p in
          g.execute("SELECT id, data, prov FROM gk_hypotheses WHERE part=?", (part,))}
    rh, bh = R.rows("hypotheses"), B.rows("hypotheses")
    hadd = hupd = 0
    order = {"proposed": 0, "testing": 1, "retired": 2, "promoted": 3}
    for hid, d in sorted(rh.items()):
        if d.get("kind", "slot") != "slot":
            continue
        b = bh.get(hid)
        if b is not None and _j(b) == _j(d):
            continue
        battempts = {a.get("outcome_id") for a in (b or {}).get("attempts", [])}
        new_att = [dict(a, problem_id=ns(prob, a.get("problem_id"))) for a in d.get("attempts", [])
                   if a.get("outcome_id") not in battempts]
        gid = hid if hid in gh else None
        if gid is None and b is None:
            for oid, (od, _) in gh.items():
                if od.get("name") == d.get("name") or similar_mechanism(od.get("mechanism", ""), d.get("mechanism", "")):
                    gid = oid
                    break
        if gid is None:
            nd = dict(d, attempts=[dict(a, problem_id=ns(prob, a.get("problem_id"))) for a in d.get("attempts", [])])
            nd["states"] = [mstate(s) for s in nd.get("states", [])]
            gh[hid] = (nd, _prov_add(None, prov, seq))
            g.execute("INSERT OR REPLACE INTO gk_hypotheses VALUES (?,?,?,?,?)",
                      (part, hid, nd.get("status"), _j(nd), _j(gh[hid][1])))
            hmap[hid] = hid
            hadd += 1
            continue
        hmap[hid] = gid
        od, op = gh[gid]
        od = dict(od)
        have = {a.get("outcome_id") for a in od.get("attempts", [])}
        od["attempts"] = od.get("attempts", []) + [a for a in new_att if a.get("outcome_id") not in have]
        rs_, os_ = d.get("status", "proposed"), od.get("status", "proposed")
        if order.get(rs_, 0) > order.get(os_, 0) and (b is None or rs_ != b.get("status")):
            od["status"], od["reason"] = rs_, d.get("reason", od.get("reason", ""))
            if d.get("promoted_to"):
                od["promoted_to"] = d["promoted_to"]
        op = _prov_add(op, prov, seq)
        if new_att:
            op["last_evidence"] = seq
        gh[gid] = (od, op)
        g.execute("INSERT OR REPLACE INTO gk_hypotheses VALUES (?,?,?,?,?)", (part, gid, od.get("status"), _j(od), _j(op)))
        hupd += 1
    rep["hypotheses"] = {"added": hadd, "updated": hupd, "mapped": {k: v for k, v in hmap.items() if k != v}}

    # -- techniques (promoted hypotheses and their later demotion / retirement) ----------------------------
    tmap: dict = {}
    rt, bt = R.techniques(), B.techniques()
    gt = {tid: json.loads(d) for tid, d in g.execute("SELECT id, data FROM gk_techniques WHERE part=?", (part,))}
    tadd = tupd = 0
    for tid, d in sorted(rt.items()):
        src = d.get("source") or ""
        if not src.startswith("hypothesis:"):
            continue
        b = bt.get(tid)
        if b is not None and _j(b) == _j(d):
            continue
        hsrc = src.split(":", 1)[1]
        gh_ = gh.get(hmap.get(hsrc, hsrc))
        if b is None and gh_ and gh_[0].get("promoted_to") in gt and gh_[0].get("promoted_to") != tid:
            tmap[tid] = gh_[0]["promoted_to"]                # the same hypothesis was promoted before
            continue
        d = dict(d, source=f"hypothesis:{hmap.get(hsrc, hsrc)}", states=[mstate(s) for s in d.get("states", [])])
        if b is None and tid in gt and not similar_mechanism(gt[tid].get("mechanism", ""), d.get("mechanism", "")):
            nid = f"{tid}_{hashlib.sha1((info.run_uid + tid).encode()).hexdigest()[:4]}"
            tmap[tid] = nid
            d["id"] = nid
            tid = nid
        if b is None and tid in gt:                       # promoted again elsewhere: keep the first, merged stats
            continue
        g.execute("INSERT OR REPLACE INTO gk_techniques VALUES (?,?,?,?,?)",
                  (part, tid, d.get("status", "active"), _j(d), _j(_prov_add(None, prov, seq))))
        if b is None:
            tadd += 1
        else:
            tupd += 1
    rep["techniques"] = {"added": tadd, "updated": tupd, "renamed": tmap}

    def mitem(i):
        if i in tmap:
            return tmap[i]
        return hmap.get(i, i)

    # -- outcomes (append once, namespaced) ---------------------------------------------------------------
    bo = B.outcome_ids()
    nout = known = 0
    for oid, (rec, v2) in R.outcomes().items():
        if oid in bo:
            continue
        try:
            rec = dict(rec, problem_id=ns(prob, rec.get("problem_id")),
                       state_before=mstate(rec.get("state_before")), state_after=mstate(rec.get("state_after")),
                       technique=mitem(rec.get("technique")))
            v2 = dict(v2)
            if v2.get("chain"):
                v2["chain"] = [mstate(s) for s in v2["chain"]]
            if v2.get("item_id"):
                v2["item_id"] = mitem(v2["item_id"])
            if v2.get("hypothesis_id"):
                v2["hypothesis_id"] = hmap.get(v2["hypothesis_id"], v2["hypothesis_id"])
            v2["gk"] = {"run_uid": info.run_uid, "problem": prob}
            cur = g.execute("INSERT OR IGNORE INTO gk_outcomes VALUES (?,?,?,?,?,?,?)",
                            (oid, part, prob, info.run_uid, rec.get("ts"), _j(rec), _j(v2)))
            nout += cur.rowcount
            known += 1 - cur.rowcount
        except (TypeError, ValueError, AttributeError):
            skip("outcomes", "malformed")
    rep["outcomes"] = nout
    if known:
        rep["outcomes_already_in_store"] = known

    # -- statistics: the delta only ---------------------------------------------------------------------------
    rs_meta = (R.meta("seed_version"), R.meta("mined_seed"))
    bs_meta = (B.meta("seed_version"), B.meta("mined_seed"))
    if rs_meta != bs_meta:
        rep["stats"] = {"skipped": f"repo seed changed during the run {bs_meta} -> {rs_meta}; stats delta unsafe"}
    elif known:
        # outcomes this delta calls new are already in the store: the same run merged before under another
        # run uid (or against the wrong base); its statistics would count twice
        rep["stats"] = {"skipped": f"{known} of the run's new outcomes are already in the store (a re-merge); "
                                   "stats delta not added"}
    else:
        Rs, Bs = R.stats(), B.stats()
        nadd = neg = 0
        for k_, (n, s, ss) in Rs.items():
            bn, bsum, bss = Bs.get(k_, (0.0, 0.0, 0.0))
            dn, ds, dss = n - bn, s - bsum, ss - bss
            if dn <= 1e-9:
                if dn < -1e-9:
                    neg += 1
                continue
            item, level, scope, op, state = k_
            level = "node" if level == "family" else level
            g.execute("INSERT INTO gk_stats VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT (part, item, level, scope, op, state) "
                      "DO UPDATE SET n=n+excluded.n, s=s+excluded.s, ss=ss+excluded.ss",
                      (part, mitem(item), level, scope, op, mstate(state), dn, ds, dss))
            g.execute("INSERT INTO gk_contrib VALUES (?,?,?,?,?,?)",
                      (seq, info.run_uid, part, "stats", _j([mitem(item), level, scope, op, mstate(state)]),
                       _j([dn, ds, dss])))
            nadd += 1
        rep["stats"] = {"rows": nadd, "negative_deltas_skipped": neg}

    # -- claims --------------------------------------------------------------------------------------------------
    evs = _review_events(review_path)
    rc_, bc_ = R.rows("claims"), B.rows("claims")
    gclaims = {cid: [json.loads(d), json.loads(p or "{}"), st, bck] for cid, st, bck, d, p in
               g.execute("SELECT id, status, base_ck, data, prov FROM gk_claims WHERE part=?", (part,))}
    # base claims of this seed that are not (yet) rows of the store: matched by content
    absorbed_into: dict = {}
    for cid, d in rc_.items():
        if cid in bc_ and d.get("status") == RETIRED and bc_[cid].get("status") == ACTIVE:
            m = (d.get("data") or {}).get("merged_into")
            if m:
                absorbed_into.setdefault(m, []).append(cid)
    cadd = cupd = cfold = cret = 0

    def localize(d):
        d = dict(d)
        d["problem_id"] = ns(prob, d.get("problem_id"))
        d["state"] = mstate(d.get("state"))
        if d.get("technique"):
            d["technique"] = mitem(d["technique"])
        if d.get("hypothesis_id"):
            d["hypothesis_id"] = hmap.get(d["hypothesis_id"], d["hypothesis_id"])
        return d

    def fold(gd, sup_ids, con_ids, dsup, dcon):
        """Add evidence; ids the row already cites are not counted again."""
        have = set(gd.get("support_ids", [])) | set(gd.get("contradict_ids", []))
        ns_ = [i for i in dict.fromkeys(sup_ids) if i not in have]
        nc_ = [i for i in dict.fromkeys(con_ids) if i not in have and i not in ns_]
        dsup -= sum(1 for i in dict.fromkeys(sup_ids) if i in have)
        dcon -= sum(1 for i in dict.fromkeys(con_ids) if i in have)
        gd["support_ids"] = (list(gd.get("support_ids", [])) + ns_)[-MAX_CITED_IDS:]
        gd["contradict_ids"] = (list(gd.get("contradict_ids", [])) + nc_)[-MAX_CITED_IDS:]
        gd["support"] = int(gd.get("support", 0)) + max(0, dsup)
        gd["contradict"] = int(gd.get("contradict", 0)) + max(0, dcon)
        gd["updated_ts"] = _now()
        return gd

    def put(cid, gd, pv, bck=None):
        g.execute("INSERT OR REPLACE INTO gk_claims VALUES (?,?,?,?,?,?)",
                  (part, cid, gd.get("status"), bck, _j(gd), _j(pv)))
        gclaims[cid] = [gd, pv, gd.get("status"), bck]

    # claims of the base matched by content (a run that built its own repo seed has other ids for them)
    base_by_ck: dict = {}
    for bid, bd in bc_.items():
        base_by_ck.setdefault(_claim_ck(bd), bid)
    overlay_by_ck = {v[3]: k for k, v in gclaims.items() if v[3]}
    same = ("support", "contradict", "status", "text", "level", "family", "arch", "state", "op_class", "technique")
    for cid, d in sorted(rc_.items()):
        try:
            Claim.from_dict(d)
        except (TypeError, ValueError):
            skip("claims", "malformed")
            continue
        b, bid = bc_.get(cid), cid
        if b is None:
            bid = base_by_ck.get(_claim_ck(d))
            b = bc_.get(bid) if bid else None
        if b is not None and all(b.get(f) == d.get(f) for f in same) and \
                sorted(b.get("support_ids", [])) == sorted(d.get("support_ids", [])):
            continue
        d = localize(d)
        if b is not None:                                          # evidence on a claim the run started with
            bck = _claim_ck(b)
            b = localize(b)
            gid = cid if cid in gclaims else bid if bid in gclaims else overlay_by_ck.get(bck)
            if gid is None:                                        # a repo-seed claim: start an overlay row
                gid = bid
                put(gid, dict(b, id=gid), _prov_add(None, prov, seq), bck)
                overlay_by_ck[bck] = gid
            # a claim consolidated into another one (consolidate.py) passes its new evidence to the survivor
            hops = 0
            while hops < 8 and ((gclaims[gid][0].get("data") or {}).get("gk_consolidated_into") in gclaims) \
                    and gclaims[gid][2] != ACTIVE:
                gid = gclaims[gid][0]["data"]["gk_consolidated_into"]
                hops += 1
            gd, pv, _, bck = gclaims[gid]
            gd = dict(gd)
            bsup = set(b.get("support_ids", []))
            bcon = set(b.get("contradict_ids", []))
            dsup = int(d.get("support", 0)) - int(b.get("support", 0))
            dcon = int(d.get("contradict", 0)) - int(b.get("contradict", 0))
            gd = fold(gd, [i for i in d.get("support_ids", []) if i not in bsup],
                      [i for i in d.get("contradict_ids", []) if i not in bcon], dsup, dcon)
            for f in ("text", "level", "family", "target", "arch", "state", "op_class", "technique"):
                if hops:                                           # evidence only: the survivor keeps its own text
                    break
                if d.get(f) != b.get(f):                           # lifted / demoted / revised in the run
                    gd[f] = d.get(f)
            pv = _prov_add(pv, prov, seq)
            if dsup > 0 or dcon > 0:
                pv["last_evidence"] = seq
            if d.get("status") == RETIRED and b.get("status") == ACTIVE and not hops:
                kinds = evs.get(cid, set())
                why = None
                if (d.get("data") or {}).get("merged_into"):
                    why = f"merged into {(d.get('data') or {})['merged_into']} in a run"
                elif "claim_retired" in kinds:
                    why = "retired by revalidation in a run"
                elif dcon > 0:
                    why = "retired in a run after new contradictions"
                if why:                                            # age-out / cap in one run is not evidence
                    gd["status"] = RETIRED
                    gd.setdefault("data", {})
                    gd["data"] = dict(gd["data"], gk_retired=why)
                    if (d.get("data") or {}).get("merged_into"):
                        gd["data"]["merged_into"] = d["data"]["merged_into"]
                    cret += 1
            if "claim_lifted" in evs.get(cid, set()) or "claim_demoted" in evs.get(cid, set()):
                pv["events"] = (pv.get("events", []) + sorted(evs[cid]))[-10:]
            put(gid, gd, pv, bck)
            cupd += 1
            continue
        # a claim the run created: fold into a near-duplicate of the same partition, or add
        if d.get("status") != ACTIVE and not d.get("support"):
            continue
        absorbed = sum(int(bc_[a].get("support", 0)) for a in absorbed_into.get(cid, []) if a in bc_)
        dsup = max(0, int(d.get("support", 0)) - absorbed)
        sup_ids = list(d.get("support_ids", []))
        try:
            cl = Claim.from_dict(d)
        except (TypeError, ValueError):
            skip("claims", "malformed")
            continue
        dup = None
        from ..claims import similar
        key, norm = cl.dedupe_key(), cl.norm
        for gid, (gd, pv, st, _) in gclaims.items():
            if st != ACTIVE:
                continue
            try:
                oc = Claim.from_dict(gd)
            except (TypeError, ValueError):
                continue
            if oc.dedupe_key() == key and similar(norm, oc.norm) and (dup is None or oc.support > dup[1]):
                dup = (gid, oc.support)
        if dup is not None and d.get("status") == ACTIVE:
            gd, pv, _, bck = gclaims[dup[0]]
            gd = fold(dict(gd), sup_ids, d.get("contradict_ids", []), dsup, int(d.get("contradict", 0)))
            pv = _prov_add(pv, prov, seq)
            pv["last_evidence"] = seq
            put(dup[0], gd, pv, bck)
            cfold += 1
            continue
        gid = cid if cid not in gclaims else f"c_{uuid.uuid4().hex[:10]}"
        d = dict(d, id=gid, support=dsup if absorbed else int(d.get("support", 0)))
        pv = _prov_add(None, prov, seq)
        pv["last_evidence"] = seq
        put(gid, d, pv)
        cadd += 1
    rep["claims"] = {"added": cadd, "evidence_updates": cupd, "folded_into_existing": cfold, "retired": cret}

    # -- prediction overrides ----------------------------------------------------------------------------------
    ro, bo_ = R.meta("prediction_overrides", {}) or {}, B.meta("prediction_overrides", {}) or {}
    oadd = orem = ocon = 0
    for tech in sorted(set(ro) | set(bo_)):
        rk = {(o.get("state"), o.get("metric")): o for o in ro.get(tech, [])}
        bk = {(o.get("state"), o.get("metric")): o for o in bo_.get(tech, [])}
        gtech = mitem(tech)
        row = g.execute("SELECT data FROM gk_overrides WHERE part=? AND tech=?", (part, gtech)).fetchone()
        cur = {(o.get("state"), o.get("metric")): o for o in (json.loads(row[0]) if row else [])}
        changed_ = False
        for kk, o in rk.items():
            if kk in bk and bk[kk].get("dir") == o.get("dir"):
                continue
            o = dict(o, state=mstate(o.get("state")), gk={"run_uid": info.run_uid, "merge": seq})
            o.pop("since_seq", None)
            k2 = (o["state"], o.get("metric"))
            if k2 in cur and cur[k2].get("dir") != o.get("dir"):
                ocon += 1
                if int(o.get("n", 0)) <= int(cur[k2].get("n", 0)):
                    continue
            cur[k2] = o
            changed_ = True
            oadd += 1
        for kk, o in bk.items():
            if kk not in rk:                                    # reverted in the run (evidence-based)
                k2 = (mstate(kk[0]), kk[1])
                if k2 in cur and cur[k2].get("dir") == o.get("dir"):
                    del cur[k2]
                    changed_ = True
                    orem += 1
        if changed_:
            g.execute("INSERT OR REPLACE INTO gk_overrides VALUES (?,?,?)",
                      (part, gtech, _j(sorted(cur.values(), key=lambda o: (str(o.get("state")), o.get("metric"))))))
    rep["prediction_overrides"] = {"added": oadd, "reverted": orem, "conflicts": ocon}

    # -- handled hypothesis triggers (general ones only: no problem / outcome specific keys) ------------------------
    rt_ = set(R.meta("hyp_triggers_seen", []) or []) - set(B.meta("hyp_triggers_seen", []) or [])
    ntr = 0
    for key in sorted(rt_):
        p = key.split("|")
        general = p[0] in ("port", "combine", "profiler") or (p[0] == "gap" and len(p) == 4)
        if general:
            g.execute("INSERT OR IGNORE INTO gk_triggers VALUES (?, ?)", (part, key))
            ntr += 1
    rep["triggers"] = ntr

    # -- state proposals (rejected ones keep the discovery prompt from repeating them) ----------------------------
    rp, bp = R.rows("proposals"), B.rows("proposals")
    npr = 0
    for pid, d in sorted(rp.items()):
        if pid in bp:
            continue
        gid = f"{info.run_uid[:8]}:{pid}"[:120]
        g.execute("INSERT OR IGNORE INTO gk_proposals VALUES (?,?,?,?,?,?)",
                  (part, gid, d.get("key"), d.get("status"), _j(dict(d, id=gid)), _j(prov)))
        npr += 1
    for (key,) in list(g.execute("SELECT DISTINCT key FROM gk_proposals WHERE part=?", (part,))):
        ids = [r[0] for r in g.execute("SELECT id FROM gk_proposals WHERE part=? AND key IS ? AND status='rejected' "
                                       "ORDER BY rowid", (part, key))]
        for pid in ids[:-pol.proposals_per_key]:
            g.execute("DELETE FROM gk_proposals WHERE part=? AND id=?", (part, pid))
    rep["proposals"] = npr
    # runtime scope nodes / targets
    for nid, parent, data in (R.c.execute("SELECT id, parent, data FROM scopes") if "scopes" in R.t else []):
        g.execute("INSERT OR IGNORE INTO gk_scopes VALUES (?,?,?)", (nid, parent, data))
    bt_keys = {r[0] for r in B.c.execute("SELECT key FROM targets")} if "targets" in B.t else set()
    for key, node, data in (R.c.execute("SELECT key, family, data FROM targets") if "targets" in R.t else []):
        if key not in bt_keys:
            g.execute("INSERT OR IGNORE INTO gk_targets VALUES (?,?,?)", (key, node, data))
    if not rep["skipped"]:
        rep.pop("skipped")
    return rep


def _impl_records(d: Optional[Path]) -> list[dict]:
    if not d:
        return []
    p = Path(d) / "index.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _merge_impl(g, store: GlobalStore, impl_dir, base_impl_dir, *, info: RunInfo, part: str, seq: int,
                pol: Policy, exclude=()) -> dict:
    """Keep per (op, signature, arch, dtype, shape set) the fastest verified record (and at most
    ``impl_max_per_op`` shape sets per op); sources go to the content-addressed ``impl_code/``."""
    recs = _impl_records(impl_dir)
    if not recs:
        return {"records": 0}
    base_ids = {r.get("id") for r in _impl_records(base_impl_dir)} | set(exclude)   # + imported mid-run
    tk, dt = parse_part(part)
    arch = tk.split(":", 1)[-1]
    store.codedir.mkdir(parents=True, exist_ok=True)
    kept = considered = 0
    best_run: dict = {}                                  # this run's fastest record per key first
    for r in recs:
        if r.get("id") in base_ids or r.get("arch") != arch or (r.get("dtype") or "fp32") != dt:
            continue
        sp = r.get("speedup_vs_seed")
        if sp is None or sp < pol.impl_min_speedup:
            continue
        considered += 1
        if not (Path(impl_dir) / "code" / f"{r.get('code_sha')}.cu").exists():
            continue
        kkey = hashlib.sha1(_j([r.get("op"), r.get("signature_sha"), r.get("arch"), r.get("dtype"),
                                r.get("shapes")]).encode()).hexdigest()[:20]
        if kkey not in best_run or float(r.get("slot_us") or 1e30) < float(best_run[kkey].get("slot_us") or 1e30):
            best_run[kkey] = r
    for kkey, r in sorted(best_run.items()):
        src = Path(impl_dir) / "code" / f"{r.get('code_sha')}.cu"
        best = g.execute("SELECT id, slot_us FROM gk_impl WHERE part=? AND kkey=? ORDER BY slot_us", (part, kkey)).fetchall()
        if len(best) >= pol.impl_per_key and best[-1][1] <= float(r.get("slot_us") or 1e30):
            continue
        dst = store.codedir / f"{r['code_sha']}.cu"
        if not dst.exists():
            tmp = dst.with_suffix(f".tmp{os.getpid()}")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
        d = dict(r, problem_id=ns(info.problem, r.get("problem_id")))
        g.execute("INSERT OR REPLACE INTO gk_impl VALUES (?,?,?,?,?,?)",
                  (r["id"], part, kkey, float(r.get("slot_us") or 0.0), _j(d), _j(info.prov())))
        g.execute("INSERT INTO gk_contrib VALUES (?,?,?,?,?,?)", (seq, info.run_uid, part, "impl", r["id"], "1"))
        kept += 1
        for rid, _ in g.execute("SELECT id, slot_us FROM gk_impl WHERE part=? AND kkey=? ORDER BY slot_us",
                                (part, kkey)).fetchall()[pol.impl_per_key:]:
            g.execute("DELETE FROM gk_impl WHERE id=?", (rid,))
        # at most impl_max_per_op shape sets per op (fastest per-instance first)
        opk = (r.get("op"), r.get("signature_sha"))
        same = []
        for rid, data in g.execute("SELECT id, data FROM gk_impl WHERE part=?", (part,)):
            x = json.loads(data)
            if (x.get("op"), x.get("signature_sha")) == opk:
                same.append((float(x.get("slot_us") or 0) / max(len(x.get("shapes") or [1]), 1), rid))
        for _, rid in sorted(same)[pol.impl_max_per_op:]:
            g.execute("DELETE FROM gk_impl WHERE id=?", (rid,))
    return {"records": len(recs), "considered": considered, "kept": kept}


# ---------------------------------------------------------------- store upkeep (no LLM)

def upkeep(g, part: str, seq: int, pol: Policy) -> dict:
    """Retire contradicted / stale claims, cap active claims, decide hypotheses on their union of attempts."""
    from ..claims import ACTIVE, RETIRED, Claim
    from .hypotheses import HypothesisBook, Hypothesis
    out = {"claims_contradicted": 0, "claims_stale": 0, "claims_capped": 0, "hyp_promoted": 0, "hyp_retired": 0}
    rows = list(g.execute("SELECT id, data, prov FROM gk_claims WHERE part=? AND status='active'", (part,)))
    alive = []
    for cid, data, prov in rows:
        d, pv = json.loads(data), json.loads(prov or "{}")
        sup, con = int(d.get("support", 0)), int(d.get("contradict", 0))
        why = None
        if con >= 2 and con >= sup:
            why, key = f"contradicted ({sup}/{con})", "claims_contradicted"
        else:
            try:
                eligible = Claim.from_dict(d).eligible()
            except (TypeError, ValueError):
                eligible = False
            age = seq - int(pv.get("last_evidence", pv.get("last_merge", seq)))
            if not eligible and age >= pol.stale_merges:
                why, key = f"stale: ineligible, no new evidence for {age} merges", "claims_stale"
        if why:
            d["status"] = RETIRED
            d["data"] = dict(d.get("data") or {}, gk_retired=why)
            g.execute("UPDATE gk_claims SET status=?, data=? WHERE part=? AND id=?", (RETIRED, _j(d), part, cid))
            out[key] += 1
        else:
            alive.append((_rank(d, pv, seq, pol), cid, d))
    # caps: general claims per partition; problem-local ones (impl notes ...) per problem
    groups: dict = {}
    for x in alive:
        pid = x[2].get("problem_id")
        groups.setdefault(pid.split(NS, 1)[0] if pid else None, []).append(x)
    for key, lst in groups.items():
        cap = pol.max_active_claims if key is None else pol.max_local_claims_per_problem
        if len(lst) <= cap:
            continue
        lst.sort(key=lambda x: (x[0], x[1]))
        for _, cid, d in lst[: len(lst) - cap]:
            d["status"] = RETIRED
            d["data"] = dict(d.get("data") or {}, gk_retired="capped (lowest global rank)")
            g.execute("UPDATE gk_claims SET status=?, data=? WHERE part=? AND id=?", (RETIRED, _j(d), part, cid))
            out["claims_capped"] += 1
    # hypotheses: the union of attempts across runs decides (wins on >= 2 problems now counts across problems)
    from .targets import ROOT, TARGET, TargetRegistry
    reg = None
    for hid, data, prov in list(g.execute("SELECT id, data, prov FROM gk_hypotheses WHERE part=? AND "
                                          "status IN ('proposed','testing')", (part,))):
        d, pv = json.loads(data), json.loads(prov or "{}")
        try:
            h = Hypothesis.from_dict(d)
        except (TypeError, ValueError):
            continue
        ok, why = HypothesisBook.verdict(h) if h.attempts else (False, "untested")
        if ok:
            reg = reg or TargetRegistry()
            tks = sorted({a["target_key"] for a in h.attempts if a.get("reward", 0) > 0}) or [h.target_key]
            scope = reg.lca(tks)
            level = reg.level_of(scope)
            have = {r[0] for r in g.execute("SELECT id FROM gk_techniques WHERE part=?", (part,))}
            tid = h.name if h.name not in have else f"{h.name}_{h.id[2:6]}"
            rscope = (f"target:{scope}" if level == TARGET else
                      f"node:{scope}" if scope != ROOT else f"node:{reg.family_of(h.target_key)}")
            t = {"id": tid, "mechanism": h.mechanism, "level": level, "scope": None if scope == ROOT else scope,
                 "aliases": [], "preconditions": h.preconditions, "predicts": h.predicts, "risks": h.risks, "params": {},
                 "relations": {"based_on": h.based_on} if h.based_on else {},
                 "realizations": [{"scope": rscope, "how": h.how}], "states": h.states, "op_classes": h.op_classes,
                 "op_kinds": [], "prior": {}, "slot": True, "source": f"hypothesis:{h.id}", "status": "active"}
            g.execute("INSERT OR REPLACE INTO gk_techniques VALUES (?,?,?,?,?)", (part, tid, "active", _j(t), _j(pv)))
            # the technique starts from the hypothesis' statistics (as KnowledgeV2._copy_stats does in a run)
            for row in list(g.execute("SELECT level, scope, op, state, n, s, ss FROM gk_stats WHERE part=? AND item=?",
                                      (part, h.id))):
                g.execute("INSERT OR REPLACE INTO gk_stats VALUES (?,?,?,?,?,?,?,?,?)", (part, tid, *row))
            d.update(status="promoted", reason=f"store: {why}", promoted_to=tid)
            out["hyp_promoted"] += 1
        elif len(h.attempts) >= h.budget:
            d.update(status="retired", reason=f"store: budget of {h.budget} attempts spent: {why}")
            out["hyp_retired"] += 1
        elif seq - int(pv.get("last_evidence", pv.get("first_merge", seq))) >= pol.stale_hyp_merges:
            d.update(status="retired", reason=f"store: untested for {pol.stale_hyp_merges} merges")
            out["hyp_retired"] += 1
        else:
            continue
        g.execute("UPDATE gk_hypotheses SET status=?, data=? WHERE part=? AND id=?", (d["status"], _j(d), part, hid))
    return out


# ---------------------------------------------------------------- convenience for run dirs

def merge_knowledge_dir(store: GlobalStore, kdir, seed_dir=None, *, info: RunInfo, policy=None, label="") -> dict:
    """Merge a run's knowledge dir (knowledge.db, impl/, review.jsonl) against its seed dir
    (knowledge.db + impl/ as materialized; None: the repo seed)."""
    kdir = Path(kdir)
    base_db = base_impl = None
    if seed_dir is not None:
        sd = Path(seed_dir)
        base_db = sd if sd.suffix == ".db" else sd / "knowledge.db"
        base_impl = (sd.parent if sd.suffix == ".db" else sd) / "impl"
    sp = kdir.parent / "state.json"
    return merge_run(store, kdir / "knowledge.db", base_db, info=info, impl_dir=kdir / "impl",
                     base_impl_dir=base_impl, policy=policy, review_path=kdir / "review.jsonl", label=label,
                     struct_path=sp if sp.exists() else None)


def seed_meta(db_path) -> dict:
    if not Path(db_path).exists():
        raise MergeError(f"{db_path}: no such database")
    c = _connect(Path(db_path), ro=True)
    try:
        return _seed_info(c)
    finally:
        c.close()


def show(store: GlobalStore, part: Optional[str], kind: str, top: int, vid: Optional[str] = None) -> list:
    vid = vid or store.current_id()
    if not vid:
        return []
    c = _connect(store.path(vid), ro=True)
    try:
        seq = int(_meta(c, "merge_seq", 0, "gk_meta") or 0)
        if kind == "claims":
            q = "SELECT part, id, status, data, prov FROM gk_claims" + (" WHERE part=?" if part else "")
            rows = []
            for p, cid, st, d, pv in c.execute(q, (part,) if part else ()):
                d, pv = json.loads(d), json.loads(pv or "{}")
                rows.append({"part": p, "id": cid, "status": st, "rank": round(_rank(d, pv, seq, Policy()), 3),
                             "support": d.get("support"), "contradict": d.get("contradict"), "kind": d.get("kind"),
                             "op_class": d.get("op_class"), "technique": d.get("technique"), "text": d.get("text"),
                             "problems": pv.get("problems"), "first": (pv.get("first") or {}).get("problem")})
            rows.sort(key=lambda r: (r["status"] != "active", -r["rank"]))
            return rows[:top]
        tb = {"hypotheses": "gk_hypotheses", "states": "gk_states", "techniques": "gk_techniques"}[kind]
        q = f"SELECT part, id, status, data FROM {tb}" + (" WHERE part=?" if part else "")
        out = []
        for p, rid, st, d in c.execute(q, (part,) if part else ()):
            d = json.loads(d)
            out.append({"part": p, "id": rid, "status": st, "name": d.get("name"),
                        "mechanism": (d.get("mechanism") or "")[:160], "attempts": len(d.get("attempts", []) or [])})
        return out[:top]
    finally:
        c.close()


# ---------------------------------------------------------------- CLI

def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kernelblaster.knowledge.v2.merge", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=None, help=f"store root (default $KB_GLOBAL_KNOWLEDGE or {DEFAULT_ROOT})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("verify")
    s = sub.add_parser("seed", help="materialize a run's starting knowledge dir from a snapshot")
    s.add_argument("--part", required=True, help="partition, e.g. cuda:sm_89|fp16")
    s.add_argument("--out", required=True)
    s.add_argument("--problem")
    s.add_argument("--snapshot", help="snapshot id (default: current)")
    s.add_argument("--run-uid")
    s.add_argument("--policy", help="JSON Policy overrides")
    m = sub.add_parser("merge", help="merge a finished run's knowledge dir into a new snapshot")
    m.add_argument("--run", required=True, help="the run's knowledge dir (knowledge.db, impl/, review.jsonl)")
    m.add_argument("--base", help="the seed dir / db it started from, or the copy merged last (default: repo seed)")
    m.add_argument("--run-uid", help="default: the run db's gk_seed run_uid")
    m.add_argument("--problem")
    m.add_argument("--attempt")
    m.add_argument("--part", help="default: the run db's gk_seed partition")
    m.add_argument("--model")
    m.add_argument("--source", default="run")
    m.add_argument("--label", default="")
    m.add_argument("--policy")
    rf = sub.add_parser("refresh", help="build a mid-run refresh package for a running run (refresh.py)")
    rf.add_argument("--part", help="default: the run db's gk_seed partition")
    rf.add_argument("--out", required=True)
    rf.add_argument("--run-db", help="the run's latest pulled knowledge.db: its run uid, partition and sync point")
    rf.add_argument("--run-uid")
    rf.add_argument("--problem")
    rf.add_argument("--since-snapshot")
    rf.add_argument("--since-seq", type=int)
    rf.add_argument("--policy")
    im = sub.add_parser("import", help="apply a refresh package to a knowledge dir (node side, at a round boundary)")
    im.add_argument("--kdir", required=True)
    im.add_argument("--pkg", required=True)
    r = sub.add_parser("rollback")
    r.add_argument("version")
    r.add_argument("--reason", default="")
    p = sub.add_parser("prune")
    p.add_argument("--keep", type=int, default=20)
    co = sub.add_parser("consolidate", help="A5: merge near-duplicate claims into a new snapshot (consolidate.py)")
    co.add_argument("--part")
    co.add_argument("--threshold", type=float, default=0.35)
    co.add_argument("--include-local", action="store_true")
    co.add_argument("--dry-run", action="store_true")
    co.add_argument("--llm", help="cheap model for borderline pairs through the relay (OPENAI_BASE_URL)")
    co.add_argument("--llm-low", type=float, default=0.2)
    co.add_argument("--llm-max-pairs", type=int, default=100)
    co.add_argument("--label", default="consolidate")
    pb = sub.add_parser("pool-build", help="A5: build implementation pools (current + gk_impl + run impl dirs)")
    pb.add_argument("--part", required=True)
    pb.add_argument("--impl", nargs="*", default=[], help="run impl dirs (results/*/work/knowledge/impl)")
    pb.add_argument("--no-gk-impl", action="store_true")
    pb.add_argument("--policy")
    pb.add_argument("--label", default="pool bootstrap")
    si = sub.add_parser("struct-ingest", help="A5: structural decisions of ModelBlaster work dirs -> gk_struct")
    si.add_argument("--part", required=True)
    si.add_argument("--work", nargs="+", required=True)
    si.add_argument("--label", default="struct bootstrap")
    pr = sub.add_parser("prior", help="A5: structural-outcome prior (priors.StructPriors)")
    pr.add_argument("kind")
    pr.add_argument("context", nargs="?", default="")
    pr.add_argument("--part")
    sh = sub.add_parser("show")
    sh.add_argument("kind", choices=["claims", "hypotheses", "states", "techniques"])
    sh.add_argument("--part")
    sh.add_argument("--top", type=int, default=20)
    sh.add_argument("--version")
    a = ap.parse_args(argv)
    st = GlobalStore(a.store)
    try:
        if a.cmd == "status":
            res = st.status()
        elif a.cmd == "verify":
            res = st.verify()
        elif a.cmd == "seed":
            st.ensure()
            res = materialize(st, a.part, a.out, problem=a.problem, snapshot=a.snapshot,
                              policy=Policy.from_json(a.policy), run_uid=a.run_uid)
        elif a.cmd == "merge":
            kdir = Path(a.run)
            meta = seed_meta(kdir / "knowledge.db")
            uid = a.run_uid or meta.get("run_uid")
            if not uid:
                raise MergeError("no run uid (the run db has no gk_seed meta): pass --run-uid")
            info = RunInfo(run_uid=uid, problem=a.problem or meta.get("problem"), attempt=a.attempt,
                           part=a.part or meta.get("part"), model=a.model, source=a.source)
            res = merge_knowledge_dir(st, kdir, a.base, info=info, policy=Policy.from_json(a.policy), label=a.label)
        elif a.cmd == "refresh":
            from .refresh import build_package, run_sync_point
            sp = run_sync_point(a.run_db) if a.run_db else {}
            seed = seed_meta(a.run_db) if a.run_db else {}
            res = build_package(st, a.part or seed.get("part"), a.out, run_uid=a.run_uid or seed.get("run_uid"),
                                problem=a.problem or seed.get("problem"),
                                since_snapshot=a.since_snapshot or sp.get("snapshot"),
                                since_seq=a.since_seq if a.since_seq is not None else sp.get("seq"),
                                policy=Policy.from_json(a.policy))
        elif a.cmd == "import":
            from .refresh import import_package
            res = import_package(a.kdir, a.pkg)
        elif a.cmd == "consolidate":
            from .consolidate import consolidate_store, relay_judge
            judge = relay_judge(a.llm) if a.llm else None
            res = consolidate_store(st, a.part, label=a.label, dry_run=a.dry_run, threshold=a.threshold,
                                    include_local=a.include_local, judge=judge, llm_low=a.llm_low,
                                    llm_max_pairs=a.llm_max_pairs)
        elif a.cmd == "pool-build":
            from .pool_store import build_from_runs
            pol = Policy.from_json(a.policy)
            pol.impl_pool = True
            res = build_from_runs(st, a.part, a.impl, policy=pol, label=a.label, from_gk_impl=not a.no_gk_impl)
        elif a.cmd == "struct-ingest":
            from .priors import ingest_runs
            res = ingest_runs(st, a.part, a.work, label=a.label)
        elif a.cmd == "prior":
            from .priors import StructPriors
            res = StructPriors.load(st, part=a.part).prior(a.kind, a.context or None, a.part).to_dict()
        elif a.cmd == "rollback":
            res = st.rollback(a.version, reason=a.reason)
        elif a.cmd == "prune":
            res = {"deleted": st.prune(a.keep)}
        else:
            res = show(st, a.part, a.kind, a.top, a.version)
    except (MergeError, sqlite3.Error, OSError) as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}), flush=True)
        return 2
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
