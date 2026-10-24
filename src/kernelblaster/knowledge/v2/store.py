# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One SQLite knowledge database (design section 9).

``knowledge.db`` in WAL mode (concurrent readers, serialised writers; every
write is a short ``BEGIN IMMEDIATE`` transaction with a busy timeout), tables::

    meta        key -> json                 epoch, reflect cursor, seed version, upkeep cursors
    targets     key, family, json
    states      id, status, json            seed + discovered states
    techniques  id, source, json            ontology (realizations in their own table)
    realizations technique_id, scope, json
    hypotheses  id, status, json
    claims      id, status, json
    outcomes    seq, id, ..., rec json, v2 json      append-only
    stats       (item, level, scope, op, state) -> n, sum, sumsq
    proposals   id, status, json            state-discovery proposals (admitted / rejected + reason)
    scopes      id, parent, json            scope nodes added at run time (the seed tree is targets.json)
    review      seq, ts, kind, ref, json    audit log of every automatic change (mirrored to
                                            review.jsonl); nothing waits on it -- the table keeps
                                            its v1 name so old databases stay readable

On top: :class:`SQLiteOutcomeLog` (the :class:`OutcomeLog` API) and
:class:`SQLiteClaimStore` (the :class:`ClaimStore` API, level-aware
retrieval, evidence unioned with concurrent writers on save).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from ..claims import ACTIVE, Claim, ClaimStore, IMPL_NOTE, IMPROVED
from ..outcomes import OutcomeRecord
from .targets import FAMILY, NODE, TARGET, UNIVERSAL, TargetRegistry, target_key

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS targets (key TEXT PRIMARY KEY, family TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS states (id TEXT PRIMARY KEY, status TEXT, data TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS techniques (id TEXT PRIMARY KEY, source TEXT, data TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS realizations (technique_id TEXT, scope TEXT, data TEXT,
                                         PRIMARY KEY (technique_id, scope));
CREATE TABLE IF NOT EXISTS hypotheses (id TEXT PRIMARY KEY, status TEXT, data TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS claims (id TEXT PRIMARY KEY, status TEXT, data TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS outcomes (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, ts TEXT, target_key TEXT,
    problem_id TEXT, slot TEXT, technique TEXT, item_id TEXT, hypothesis_id TEXT, op_class TEXT,
    status TEXT, reward REAL, state TEXT, mechanism TEXT, rec TEXT, v2 TEXT);
CREATE INDEX IF NOT EXISTS outcomes_ps ON outcomes (problem_id, slot);
CREATE INDEX IF NOT EXISTS outcomes_item ON outcomes (item_id);
CREATE TABLE IF NOT EXISTS stats (item TEXT, level TEXT, scope TEXT, op TEXT, state TEXT,
                                  n REAL, s REAL, ss REAL, PRIMARY KEY (item, level, scope, op, state));
CREATE TABLE IF NOT EXISTS proposals (id TEXT PRIMARY KEY, status TEXT, data TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS scopes (id TEXT PRIMARY KEY, parent TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS review (seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT,
                                   ref TEXT, data TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class KnowledgeDB:
    def __init__(self, path: Path, timeout_s: float = 30.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self._local = threading.local()
        self.conn.executescript(SCHEMA)          # idempotent (IF NOT EXISTS); commits itself

    # -- connections ------------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(str(self.path), timeout=self.timeout_s, isolation_level=None)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute(f"PRAGMA busy_timeout={int(self.timeout_s * 1000)}")
            c.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = c
        return c

    @contextmanager
    def tx(self):
        """One serialised write transaction (BEGIN IMMEDIATE, retried while locked)."""
        c = self.conn
        for attempt in range(50):
            try:
                c.execute("BEGIN IMMEDIATE")
                break
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) or attempt == 49:
                    raise
                time.sleep(0.05 * (attempt + 1))
        try:
            yield c
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise

    def query(self, sql: str, args=()) -> list[tuple]:
        return self.conn.execute(sql, args).fetchall()

    # -- meta -------------------------------------------------------------
    def get_meta(self, key: str, default=None):
        r = self.query("SELECT value FROM meta WHERE key=?", (key,))
        return json.loads(r[0][0]) if r else default

    def set_meta(self, key: str, value) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))

    # -- generic json rows -------------------------------------------------
    def put(self, table: str, rid: str, data: dict, *, status: Optional[str] = None,
            replace: bool = True) -> None:
        verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
        col2 = "source" if table == "techniques" else "status"
        val2 = data.get("source") if table == "techniques" else (status or data.get("status"))
        with self.tx() as c:
            c.execute(f"{verb} INTO {table} (id, {col2}, data, updated) VALUES (?, ?, ?, ?)",
                      (rid, val2, json.dumps(data, sort_keys=True), _now()))
            if table == "techniques":
                if replace:
                    c.execute("DELETE FROM realizations WHERE technique_id=?", (rid,))
                for r in data.get("realizations", []):
                    c.execute("INSERT OR IGNORE INTO realizations VALUES (?, ?, ?)",
                              (rid, r.get("scope", "universal"), json.dumps(r, sort_keys=True)))

    def rows(self, table: str, status: Optional[str] = None) -> list[dict]:
        sql = f"SELECT data FROM {table}" + (" WHERE status=?" if status else "")
        return [json.loads(r[0]) for r in self.query(sql, (status,) if status else ())]

    def techniques(self) -> list[dict]:
        real: dict[str, list] = {}
        for tid, data in self.query("SELECT technique_id, data FROM realizations"):
            real.setdefault(tid, []).append(json.loads(data))
        out = []
        for (data,) in self.query("SELECT data FROM techniques"):
            d = json.loads(data)
            d["realizations"] = real.get(d["id"], d.get("realizations", []))
            out.append(d)
        return out

    # -- stats --------------------------------------------------------------
    def add_stats(self, nodes: list[tuple], reward: float) -> None:
        with self.tx() as c:
            for node in nodes:
                c.execute("INSERT INTO stats VALUES (?, ?, ?, ?, ?, 1, ?, ?) ON CONFLICT"
                          "(item, level, scope, op, state) DO UPDATE SET n=n+1, s=s+excluded.s, "
                          "ss=ss+excluded.ss", (*node, reward, reward * reward))

    def add_stats_bulk(self, agg: dict) -> None:
        """{(item, level, scope, op, state): [n, sum, sumsq]} added in one transaction."""
        with self.tx() as c:
            for k, (n, sm, ss) in agg.items():
                c.execute("INSERT INTO stats VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT"
                          "(item, level, scope, op, state) DO UPDATE SET n=n+excluded.n, s=s+excluded.s, "
                          "ss=ss+excluded.ss", (*k, n, sm, ss))

    def all_stats(self) -> dict[tuple, list]:
        return {tuple(r[:5]): [r[5], r[6], r[7]] for r in self.query("SELECT * FROM stats")}

    # -- audit log (no human gate: changes take effect when they are logged) ----
    def audit(self, kind: str, ref: str, data: dict, mirror: Optional[Path] = None) -> None:
        ts = _now()
        with self.tx() as c:
            c.execute("INSERT INTO review (ts, kind, ref, data) VALUES (?, ?, ?, ?)",
                      (ts, kind, ref, json.dumps(data, sort_keys=True, default=str)))
        if mirror is not None:
            with open(mirror, "a", encoding="utf-8") as f:
                f.write(json.dumps({**data, "ts": ts, "kind": kind, "ref": ref}, default=str) + "\n")

    def audit_items(self) -> list[dict]:
        return [{**json.loads(d), "seq": s, "ts": ts, "kind": k, "ref": r}
                for s, ts, k, r, d in self.query("SELECT * FROM review ORDER BY seq")]

    review, review_items = audit, audit_items          # v1 names

    # -- runtime scope tree --------------------------------------------------------
    def put_scope(self, nid: str, parent: str, data: dict) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO scopes VALUES (?, ?, ?)", (nid, parent, json.dumps(data, sort_keys=True)))

    def scopes(self) -> list[dict]:
        return [json.loads(d) for (d,) in self.query("SELECT data FROM scopes ORDER BY rowid")]

    def put_target(self, key: str, node: str, data: dict) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO targets VALUES (?, ?, ?)", (key, node, json.dumps(data, sort_keys=True)))

    def target_rows(self) -> list[tuple]:
        return [(k, n, json.loads(d)) for k, n, d in self.query("SELECT key, family, data FROM targets")]


# ---------------------------------------------------------------- outcomes

class SQLiteOutcomeLog:
    """:class:`OutcomeLog` API over the ``outcomes`` table; ``append`` takes v2 extras."""

    def __init__(self, db: KnowledgeDB):
        self.db = db

    def append(self, rec: OutcomeRecord, v2: Optional[dict] = None) -> OutcomeRecord:
        v2 = v2 or {}
        with self.db.tx() as c:
            c.execute("INSERT OR IGNORE INTO outcomes (id, ts, target_key, problem_id, slot, technique, item_id,"
                      " hypothesis_id, op_class, status, reward, state, mechanism, rec, v2)"
                      " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (rec.id, rec.ts, target_key(rec.target, rec.arch), rec.problem_id, rec.slot,
                       rec.technique, v2.get("item_id") or rec.technique, v2.get("hypothesis_id"), rec.op_class,
                       rec.status, v2.get("reward"), rec.state_before, (v2.get("mechanism") or {}).get("label"),
                       json.dumps(asdict(rec), sort_keys=True), json.dumps(v2, sort_keys=True, default=str)))
        return rec

    def __iter__(self) -> Iterator[OutcomeRecord]:
        for (rec,) in self.db.query("SELECT rec FROM outcomes ORDER BY seq"):
            yield OutcomeRecord(**json.loads(rec))

    def rows(self, where: str = "", args=()) -> list[dict]:
        """Outcome dicts with ``v2`` extras and ``seq`` (``where`` is an SQL condition)."""
        sql = "SELECT seq, rec, v2 FROM outcomes" + (f" WHERE {where}" if where else "") + " ORDER BY seq"
        out = []
        for seq, rec, v2 in self.db.query(sql, args):
            d = json.loads(rec)
            d["v2"] = json.loads(v2) if v2 else {}
            d["seq"] = seq
            out.append(d)
        return out

    def get(self, ids) -> list[OutcomeRecord]:
        want = set(ids)
        return [r for r in self if r.id in want]

    def since(self, cursor: int) -> tuple[list[OutcomeRecord], int]:
        recs = list(self)
        return recs[cursor:], len(recs)

    def __len__(self) -> int:
        return self.db.query("SELECT COUNT(*) FROM outcomes")[0][0]


# ---------------------------------------------------------------- claims

def claim_scope(c: Claim, registry: TargetRegistry) -> Optional[str]:
    """A claim's scope id: universal | node id (``level`` node, or v1 ``family``, with the
    node in ``c.family``) | None for target-level claims (backend + optional arch, as in v1)."""
    level = getattr(c, "level", TARGET) or TARGET
    if level == UNIVERSAL:
        return UNIVERSAL
    if level in (FAMILY, NODE):
        return c.family or UNIVERSAL
    return None


class SQLiteClaimStore(ClaimStore):
    """Claims in the ``claims`` table with level-aware retrieval (design section 8).

    * a ``universal`` claim applies everywhere, a ``node`` claim (v1:
      ``family``; the node id is in ``Claim.family``) to every target whose
      scope chain contains that node, a ``target`` claim as in v1 (backend + arch);
    * a claim's ``state`` matches the query state or any of its ancestors;
    * ``save`` writes only changed claims and unions their cited ids with
      what concurrent writers stored meanwhile;
    * "already tried" comes from the outcomes table (no separate tally).
    """

    def __init__(self, db: KnowledgeDB, *, registry: Optional[TargetRegistry] = None, hierarchy=None,
                 min_support: Optional[int] = None):
        self.db = db
        self.registry = registry or TargetRegistry()
        self.hierarchy = hierarchy
        self._snap: dict[str, str] = {}
        super().__init__(db.path.with_suffix(".claims"), min_support=min_support)

    def load(self) -> None:
        with self._lock:
            self.meta.update(self.db.get_meta("claims_meta", {}) or {})
            self.claims = {}
            for d in self.db.rows("claims"):
                try:
                    c = Claim.from_dict(d)
                except (TypeError, ValueError):
                    continue
                self.claims[c.id] = c
                self._snap[c.id] = json.dumps(asdict(c), sort_keys=True)
            self.tried = {}

    def save(self) -> None:
        with self._lock:
            changed = []
            for c in self.claims.values():
                js = json.dumps(asdict(c), sort_keys=True)
                if self._snap.get(c.id) != js:
                    changed.append(c)
            with self.db.tx() as conn:
                for c in changed:
                    row = conn.execute("SELECT data FROM claims WHERE id=?", (c.id,)).fetchone()
                    if row and self._snap.get(c.id) != row[0]:
                        other = Claim.from_dict(json.loads(row[0]))    # written by someone else meanwhile
                        extra_s = [i for i in other.support_ids if i not in c.support_ids and i not in c.contradict_ids]
                        extra_c = [i for i in other.contradict_ids if i not in c.contradict_ids and i not in c.support_ids]
                        c.add_support(extra_s)
                        c.add_contradiction(extra_c)
                    js = json.dumps(asdict(c), sort_keys=True)
                    conn.execute("INSERT OR REPLACE INTO claims VALUES (?, ?, ?, ?)", (c.id, c.status, js, c.updated_ts))
                    self._snap[c.id] = js
                cur = conn.execute("SELECT value FROM meta WHERE key='claims_meta'").fetchone()
                meta = dict(self.meta)
                if cur:
                    old = json.loads(cur[0])
                    for k in ("epoch", "reflect_cursor", "reflections"):
                        meta[k] = max(int(meta.get(k, 0)), int(old.get(k, 0)))
                    self.meta.update(meta)
                conn.execute("INSERT OR REPLACE INTO meta VALUES ('claims_meta', ?)", (json.dumps(meta),))

    # -- level-aware scoping ------------------------------------------------
    def _state_ok(self, cs: Optional[str], qs: Optional[str]) -> bool:
        if cs is None or qs is None:
            return True
        if cs == qs:
            return True
        return self.hierarchy is not None and cs in self.hierarchy.ancestors(qs)

    def _matches(self, c: Claim, q: dict) -> Optional[int]:
        tk = target_key(q["target"], q.get("arch"))
        sid = claim_scope(c, self.registry)
        if sid is not None:
            if sid not in self.registry.chain(tk):
                return None
            spec = self.registry.depth(sid)           # deeper node = more specific
        else:
            if c.target != q["target"]:
                return None
            if c.arch is not None and q.get("arch") is not None and c.arch != q.get("arch"):
                return None
            spec = 20 + (1 if c.arch is not None and q.get("arch") else 0)
        for f in ("op_class", "technique", "problem_id", "slot", "impl_id"):
            cv, qv = getattr(c, f), q.get(f)
            if cv is None:
                continue
            if f in ("problem_id", "slot"):
                if qv != cv:
                    return None
            elif qv is not None and qv != cv:
                return None
            spec += 1 if qv is not None else 0
        if c.state is not None:
            if not self._state_ok(c.state, q.get("state")):
                return None
            spec += 1 if q.get("state") else 0
        return spec

    def level_label(self, c: Claim) -> str:
        sid = claim_scope(c, self.registry)
        if sid is None:
            return f"{c.arch or c.target} only"
        return self.registry.label(sid)

    def render(self, claims) -> str:
        lines = []
        for c in claims:
            extra = ""
            if c.kind == "pitfall" and c.data.get("fix"):
                extra = f" FIX: {c.data['fix']}"
            elif c.kind == "parameter" and c.data.get("params"):
                extra = " " + json.dumps(c.data["params"], sort_keys=True)
            lines.append(f"- (+{c.support}/-{c.contradict}, {self.level_label(c)}) [{self._label(c)}] {c.text}{extra}")
        return "\n".join(lines)

    # -- "already tried" from the outcomes table ----------------------------
    def observe(self, rec: OutcomeRecord) -> None:
        pass                                    # the outcome row already is the tally

    def already_tried(self, problem_id: Optional[str], slot: Optional[str] = None, *, max_items: int = 12) -> str:
        rows = self.db.query("SELECT technique, status, rec FROM outcomes WHERE problem_id=? AND "
                             "COALESCE(slot,'')=?", (problem_id, slot or ""))
        tally: dict[str, dict] = {}
        for tech, status, rec in rows:
            r = json.loads(rec)
            t = tally.setdefault(tech, {"n": 0, "statuses": {}, "best_speedup": None, "last_error": None})
            t["n"] += 1
            t["statuses"][status] = t["statuses"].get(status, 0) + 1
            sp = r.get("speedup")
            if sp is not None and (t["best_speedup"] is None or sp > t["best_speedup"]):
                t["best_speedup"] = sp
            if r.get("error"):
                t["last_error"] = r["error"].strip().splitlines()[0][:160] if r["error"].strip() else None
        self.tried = {self._tkey(problem_id, slot): tally}
        lines = []
        for tech, t in sorted(tally.items(), key=lambda kv: -kv[1]["n"])[:max_items]:
            st = ", ".join(f"{k} {v}" for k, v in sorted(t["statuses"].items(), key=lambda kv: -kv[1]))
            best = f"; best {t['best_speedup']:.2f}x" if t.get("best_speedup") else ""
            no_win = "" if t["statuses"].get(IMPROVED) else " -- no improvement yet"
            err = f'; last error: "{t["last_error"]}"' if t.get("last_error") else ""
            lines.append(f"- {tech}: {t['n']} tries ({st}){best}{no_win}{err}")
        notes = [c for c in self.claims.values()
                 if c.kind == IMPL_NOTE and c.status == ACTIVE and self.eligible(c)
                 and c.problem_id == problem_id and (c.slot or None) == (slot or None)]
        notes.sort(key=lambda c: -c.rank(self.epoch))
        if notes:
            lines.append(self.render(notes[:max_items]))
        return "\n".join(lines)
