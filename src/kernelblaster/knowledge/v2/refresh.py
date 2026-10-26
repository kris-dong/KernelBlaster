# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mid-run refresh: a RUNNING run imports what OTHER runs merged into the global store since it last
synchronised (the reverse of ``merge.merge_run``). Off unless a caller asks for it (``kb_graph_opt
--knowledge-global-refresh DIR``; the L3 farm's ``global_knowledge.refresh: round``).

    controller   build_package(store, part, out, run_uid=, problem=, since_seq=)   -> out/refresh.json (+ impl/code)
    node         import_package(kdir, pkg)   at a round boundary (no KernelBlaster process has the db open)
    controller   merge_run(...) later: the run db carries the exact log of what was imported (table
                 ``gk_refresh``); the merge applies it to its private base copy, so imported rows are never
                 merged back as this run's own evidence (``adjust_base``)

What a package carries (one partition, ``share: strict``; never this run's own contributions):

| item | selected (controller) | imported (node) |
|---|---|---|
| statistics | ``gk_contrib`` rows (one per merge and key, written by ``merge_run``) with ``from_seq < seq <= to_seq`` and another run uid | rows with ``seq`` > the run's sync seq, added to ``stats`` (exactly once per merge seq) |
| new claims | active, not problem-local (no ``problem_id``), first merged after ``from_seq`` by another run, ranked support - contradict, capped (``refresh_max_claims``) | ids the run does not have; state known to the run (or none); epochs set to the run's current epoch |
| claim evidence | claims merged after ``from_seq``: their cited outcome ids whose outcome is NOT this run's | on claims the run has: ids it does not cite yet, support / contradict + that many |
| outcomes | the foreign outcomes the shipped claims cite (capped) | only when the run's reflection cursor is at the end of its outcome log (imported records are never reflected on again), cursors advanced past them; else skipped (claims still cite them) |
| hypothesis triggers | the partition's handled-trigger keys | keys the run has not seen |
| impl winners | ``gk_contrib`` impl ids after ``from_seq`` by another run, still winners | ids not in the run's ``impl/index.jsonl`` (+ code) |

Not imported: hypotheses, states, techniques, prediction overrides, rejected proposals (they stay as the seed
gave them; the store decides them across runs at merge time).

Consistency. The import is one sqlite transaction and logs every row it changed in ``gk_refresh`` (the
impl index append is undone if the transaction fails). ``meta.gk_refresh_state`` = the run's sync point
(snapshot, merge seq, the packages applied). A package whose lineage does not contain the run's sync
snapshot (the store was rolled back past it) is refused. Packages are idempotent: by id for rows, by
merge seq for statistics, so a newer package that overlaps an applied one adds only what is new.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

FORMAT = "kb-global-refresh/1"
REFRESH_SCHEMA = """
CREATE TABLE IF NOT EXISTS gk_refresh (rid TEXT, kind TEXT, key TEXT, data TEXT, PRIMARY KEY (rid, kind, key));
"""


def _j(x) -> str:
    return json.dumps(x, sort_keys=True, default=str)


def _meta(c, key, default=None, table="meta"):
    try:
        r = c.execute(f"SELECT value FROM {table} WHERE key=?", (key,)).fetchone()
    except sqlite3.Error:
        return default
    return json.loads(r[0]) if r else default


def _set_meta(c, key, value, table="meta"):
    c.execute(f"INSERT OR REPLACE INTO {table} VALUES (?, ?)", (key, json.dumps(value, sort_keys=True)))


def _tables(c) -> set:
    return {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}


# ---------------------------------------------------------------- controller side

def lineage(store, vid: Optional[str], limit: int = 2000) -> list:
    """The snapshot ids from the oldest known ancestor to ``vid`` (``gk_meta.lineage`` when the snapshot has
    it, else the manifests' parent chain)."""
    if not vid:
        return []
    from .merge import _connect
    p = store.path(vid)
    if p.exists():
        c = _connect(p, ro=True)
        try:
            lin = _meta(c, "lineage", None, "gk_meta")
        finally:
            c.close()
        if lin:
            return list(lin)[-limit:]
    out, cur, seen = [], vid, set()
    while cur and cur not in seen and len(out) < limit:
        seen.add(cur)
        out.append(cur)
        cur = store.manifest(cur).get("parent")
    return out[::-1]


def run_sync_point(db) -> dict:
    """{snapshot, seq} the run db last synchronised with (refresh state, else its seed)."""
    c = sqlite3.connect(f"file:{Path(db).resolve()}?mode=ro", uri=True)
    try:
        st = _meta(c, "gk_refresh_state", None)
        if st:
            return {"snapshot": st.get("snapshot"), "seq": st.get("seq")}
        seed = _meta(c, "gk_seed", {}) or {}
        return {"snapshot": seed.get("snapshot"), "seq": seed.get("merge_seq"), "run_uid": seed.get("run_uid"),
                "part": seed.get("part"), "problem": seed.get("problem")}
    finally:
        c.close()


def build_package(store, part: str, out_dir, *, run_uid: str, problem: Optional[str] = None,
                  since_snapshot: Optional[str] = None, since_seq: Optional[int] = None,
                  policy=None) -> dict:
    """Write ``out_dir/refresh.json`` (+ ``impl/code``) with what OTHER runs merged into ``part`` after
    ``since_seq`` (default: the merge seq of ``since_snapshot``). Status: ready | up_to_date | refused."""
    from .merge import Policy, _connect
    pol = policy or Policy()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cur = store.current_id()
    man = {"format": FORMAT, "rid": uuid.uuid4().hex[:16], "part": part, "run_uid": run_uid, "problem": problem,
           "to": {"snapshot": cur}, "status": "ready"}
    if not cur:
        man["status"] = "up_to_date"
        return man
    lin = lineage(store, cur)
    if since_seq is None:
        if since_snapshot and store.path(since_snapshot).exists():
            c = _connect(store.path(since_snapshot), ro=True)
            try:
                since_seq = int(_meta(c, "merge_seq", 0, "gk_meta") or 0)
            finally:
                c.close()
        else:
            since_seq = None
    if since_snapshot and since_snapshot not in lin:
        man.update(status="refused", reason=f"{since_snapshot} is not an ancestor of {cur} (store rolled back?)")
        return man
    if since_seq is None:
        man.update(status="refused", reason="unknown sync point (no merge seq for the run's snapshot)")
        return man
    g = _connect(store.path(cur), ro=True)
    try:
        to_seq = int(_meta(g, "merge_seq", 0, "gk_meta") or 0)
        man["to"].update(seq=to_seq)
        man["from"] = {"snapshot": since_snapshot, "seq": since_seq}
        man["lineage"] = lin[-500:]
        if to_seq <= since_seq:
            man["status"] = "up_to_date"
            (out / "refresh.json").write_text(json.dumps(man, indent=1))
            return man
        t = _tables(g)
        own_out = set()

        def foreign(ids):
            ids = list(dict.fromkeys(ids or []))
            if not ids:
                return []
            known = {}
            for i in range(0, len(ids), 500):
                ch = ids[i:i + 500]
                q = ",".join("?" * len(ch))
                for oid, ru in g.execute(f"SELECT id, run_uid FROM gk_outcomes WHERE id IN ({q})", ch):
                    known[oid] = ru
            # only outcomes a merge stored (evidence added after the sync point always is); ids the store does
            # not know are repo-seed / pre-store citations the run already has
            res = []
            for i in ids:
                if i not in known:
                    continue
                if known[i] == run_uid:
                    own_out.add(i)
                    continue
                res.append(i)
            return res

        # statistics: per merge seq, other runs only
        stats = []
        if "gk_contrib" in t:
            for seq, item, level, scope, op, state, n, s, ss in g.execute(
                    "SELECT seq, json_extract(key, '$[0]'), json_extract(key, '$[1]'), json_extract(key, '$[2]'), "
                    "json_extract(key, '$[3]'), json_extract(key, '$[4]'), "
                    "SUM(json_extract(data, '$[0]')), SUM(json_extract(data, '$[1]')), SUM(json_extract(data, '$[2]')) "
                    "FROM gk_contrib WHERE kind='stats' AND part=? AND seq>? AND seq<=? AND run_uid<>? "
                    "GROUP BY seq, key ORDER BY seq, key", (part, since_seq, to_seq, run_uid)):
                stats.append([seq, [item, level, scope, op, state], [n, s, ss]])
        # claims: new (by another run) and evidence on older ones
        new_claims, evidence = [], []
        for cid, status, bck, data, prov in g.execute(
                "SELECT id, status, base_ck, data, prov FROM gk_claims WHERE part=?", (part,)):
            pv = json.loads(prov or "{}")
            if int(pv.get("last_merge", 0)) <= since_seq:
                continue
            d = json.loads(data)
            if d.get("problem_id"):                        # problem-local: never to another problem
                continue
            if int(pv.get("first_merge", 0)) > since_seq and not bck:
                if (pv.get("first") or {}).get("run_uid") == run_uid or status != "active":
                    continue
                new_claims.append((int(d.get("support", 0)) - int(d.get("contradict", 0)), cid, d))
            else:
                sup, con = foreign(d.get("support_ids")), foreign(d.get("contradict_ids"))
                if sup or con:
                    evidence.append({"id": cid, "sup": sup, "con": con})
        new_claims.sort(key=lambda x: (-x[0], x[1]))
        cap = int(getattr(pol, "refresh_max_claims", 300))
        new_claims = [dict(d, id=cid) for _, cid, d in new_claims[:cap]]
        for d in new_claims:                               # own outcomes never ride along
            d["support_ids"] = foreign(d.get("support_ids"))
            d["contradict_ids"] = foreign(d.get("contradict_ids"))
        # outcomes cited by what ships (foreign only)
        want = []
        for d in new_claims:
            want += d.get("support_ids", []) + d.get("contradict_ids", [])
        for e in evidence:
            want += e["sup"] + e["con"]
        want = list(dict.fromkeys(want))[: int(getattr(pol, "seed_max_cited_outcomes", 3000))]
        outcomes = []
        for i in range(0, len(want), 500):
            ch = want[i:i + 500]
            q = ",".join("?" * len(ch))
            for oid, ts, rec, v2, ru in g.execute(
                    f"SELECT id, ts, rec, v2, run_uid FROM gk_outcomes WHERE part=? AND id IN ({q})", [part] + ch):
                if ru != run_uid:
                    outcomes.append([oid, ts, rec, v2])
        outcomes.sort(key=lambda r: (r[1] or "", r[0]))
        triggers = sorted(r[0] for r in g.execute("SELECT key FROM gk_triggers WHERE part=?", (part,)))
        impl = []
        if "gk_contrib" in t:
            ids = [r[0] for r in g.execute("SELECT DISTINCT key FROM gk_contrib WHERE kind='impl' AND part=? AND "
                                           "seq>? AND seq<=? AND run_uid<>?", (part, since_seq, to_seq, run_uid))]
            for i in range(0, len(ids), 500):
                ch = ids[i:i + 500]
                q = ",".join("?" * len(ch))
                impl += [json.loads(r[0]) for r in g.execute(f"SELECT data FROM gk_impl WHERE id IN ({q})", ch)]
    finally:
        g.close()
    (out / "impl" / "code").mkdir(parents=True, exist_ok=True)
    shipped = []
    for r in impl:
        src = store.codedir / f"{r.get('code_sha')}.cu"
        if src.exists():
            shutil.copyfile(src, out / "impl" / "code" / src.name)
            shipped.append(r)
    pkg = {"stats": stats, "claims": new_claims, "evidence": evidence, "outcomes": outcomes,
           "triggers": triggers, "impl": shipped}
    man["counts"] = {k: len(v) for k, v in pkg.items()}
    (out / "refresh.json").write_text(json.dumps(dict(man, **pkg)))
    return man


# ---------------------------------------------------------------- node side

def _connect_rw(db: Path) -> sqlite3.Connection:
    c = sqlite3.connect(str(db), isolation_level=None, timeout=60)
    c.execute("PRAGMA busy_timeout=60000")
    return c


def _impl_ids(kdir: Path) -> set:
    p = kdir / "impl" / "index.jsonl"
    out = set()
    if p.exists():
        for line in p.read_text(errors="replace").splitlines():
            try:
                out.add(json.loads(line).get("id"))
            except ValueError:
                continue
    return out


def import_package(kdir, pkg_dir, *, fail_after: Optional[str] = None) -> dict:
    """Apply a refresh package to ``kdir/knowledge.db`` (+ ``impl/``) in one transaction. -> report with
    ``status`` applied | already_applied | up_to_date | refused | error. ``fail_after`` (tests): raise after
    that step, to check the rollback."""
    from ..claims import MAX_CITED_IDS
    kdir, pkg_dir = Path(kdir), Path(pkg_dir)
    db = kdir / "knowledge.db"
    try:
        pkg = json.loads((pkg_dir / "refresh.json").read_text())
    except (OSError, ValueError) as e:
        return {"status": "refused", "reason": f"unreadable package: {e!r}"}
    if pkg.get("format") != FORMAT:
        return {"status": "refused", "reason": f"format {pkg.get('format')}"}
    if not db.exists():
        return {"status": "refused", "reason": "no knowledge.db"}
    c = _connect_rw(db)
    idx = kdir / "impl" / "index.jsonl"
    idx_size = idx.stat().st_size if idx.exists() else None
    new_code: list = []
    rep = {"rid": pkg.get("rid"), "to": pkg.get("to"), "from": pkg.get("from")}
    try:
        seed = _meta(c, "gk_seed", {}) or {}
        st = _meta(c, "gk_refresh_state", None) or {"snapshot": seed.get("snapshot"), "seq": seed.get("merge_seq"),
                                                     "applied": []}
        if seed.get("run_uid") and pkg.get("run_uid") != seed.get("run_uid"):
            return dict(rep, status="refused", reason="package built for another run")
        if seed.get("part") and pkg.get("part") != seed.get("part"):
            return dict(rep, status="refused", reason=f"partition {pkg.get('part')} != {seed.get('part')}")
        if pkg.get("rid") in (st.get("applied") or []):
            return dict(rep, status="already_applied")
        if pkg.get("status") == "up_to_date":
            return dict(rep, status="up_to_date")
        if st.get("snapshot") and st["snapshot"] not in (pkg.get("lineage") or []):
            return dict(rep, status="refused", reason=f"the run's sync snapshot {st['snapshot']} is not in the "
                                                      "package's lineage (store rolled back?)")
        sync_seq = st.get("seq")
        if sync_seq is None:
            sync_seq = (pkg.get("from") or {}).get("seq") or 0
        to_seq = int((pkg.get("to") or {}).get("seq") or 0)
        if to_seq <= int(sync_seq):
            return dict(rep, status="up_to_date")
        if int((pkg.get("from") or {}).get("seq") or 0) > int(sync_seq):
            return dict(rep, status="refused", reason=f"package starts after the run's sync seq {sync_seq}")
        rid = pkg["rid"]
        c.executescript(REFRESH_SCHEMA)
        c.execute("BEGIN IMMEDIATE")
        logrows = []

        def log(kind, key, data):
            logrows.append((rid, kind, key if isinstance(key, str) else _j(key), _j(data)))

        counts = {"stats": 0, "claims": 0, "claims_unknown_state": 0, "evidence": 0, "outcomes": 0,
                  "outcomes_skipped": 0, "triggers": 0, "impl": 0}
        # statistics (exactly once per merge seq)
        for seq, key, (n, s, ss) in pkg.get("stats", []):
            if int(seq) <= int(sync_seq):
                continue
            c.execute("INSERT INTO stats VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (item, level, scope, op, state) "
                      "DO UPDATE SET n=n+excluded.n, s=s+excluded.s, ss=ss+excluded.ss", (*key, n, s, ss))
            log("stats", [seq] + list(key), [n, s, ss])
            counts["stats"] += 1
        if fail_after == "stats":
            raise RuntimeError("injected failure after stats")
        # claims
        have = {cid: json.loads(d) for cid, d in c.execute("SELECT id, data FROM claims")}
        states = {r[0] for r in c.execute("SELECT id FROM states")}
        cm = _meta(c, "claims_meta", {}) or {}
        epoch = int(cm.get("epoch", 0))
        for d in pkg.get("claims", []):
            cid = d.get("id")
            if not cid or cid in have:
                continue
            if d.get("state") and d["state"] not in states:
                counts["claims_unknown_state"] += 1
                continue
            d = dict(d, created_epoch=epoch, last_epoch=epoch)
            c.execute("INSERT INTO claims VALUES (?, ?, ?, ?)", (cid, d.get("status"), _j(d), d.get("updated_ts")))
            have[cid] = d
            log("claim", cid, d)
            counts["claims"] += 1
        for e in pkg.get("evidence", []):
            d = have.get(e["id"])
            if d is None:
                continue
            cited = set(d.get("support_ids", [])) | set(d.get("contradict_ids", []))
            ns_ = [i for i in e.get("sup", []) if i not in cited]
            nc_ = [i for i in e.get("con", []) if i not in cited and i not in ns_]
            if not ns_ and not nc_:
                continue
            d = _fold(d, ns_, nc_, MAX_CITED_IDS)
            c.execute("UPDATE claims SET data=?, status=? WHERE id=?", (_j(d), d.get("status"), e["id"]))
            have[e["id"]] = d
            log("evidence", e["id"], {"sup": ns_, "con": nc_})
            counts["evidence"] += 1
        if fail_after == "claims":
            raise RuntimeError("injected failure after claims")
        # outcomes: only where imported records cannot be reflected on (cursor at the end)
        n_out = c.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
        cursor = int(cm.get("reflect_cursor", 0))
        ucur = _meta(c, "upkeep_cursor", 0)
        rows = [r for r in pkg.get("outcomes", [])]
        if rows and cursor == n_out:
            have_o = {r[0] for r in c.execute("SELECT id FROM outcomes")}
            for oid, ts, rec, v2 in rows:
                if oid in have_o:
                    continue
                r = json.loads(rec)
                v = json.loads(v2) if v2 else {}
                c.execute("INSERT OR IGNORE INTO outcomes (id, ts, target_key, problem_id, slot, technique, item_id,"
                          " hypothesis_id, op_class, status, reward, state, mechanism, rec, v2)"
                          " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (oid, r.get("ts"), f"{r.get('target')}:{r.get('arch')}" if r.get("arch") else r.get("target"),
                           r.get("problem_id"), r.get("slot"), r.get("technique"), v.get("item_id") or r.get("technique"),
                           v.get("hypothesis_id"), r.get("op_class"), r.get("status"), v.get("reward"),
                           r.get("state_before"), (v.get("mechanism") or {}).get("label"), rec, v2 or None))
                log("outcome", oid, [oid, ts, rec, v2])
                counts["outcomes"] += 1
            n2 = c.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
            cm["reflect_cursor"] = n2
            _set_meta(c, "claims_meta", cm)
            if int(ucur or 0) == n_out:
                _set_meta(c, "upkeep_cursor", n2)
        elif rows:
            counts["outcomes_skipped"] = len(rows)
        # handled hypothesis triggers
        seen = set(_meta(c, "hyp_triggers_seen", []) or [])
        add = [k for k in pkg.get("triggers", []) if k not in seen]
        if add:
            _set_meta(c, "hyp_triggers_seen", sorted(seen | set(add)))
            for k in add:
                log("trigger", k, 1)
            counts["triggers"] = len(add)
        # implementation winners
        ids = _impl_ids(kdir)
        lines = []
        for r in pkg.get("impl", []):
            if r.get("id") in ids:
                continue
            src = pkg_dir / "impl" / "code" / f"{r.get('code_sha')}.cu"
            if not src.exists():
                continue
            dst = kdir / "impl" / "code" / src.name
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                new_code.append(dst)
            lines.append(json.dumps(r, sort_keys=True))
            ids.add(r.get("id"))
            log("impl", r["id"], 1)
            counts["impl"] += 1
        if lines:
            idx.parent.mkdir(parents=True, exist_ok=True)
            with open(idx, "a") as f:
                f.write("\n".join(lines) + "\n")
        if fail_after == "impl":
            raise RuntimeError("injected failure after impl")
        c.executemany("INSERT OR REPLACE INTO gk_refresh VALUES (?, ?, ?, ?)", logrows)
        st = {"snapshot": pkg["to"]["snapshot"], "seq": to_seq,
              "applied": ((st.get("applied") or []) + [rid])[-100:],
              "last": {"rid": rid, "counts": counts, "from_seq": int(sync_seq)}}
        _set_meta(c, "gk_refresh_state", st)
        c.execute("COMMIT")
        return dict(rep, status="applied", counts=counts, sync=st)
    except Exception as e:  # noqa: BLE001 -- roll back everything, report
        try:
            c.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        if idx_size is not None and idx.exists():
            with open(idx, "r+b") as f:
                f.truncate(idx_size)
        elif idx_size is None and idx.exists():
            idx.unlink()
        for p in new_code:
            p.unlink(missing_ok=True)
        return dict(rep, status="error", error=f"{type(e).__name__}: {e}")
    finally:
        c.close()


def _fold(d: dict, sup: list, con: list, cap: int) -> dict:
    d = dict(d)
    d["support_ids"] = (list(d.get("support_ids", [])) + list(sup))[-cap:]
    d["contradict_ids"] = (list(d.get("contradict_ids", [])) + list(con))[-cap:]
    d["support"] = int(d.get("support", 0)) + len(sup)
    d["contradict"] = int(d.get("contradict", 0)) + len(con)
    return d


# ---------------------------------------------------------------- merge side

def imported_rids(c: sqlite3.Connection) -> list:
    if "gk_refresh" not in _tables(c):
        return []
    return [r[0] for r in c.execute("SELECT rid FROM gk_refresh GROUP BY rid ORDER BY MIN(rowid)")]


def imported_impl_ids(c: sqlite3.Connection) -> set:
    if "gk_refresh" not in _tables(c):
        return set()
    return {r[0] for r in c.execute("SELECT key FROM gk_refresh WHERE kind='impl'")}


def adjust_base(base: sqlite3.Connection, run: sqlite3.Connection) -> dict:
    """Apply to a private copy of the merge base every refresh the run imported after that base was taken,
    so ``run - base`` is the run's own learning only. -> {rids, rows}."""
    from ..claims import MAX_CITED_IDS
    have = set(imported_rids(base))
    todo = [r for r in imported_rids(run) if r not in have]
    if not todo:
        return {"rids": 0, "rows": 0}
    base.executescript(REFRESH_SCHEMA)
    base.execute("BEGIN")
    n = 0
    trig = set(_meta(base, "hyp_triggers_seen", []) or [])
    for rid in todo:
        for kind, key, data in run.execute("SELECT kind, key, data FROM gk_refresh WHERE rid=? ORDER BY rowid", (rid,)):
            d = json.loads(data)
            if kind == "stats":
                k = json.loads(key)[1:]
                base.execute("INSERT INTO stats VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (item, level, scope, op, "
                             "state) DO UPDATE SET n=n+excluded.n, s=s+excluded.s, ss=ss+excluded.ss", (*k, *d))
            elif kind == "claim":
                base.execute("INSERT OR IGNORE INTO claims VALUES (?, ?, ?, ?)", (key, d.get("status"), _j(d),
                                                                                  d.get("updated_ts")))
            elif kind == "evidence":
                row = base.execute("SELECT data FROM claims WHERE id=?", (key,)).fetchone()
                if row:
                    cd = json.loads(row[0])
                    cited = set(cd.get("support_ids", [])) | set(cd.get("contradict_ids", []))
                    cd = _fold(cd, [i for i in d["sup"] if i not in cited], [i for i in d["con"] if i not in cited],
                               MAX_CITED_IDS)
                    base.execute("UPDATE claims SET data=?, status=? WHERE id=?", (_j(cd), cd.get("status"), key))
            elif kind == "outcome":
                oid, ts, rec, v2 = d
                r = json.loads(rec)
                base.execute("INSERT OR IGNORE INTO outcomes (id, ts, problem_id, technique, op_class, status, rec, v2)"
                             " VALUES (?,?,?,?,?,?,?,?)", (oid, ts, r.get("problem_id"), r.get("technique"),
                                                           r.get("op_class"), r.get("status"), rec, v2 or None))
            elif kind == "trigger":
                trig.add(key)
            n += 1
            base.execute("INSERT OR IGNORE INTO gk_refresh VALUES (?, ?, ?, ?)", (rid, kind, key, data))
    _set_meta(base, "hyp_triggers_seen", sorted(trig))
    base.execute("COMMIT")
    return {"rids": len(todo), "rows": n}
