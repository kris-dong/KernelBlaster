# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Implementation pools in the global store (opt-in: ``Policy.impl_pool`` / ``seed_pool``; A5).

The store's ``gk_impl`` keeps one winner per (op, signature, arch, dtype, shape set) -- 25,741 sweep records became
362 winners. A pool keeps, per (op, signature, arch, dtype), every useful / distinct verified kernel and its sparse
cross-shape matrix (``knowledge/impl_pool.py``), bounded by ``Policy.pool_max_kernels`` with usefulness + diversity
retention (``Pool.retain``). Tables (created only when the feature is used; a snapshot without them loads as before,
and code that does not know them ignores them)::

    gk_pool      (part, pkey, kid, data, prov)         kernel metadata (code_sha, op, signature, kind, lineage, tunables,
                                                       records, problems, techniques, best_speedup_vs_seed)
    gk_pool_obs  (part, pkey, kid, skey, data)         one matrix cell: {us, ok, n, n_fail, src, config, err}

Sources live in the store's content-addressed ``impl_code/`` (``GlobalStore.gc_code`` keeps pool references).
A seed (``materialize`` with ``seed_pool``) writes the partition's pools into the run's ``impl/pool/kernels.jsonl``
(kernel metadata with its cells) + ``impl/code``; ``ImplStore.pool`` reads them next to the run's own records.
A merge (``impl_pool``) adds the run's verified records (``speedup_vs_seed >= pool_min_speedup``) and its
``impl/pool/obs.jsonl`` observations (lines the base did not have) and re-applies retention to the touched pools.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

POOL_FORMAT = "kb-impl-pool/1"
POOL_SCHEMA = """
CREATE TABLE IF NOT EXISTS gk_pool (part TEXT, pkey TEXT, kid TEXT, data TEXT, prov TEXT, PRIMARY KEY (part, pkey, kid));
CREATE TABLE IF NOT EXISTS gk_pool_obs (part TEXT, pkey TEXT, kid TEXT, skey TEXT, data TEXT,
    PRIMARY KEY (part, pkey, kid, skey));
"""
_META_FIELDS = ("op", "signature_sha", "arch", "dtype", "target")


def _j(x) -> str:
    return json.dumps(x, sort_keys=True, default=str)


def has_pool(c) -> bool:
    return bool(c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='gk_pool'").fetchone())


def ensure(c) -> None:
    # statement by statement: executescript() would COMMIT the caller's open transaction
    for stmt in POOL_SCHEMA.split(";"):
        if stmt.strip():
            c.execute(stmt)
    c.execute("INSERT OR REPLACE INTO gk_meta VALUES (?, ?)", ("pool_format", json.dumps(POOL_FORMAT)))


def pool_keys(c, part: str) -> list:
    if not has_pool(c):
        return []
    return [r[0] for r in c.execute("SELECT DISTINCT pkey FROM gk_pool WHERE part=? ORDER BY pkey", (part,))]


def load(c, part: str, pkey: str):
    """The stored pool of ``pkey`` (a Pool; empty when absent)."""
    from ..impl_pool import Pool
    p = Pool()
    if not has_pool(c):
        return p
    for kid, data in c.execute("SELECT kid, data FROM gk_pool WHERE part=? AND pkey=?", (part, pkey)):
        d = json.loads(data)
        if not p.op:
            p.op, p.sig, p.arch, p.dtype, p.target = (d.get("op", ""), d.get("signature_sha", ""), d.get("arch", ""),
                                                      d.get("dtype", "fp32"), d.get("target", "cuda"))
        p.kernels[kid] = {k: v for k, v in d.items() if k not in _META_FIELDS}
    for kid, skey, data in c.execute("SELECT kid, skey, data FROM gk_pool_obs WHERE part=? AND pkey=?", (part, pkey)):
        p.obs.setdefault(kid, {})[skey] = json.loads(data)
        p.shapes.setdefault(skey, json.loads(skey))
    return p


def save(c, part: str, pool, prov: Optional[dict] = None) -> None:
    pk = pool.key
    c.execute("DELETE FROM gk_pool WHERE part=? AND pkey=?", (part, pk))
    c.execute("DELETE FROM gk_pool_obs WHERE part=? AND pkey=?", (part, pk))
    meta = {"op": pool.op, "signature_sha": pool.sig, "arch": pool.arch, "dtype": pool.dtype, "target": pool.target}
    for kid, k in sorted(pool.kernels.items()):
        c.execute("INSERT INTO gk_pool VALUES (?,?,?,?,?)", (part, pk, kid, _j({**k, **meta}), _j(prov or {})))
        for skey, cell in sorted((pool.obs.get(kid) or {}).items()):
            c.execute("INSERT INTO gk_pool_obs VALUES (?,?,?,?,?)", (part, pk, kid, skey, _j(cell)))


def _merge_into(dst, src) -> None:
    """Add ``src``'s kernels / cells to ``dst`` (cells: min time, failures summed, validity AND)."""
    for kid, k in src.kernels.items():
        d = dst.kernels.get(kid)
        if d is None:
            dst.kernels[kid] = json.loads(_j(k))
            continue
        for f in ("records", "problems", "validated"):
            for x in k.get(f) or []:
                if x not in d.setdefault(f, []):
                    d[f].append(x)
        for f in ("techniques", "tunables", "lineage", "kind", "stages", "composition"):
            if not d.get(f) and k.get(f):
                d[f] = k[f]
        if (k.get("best_speedup_vs_seed") or 0) > (d.get("best_speedup_vs_seed") or 0):
            d["best_speedup_vs_seed"] = k["best_speedup_vs_seed"]
        for sk, u in (k.get("set_us") or {}).items():
            d.setdefault("set_us", {})[sk] = min(u, d["set_us"].get(sk, u))
    for kid, row in src.obs.items():
        for sk, c in row.items():
            d = dst.obs.setdefault(kid, {}).get(sk)
            dst.shapes.setdefault(sk, src.shapes.get(sk) or json.loads(sk))
            if d is None:
                dst.obs[kid][sk] = dict(c)
                continue
            d["n"] = int(d.get("n", 0)) + int(c.get("n", 0))
            d["n_fail"] = int(d.get("n_fail", 0)) + int(c.get("n_fail", 0))
            d["ok"] = bool(d.get("ok", True)) and bool(c.get("ok", True))
            if c.get("us") is not None and (d.get("us") is None or c["us"] < d["us"]):
                d["us"] = c["us"]
                if c.get("config") is not None:
                    d["config"] = c["config"]
                d["src"] = c.get("src", d.get("src"))


def _records(d: Optional[Path]) -> list:
    if not d or not (Path(d) / "index.jsonl").exists():
        return []
    out = []
    for line in (Path(d) / "index.jsonl").read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _obs_lines(d: Optional[Path]) -> list:
    if not d or not (Path(d) / "pool" / "obs.jsonl").exists():
        return []
    out = []
    for line in (Path(d) / "pool" / "obs.jsonl").read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _has_code(store, impl_dir: Path, code_sha: str) -> bool:
    return (store.codedir / f"{code_sha}.cu").exists() or (Path(impl_dir) / "code" / f"{code_sha}.cu").exists()


def _copy_code(store, impl_dir: Path, code_sha: str) -> bool:
    src = Path(impl_dir) / "code" / f"{code_sha}.cu"
    dst = store.codedir / f"{code_sha}.cu"
    if dst.exists():
        return True
    if not src.exists():
        return False
    store.codedir.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(f".tmp{os.getpid()}")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return True


def merge_pool(g, store, impl_dirs: Iterable, base_impl_dir=None, *, part: str, pol, prov: Optional[dict] = None,
               exclude: Iterable = (), problem: Optional[str] = None) -> dict:
    """Add the runs' verified records + pool observations of partition ``part`` to the store's pools, then retain.
    ``impl_dirs``: one run's impl dir (a merge) or many (a bootstrap); ``base_impl_dir``: what the run started from
    (its records / observation lines are not the run's own)."""
    from ..impl_pool import Pool, kernel_id, pool_key
    from .merge import NS, ns, parse_part
    ensure(g)
    tk, dt = parse_part(part)
    arch = tk.split(":", 1)[-1]
    base_ids = {r.get("id") for r in _records(base_impl_dir)} | set(exclude)
    base_obs = {_j(o) for o in _obs_lines(base_impl_dir)}
    fresh: dict = {}                                     # pkey -> Pool of this merge's additions
    src_of: dict = {}                                    # code sha -> the impl dir holding it (copied if kept)
    n_rec = n_obs = n_missing = 0
    for d in impl_dirs:
        d = Path(d)
        recs = _records(d)
        for r in recs:
            if r.get("id") in base_ids or r.get("arch") != arch or (r.get("dtype") or "fp32") != dt:
                continue
            sp = r.get("speedup_vs_seed")
            if sp is None or sp < pol.pool_min_speedup:
                continue
            if not _has_code(store, d, r.get("code_sha") or ""):
                n_missing += 1
                continue
            src_of.setdefault(r["code_sha"], d)
            pk = pool_key(r["op"], r["signature_sha"], arch, dt, r.get("target", "cuda"))
            p = fresh.setdefault(pk, Pool(r["op"], r["signature_sha"], arch, dt, r.get("target", "cuda")))
            pname = problem or (d.parts[-4] if len(d.parts) >= 4 else None)
            rr = dict(r, problem_id=ns(pname, r.get("problem_id")) if pname and NS not in str(r.get("problem_id"))
                      else r.get("problem_id"))
            p.add_kernel(r["code_sha"], record=rr)
            n_rec += 1
        for o in _obs_lines(d):
            if _j(o) in base_obs or o.get("arch") != arch or (o.get("dtype") or "fp32") != dt:
                continue
            if not _has_code(store, d, o.get("code_sha") or ""):
                n_missing += 1
                continue
            src_of.setdefault(o["code_sha"], d)
            p = fresh.setdefault(o["pkey"], Pool(o["op"], o["signature_sha"], arch, dt, o.get("target", "cuda")))
            kid = kernel_id(o["code_sha"])
            if kid not in p.kernels:
                p.add_kernel(o["code_sha"])
            p.observe(kid, o["shape"], us=o.get("us"), ok=o.get("ok", True), src=o.get("src", "measured"),
                      config=o.get("config"), err=o.get("err"))
            n_obs += 1
    rep = {"records": n_rec, "observations": n_obs, "missing_code": n_missing, "pools": len(fresh), "kept": 0,
           "dropped": 0}
    for pk, p in sorted(fresh.items()):
        cur = load(g, part, pk)
        if not cur.op:
            cur.op, cur.sig, cur.arch, cur.dtype, cur.target = p.op, p.sig, p.arch, p.dtype, p.target
        _merge_into(cur, p)
        r = retain(store, cur, pol, src_of=src_of)
        rep["kept"] += len(r["kept"])
        rep["dropped"] += len(r["dropped"])
        for kid, k in cur.kernels.items():               # only kept kernels' sources enter impl_code/
            if not (store.codedir / f"{k['code_sha']}.cu").exists() and k["code_sha"] in src_of:
                _copy_code(store, src_of[k["code_sha"]], k["code_sha"])
        save(g, part, cur, prov)
    return rep


def retain(store, pool, pol, src_of: Optional[dict] = None) -> dict:
    """Pool.retain with sources from the store / the run dirs (diversity uses them); tunables from the source."""
    from ..impl_pool import SourceIndex
    from ..tunables import extract_tunables
    codes = {}
    for kid, k in pool.kernels.items():
        f = store.codedir / f"{k['code_sha']}.cu"
        if not f.exists() and src_of and k["code_sha"] in src_of:
            f = Path(src_of[k["code_sha"]]) / "code" / f"{k['code_sha']}.cu"
        if f.exists():
            codes[kid] = f.read_text(errors="replace")
            if "tunables" not in k:
                t = extract_tunables(codes[kid])
                k["tunables"] = t if t["tunable"] else None
    idx = SourceIndex(codes) if len(pool.kernels) > pol.pool_max_kernels else None
    return pool.retain(pol.pool_max_kernels, src_index=idx)


def seed_pool(g, store, part: str, out_impl: Path) -> dict:
    """Write the partition's pools into a seed's ``impl/pool/kernels.jsonl`` (+ code)."""
    if not has_pool(g):
        return {"pools": 0, "kernels": 0}
    out_impl = Path(out_impl)
    (out_impl / "code").mkdir(parents=True, exist_ok=True)
    (out_impl / "pool").mkdir(parents=True, exist_ok=True)
    n = miss = 0
    keys = pool_keys(g, part)
    with open(out_impl / "pool" / "kernels.jsonl", "a") as f:
        for pk in keys:
            p = load(g, part, pk)
            for kid, k in sorted(p.kernels.items()):
                src = store.codedir / f"{k['code_sha']}.cu"
                if not src.exists():
                    miss += 1
                    continue
                dst = out_impl / "code" / src.name
                if not dst.exists():
                    shutil.copyfile(src, dst)
                meta = {kk: v for kk, v in k.items() if kk not in ("validated", "set_us")}
                meta["obs"] = p.obs.get(kid) or {}
                f.write(_j({"pkey": pk, "op": p.op, "signature_sha": p.sig, "arch": p.arch, "dtype": p.dtype,
                            "target": p.target, "code_sha": k["code_sha"], "meta": meta}) + "\n")
                n += 1
    return {"pools": len(keys), "kernels": n, "missing_code": miss}


def code_refs(c) -> set:
    if not has_pool(c):
        return set()
    return {r[0] for r in c.execute("SELECT json_extract(data, '$.code_sha') FROM gk_pool")}


def build_from_runs(store, part: str, impl_dirs: list, *, policy=None, label: str = "pool bootstrap",
                    from_gk_impl: bool = True) -> dict:
    """Migration / bootstrap: a new snapshot = current + pools built from ``gk_impl`` winners (``from_gk_impl``)
    and the given run impl dirs (e.g. every ``results/*/work/knowledge/impl`` of a sweep). Under the store lock,
    with a backup; the current snapshot is never modified (a new version is committed)."""
    import contextlib
    from .merge import Policy, _connect, _j as mj, _now
    pol = policy or Policy(impl_pool=True)
    with store.lock():
        cur = store.current_id()
        store._backup(f"{label}")
        tmp = store._start_from(cur)
        g = _connect(tmp)
        try:
            g.execute("BEGIN IMMEDIATE")
            ensure(g)
            rep = {"from_gk_impl": 0}
            if from_gk_impl:
                import tempfile
                with tempfile.TemporaryDirectory(prefix="gkpool_") as td:
                    tdir = Path(td) / "impl"
                    (tdir / "code").mkdir(parents=True)
                    with open(tdir / "index.jsonl", "w") as f:
                        for (data,) in g.execute("SELECT data FROM gk_impl WHERE part=?", (part,)):
                            d = json.loads(data)
                            src = store.codedir / f"{d.get('code_sha')}.cu"
                            if src.exists():
                                if not (tdir / "code" / src.name).exists():
                                    os.symlink(src, tdir / "code" / src.name)
                                f.write(json.dumps(d) + "\n")
                                rep["from_gk_impl"] += 1
                    rep["gk_impl"] = merge_pool(g, store, [tdir], part=part, pol=pol, prov={"source": "gk_impl"})
            dirs = [Path(d) for d in impl_dirs]
            rep["runs"] = merge_pool(g, store, dirs, part=part, pol=pol, prov={"source": label}) if dirs else {}
            g.execute("INSERT INTO gk_log (ts, kind, ref, part, data) VALUES (?,?,?,?,?)",
                      (_now(), "pool_build", label, part, mj(rep)))
            vid_next = store._next_id()
            from .refresh import lineage
            from .merge import _meta, _set_meta
            lin = _meta(g, "lineage", None, "gk_meta")
            if lin is None:
                lin = lineage(store, cur)
            _set_meta(g, "lineage", (list(lin) + [vid_next])[-2000:], "gk_meta")
            g.execute("COMMIT")
        except BaseException:
            with contextlib.suppress(Exception):
                g.execute("ROLLBACK")
            g.close()
            tmp.unlink(missing_ok=True)
            raise
        g.close()
        vid = store._commit(tmp, parent=cur, label=label, extra={"pool_build": rep})
        with open(store.root / "history.jsonl", "a") as f:
            f.write(mj({"ts": _now(), "op": "pool_build", "version": vid, "parent": cur, "label": label}) + "\n")
        rep.update(status="built", version=vid, parent=cur)
        return rep
