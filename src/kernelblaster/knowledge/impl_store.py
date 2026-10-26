# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Implementation knowledgebase: verified kernel sources, shared across problems.

Every slot implementation that passes the driver's partition and isolated
checks is stored (code content-addressed, one JSONL index line per record)
with what it implements (target, arch, op, signature, shapes, op class), how
fast it was, which techniques produced it and what it was derived from.

Every tier (and the winners) is restricted to the asking slot's storage
dtype: an fp16 kernel is a different implementation of the same op, and its
code and timings say nothing reliable about the fp32 one (or vice versa).

Retrieval for a slot (see :meth:`ImplStore.retrieve`):
  seed     same op + signature + shape bucket on this arch: start from it
  adapt    same op + signature, other shapes: start from it, say what changed
  examples best records of the same op class (other ops): few-shot context

ModelBlaster's write-back reads :meth:`ImplStore.winners`.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _mag(v) -> str:
    """Order-of-magnitude bucket (powers of 4) of a positive size."""
    v = max(int(v), 1)
    return f"{4 ** int(math.log(v, 4)):d}"


# shape keys that set a kernel's work, per op class; others ignored for bucketing
_BUCKET_KEYS = {
    "gemm": ("M", "N", "K", "batch"),
    "conv": ("N", "IC", "OC", "OH", "OW", "KH", "KW", "G"),
    "attention": ("BH", "S", "D"),
    "recurrent": ("T", "B", "I", "H"),          # gru_seq / lstm_seq with ModelBlaster --recurrent-class on
}


def shape_bucket(op_class: str, shape: dict) -> str:
    """Coarse shape key: winners usually transfer within a bucket.

    GEMM/conv/attention use their work dimensions; everything else the total
    element count (the shape dict's ``n`` or the product of its dims). A
    fused op ({"a": stage 0 shape, ...}) is bucketed by its first stage.
    """
    if isinstance(shape.get("a"), dict):
        shape = shape["a"]
    keys = _BUCKET_KEYS.get(op_class)
    if keys:
        parts = [f"{k}{_mag(shape[k])}" for k in keys if k in shape]
        if parts:
            return op_class + ":" + ",".join(parts)
    n = shape.get("n")
    if n is None:
        ints = [v for v in shape.values() if isinstance(v, int) and v > 0]
        n = math.prod(ints) if ints else 1
    return f"{op_class}:n{_mag(n)}"


def signature_sha(signature: str) -> str:
    return hashlib.sha1(" ".join(signature.split()).encode()).hexdigest()[:12]


@dataclass
class ImplRecord:
    target: str
    arch: str
    op: str
    op_class: str
    signature_sha: str
    code_sha: str
    shapes: list                                 # instance shape dicts it was verified on
    buckets: list                                # their shape_bucket()s
    slot_us: float                               # summed over the instances, on `arch`
    baseline_us: Optional[float] = None          # its parent's time (the step it improved on)
    speedup: Optional[float] = None              # baseline_us / slot_us (vs parent)
    seed_us: Optional[float] = None              # the problem's seed (hand kernel) time
    speedup_vs_seed: Optional[float] = None      # seed_us / slot_us -- what write-back filters on
    techniques: list = field(default_factory=list)   # lineage, oldest first
    parent_id: Optional[str] = None
    problem_id: Optional[str] = None
    run_id: Optional[str] = None
    slot: Optional[str] = None
    max_rel_err: Optional[float] = None
    notes: str = ""
    dtype: str = "fp32"                          # storage dtype; records never cross dtypes
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


_FIELDS = None


def _record(d: dict) -> ImplRecord:
    """An index line -> ImplRecord; keys this version does not know (newer writers) are dropped."""
    global _FIELDS
    if _FIELDS is None:
        from dataclasses import fields
        _FIELDS = {f.name for f in fields(ImplRecord)}
    return ImplRecord(**{k: v for k, v in d.items() if k in _FIELDS})


class ImplStore:
    """``<root>/index.jsonl`` + ``<root>/code/<sha>.cu``. Append-only; thread-safe.

    Optional (A5, additive): ``<root>/pool/obs.jsonl`` (cross-shape observations) and ``pool/kernels.jsonl``
    (kernel metadata: structural lineage, tunables, links) behind :meth:`pool`, :meth:`observe`,
    :meth:`annotate`, :meth:`retrieve_similar`, :meth:`retrieve_structural`; ``retrieve(similar_k=0)`` is the
    unchanged tier retrieval unless ``similar_k`` > 0. Stores without ``pool/`` load as before."""

    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "code").mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"
        self._lock = threading.Lock()

    # ---- write
    def add(self, code: str, **kw) -> ImplRecord:
        sha = hashlib.sha1(code.encode()).hexdigest()
        p = self.root / "code" / f"{sha}.cu"
        if not p.exists():
            p.write_text(code)
        rec = ImplRecord(code_sha=sha, **kw)
        with self._lock, open(self.index_path, "a") as f:
            f.write(json.dumps(asdict(rec), sort_keys=True) + "\n")
        return rec

    # ---- read
    def records(self) -> list[ImplRecord]:
        if not self.index_path.exists():
            return []
        out = []
        for line in self.index_path.read_text().splitlines():
            if line.strip():
                out.append(_record(json.loads(line)))
        return out

    def code(self, rec: ImplRecord) -> str:
        return (self.root / "code" / f"{rec.code_sha}.cu").read_text()

    def get(self, rec_id: str) -> Optional[ImplRecord]:
        return next((r for r in self.records() if r.id == rec_id), None)

    def retrieve(self, *, target: str, arch: str, op: str, op_class: str, signature: str,
                 shapes: list, n_examples: int = 2, exclude_problem: Optional[str] = None,
                 far_same_op: bool = False, dtype: str = "fp32", similar_k: int = 0) -> dict:
        """-> {"seed": rec|None, "adapt": rec|None, "examples": [rec, ...]}

        ``exclude_problem`` drops the asking problem's own records: during a
        run they are its current state, and its baseline seeds would
        otherwise always win the exact tier and hide other problems' work.

        ``similar_k`` > 0 (opt-in) also returns ``"similar"``: :meth:`retrieve_similar`'s top k (pool-based
        nearest-shape / cross-shape ranking), and when neither the seed nor the adapt tier found anything,
        its top record becomes ``adapt`` (``adapt_from: "similar"``). 0 = the unchanged tiers."""
        out = self._retrieve_tiers(target=target, arch=arch, op=op, op_class=op_class, signature=signature,
                                   shapes=shapes, n_examples=n_examples, exclude_problem=exclude_problem,
                                   far_same_op=far_same_op, dtype=dtype)
        if similar_k and similar_k > 0:
            sim = self.retrieve_similar(target=target, arch=arch, op=op, signature=signature, shapes=shapes,
                                        k=similar_k, exclude_problem=exclude_problem, dtype=dtype)
            out["similar"] = sim
            recs = [x["record"] for x in sim if isinstance(x["record"], ImplRecord)]
            if out["seed"] is None and out["adapt"] is None and recs:
                out["adapt"] = recs[0]                    # an index record (callers read .shapes / code(rec))
                out["adapt_from"] = "similar"
        return out

    def _retrieve_tiers(self, *, target: str, arch: str, op: str, op_class: str, signature: str,
                        shapes: list, n_examples: int = 2, exclude_problem: Optional[str] = None,
                        far_same_op: bool = False, dtype: str = "fp32") -> dict:
        sig = signature_sha(signature)
        want = {shape_bucket(op_class, s) for s in shapes}
        recs = [r for r in self.records() if r.target == target and r.dtype == dtype
                and (exclude_problem is None or r.problem_id != exclude_problem)]
        same_op = [r for r in recs if r.op == op and r.signature_sha == sig]
        exact = [r for r in same_op if r.arch == arch and want <= set(r.buckets)]
        seed = min(exact, key=lambda r: r.slot_us / max(len(r.shapes), 1), default=None)
        # adapt only from records that share a shape bucket with this slot: a
        # same-op winner tuned for unrelated shapes anchored the LLM on the
        # wrong blocking (ResNet18 given a ResNet-block conv: 5.4x as an adapt
        # seed, 7.5x as an ideas-only example, 25x without; one run each), so
        # such records are only offered (as examples) when far_same_op is set
        others = [r for r in same_op if r is not seed and (r.speedup_vs_seed or 0) > 1.05]
        near = [r for r in others if want & set(r.buckets)]
        adapt = None if seed else max(near, key=lambda r: (r.arch == arch, r.speedup_vs_seed or 0), default=None)
        far = [] if seed or adapt or not far_same_op else \
            sorted(others, key=lambda r: -(r.speedup_vs_seed or 0))[:1]
        ex = [r for r in recs if r.op_class == op_class and r.op != op and (r.speedup_vs_seed or 0) > 1.05]
        best_per_op = {}
        for r in sorted(ex, key=lambda r: -(r.speedup_vs_seed or 0)):
            best_per_op.setdefault(r.op, r)
        return {"seed": seed, "adapt": adapt,
                "examples": (far + list(best_per_op.values()))[:n_examples]}

    def winners(self, *, target: str = "cuda", arch: Optional[str] = None) -> list[ImplRecord]:
        """Fastest verified record per (op, signature, arch, dtype, shape set)."""
        best: dict = {}
        for r in self.records():
            if r.target != target or (arch and r.arch != arch):
                continue
            k = (r.op, r.signature_sha, r.arch, r.dtype, json.dumps(r.shapes, sort_keys=True))
            if k not in best or r.slot_us < best[k].slot_us:
                best[k] = r
        return list(best.values())

    def export_winners(self, out: Path, *, target: str = "cuda", arch: Optional[str] = None,
                       min_speedup: float = 1.05) -> Path:
        """Write ``winners.json`` + sources for ModelBlaster's ingest (records
        at least ``min_speedup`` over their problem's seed; seeds excluded)."""
        out = Path(out)
        (out / "code").mkdir(parents=True, exist_ok=True)
        rows = []
        for r in self.winners(target=target, arch=arch):
            if r.speedup_vs_seed is None or r.speedup_vs_seed < min_speedup:
                continue
            (out / "code" / f"{r.code_sha}.cu").write_text(self.code(r))
            rows.append({**asdict(r), "file": f"code/{r.code_sha}.cu"})
        (out / "winners.json").write_text(json.dumps({"format": "kb-winners/1", "winners": rows}, indent=1))
        return out / "winners.json"

    # ------------------------------------------------------------------ pool / similarity (A5, opt-in)
    @property
    def pool_dir(self) -> Path:
        return self.root / "pool"

    def _jsonl(self, name: str) -> list:
        p = self.pool_dir / name
        if not p.exists():
            return []
        out = []
        for line in p.read_text(errors="replace").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def _append(self, name: str, row: dict) -> None:
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self.pool_dir / name, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def _put_code(self, code: str) -> str:
        sha = hashlib.sha1(code.encode()).hexdigest()
        p = self.root / "code" / f"{sha}.cu"
        if not p.exists():
            p.write_text(code)
        return sha

    def code_by_sha(self, code_sha: str) -> Optional[str]:
        p = self.root / "code" / f"{code_sha}.cu"
        return p.read_text() if p.exists() else None

    def observe(self, op: str, signature_sha_: str, arch: str, dtype: str, code_sha: Optional[str], shape: dict,
                us: Optional[float] = None, ok: bool = True, *, config=None, err: Optional[str] = None,
                src: str = "measured", problem: Optional[str] = None, code: Optional[str] = None,
                target: str = "cuda") -> dict:
        """Record one cross-shape measurement of a kernel (e.g. kb_reuse's matrix runs, the family tuner):
        appended to ``pool/obs.jsonl``; ``code`` (optional) is stored content-addressed."""
        from .impl_pool import pool_key, shape_key
        if code is not None:
            code_sha = self._put_code(code)
        if not code_sha:
            raise ValueError("observe needs code_sha or code")
        row = {"pkey": pool_key(op, signature_sha_, arch, dtype, target), "op": op, "signature_sha": signature_sha_,
               "arch": arch, "dtype": dtype or "fp32", "target": target, "code_sha": code_sha, "shape": shape,
               "skey": shape_key(shape), "us": us, "ok": bool(ok), "config": config, "err": (err or None) and str(err)[:300],
               "src": src, "problem": problem,
               "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self._append("obs.jsonl", row)
        return row

    def annotate(self, op: str, signature_sha_: str, arch: str, dtype: str, code_sha: str, *,
                 target: str = "cuda", **meta) -> dict:
        """Kernel metadata (structural lineage ``lineage={...}``, ``tunables``, links, ``kind`` ...): appended to
        ``pool/kernels.jsonl``; later lines update earlier ones."""
        from .impl_pool import pool_key
        row = {"pkey": pool_key(op, signature_sha_, arch, dtype, target), "op": op, "signature_sha": signature_sha_,
               "arch": arch, "dtype": dtype or "fp32", "target": target, "code_sha": code_sha, "meta": meta}
        self._append("kernels.jsonl", row)
        return row

    def pool(self, op: str, signature_sha_: str, arch: str, dtype: str = "fp32", *, target: str = "cuda",
             exclude_problem: Optional[str] = None, with_code: bool = False, records: Optional[list] = None):
        """The :class:`~kernelblaster.knowledge.impl_pool.Pool` of (op, signature, arch, dtype): this store's
        verified records (each a measurement on its own shapes) + ``pool/kernels.jsonl`` kernels (e.g. seeded
        from the global pool) + ``pool/obs.jsonl`` observations. ``signature_sha_`` may also be a full
        signature. ``exclude_problem`` drops that problem's records and observations."""
        from .impl_pool import Pool, kernel_id, pool_key
        if "(" in signature_sha_ or len(signature_sha_) != 12:
            signature_sha_ = signature_sha(signature_sha_)
        pk = pool_key(op, signature_sha_, arch, dtype, target)
        recs = records if records is not None else self.records()
        p = Pool(op, signature_sha_, arch, dtype, target)
        for r in recs:
            if (r.op, r.signature_sha, r.arch, r.dtype or "fp32", r.target) != (op, signature_sha_, arch, dtype or "fp32", target):
                continue
            if exclude_problem is not None and r.problem_id == exclude_problem:
                continue
            p.add_kernel(r.code_sha, record=asdict(r), code=self.code_by_sha(r.code_sha) if with_code else None)
        for row in self._jsonl("kernels.jsonl"):
            if row.get("pkey") != pk:
                continue
            meta = dict(row.get("meta") or {})
            if exclude_problem is not None and exclude_problem in (meta.get("problems") or []) and \
                    len(meta.get("problems") or []) == 1:
                continue
            obs = meta.pop("obs", None)                 # a seeded global pool kernel carries its cells
            kid = p.add_kernel(row["code_sha"], code=self.code_by_sha(row["code_sha"]) if with_code else None,
                               **{k: v for k, v in meta.items() if k not in ("records", "validated", "set_us", "code_sha",
                                                                              "problems")})
            k = p.kernels[kid]
            for f in ("records", "problems"):
                for x in meta.get(f) or []:
                    if x not in k[f]:
                        k[f].append(x)
            for sk, c in (obs or {}).items():
                shape = json.loads(sk)
                for _ in range(max(int(c.get("n_fail", 0)), 0)):
                    p.observe(kid, shape, ok=False, src=c.get("src", "global"), err=c.get("err"))
                if c.get("ok", True):
                    p.observe(kid, shape, us=c.get("us"), ok=True, src=c.get("src", "global"), config=c.get("config"))
        for row in self._jsonl("obs.jsonl"):
            if row.get("pkey") != pk or (exclude_problem is not None and row.get("problem") == exclude_problem):
                continue
            kid = kernel_id(row["code_sha"])
            if kid not in p.kernels:
                p.add_kernel(row["code_sha"], code=self.code_by_sha(row["code_sha"]) if with_code else None)
            p.observe(kid, row["shape"], us=row.get("us"), ok=row.get("ok", True), src=row.get("src", "measured"),
                      config=row.get("config"), err=row.get("err"))
        return p

    def retrieve_similar(self, *, target: str = "cuda", arch: str, op: str, shapes: list,
                         signature: Optional[str] = None, sig_sha: Optional[str] = None, op_class: Optional[str] = None,
                         k: int = 4, exclude_problem: Optional[str] = None, dtype: str = "fp32",
                         query_code: Optional[str] = None, w_src: float = 0.0, profile: Optional[dict] = None,
                         w_prof: float = 0.0, profiles: Optional[dict] = None, links=None, **rank_kw) -> list:
        """Pool-based ranking of this op+signature's kernels for ``shapes`` (all instances of the slot):
        measured cells on the very shapes first, then predicted slowdown from nearby shapes (see
        ``impl_pool.rank_kernels``; several shapes: mean log ratio). -> [{record, kernel, pred_ratio, score, why}]
        (one record per kernel: its fastest per instance; ``record`` is an ImplRecord when the kernel has one
        in this store's index, else a dict). ``links`` (an ``ImplLinks``) adds negative evidence: kernels with
        failed / regressed outcomes on these shape buckets are pushed down."""
        import math as _m
        from .impl_pool import rank_kernels
        sig = sig_sha or signature_sha(signature or "")
        recs = self.records()
        p = self.pool(op, sig, arch, dtype, target=target, exclude_problem=exclude_problem,
                      with_code=bool(query_code and w_src), records=recs)
        if not p.kernels or not shapes:
            return []
        agg: dict = {}
        for s in shapes:
            for x in rank_kernels(p, s, query_code=query_code, w_src=w_src, profile=profile, w_prof=w_prof,
                                  profiles=profiles, **rank_kw):
                a = agg.setdefault(x["kernel"], {"lr": 0.0, "score": 0.0, "why": x["why"]})
                a["lr"] += _m.log(x["pred_ratio"]) / len(shapes)
                a["score"] += x["score"] / len(shapes)
        if links is not None:
            for kid, a in agg.items():
                pen = links.penalty(p.kernels[kid], shapes=shapes)
                if pen:
                    a["score"] += pen
                    a["why"] += f"; negative evidence +{pen:.2f}"
        by_kid: dict = {}
        for r in recs:
            if (r.op, r.signature_sha, r.arch, r.dtype or "fp32") != (op, sig, arch, dtype or "fp32"):
                continue
            if exclude_problem is not None and r.problem_id == exclude_problem:
                continue
            kid = r.code_sha[:16]
            per = r.slot_us / max(len(r.shapes), 1)
            if kid not in by_kid or per < by_kid[kid][0]:
                by_kid[kid] = (per, r)
        ranked = sorted(agg.items(), key=lambda kv: (kv[1]["score"], kv[0]))[:max(0, int(k))]
        out = []
        for kid, a in ranked:
            km = p.kernels[kid]
            rec = by_kid.get(kid, (None, None))[1]
            if rec is None:
                rec = {"code_sha": km["code_sha"], "op": op, "signature_sha": sig, "arch": arch, "dtype": dtype,
                       "shapes": [p.shapes[s] for s in km["validated"][:4] if s in p.shapes], "id": f"pool:{kid}",
                       "problem_id": (km.get("problems") or [None])[-1], "techniques": km.get("techniques") or []}
            out.append({"record": rec, "kernel": kid, "pred_ratio": round(_m.exp(a["lr"]), 4),
                        "score": round(a["score"], 4), "why": a["why"], "kind": km.get("kind"),
                        "tunables": bool(km.get("tunables"))})
        return out

    def lineage(self, rec_id: str, records: Optional[list] = None, limit: int = 64) -> list:
        """The record's ancestors in this store, oldest first, ending with the record (parent_id chain; a fused
        kernel's chain starts at its composite seed)."""
        by = {r.id: r for r in (records if records is not None else self.records())}
        chain, cur, seen = [], by.get(rec_id), set()
        while cur is not None and cur.id not in seen and len(chain) < limit:
            chain.append(cur)
            seen.add(cur.id)
            cur = by.get(cur.parent_id) if cur.parent_id else None
        return list(reversed(chain))

    def retrieve_structural(self, *, target: str = "cuda", arch: str, op: str, shapes: list,
                            signature: Optional[str] = None, dtype: str = "fp32", k: int = 3,
                            exclude_problem: Optional[str] = None) -> dict:
        """Structural (fused / sibling / attention / layout) kernels as first-class records:
        ``same`` (this op, :meth:`retrieve_similar`), ``composition`` (other ops with the same stage composition,
        e.g. another fusion argument tag, best per op), ``stages`` (per stage op: the best records -- the
        ingredients of a composite seed), ``lineage`` (the top same-op record's chain back to its seed)."""
        from .impl_pool import struct_info
        info = struct_info(op)
        recs = [r for r in self.records() if r.target == target and (r.dtype or "fp32") == (dtype or "fp32")
                and (exclude_problem is None or r.problem_id != exclude_problem)]
        out = {"info": info, "same": [], "composition": [], "stages": {}, "lineage": []}
        if signature is not None:
            out["same"] = self.retrieve_similar(target=target, arch=arch, op=op, signature=signature, shapes=shapes,
                                                k=k, exclude_problem=exclude_problem, dtype=dtype)
        best_op: dict = {}
        for r in recs:
            if r.op == op or r.arch != arch or struct_info(r.op)["composition"] != info["composition"]:
                continue
            if r.op not in best_op or (r.speedup_vs_seed or 0) > (best_op[r.op].speedup_vs_seed or 0):
                best_op[r.op] = r
        out["composition"] = sorted(best_op.values(), key=lambda r: -(r.speedup_vs_seed or 0))[:k]
        if info["kind"] in ("fused", "sibling") and len(info["stages"]) > 1:
            from .impl_pool import shape_distance
            tags = "abcdefghij"
            for i, st in enumerate(info["stages"]):
                want = [s.get(tags[i]) for s in shapes if isinstance(s.get(tags[i]), dict)] if shapes else []
                cands = [r for r in recs if r.op == st and r.arch == arch]
                def d(r):
                    if not want or not r.shapes:
                        return 0.0
                    return min(shape_distance(w, x) for w in want for x in r.shapes)
                cands.sort(key=lambda r: (round(d(r), 3), r.slot_us / max(len(r.shapes), 1)))
                out["stages"][st] = cands[:k]
        top = out["same"][0]["record"] if out["same"] else None
        if top is not None and hasattr(top, "id"):
            out["lineage"] = self.lineage(top.id, records=recs)
        return out
