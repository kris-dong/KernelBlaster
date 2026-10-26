# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Priors for STRUCTURAL outcomes (graph-level decisions: fusion, split, layout fold, attention fusion, sibling
fusion, epilogues) across runs -- the ``prior()`` API for graph search (A4). Opt-in (A5).

Data: every ModelBlaster round records its graph decisions in ``state.json`` ``rounds[].decisions`` (hint id
``<kind>:<detail>``, status accepted / rejected, paired gain, original / variant time). The Level 3 sweep had
139 (fuse 136, split 3), e.g. bn+relu 6/6 accepted, conv+relu 4/8. A row per decision::

    {id, part, kind, context: [op kinds], composition, classes, status, accepted, gain, original_us, variant_us,
     round, first_round, rounds_to_decide, problem, run_uid, hint}

``StructPriors.prior(kind, context, part=None)`` -> :class:`Prior` with counts AND intervals, so a caller can explore
(Thompson ``sample``, ``ucb``) instead of trusting a point rate. Backoff hierarchy, most specific first:
``exact`` (kind + op-kind composition, e.g. fuse conv2d+batchnorm2d+relu) -> ``classes`` (kind + op-class
composition, conv+norm+ewise) -> ``head`` (kind + class of the first stage, conv) -> ``kind`` -> ``all``. The Beta
posterior of a level is shrunk towards its parent level's rate with ``strength`` pseudo-observations, so a context
seen once does not get a 0% / 100% prior.

Store: table ``gk_struct`` (created only when used; ``Policy.struct_outcomes`` merges a run's state.json,
``Policy.seed_struct`` ships ``struct_priors.json`` in a seed; ``merge struct-ingest`` bootstraps from work dirs).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

FORMAT = "kb-struct-priors/1"
STRUCT_SCHEMA = "CREATE TABLE IF NOT EXISTS gk_struct (part TEXT, id TEXT, data TEXT, PRIMARY KEY (part, id));"
Z90 = 1.6449

_CLASS = [(r"^conv", "conv"), (r"^(linear|matmul|bmm|gemm|addmm|mm)", "gemm"),
          (r"^(batchnorm|layernorm|groupnorm|instancenorm|rmsnorm|norm)", "norm"),
          (r"^(relu|relu6|sigmoid|gelu|silu|tanh|hardswish|swish|elu|leaky|mish|add|mul|sub|div|ewise|scale|mul_scalar|"
           r"clamp|dropout|exp)", "ewise"),
          (r"^cat", "cat"), (r"^(maxpool|avgpool|adaptive|pool)", "pool"), (r"^softmax", "softmax"),
          (r"attention", "attention"), (r"^(transpose|permute|strided|copy|view|reshape|contig)", "layout"),
          (r"^(reduce|sum|mean)", "reduce")]


def _base(op: str) -> str:
    return re.sub(r"(_(f16|f32|bf16|fp16|fp32|i8))+$", "", op)


def op_class(op: str) -> str:
    b = _base(op)
    for rx, c in _CLASS:
        if re.search(rx, b):
            return c
    return b


def parse_hint(hint: str) -> dict:
    """``fuse:fused__conv2d_f16__relu_f16`` -> {kind: fuse, context: [conv2d, relu], composition, classes, head}."""
    kind, _, detail = (hint or "").partition(":")
    kind = kind or "unknown"
    if detail.startswith("fused__"):
        parts = [p for p in detail.split("__")[1:] if p]
        if parts and parts[0] == "par":
            parts = parts[1:]
            if kind == "fuse":
                kind = "fuse_siblings"
        ctx = [_base(p) for p in parts if len(p) > 1]
    elif detail:
        ctx = [_base(x) for x in re.split(r"[+,|]", detail) if x]
    else:
        ctx = []
    cls = [op_class(c) for c in ctx]
    return {"kind": kind, "context": ctx, "composition": "+".join(ctx), "classes": "+".join(cls),
            "head": cls[0] if cls else ""}


def rows_from_state(state, *, part: str, problem: Optional[str] = None, run_uid: Optional[str] = None) -> list:
    """Decision rows of one ModelBlaster ``state.json`` (dict or path)."""
    if not isinstance(state, dict):
        try:
            state = json.loads(Path(state).read_text())
        except (OSError, ValueError):
            return []
    out, first = [], {}
    for ri, r in enumerate(state.get("rounds") or []):
        if not isinstance(r, dict):
            continue
        rnd = r.get("round", ri)
        for d in r.get("decisions") or []:
            h = d.get("hint")
            if not h:
                continue
            first.setdefault(h, rnd)
            st = d.get("status") or d.get("hint_status")
            if st not in ("accepted", "rejected"):
                continue                                  # deferred / pending: not an outcome yet
            try:
                gain = float(d.get("gain")) if d.get("gain") is not None else None
            except (TypeError, ValueError):
                gain = None
            ph = parse_hint(h)
            rid = hashlib.sha1(f"{run_uid}|{problem}|{rnd}|{h}".encode()).hexdigest()[:16]
            out.append({"id": rid, "part": part, **ph, "status": st, "accepted": st == "accepted", "gain": gain,
                        "original_us": d.get("original_us"), "variant_us": d.get("variant_us"), "round": rnd,
                        "first_round": first[h], "rounds_to_decide": rnd - first[h] + 1,
                        "steps": d.get("steps") or d.get("llm_calls"), "problem": problem, "run_uid": run_uid,
                        "hint": h, "n_ops": len(d.get("ops") or [])})
    return out


def _wilson(k: float, n: float, z: float = Z90) -> tuple:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


@dataclass
class Prior:
    kind: str
    context: list
    level: str                       # exact | classes | head | kind | all | none
    n: int = 0                       # decisions at that level
    accepted: int = 0
    rejected: int = 0
    rate: Optional[float] = None     # posterior mean acceptance (shrunk to the parent level)
    lo: float = 0.0                  # Wilson 90% interval of the raw rate at that level
    hi: float = 1.0
    alpha: float = 1.0               # Beta posterior (for Thompson sampling)
    beta: float = 1.0
    gain_mean: Optional[float] = None    # paired gain (fraction of time saved) of ACCEPTED decisions
    gain_lo: Optional[float] = None
    gain_hi: Optional[float] = None
    gain_all_mean: Optional[float] = None  # over all decisions (rejected ones have <= 0 gains)
    rounds_to_decide: Optional[float] = None
    steps_mean: Optional[float] = None
    levels: dict = field(default_factory=dict)   # counts at every level, for transparency

    def sample(self, rng: Optional[random.Random] = None) -> float:
        return (rng or random).betavariate(self.alpha, self.beta)

    def ucb(self, c: float = 1.0) -> float:
        a, b = self.alpha, self.beta
        return a / (a + b) + c * math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

    def to_dict(self) -> dict:
        return asdict(self)


class StructPriors:
    LEVELS = ("exact", "classes", "head", "kind", "all")

    def __init__(self, rows: Iterable = ()):
        self.rows = [r for r in rows if r.get("status") in ("accepted", "rejected")]

    # ---- loading
    @classmethod
    def load(cls, src=None, *, part: Optional[str] = None) -> "StructPriors":
        """``src``: a ``struct_priors.json`` / a dir holding one (a seed dir, a knowledge dir) / a GlobalStore /
        a list of row dicts / a list of ModelBlaster work dirs (their state.json)."""
        if src is None:
            return cls([])
        if isinstance(src, (list, tuple)):
            rows = []
            for x in src:
                if isinstance(x, dict):
                    rows.append(x)
                else:
                    p = Path(x)
                    st = p / "state.json" if p.is_dir() else p
                    rows += rows_from_state(st, part=part or "", problem=p.name if p.is_dir() else None,
                                            run_uid=f"workdir:{p}")
            return cls(rows)
        if hasattr(src, "current_id"):                 # a GlobalStore
            from .merge import _connect
            vid = src.current_id()
            if not vid:
                return cls([])
            c = _connect(src.path(vid), ro=True)
            try:
                return cls(store_rows(c, part))
            finally:
                c.close()
        p = Path(src)
        for f in (p, p / "struct_priors.json", p.parent / "struct_priors.json"):
            if f.is_file():
                try:
                    d = json.loads(f.read_text())
                except ValueError:
                    continue
                rows = d.get("rows", []) if isinstance(d, dict) else d
                return cls([r for r in rows if not part or r.get("part") in (None, "", part)])
        return cls([])

    def add(self, rows: Iterable) -> "StructPriors":
        have = {r.get("id") for r in self.rows}
        self.rows += [r for r in rows if r.get("status") in ("accepted", "rejected") and r.get("id") not in have]
        return self

    # ---- queries
    def _match(self, level: str, kind: str, ph: dict, part: Optional[str]):
        for r in self.rows:
            if part and r.get("part") and r["part"] != part:
                continue
            if level == "all":
                yield r
                continue
            if r.get("kind") != kind:
                continue
            if level == "kind" or (level == "head" and r.get("head") == ph["head"]) or \
                    (level == "classes" and r.get("classes") == ph["classes"]) or \
                    (level == "exact" and r.get("composition") == ph["composition"]):
                yield r

    def prior(self, kind: str, context=None, part: Optional[str] = None, *, strength: float = 2.0) -> Prior:
        """Acceptance / gain prior of a structural decision ``kind`` on ``context`` (op kinds, e.g.
        ["conv2d", "batchnorm2d"], or a fused kind name / hint id). See the module doc."""
        if isinstance(context, str):
            ph = parse_hint(context if ":" in context else f"{kind}:{context}")
            if ph["kind"] != kind and ph["kind"] == "fuse_siblings":
                kind = "fuse_siblings"
        else:
            ph = parse_hint(f"{kind}:" + "+".join(_base(c) for c in (context or [])))
        stats = {}
        for lv in self.LEVELS:
            rs = list(self._match(lv, kind, ph, part))
            acc = sum(1 for r in rs if r.get("accepted"))
            stats[lv] = (rs, acc, len(rs) - acc)
        # shrink from the top: all -> kind -> head -> classes -> exact
        rate = 0.5
        post = {}
        for lv in reversed(self.LEVELS):
            _, acc, rej = stats[lv]
            # Jeffreys base (0.5 / 0.5) + ``strength`` pseudo-observations at the parent level's rate: 6/6 accepted
            # gives ~0.94, not 1.0, and an unseen context inherits its parent's rate with little confidence
            a, b = acc + strength * rate + 0.5, rej + strength * (1 - rate) + 0.5
            post[lv] = (a, b)
            rate = a / (a + b) if (a + b) > 0 else rate
        level = next((lv for lv in self.LEVELS if stats[lv][0]), "none")
        rs, acc, rej = stats.get(level, ([], 0, 0)) if level != "none" else ([], 0, 0)
        a, b = post[level] if level != "none" else (1.0, 1.0)
        lo, hi = _wilson(acc, acc + rej)
        gains = [r["gain"] for r in rs if r.get("accepted") and r.get("gain") is not None]
        gall = [r["gain"] for r in rs if r.get("gain") is not None]
        rtd = [r["rounds_to_decide"] for r in rs if r.get("accepted") and r.get("rounds_to_decide") is not None]
        steps = [r["steps"] for r in rs if r.get("accepted") and isinstance(r.get("steps"), (int, float))]
        gm = sum(gains) / len(gains) if gains else None
        gse = (math.sqrt(sum((g - gm) ** 2 for g in gains) / (len(gains) - 1)) / math.sqrt(len(gains))
               if len(gains) >= 2 else None)
        return Prior(kind=kind, context=ph["context"], level=level, n=acc + rej, accepted=acc, rejected=rej,
                     rate=round(a / (a + b), 4), lo=round(lo, 4), hi=round(hi, 4), alpha=round(a, 4), beta=round(b, 4),
                     gain_mean=None if gm is None else round(gm, 4),
                     gain_lo=None if gse is None else round(gm - Z90 * gse, 4),
                     gain_hi=None if gse is None else round(gm + Z90 * gse, 4),
                     gain_all_mean=round(sum(gall) / len(gall), 4) if gall else None,
                     rounds_to_decide=round(sum(rtd) / len(rtd), 3) if rtd else None,
                     steps_mean=round(sum(steps) / len(steps), 2) if steps else None,
                     levels={lv: {"n": stats[lv][1] + stats[lv][2], "accepted": stats[lv][1]} for lv in self.LEVELS})

    def table(self, kind: Optional[str] = None, part: Optional[str] = None) -> list:
        """Every (kind, composition) seen with its prior (sorted by n)."""
        seen = {}
        for r in self.rows:
            if (kind and r.get("kind") != kind) or (part and r.get("part") and r["part"] != part):
                continue
            seen.setdefault((r["kind"], r.get("composition", "")), r.get("context", []))
        out = [{"kind": k, "composition": c, **{f: v for f, v in self.prior(k, ctx, part).to_dict().items()
                                                 if f in ("n", "accepted", "rate", "lo", "hi", "gain_mean", "level")}}
               for (k, c), ctx in seen.items()]
        return sorted(out, key=lambda x: (-x["n"], x["kind"], x["composition"]))

    def to_json(self) -> dict:
        return {"format": FORMAT, "rows": self.rows}


# ---------------------------------------------------------------- store side

def ensure(c) -> None:
    c.execute(STRUCT_SCHEMA.rstrip(";"))         # not executescript(): it would COMMIT the caller's transaction


def store_rows(c, part: Optional[str] = None) -> list:
    if not c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='gk_struct'").fetchone():
        return []
    q = "SELECT data FROM gk_struct" + (" WHERE part=?" if part else "") + " ORDER BY id"
    return [json.loads(d) for (d,) in c.execute(q, (part,) if part else ())]


def merge_struct(g, state_path, *, part: str, run_uid: str, problem: Optional[str] = None) -> dict:
    """A run's decisions into ``gk_struct`` (idempotent by row id)."""
    ensure(g)
    rows = rows_from_state(state_path, part=part, problem=problem, run_uid=run_uid)
    n = 0
    for r in rows:
        cur = g.execute("INSERT OR IGNORE INTO gk_struct VALUES (?,?,?)", (part, r["id"], json.dumps(r, sort_keys=True)))
        n += cur.rowcount
    return {"decisions": len(rows), "added": n}


def seed_struct(g, part: str, out_path) -> dict:
    rows = store_rows(g, part)
    if not rows:
        return {"rows": 0}
    Path(out_path).write_text(json.dumps({"format": FORMAT, "part": part, "rows": rows}, indent=0))
    return {"rows": len(rows), "path": str(out_path)}


def ingest_runs(store, part: str, work_dirs: list, *, label: str = "struct bootstrap") -> dict:
    """Bootstrap: a new snapshot with the decisions of the given ModelBlaster work dirs (under the lock, backup)."""
    import contextlib
    from .merge import _connect, _j, _now
    with store.lock():
        cur = store.current_id()
        store._backup(label)
        tmp = store._start_from(cur)
        g = _connect(tmp)
        rep = {"runs": 0, "decisions": 0, "added": 0}
        try:
            g.execute("BEGIN IMMEDIATE")
            for w in work_dirs:
                w = Path(w)
                st = w / "state.json"
                if not st.exists():
                    continue
                prob = w.parent.name if w.name == "work" else w.name
                grp = w.parent.parent.name if w.name == "work" else w.parent.name
                r = merge_struct(g, st, part=part, run_uid=f"bootstrap:{grp}/{prob}", problem=prob)
                rep["runs"] += 1
                rep["decisions"] += r["decisions"]
                rep["added"] += r["added"]
            g.execute("INSERT INTO gk_log (ts, kind, ref, part, data) VALUES (?,?,?,?,?)",
                      (_now(), "struct_ingest", label, part, _j(rep)))
            from .merge import _meta, _set_meta
            from .refresh import lineage
            vid_next = store._next_id()
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
        vid = store._commit(tmp, parent=cur, label=label, extra={"struct_ingest": rep})
        with open(store.root / "history.jsonl", "a") as f:
            f.write(_j({"ts": _now(), "op": "struct_ingest", "version": vid, "parent": cur}) + "\n")
        rep.update(status="ingested", version=vid, parent=cur)
        return rep
