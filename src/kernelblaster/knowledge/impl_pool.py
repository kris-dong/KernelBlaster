# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Implementation POOL: every verified kernel of one (op, signature, arch, dtype) with a cross-shape matrix.

The impl store's tiers keep only *the fastest record per shape set*. The shape-family pilot
(``l3_farm/analysis/shape_family.md``) showed what that loses: of DenseNet121's 60 per-shape conv winners, 40 run
correctly on every shape, and for 43 of 60 shapes ANOTHER shape's winner is >10% faster than the shape's own; a
per-shape routing over the valid kernels (815 us family time, zero LLM calls) beats the as-deployed set and the
best single kernel (2,242 us). A :class:`Pool` keeps the kernels and a sparse **kernel x shape matrix** (time,
validity, config) instead of the winner only, and answers:

* :meth:`Pool.best_for` (one shape) / :meth:`Pool.route` (instances -> per-instance kernel choice with correctness
  guard metadata): measured cells first; unmeasured shapes get a *predicted* choice (marked ``measured: False``,
  ``guard.needs_check``) from :func:`rank_kernels` -- the caller must validate before deploying it;
* :meth:`Pool.pareto`, :meth:`Pool.oracle`, :meth:`Pool.matrix`;
* :meth:`Pool.retain` -- bounded retention by usefulness (greedy marginal time saved over the matrix, coverage of
  shapes nobody else serves) and diversity (behavioural distinctness over co-measured shapes + source
  shingle distance), with caps.

Similarity ranking (:func:`rank_kernels`): a kernel's predicted slowdown on a query shape is the evidence-weighted
geometric mean of its measured slowdown ratios (time / best known time) on nearby shapes (Gaussian kernel over a
log-dimension shape distance + log work), shrunk towards ``prior_ratio`` when there is little nearby evidence;
failures count as ``fail_ratio``. Optional extra signals: source similarity to a query kernel (TF-IDF over
normalised token shingles, :class:`SourceIndex`) and profile similarity (cosine over shared metric names).

Pure Python, no new dependencies, deterministic. Persistence for a run's impl store: ``<impl root>/pool/obs.jsonl``
(observations) and ``pool/kernels.jsonl`` (kernel metadata / annotations); see ``ImplStore.pool``.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from typing import Iterable, Optional

FORMAT = "kb-impl-pool/1"

# conv / gemm keys that change the algorithm more than the size does: weighted up in the shape distance
_STRUCT_KEYS = {"KH", "KW", "SH", "SW", "DH", "DW", "PH", "PW", "G", "groups", "stride", "kernel", "causal", "reverse"}


def shape_key(shape: dict) -> str:
    """The canonical key of one instance shape (same as ``kb_reuse._skey`` / the merge's shape-set key)."""
    return json.dumps(shape, sort_keys=True)


def set_key(shapes: list) -> str:
    return json.dumps(sorted(shape_key(s) for s in shapes))


def pool_key(op: str, signature_sha: str, arch: str, dtype: str = "fp32", target: str = "cuda") -> str:
    """``(target, op, signature, arch, dtype)`` -> short stable key (file / table key)."""
    return hashlib.sha1(json.dumps([target, op, signature_sha, arch, dtype or "fp32"]).encode()).hexdigest()[:20]


def kernel_id(code_sha: str) -> str:
    return code_sha[:16]


# ---------------------------------------------------------------- structural kinds

def struct_info(op: str) -> dict:
    """Structural kind of an op name: plain | fused (``fused__a__b[__a]``) | sibling (``fused__par__``) |
    attention (``attention_f16``) | layout (``bmm_strided``, ``strided_copy``) -> {kind, stages, composition}.
    ``composition`` ignores dtype suffixes and fusion argument tags, so ``fused__conv2d_f16__relu_f16`` and its
    fp32 twin share it (retrieval of structural records by composition)."""
    if op.startswith("fused__"):
        parts = [p for p in op.split("__")[1:] if p]
        kind = "fused"
        if parts and parts[0] == "par":
            kind, parts = "sibling", parts[1:]
        stages = [p for p in parts if len(p) > 1]              # drop argument tags such as "a"
        comp = [re.sub(r"_(f16|f32|bf16|fp16|fp32|i8)$", "", s) for s in stages]
        return {"kind": kind, "stages": stages, "composition": "+".join(comp)}
    base = re.sub(r"_(f16|f32|bf16|fp16|fp32|i8)$", "", op)
    if "attention" in op:
        return {"kind": "attention", "stages": [op], "composition": base}
    if "strided" in op:
        return {"kind": "layout", "stages": [op], "composition": base}
    return {"kind": "plain", "stages": [op], "composition": base}


# ---------------------------------------------------------------- shape features

def _flat(shape: dict, pre: str = "") -> dict:
    out = {}
    for k, v in (shape or {}).items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        elif isinstance(v, bool):
            out[pre + k] = int(v)
        elif isinstance(v, (int, float)):
            out[pre + k] = v
    return out


def work(shape: dict) -> float:
    """A size measure for the roofline-ish feature: conv / gemm FLOPs when the keys are there, else elements."""
    s = shape.get("a") if isinstance(shape.get("a"), dict) else shape
    g = lambda k, d=1: max(int(s.get(k, d) or d), 1)                       # noqa: E731
    if {"OC", "OH", "OW", "KH"} <= set(s):
        ic = g("IC") // max(g("G"), 1) if "IC" in s else 1
        return 2.0 * g("N") * g("OC") * g("OH") * g("OW") * max(ic, 1) * g("KH") * g("KW")
    if {"M", "N", "K"} <= set(s):
        return 2.0 * g("M") * g("N") * g("K") * g("batch")
    if "n" in s:
        return float(g("n"))
    ints = [v for v in _flat(s).values() if isinstance(v, (int, float)) and v > 0]
    return float(math.prod(ints)) if ints else 1.0


def shape_distance(a: dict, b: dict) -> float:
    """RMS of log2 differences over the union of keys (structural keys x2, missing keys = 3) + log2 work."""
    fa, fb = _flat(a), _flat(b)
    keys = sorted(set(fa) | set(fb))
    if not keys:
        return 0.0
    tot = wsum = 0.0
    for k in keys:
        w = 2.0 if k.split(".")[-1] in _STRUCT_KEYS else 1.0
        if k in fa and k in fb:
            d = math.log2(abs(fa[k]) + 1) - math.log2(abs(fb[k]) + 1)
        else:
            d = 3.0
        tot += w * d * d
        wsum += w
    dw = math.log2(work(a) + 1) - math.log2(work(b) + 1)
    tot += dw * dw
    wsum += 1.0
    return math.sqrt(tot / wsum)


# ---------------------------------------------------------------- source features

_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_TOK = re.compile(r"[A-Za-z_]\w*|\d+(?:\.\d+)?[fFuU]?|\S")


def tokens(src: str) -> list:
    """Comment-free tokens; numbers -> N (tile sizes vary inside one algorithm family)."""
    out = []
    for t in _TOK.findall(_COMMENT.sub(" ", src or "")):
        out.append("N" if t[0].isdigit() else t)
    return out


def shingles(src: str, k: int = 4) -> Counter:
    t = tokens(src)
    return Counter(" ".join(t[i:i + k]) for i in range(max(len(t) - k + 1, 1)))


class SourceIndex:
    """TF-IDF over token shingles; cosine similarity between kernels / to a query source."""

    def __init__(self, docs: dict, k: int = 4):
        self.k = k
        self.tf = {i: shingles(c, k) for i, c in docs.items() if c}
        df: Counter = Counter()
        for tf in self.tf.values():
            df.update(tf.keys())
        n = max(len(self.tf), 1)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self.vec = {i: self._vec(tf) for i, tf in self.tf.items()}

    def _vec(self, tf: Counter) -> dict:
        v = {t: (1 + math.log(c)) * self.idf.get(t, math.log(1 + max(len(self.tf), 1)) + 1.0) for t, c in tf.items()}
        nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / nrm for t, x in v.items()}

    @staticmethod
    def _cos(a: dict, b: dict) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(x * b.get(t, 0.0) for t, x in a.items())

    def sim(self, i, j) -> float:
        if i not in self.vec or j not in self.vec:
            return 0.0
        return self._cos(self.vec[i], self.vec[j])

    def query(self, code: str) -> dict:
        q = self._vec(shingles(code, self.k))
        return {i: self._cos(q, v) for i, v in self.vec.items()}


def profile_similarity(a: Optional[dict], b: Optional[dict]) -> Optional[float]:
    """Cosine over the numeric metrics both profiles have (NCU summary dicts, roofline fractions ...)."""
    if not a or not b:
        return None
    ks = [k for k in a if k in b and isinstance(a[k], (int, float)) and isinstance(b[k], (int, float))]
    if len(ks) < 2:
        return None
    na = math.sqrt(sum(float(a[k]) ** 2 for k in ks)) or 1.0
    nb = math.sqrt(sum(float(b[k]) ** 2 for k in ks)) or 1.0
    return sum(float(a[k]) * float(b[k]) for k in ks) / (na * nb)


# ---------------------------------------------------------------- the pool

class Pool:
    """Kernels of one (op, signature, arch, dtype) + a sparse kernel x shape matrix.

    ``kernels[kid]``: {code_sha, records: [ids], problems: [..], techniques, kind, stages, lineage, tunables,
    validated: [shape keys], set_us: {set key: us}, best_speedup_vs_seed, meta}.
    ``obs[kid][skey]``: {us (min seen, None = validated only), ok, n, n_fail, src, config, err, shape}."""

    def __init__(self, op: str = "", signature_sha: str = "", arch: str = "", dtype: str = "fp32", target: str = "cuda"):
        self.op, self.sig, self.arch, self.dtype, self.target = op, signature_sha, arch, dtype or "fp32", target
        self.kernels: dict = {}
        self.obs: dict = {}
        self.shapes: dict = {}                       # skey -> shape dict
        self._code: dict = {}                        # kid -> source (optional, for source similarity)

    @property
    def key(self) -> str:
        return pool_key(self.op, self.sig, self.arch, self.dtype, self.target)

    # ---- building
    def add_kernel(self, code_sha: str, *, record: Optional[dict] = None, code: Optional[str] = None,
                   **meta) -> str:
        kid = kernel_id(code_sha)
        k = self.kernels.setdefault(kid, {"code_sha": code_sha, "records": [], "problems": [], "techniques": [],
                                          "validated": [], "set_us": {}, "best_speedup_vs_seed": None,
                                          **struct_info(self.op)})
        if record:
            if record.get("id") and record["id"] not in k["records"]:
                k["records"].append(record["id"])
            p = record.get("problem_id")
            if p and p not in k["problems"]:
                k["problems"].append(p)
            if not k["techniques"] and record.get("techniques"):
                k["techniques"] = list(record["techniques"])
            sp = record.get("speedup_vs_seed")
            if sp is not None and (k["best_speedup_vs_seed"] is None or sp > k["best_speedup_vs_seed"]):
                k["best_speedup_vs_seed"] = sp
            if record.get("parent_id") and "parent_ids" not in k:
                k["parent_ids"] = [record["parent_id"]]
            shapes = record.get("shapes") or []
            us = record.get("slot_us")
            if len(shapes) == 1:
                self.observe(kid, shapes[0], us=us, ok=True, src="record")
            elif shapes:
                for s in shapes:
                    self.observe(kid, s, us=None, ok=True, src="set")
                if us is not None:
                    sk = set_key(shapes)
                    k["set_us"][sk] = min(us, k["set_us"].get(sk, float("inf")))
        for f, v in meta.items():
            if v is None:
                continue
            if f == "lineage" and isinstance(v, dict):
                k["lineage"] = {**(k.get("lineage") or {}), **v}
            else:
                k[f] = v
        if code is not None:
            self._code[kid] = code
            if "tunables" not in k:
                from .tunables import extract_tunables
                t = extract_tunables(code)
                k["tunables"] = t if t["tunable"] else None
        return kid

    def observe(self, kid: str, shape: dict, *, us: Optional[float] = None, ok: bool = True, src: str = "measured",
                config=None, err: Optional[str] = None) -> None:
        """One measurement of kernel ``kid`` on ``shape``: ``us`` (None = validated only) / ``ok``. Repeated
        observations keep the minimum time and count failures (``n_fail``); a cell with any failure is not
        routed to (conservative guard)."""
        if kid not in self.kernels:
            self.kernels[kid] = {"code_sha": kid, "records": [], "problems": [], "techniques": [], "validated": [],
                                 "set_us": {}, "best_speedup_vs_seed": None, **struct_info(self.op)}
        sk = shape_key(shape)
        self.shapes.setdefault(sk, shape)
        c = self.obs.setdefault(kid, {}).setdefault(sk, {"us": None, "ok": True, "n": 0, "n_fail": 0, "src": src})
        c["n"] += 1
        if not ok:
            c["n_fail"] += 1
            c["ok"] = False
            if err:
                c["err"] = str(err)[:300]
        elif us is not None and (c["us"] is None or us < c["us"]):
            c["us"] = float(us)
            if config is not None:
                c["config"] = config
            if src == "measured" or c["src"] != "measured":
                c["src"] = src
        if ok and sk not in self.kernels[kid]["validated"]:
            self.kernels[kid]["validated"].append(sk)

    def set_code(self, kid: str, code: str) -> None:
        self._code[kid] = code

    # ---- matrix views
    def cell(self, kid: str, sk: str) -> Optional[dict]:
        return (self.obs.get(kid) or {}).get(sk)

    def best_known(self, sk: str, exclude: Iterable = ()) -> Optional[float]:
        ex = set(exclude)
        ts = [c["us"] for kid, row in self.obs.items() if kid not in ex
              for k2, c in row.items() if k2 == sk and c["ok"] and c["us"] is not None]
        return min(ts) if ts else None

    def measured_shapes(self) -> list:
        return sorted({sk for row in self.obs.values() for sk, c in row.items() if c["us"] is not None})

    def matrix(self) -> dict:
        ks, ss = sorted(self.kernels), self.measured_shapes()
        return {"format": FORMAT, "kernels": ks, "shapes": ss,
                "us": [[(self.cell(k, s) or {}).get("us") for s in ss] for k in ks],
                "ok": [[(self.cell(k, s) or {}).get("ok") for s in ss] for k in ks]}

    def oracle(self) -> float:
        return sum(self.best_known(s) or 0.0 for s in self.measured_shapes())

    def pareto(self) -> list:
        """Kernels not dominated by another kernel valid and at least as fast on every shape they were timed on."""
        timed = {k: {s: c["us"] for s, c in row.items() if c["ok"] and c["us"] is not None}
                 for k, row in self.obs.items()}
        out = []
        for k, tk in timed.items():
            if not tk:
                continue
            dom = False
            for k2, t2 in timed.items():
                if k2 == k or not set(tk) <= set(t2):
                    continue
                if all(t2[s] <= tk[s] for s in tk) and any(t2[s] < tk[s] for s in tk):
                    dom = True
                    break
            if not dom:
                out.append(k)
        return sorted(out)

    # ---- choice
    def best_for(self, shape: dict, *, measured_only: bool = False, exclude: Iterable = (),
                 **rank_kw) -> Optional[dict]:
        sk = shape_key(shape)
        ex = set(exclude)
        timed = [(c["us"], kid) for kid, row in self.obs.items() if kid not in ex
                 for k2, c in row.items() if k2 == sk and c["ok"] and c["us"] is not None]
        if timed:
            us, kid = min(timed)
            return self._choice(kid, sk, us=us, measured=True, pred_ratio=1.0)
        if measured_only:
            return None
        r = rank_kernels(self, shape, exclude=ex, **rank_kw)
        if not r:
            return None
        top = r[0]
        return self._choice(top["kernel"], sk, us=None, measured=False, pred_ratio=top["pred_ratio"],
                            why=top["why"])

    def _choice(self, kid: str, sk: str, *, us, measured: bool, pred_ratio: float, why=None) -> dict:
        k = self.kernels[kid]
        c = self.cell(kid, sk) or {}
        failed = sorted(s for s, x in (self.obs.get(kid) or {}).items() if not x["ok"])
        return {"kernel": kid, "code_sha": k["code_sha"], "us": us, "measured": measured,
                "validated": bool(c.get("ok") and c.get("n")), "pred_ratio": round(pred_ratio, 4),
                "config": c.get("config"), "kind": k.get("kind"),
                "guard": {"shape": sk, "validated_shapes": len(k["validated"]), "failed_on": failed[:8],
                          "n_failed": len(failed), "needs_check": not (c.get("ok") and c.get("n"))},
                **({"why": why} if why else {})}

    def route(self, instances: list, *, measured_only: bool = False, incumbent: Optional[dict] = None,
              min_gain: float = 0.0, exclude: Iterable = (), **rank_kw) -> dict:
        """Per-instance kernel choice (the zero-LLM routing of the shape-family pilot).

        ``instances``: shape dicts or {"op_name"/"name", "shape"} dicts; ``incumbent``: {shape key or op name: us}
        -- a choice replaces the incumbent only when measured faster by ``min_gain`` (fraction), else the
        instance keeps the incumbent (``choice: None``). Predicted (unmeasured) choices are never compared with an
        incumbent; they carry ``guard.needs_check`` and must be validated by the caller before deployment."""
        out, tot, used, unval = [], 0.0, Counter(), []
        for inst in instances:
            shape = inst.get("shape", inst) if isinstance(inst, dict) else inst
            name = inst.get("op_name") or inst.get("name") if isinstance(inst, dict) and "shape" in inst else None
            ch = self.best_for(shape, measured_only=measured_only, exclude=exclude, **rank_kw)
            sk = shape_key(shape)
            inc = None
            if incumbent:
                inc = incumbent.get(name) if name and name in incumbent else incumbent.get(sk)
            if ch is not None and ch["measured"] and inc is not None and not ch["us"] < inc * (1 - min_gain):
                ch = None
            row = {"instance": name or sk, "shape": sk, "choice": ch, "incumbent_us": inc}
            if ch is not None:
                used[ch["kernel"]] += 1
                if ch["measured"]:
                    tot += ch["us"]
                else:
                    unval.append(name or sk)
            elif inc is not None:
                tot += inc
            out.append(row)
        return {"format": FORMAT, "pool": self.key, "choices": out, "sum_us": round(tot, 3),
                "kernels": dict(used), "unvalidated": unval}

    def tuned_table(self, kid: str) -> dict:
        """Per shape the best config seen for kernel ``kid`` (autotuner winning settings)."""
        return {sk: {"config": c.get("config"), "us": c["us"]} for sk, c in (self.obs.get(kid) or {}).items()
                if c.get("config") is not None and c["ok"] and c["us"] is not None}

    # ---- retention
    def retain(self, max_kernels: int = 16, *, useful_frac: float = 0.75, protect: Iterable = (),
               min_gain_frac: float = 0.01, src_index: Optional[SourceIndex] = None) -> dict:
        """Keep at most ``max_kernels``: (1) usefulness -- greedy by marginal time saved on the measured shapes
        (a shape nobody kept covers counts as its full best time), up to ``useful_frac`` of the cap, stopping
        when a kernel saves < ``min_gain_frac`` of the oracle; (2) diversity -- the rest by max-min distance to
        the kept set (behaviour: mean |log ratio| difference over co-measured shapes; source: 1 - TF-IDF cosine;
        unmeasured kernels by source only), tunable kernels and new structural kinds first on ties;
        ``protect`` ids are always kept. Mutates the pool; returns {kept, dropped: {kid: reason}}."""
        ks = sorted(self.kernels)
        if len(ks) <= max_kernels:
            return {"kept": ks, "dropped": {}}
        prot = [k for k in ks if k in set(protect)]
        kept = list(prot)
        shapes = self.measured_shapes()
        best = {s: self.best_known(s) for s in shapes}
        oracle = sum(v for v in best.values() if v) or 1.0
        t = {k: {s: c["us"] for s, c in (self.obs.get(k) or {}).items() if c["ok"] and c["us"] is not None}
             for k in ks}
        # an uncovered shape counts as served by a kernel twice as slow as the slowest one seen on it, so the greedy
        # (like kb_family_tune's pool step) both covers shapes and saves time
        worst = {s: max([t[k][s] for k in ks if s in t[k]], default=0.0) for s in shapes}
        cur = {s: min([t[k][s] for k in kept if s in t[k]], default=2.0 * worst[s]) for s in shapes}
        cap_useful = max(1, int(round(max_kernels * useful_frac)))
        reason = {k: "protected" for k in kept}
        while len(kept) < min(cap_useful, max_kernels):
            gains = []
            for k in ks:
                if k in kept or not t[k]:
                    continue
                g = sum(cur[s] - us for s, us in t[k].items() if us < cur[s])
                gains.append((g, k))
            if not gains:
                break
            g, k = max(gains)
            if g < min_gain_frac * oracle:
                break
            kept.append(k)
            reason[k] = f"useful (saves {g:.1f} us)"
            for s, us in t[k].items():
                cur[s] = min(cur[s], us)
        # diversity
        if src_index is None and self._code:
            src_index = SourceIndex({k: self._code[k] for k in ks if k in self._code})

        def ratios(k):
            return {s: math.log(us / best[s]) for s, us in t[k].items() if best.get(s)}

        R = {k: ratios(k) for k in ks}

        def dist(a, b):
            parts = []
            co = set(R[a]) & set(R[b])
            if co:
                parts.append(min(1.0, sum(abs(R[a][s] - R[b][s]) for s in co) / len(co) / math.log(4)))
            if src_index is not None and a in src_index.vec and b in src_index.vec:
                parts.append(1.0 - src_index.sim(a, b))
            return sum(parts) / len(parts) if parts else 0.5

        rest = [k for k in ks if k not in kept]
        kinds = {self.kernels[k].get("kind") for k in kept}
        while len(kept) < max_kernels and rest:
            def score(k):
                d = min((dist(k, j) for j in kept), default=1.0)
                bonus = 0.05 * bool(self.kernels[k].get("tunables")) + 0.1 * (self.kernels[k].get("kind") not in kinds)
                return (d + bonus, -(self.kernels[k].get("best_speedup_vs_seed") or 0.0), k)
            k = max(rest, key=score)
            kept.append(k)
            kinds.add(self.kernels[k].get("kind"))
            reason[k] = f"diverse (distance {score(k)[0]:.2f})"
            rest.remove(k)
        dropped = {k: "capped (least useful / least distinct)" for k in rest}
        for k in rest:
            self.kernels.pop(k, None)
            self.obs.pop(k, None)
            self._code.pop(k, None)
        return {"kept": kept, "reasons": reason, "dropped": dropped}

    # ---- (de)serialisation
    def to_json(self) -> dict:
        return {"format": FORMAT, "op": self.op, "signature_sha": self.sig, "arch": self.arch, "dtype": self.dtype,
                "target": self.target, "key": self.key, "kernels": self.kernels, "obs": self.obs,
                "shapes": self.shapes}

    @classmethod
    def from_json(cls, d: dict) -> "Pool":
        p = cls(d.get("op", ""), d.get("signature_sha", ""), d.get("arch", ""), d.get("dtype", "fp32"),
                d.get("target", "cuda"))
        p.kernels = d.get("kernels") or {}
        p.obs = d.get("obs") or {}
        p.shapes = d.get("shapes") or {}
        return p

    @classmethod
    def from_records(cls, records: Iterable, *, op: str, signature_sha: str, arch: str, dtype: str = "fp32",
                     target: str = "cuda", code_of=None) -> "Pool":
        """A pool from impl-store record dicts / ImplRecords of that key (others ignored)."""
        p = cls(op, signature_sha, arch, dtype, target)
        for r in records:
            d = r if isinstance(r, dict) else r.__dict__
            if (d.get("op"), d.get("signature_sha"), d.get("arch"), d.get("dtype") or "fp32",
                    d.get("target", "cuda")) != (op, signature_sha, arch, dtype or "fp32", target):
                continue
            code = code_of(d) if code_of else None
            p.add_kernel(d["code_sha"], record=d, code=code)
        return p


# ---------------------------------------------------------------- similarity ranking

def rank_kernels(pool: Pool, shape: dict, *, exclude: Iterable = (), tau: Optional[float] = None,
                 neighbours: int = 4, prior_ratio: Optional[float] = None,
                 prior_w: float = 0.3, fail_ratio: float = 8.0, query_code: Optional[str] = None,
                 w_src: float = 0.0, src_index: Optional[SourceIndex] = None, profile: Optional[dict] = None,
                 w_prof: float = 0.0, profiles: Optional[dict] = None, k: Optional[int] = None) -> list:
    """Kernels of ``pool`` ranked for ``shape`` (best first) -> [{kernel, pred_ratio, score, evidence, why}].

    pred_ratio = exp((sum_w w*log r + prior_w*log prior_ratio) / (sum_w + prior_w)) over the kernel's measured
    cells r = t / best_known(shape') (failures: ``fail_ratio``), w = exp(-d^2 / 2 tau^2), d = shape_distance;
    tau None = adaptive (distance to the ``neighbours``-th nearest timed shape, clamped to [0.05, 1]); prior_ratio
    None = the pool's median cell slowdown (>= 2).
    A cell measured on ``shape`` itself is used as is (w = 1, d = 0). score = log(pred_ratio) - w_src * source
    similarity to ``query_code`` - w_prof * profile similarity (lower is better)."""
    ex = set(exclude)
    best: dict = {}
    for sk in pool.measured_shapes():
        best[sk] = pool.best_known(sk, exclude=ex)
    if prior_ratio is None:
        # data-driven prior: the median slowdown of a timed cell in this pool (a kernel we know nothing about near
        # the query is assumed typical, not good); 2.0 when the pool has no comparable cells
        rs = sorted(c["us"] / best[sk] for kid, row in pool.obs.items() if kid not in ex
                    for sk, c in row.items() if c["ok"] and c["us"] is not None and best.get(sk))
        prior_ratio = max(2.0, rs[len(rs) // 2]) if len(rs) >= 8 else 2.0
    dcache: dict = {}
    for sk in pool.shapes:
        dcache[sk] = shape_distance(shape, pool.shapes[sk])
    if tau is None:
        # adaptive width: the distance to the query's ``neighbours``-th nearest timed shape, so a dense matrix
        # (every kernel timed on many shapes) is judged on the closest shapes, and a sparse one (each kernel on its
        # own shape only) still reaches the nearest few kernels
        ds = sorted(dcache[sk] for sk in best if best.get(sk) and sk in dcache)
        tau = min(1.0, max(0.05, ds[min(len(ds), max(neighbours, 1)) - 1])) if ds else 0.75
    src_sim: dict = {}
    if query_code and w_src:
        if src_index is None:
            src_index = SourceIndex({kid: c for kid, c in pool._code.items()})
        src_sim = src_index.query(query_code)
    out = []
    for kid, kmeta in pool.kernels.items():
        if kid in ex:
            continue
        sw = swl = 0.0
        near = None
        for sk, c in (pool.obs.get(kid) or {}).items():
            if sk not in dcache:
                dcache[sk] = shape_distance(shape, pool.shapes.get(sk) or json.loads(sk))
            d = dcache[sk]
            if c["ok"] and c["us"] is not None and best.get(sk):
                r = max(c["us"] / best[sk], 1e-6)
            elif not c["ok"]:
                r = fail_ratio
            else:
                continue                                           # validated only, no time
            w = math.exp(-d * d / (2 * tau * tau))
            sw += w
            swl += w * math.log(r)
            if near is None or d < near:
                near = d
        pr = math.exp((swl + prior_w * math.log(prior_ratio)) / (sw + prior_w))
        score = math.log(pr)
        why = [f"evidence {sw:.2f}", f"nearest timed shape d={near:.2f}" if near is not None else "no timed shape"]
        if src_sim:
            s = src_sim.get(kid, 0.0)
            score -= w_src * s
            why.append(f"source sim {s:.2f}")
        if profile is not None and w_prof and profiles:
            ps = profile_similarity(profile, profiles.get(kid))
            if ps is not None:
                score -= w_prof * ps
                why.append(f"profile sim {ps:.2f}")
        out.append({"kernel": kid, "pred_ratio": pr, "score": score, "evidence": sw, "why": "; ".join(why),
                    "_tie": (-(kmeta.get("best_speedup_vs_seed") or 0.0), kid)})
    out.sort(key=lambda x: (x["score"], x["_tie"]))
    for x in out:
        x.pop("_tie")
    return out[:k] if k else out
