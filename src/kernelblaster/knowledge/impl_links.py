# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Links between implementation records and the technique / state / claim / hypothesis knowledge (A5, opt-in).

The pieces were already there, unconnected: an outcome record names the impl it produced (``impl_id``) and the
impl it started from (``parent_impl_id``), its technique, states before / after and (v2) the hypothesis it tested;
a claim cites outcome ids as support / contradiction. :class:`ImplLinks` joins them per implementation:

* ``produced_by`` -- the outcome(s) that created the record: technique, hypothesis, states, speedup;
* ``claims`` -- lessons citing those outcomes (which claim / hypothesis a kernel instantiates);
* ``children`` -- what was tried ON TOP of it: improved / neutral / regressed / failed counts, and the
  **negatives** ("technique T on this kernel regressed 0.91x on shape bucket B", compile / runtime / wrong-output
  errors) with the pitfall / hurts claims that cite them.

Uses: :meth:`render` -- a size-capped prompt section for the retrieved implementations (opt_multislot, env
``KERNELBLASTER_IMPL_LINKS=1``, cap ``KERNELBLASTER_IMPL_LINKS_CHARS``, default 600); :meth:`penalty` -- negative
evidence for ``ImplStore.retrieve_similar(links=...)``. Sources: a run's ``knowledge.db`` (tables ``outcomes`` /
``claims``) or a global snapshot (``gk_outcomes`` / ``gk_claims``, one partition). Read-only, no LLM.
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

NEG = ("regressed", "compile_error", "runtime_error", "wrong_output")


class ImplLinks:
    def __init__(self, outcomes: Iterable = (), claims: Iterable = ()):
        """``outcomes``: (id, rec dict, v2 dict); ``claims``: claim dicts (with id)."""
        self.out = {}
        self.by_impl = defaultdict(list)
        self.by_parent = defaultdict(list)
        for oid, rec, v2 in outcomes:
            self.out[oid] = (rec, v2 or {})
            if rec.get("impl_id"):
                self.by_impl[rec["impl_id"]].append(oid)
            if rec.get("parent_impl_id"):
                self.by_parent[rec["parent_impl_id"]].append(oid)
        self.claims = {}
        self.cites = defaultdict(list)                       # outcome id -> [(claim id, role)]
        for c in claims:
            if c.get("status", "active") != "active":
                continue
            self.claims[c["id"]] = c
            for i in c.get("support_ids") or []:
                self.cites[i].append((c["id"], "support"))
            for i in c.get("contradict_ids") or []:
                self.cites[i].append((c["id"], "contradict"))
        self.built = time.time()

    # ---- loading
    @classmethod
    def from_db(cls, db_path, *, part: Optional[str] = None) -> "ImplLinks":
        """A run's knowledge.db or a global snapshot (``part`` selects the partition)."""
        c = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True, timeout=10)
        try:
            t = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            outs, cls_ = [], []
            if "gk_outcomes" in t:
                q = "SELECT id, rec, v2 FROM gk_outcomes" + (" WHERE part=?" if part else "")
                rows = c.execute(q, (part,) if part else ())
                cq = "SELECT id, data FROM gk_claims WHERE status='active'" + (" AND part=?" if part else "")
                crow = c.execute(cq, (part,) if part else ()).fetchall()
            else:
                rows = c.execute("SELECT id, rec, v2 FROM outcomes")
                crow = c.execute("SELECT id, data FROM claims WHERE status='active'").fetchall() if "claims" in t else []
            for oid, rec, v2 in rows:
                try:
                    outs.append((oid, json.loads(rec), json.loads(v2) if v2 else {}))
                except (TypeError, ValueError):
                    continue
            for cid, data in crow:
                try:
                    cls_.append(dict(json.loads(data), id=cid))
                except (TypeError, ValueError):
                    continue
            return cls(outs, cls_)
        finally:
            c.close()

    # ---- queries
    def _claims_of(self, oids: Iterable, kinds: Optional[tuple] = None) -> list:
        seen, out = set(), []
        for o in oids:
            for cid, role in self.cites.get(o, []):
                c = self.claims.get(cid)
                if c is None or cid in seen or (kinds and c.get("kind") not in kinds):
                    continue
                seen.add(cid)
                out.append({"id": cid, "role": role, "kind": c.get("kind"), "text": c.get("text", ""),
                            "support": c.get("support", 0), "contradict": c.get("contradict", 0),
                            "technique": c.get("technique"), "hypothesis_id": c.get("hypothesis_id"),
                            "effect": (c.get("data") or {}).get("effect")})
        out.sort(key=lambda x: (-(int(x["support"]) - int(x["contradict"])), x["id"]))
        return out

    def links(self, impl_id: str, *, k_claims: int = 4, k_neg: int = 4) -> dict:
        prod = self.by_impl.get(impl_id, [])
        kids = self.by_parent.get(impl_id, [])
        pb = []
        for o in prod[:3]:
            rec, v2 = self.out[o]
            pb.append({"outcome": o, "technique": rec.get("technique"), "hypothesis_id": v2.get("hypothesis_id"),
                       "state_before": rec.get("state_before"), "state_after": rec.get("state_after"),
                       "speedup": rec.get("speedup"), "status": rec.get("status")})
        counts = defaultdict(int)
        neg = []
        for o in kids:
            rec, _ = self.out[o]
            st = rec.get("status")
            counts[st] += 1
            if st in NEG:
                neg.append({"outcome": o, "technique": rec.get("technique"), "status": st,
                            "speedup": rec.get("speedup"), "shape_bucket": rec.get("shape_bucket"),
                            "error": (rec.get("error") or "")[:160] or None,
                            "why": [c["text"] for c in self._claims_of([o], ("pitfall", "applicability"))][:1]})
        neg.sort(key=lambda x: (x["status"] == "regressed", x["speedup"] or 0))
        return {"impl_id": impl_id, "produced_by": pb, "claims": self._claims_of(prod)[:k_claims],
                "hypotheses": sorted({p["hypothesis_id"] for p in pb if p["hypothesis_id"]}),
                "children": dict(counts), "negatives": neg[:k_neg]}

    def negatives(self, *, op_kind: Optional[str] = None, shape_bucket: Optional[str] = None, k: int = 5) -> list:
        """Failed / regressed outcomes of an op kind (optionally one shape bucket) with the claims explaining them."""
        out = []
        for oid, (rec, _) in self.out.items():
            if rec.get("status") not in NEG or (op_kind and rec.get("op_kind") != op_kind) or \
                    (shape_bucket and rec.get("shape_bucket") != shape_bucket):
                continue
            why = self._claims_of([oid])
            if not why:
                continue
            out.append({"outcome": oid, "technique": rec.get("technique"), "status": rec.get("status"),
                        "speedup": rec.get("speedup"), "parent_impl_id": rec.get("parent_impl_id"),
                        "why": why[0]["text"]})
        out.sort(key=lambda x: (x["speedup"] if x["speedup"] is not None else 0.0, x["outcome"]))
        return out[:k]

    def penalty(self, kernel: dict, *, shapes: Optional[list] = None, per: float = 0.05, cap: float = 0.5) -> float:
        """Negative evidence on a pool kernel's records (failed / regressed children explained by a claim, and
        failed cross-shape cells): added to its retrieval score (log-ratio units)."""
        n = 0
        for rid in kernel.get("records") or []:
            for o in self.by_parent.get(rid, []):
                rec, _ = self.out[o]
                if rec.get("status") in ("compile_error", "runtime_error", "wrong_output"):
                    n += 1
        return min(cap, per * n)

    def render(self, impl_ids: Iterable, *, labels: Optional[dict] = None, max_chars: int = 600) -> str:
        """Prompt section for the retrieved / current implementations, at most ``max_chars``."""
        lines = []
        for iid in dict.fromkeys(i for i in impl_ids if i):
            L = self.links(iid, k_claims=2, k_neg=3)
            if not (L["produced_by"] or L["claims"] or L["negatives"]):
                continue
            head = f"- impl {iid}" + (f" ({labels[iid]})" if labels and iid in labels else "")
            pb = L["produced_by"][0] if L["produced_by"] else None
            if pb:
                head += f": made by {pb['technique']}" + (f" ({pb['speedup']:.2f}x)" if pb.get("speedup") else "")
                if pb.get("hypothesis_id"):
                    head += f", testing hypothesis {pb['hypothesis_id']}"
            lines.append(head)
            for c in L["claims"]:
                lines.append(f"  lesson (+{c['support']}/-{c['contradict']}): {c['text'][:180]}")
            ch = L["children"]
            if ch:
                lines.append("  tried on top of it: " + ", ".join(f"{v} {k}" for k, v in sorted(ch.items())))
            for n in L["negatives"]:
                sp = f" {n['speedup']:.2f}x" if n.get("speedup") else ""
                why = f": {n['why'][0][:140]}" if n.get("why") else (f": {n['error'][:100]}" if n.get("error") else "")
                lines.append(f"  did not work: {n['technique']} {n['status']}{sp}{why}")
        if not lines:
            return ""
        text = "Knowledge linked to these implementations (from earlier steps / runs):\n"
        for ln in lines:
            if len(text) + len(ln) + 1 > max_chars:
                break
            text += ln + "\n"
        return text.rstrip()
