# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rebuild what a run learned from its text artefacts when its ``knowledge.db`` is gone.

Inputs (all under a run's knowledge dir): ``review.jsonl`` (the audit log of every automatic
change) and ``reflect_traces/*.json`` (each reflection / consolidation call: the outcome records
it read, the claims it saw with their counts and citations, the edits it applied). Output: a
knowledge dir (repo seed + the recovered rows) that :func:`merge.merge_run` merges like a run.

Recovered:
  * outcome records the reflection passes read (ids, technique, status, op class / kind, shape
    bucket, dtype, states, speedup, metric deltas, impl ids) -- without the v2 reward, the
    measured metric vectors and the classification chain, so they add NO statistics;
  * claims: every applied edit replayed in time order (added / supported / contradicted /
    merged by consolidation), then reconciled with the newest support / contradict counts and
    citations the traces show; retirements and demotions from the audit log;
  * discovered states (``state_admitted``; the predicate is parsed back from its rendered form)
    with their later ``state_core`` / ``state_retired`` status;
  * prediction overrides (``prediction_refined`` minus ``prediction_reverted``);
  * hypotheses that were decided: retired (with the reason) and promoted (name, mechanism,
    reason). Not recoverable: a hypothesis' realization ("how"), preconditions, predictions and
    attempts (the audit log does not carry them), so undecided hypotheses are NOT imported
    (they could not be tested, and importing them would block the generator from re-proposing
    them) and promoted ones cannot be rebuilt as techniques;
  * not recoverable at all: reward statistics, hypothesis attempts, state-proposal trigger
    keys, handled-trigger sets, graph hypotheses.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Optional

from . import predicates as P

_TOK = re.compile(r"\(|\)|\[[^\]]*\]|[^\s()]+")


def parse_rendered_predicate(text: str):
    """Inverse of :func:`predicates.render` (``(a and b)``, ``(a or b)``, ``not x``, ``m op v``)."""
    toks = _TOK.findall(text.strip())
    pos = 0

    def atom():
        nonlocal pos
        if pos >= len(toks):
            raise ValueError("unexpected end")
        t = toks[pos]
        if t == "always":
            pos += 1
            return True
        if t == "not":
            pos += 1
            return {"not": atom()}
        if t == "(":
            pos += 1
            parts, joiner = [expr()], None
            while toks[pos] in ("and", "or"):
                j = toks[pos]
                if joiner and j != joiner:
                    raise ValueError("mixed and/or in one group")
                joiner = j
                pos += 1
                parts.append(expr())
            if toks[pos] != ")":
                raise ValueError(f"expected ) at {toks[pos]!r}")
            pos += 1
            if joiner is None:
                return parts[0]
            return {"all" if joiner == "and" else "any": parts}
        name, op, val = toks[pos], toks[pos + 1], toks[pos + 2]
        pos += 3
        if val.startswith("["):
            v = json.loads(val.replace("'", '"'))
            return {name: op, "v": v}
        try:
            return {name: op, "v": float(val) if any(ch in val for ch in ".eE") else int(val)}
        except ValueError:
            return {name: op, "ref": val}

    def expr():
        return atom()

    p = expr()
    if pos != len(toks):
        raise ValueError(f"trailing tokens {toks[pos:]}")
    P.validate(p)
    return p


def _lines(section: str, content: str) -> list[dict]:
    out, on = [], False
    for line in content.splitlines():
        if line.startswith(section):
            on = True
            continue
        if on:
            if line.startswith("{"):
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
            elif line.strip():
                on = False
    return out


def _trace_files(kdir: Path) -> list[Path]:
    def key(p: Path):
        m = re.search(r"_(\d{6,})\.json$", p.name)
        return (int(m.group(1)) if m else 0, p.name)
    return sorted((kdir / "reflect_traces").glob("*.json"), key=key)


def read_review(kdir: Path) -> list[dict]:
    p = Path(kdir) / "review.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def trace_outcome_ids(kdir: Path) -> set:
    ids = set()
    for f in _trace_files(Path(kdir)):
        try:
            d = json.loads(f.read_text())
        except ValueError:
            continue
        for m in d.get("messages", [])[1:2]:
            ids |= {r.get("id") for r in _lines("OUTCOME RECORDS", m.get("content", ""))}
    ids.discard(None)
    return ids


def salvage(kdir, out_dir, base_db: Path, *, target_key: str = "cuda:sm_89") -> dict:
    """Build ``out_dir/knowledge.db`` (repo seed + what ``kdir``'s text artefacts carry). -> report."""
    from ..claims import ACTIVE, RETIRED, Claim
    from ..outcomes import OutcomeRecord
    from .hypotheses import Hypothesis
    from .knowledge import KnowledgeV2
    from .reflect_v2 import set_claim_scope
    from .states import PROBATIONARY, State

    kdir, out = Path(kdir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base_db, out / "knowledge.db")
    kv = KnowledgeV2(out, seed_mined=True)
    rep = {"outcomes": 0, "claims_added": 0, "claims_supported": 0, "claims_unresolved_edits": 0,
           "claims_reconciled": 0, "claims_retired": 0, "states": 0, "states_unparsed": 0,
           "hypotheses_retired": 0, "hypotheses_promoted": 0, "hypotheses_open_skipped": 0,
           "prediction_overrides": 0, "traces": 0}
    base_by_text = {(c.kind, c.norm): c.id for c in kv.claims.claims.values()}
    idmap: dict = {}                                    # run claim id -> id in this store
    latest: dict = {}                                    # run claim id -> newest EXISTING CLAIMS line

    def resolve(cid: Optional[str], hint: Optional[dict] = None) -> Optional[str]:
        if cid in idmap:
            return idmap[cid]
        if cid in kv.claims.claims:
            return cid
        h = hint or latest.get(cid)
        if h:
            from ..claims import normalise
            bid = base_by_text.get((h.get("kind"), normalise(h.get("text", ""))))
            if bid:
                idmap[cid] = bid
                return bid
            try:                                       # a claim the traces only show as context: rebuild it
                scope = {k: h.get(k) for k in ("arch", "op_class", "state", "technique", "problem_id", "slot",
                                               "impl_id") if h.get(k) is not None}
                c = Claim(kind=h["kind"], text=h["text"], data=h.get("data") or {}, source="salvaged", **scope)
                c.id = cid
                kv.claims.claims[cid] = c
                c.add_support(h.get("cites") or [])
                idmap[cid] = cid
                return cid
            except (KeyError, TypeError, ValueError):
                return None
        return None

    known_out = set()
    for f in _trace_files(kdir):
        try:
            d = json.loads(f.read_text())
        except ValueError:
            continue
        rep["traces"] += 1
        content = (d.get("messages") or [{}, {}])[1].get("content", "") if len(d.get("messages") or []) > 1 else ""
        for r in _lines("OUTCOME RECORDS", content):
            if r.get("id") in known_out:
                continue
            try:
                rec = OutcomeRecord.from_json({**r, "diff_summary": r.get("diff")})
            except (TypeError, ValueError):
                continue
            kv.outcomes.append(rec, {"item_id": rec.technique, "salvaged": True})
            known_out.add(rec.id)
            rep["outcomes"] += 1
        for h in _lines("EXISTING CLAIMS", content):
            if h.get("id"):
                latest[h["id"]] = h
        for a in d.get("applied") or []:
            kind, res = a.get("kind"), a.get("result", "")
            if kind in ("support", "contradict"):
                cid = resolve(a.get("claim_id"))
                if cid is None:
                    rep["claims_unresolved_edits"] += 1
                    continue
                c = kv.claims.claims[cid]
                (c.add_support if kind == "support" else c.add_contradiction)(a.get("evidence") or [])
                rep["claims_supported"] += 1
            elif kind == "lift":
                cid = resolve(a.get("claim_id"))
                m = re.search(r"->(\S+)$", res)
                if cid and m:
                    set_claim_scope(kv.claims.claims[cid], m.group(1))
                    kv.claims.claims[cid].add_support(a.get("evidence") or [])
            elif res in ("added", "merged"):
                if res == "merged" and resolve(a.get("claim_id")):
                    kv.claims.claims[resolve(a.get("claim_id"))].add_support(a.get("evidence") or [])
                    continue
                try:
                    c = Claim(kind=kind, text=a.get("text") or "", data=a.get("data") or {}, source="salvaged",
                              **(a.get("scope") or {}))
                except (TypeError, ValueError):
                    rep["claims_unresolved_edits"] += 1
                    continue
                c.id = a.get("claim_id") or c.id
                c.add_support(a.get("evidence") or [])
                kv.claims.claims[c.id] = c
                idmap[c.id] = c.id
                rep["claims_added"] += 1
        for r in d.get("results") or []:                 # consolidation
            if r.get("result") == "merged" and r.get("keep"):
                keep = resolve(r["keep"])
                if keep is None:
                    continue
                for ab in r.get("absorbed") or []:
                    a = resolve(ab)
                    if a and a != keep:
                        kv.claims.merge(keep, [a], text=r.get("text"))
    # newest counts / citations the traces show
    for rid, h in latest.items():
        cid = resolve(rid, h)
        if cid is None:
            continue
        c = kv.claims.claims[cid]
        c.add_support([i for i in h.get("cites") or [] if i not in c.contradict_ids])
        if int(h.get("support", 0)) > c.support or int(h.get("contradict", 0)) > c.contradict:
            c.support = max(c.support, int(h.get("support", 0)))
            c.contradict = max(c.contradict, int(h.get("contradict", 0)))
            rep["claims_reconciled"] += 1
    ev = read_review(kdir)
    for e in ev:
        k = e.get("kind")
        if k in ("claim_retired", "claim_demoted"):
            cid = resolve(e.get("claim_id") or e.get("ref"))
            if cid is None:
                continue
            if k == "claim_retired":
                kv.claims.claims[cid].status = RETIRED
                rep["claims_retired"] += 1
            elif e.get("to"):
                set_claim_scope(kv.claims.claims[cid], e["to"])
    kv.claims.save()
    # states
    status_after = {}
    for e in ev:
        if e.get("kind") in ("state_core", "state_retired"):
            status_after[e.get("ref")] = e["kind"].split("_", 1)[1]
    for e in ev:
        if e.get("kind") != "state_admitted":
            continue
        sid = e.get("ref")
        if sid in kv.hierarchy.states:
            continue
        try:
            pred = parse_rendered_predicate(e.get("predicate", ""))
            s = State(id=sid, parent=e.get("parent"), predicate=pred, mechanism=(e.get("mechanism") or "")[:400],
                      status=status_after.get(sid, PROBATIONARY), source="discovered:salvaged", order=-1,
                      evidence={"numbers": e.get("numbers"), "admitted_seq": 0, "salvaged": True})
            kv.hierarchy.add(s)
        except (ValueError, KeyError, IndexError, P.PredicateError) as exc:
            rep["states_unparsed"] += 1
            rep.setdefault("states_unparsed_examples", []).append(f"{sid}: {exc}"[:160])
            continue
        kv.db.put("states", s.id, s.to_dict(), status=s.status)
        rep["states"] += 1
    # prediction overrides
    ov: dict = {}
    for e in ev:
        if e.get("kind") == "prediction_refined":
            o = {k: e.get(k) for k in ("state", "metric", "dir", "from", "n", "disagree_frac", "claims", "since_seq")}
            ov.setdefault(e.get("technique"), []).append(o)
        elif e.get("kind") == "prediction_reverted":
            lst = ov.get(e.get("technique"), [])
            ov[e.get("technique")] = [o for o in lst if not (o["state"] == e.get("state") and o["metric"] == e.get("metric")
                                                             and o["dir"] == e.get("dir"))]
    ov = {k: v for k, v in ov.items() if v}
    if ov:
        kv.db.set_meta("prediction_overrides", ov)
    rep["prediction_overrides"] = sum(len(v) for v in ov.values())
    # hypotheses: decided ones only
    prop = {e.get("ref"): e for e in ev if e.get("kind") == "hypothesis_proposed"}
    decided = set()
    for e in ev:
        k = e.get("kind")
        if k not in ("hypothesis_retired", "hypothesis_promoted"):
            continue
        hid = e.get("ref")
        p = prop.get(hid, {})
        name = p.get("name") or e.get("name") or hid
        try:
            h = Hypothesis(name=name, mechanism=p.get("mechanism") or name, source=p.get("source") or "gap",
                           how="", target_key=p.get("target") or target_key, id=hid,
                           status="retired" if k == "hypothesis_retired" else "promoted",
                           reason=(e.get("reason") or "") + " [salvaged from the audit log; attempts not recoverable]")
        except (TypeError, ValueError):
            continue
        if k == "hypothesis_promoted":
            h.promoted_to = e.get("technique")
            rep["hypotheses_promoted"] += 1
        else:
            rep["hypotheses_retired"] += 1
        kv.db.put("hypotheses", h.id, h.to_dict(), status=h.status)
        decided.add(hid)
    rep["hypotheses_open_skipped"] = len(set(prop) - decided)
    rep["review_events"] = len(ev)
    return rep
