# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reflection for knowledge v2: the v1 edits plus ``lift`` and ``mechanism`` (design section 8).

* ``lift`` moves a claim up a level: target -> family needs consistent
  support on >= 2 targets of the family, family -> universal on >= 2
  families. Checked against the targets of the cited outcome records
  (the claim's own citations plus the edit's evidence), never taken on the
  model's word.
* ``mechanism`` ties a technique's predicted effect on one canonical metric
  to the observed measured deltas of the cited records (``verdict``:
  confirmed / contradicted / side_effect, checked against the data).

:class:`SemanticUpdaterV2` also runs the slower upkeep passes of
:class:`KnowledgeV2` (hypothesis generation and review, state discovery,
probation) after each reflection.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Optional

from ..claims import ACTIVE, APPLICABILITY, MAX_TEXT_CHARS, MECHANISM, PARAMETER, PITFALL, Claim
from ..outcomes import COMPILE_ERROR, IMPROVED, NEUTRAL, REGRESSED, RUNTIME_ERROR, WRONG_OUTPUT
from ..reflect import EditError, SemanticUpdater, apply_edit, validate_edit
from .metrics import CANONICAL, flat_band
from .ontology import direction_agrees
from .store import claim_scope
from .targets import NODE, TARGET, UNIVERSAL, target_key

LIFT = "lift"
FAILURES = (COMPILE_ERROR, WRONG_OUTPUT, RUNTIME_ERROR)
VERDICTS = ("confirmed", "contradicted", "side_effect")

V2_INSTRUCTIONS = """
Knowledge v2 additions:
- Claims have a scope in a tree: one target (cuda:sm_89), a scope node (e.g. ada > nvidia_gpu >
  simt_gpu) or universal. New claims start at target level. States are hierarchical (e.g.
  memory_bound > cache_thrashing); scope a claim to the most general state it holds for.
- lift: {"kind": "lift", "claim_id": "...", "evidence": [ids], "text": "the lesson reworded without
  target specifics (optional)"} when a claim holds on SEVERAL targets. It moves to the lowest
  common ancestor of the targets with consistent support (>= 2 targets; e.g. cuda:sm_75 + cuda:sm_89
  -> nvidia_gpu), checked against the cited records' targets; "to": "<node id>" optionally caps it.
- mechanism: {"kind": "mechanism", "text": "...", "evidence": [ids], "scope": {"technique": "..."},
  "data": {"metric": "<canonical metric>", "observed": "up|down|flat", "verdict":
  "confirmed|contradicted|side_effect"}} explains what a technique did to a MEASURED metric (records'
  metrics_delta) vs its prediction (MECHANISM CHECKS below). side_effect = a metric it does not
  predict moved and explains the outcome (e.g. spills, occupancy loss). Cite only records whose
  metrics_delta has that metric.
"""


def consistent(c: Claim, status: str) -> bool:
    """Does a record with ``status`` support claim ``c``'s polarity?"""
    if c.kind == APPLICABILITY:
        eff = (c.data or {}).get("effect")
        return {"helps": status == IMPROVED, "hurts": status in (REGRESSED,) + FAILURES,
                "no_effect": status == NEUTRAL}.get(eff, False)
    if c.kind == PITFALL:
        return status in FAILURES
    if c.kind == PARAMETER:
        return status == IMPROVED
    return True


def _claim_home(c: Claim, registry) -> str:
    """The claim's current scope id (a target key for target-level claims)."""
    sid = claim_scope(c, registry)
    if sid is not None:
        return sid
    if c.arch:
        return registry.get(target_key(c.target, c.arch)).key
    return registry.node_for_backend(f"{c.target}:x")           # backend-wide v1 claim


def per_target_evidence(c: Claim, index: dict, extra=()) -> tuple[Counter, Counter, list]:
    """(support, contradiction) per target key from the claim's cited records (+ ``extra`` ids)."""
    supp, contra, used = Counter(), Counter(), []
    for i in dict.fromkeys(list(c.support_ids) + list(extra)):
        r = index.get(i)
        if r is None or (c.technique and r.technique != c.technique) or (c.op_class and r.op_class != c.op_class):
            continue
        if consistent(c, r.status):
            supp[target_key(r.target, r.arch)] += 1
            used.append(i)
    for i in c.contradict_ids:
        r = index.get(i)
        if r is not None:
            contra[target_key(r.target, r.arch)] += 1
    return supp, contra, used


def validate_lift(edit: dict, *, index: dict, claims, registry) -> dict:
    """Lift to the lowest common ancestor of the targets with consistent support (>= 2 targets).

    Lifting is stepwise: a claim already at node N moves only if the evidence's
    common ancestor is strictly above N. ``to`` may cap the lift at a node; v1's
    ``to: family|universal`` still work (``universal`` must be what the evidence supports)."""
    c = claims.get(edit.get("claim_id")) if isinstance(edit.get("claim_id"), str) else None
    if c is None or c.status != ACTIVE:
        raise EditError(f"unknown or retired claim {edit.get('claim_id')!r}")
    ev = edit.get("evidence") or []
    if not isinstance(ev, list) or any(i not in index for i in ev):
        raise EditError("lift evidence cites ids not in the outcome log")
    cur = _claim_home(c, registry)
    supp, contra, used = per_target_evidence(c, index, ev)
    good = sorted(t for t in supp if supp[t] > contra[t])
    counts = f"(support {dict(supp)}, contradict {dict(contra)})"
    if len(good) < 2:
        raise EditError(f"lift needs consistent support on >= 2 targets, has {good or 'none'} {counts}")
    dest = registry.lca(good)
    to = edit.get("to")
    if to == UNIVERSAL and dest != UNIVERSAL:
        raise EditError(f"lift to universal needs support under >= 2 families (children of universal); "
                        f"the evidence supports {dest} {counts}")
    if isinstance(to, str) and to in registry.nodes and to != dest:
        if registry.is_above(to, dest):
            raise EditError(f"evidence supports {dest}, not {to} {counts}")
        if registry.is_above(dest, to):
            dest = to                                                 # a shorter step was asked for
    if not registry.is_above(dest, cur):
        raise EditError(f"lift needs consistent support on >= 2 targets whose common ancestor is above "
                        f"{cur}; {good} meet at {dest} {counts}")
    bad = [t for t in contra if contra[t] >= supp.get(t, 0) and dest in registry.chain(t)]
    if bad:
        raise EditError(f"targets under {dest} disagree: {sorted(bad)} {counts}")
    return {"kind": LIFT, "claim_id": c.id, "to": dest, "from": cur, "evidence": used, "targets": good,
            "text": _lift_text(edit)}


def _lift_text(edit: dict) -> Optional[str]:
    t = edit.get("text")
    return t.strip()[:MAX_TEXT_CHARS] if isinstance(t, str) and t.strip() else None


def apply_lift(e: dict, claims) -> dict:
    with claims._lock:
        c = claims.get(e["claim_id"])
        c.add_support(e["evidence"])
        frm = e.get("from") or c.level
        set_claim_scope(c, e["to"])
        if e.get("text"):                       # the general wording (no target specifics)
            c.text = e["text"]
        claims._touch(c)
    return {"kind": LIFT, "claim_id": c.id, "result": f"lifted {frm}->{e['to']}", "support": c.support,
            "contradict": c.contradict, "evidence": e["evidence"], "targets": e.get("targets")}


def set_claim_scope(c: Claim, sid: str, registry=None) -> None:
    """Move a claim to scope id ``sid`` (universal | node id | target key)."""
    if sid == UNIVERSAL:
        c.level, c.family, c.arch = UNIVERSAL, None, None
    elif ":" in sid:                            # back to one target
        c.level, c.family = TARGET, None
        c.target, c.arch = sid.split(":", 1)
    else:
        c.level, c.family, c.arch = NODE, sid, None


def _rollback_scope(registry, good: list, bad: list, supp: Counter) -> Optional[str]:
    """Where a claim goes when ``bad`` targets disagree: the node that covers the most
    supporting targets (>= 2; deeper on ties) and none of the disagreeing ones, else the
    best-supported single target, else None (retire)."""
    best, best_key = None, None
    for g in good:
        for n in registry.chain(g)[1:]:
            if any(n in registry.chain(b) for b in bad):
                break
            cov = [x for x in good if n in registry.chain(x)]
            key = (len(cov), registry.depth(n))
            if len(cov) >= 2 and (best_key is None or key > best_key):
                best, best_key = n, key
    if best is not None:
        return best
    if good:
        return sorted(good, key=lambda t: (-supp[t], t))[0]
    return None


def revalidate_lifts(claims, index: dict, registry, mirror=None, db=None) -> list[dict]:
    """Roll back lifts the later evidence no longer supports (no human review, so this is automatic).

    For every claim above target level: if some target under its scope now has
    contradictions >= support, the claim moves down to the largest node that covers
    >= 2 still-supporting targets and none of the disagreeing ones, else back to the
    best-supported single target, and is retired when none is left."""
    from ..claims import RETIRED
    events = []
    with claims._lock:
        for c in list(claims.active()):
            sid = claim_scope(c, registry)
            if sid is None:
                continue
            supp, contra, _ = per_target_evidence(c, index)
            under = lambda t: sid in registry.chain(t)                      # noqa: E731
            bad = sorted(t for t in contra if under(t) and contra[t] >= supp.get(t, 0))
            if not bad:
                continue
            good = sorted(t for t in supp if under(t) and supp[t] > contra[t])
            dest = _rollback_scope(registry, good, bad, supp)
            if dest == sid:
                continue
            if dest is None:
                c.status = RETIRED
                kind = "claim_retired"
            else:
                set_claim_scope(c, dest)
                kind = "claim_demoted"
            claims._touch(c)
            ev = {"event": kind, "claim_id": c.id, "from": sid, "to": dest, "disagreeing": bad, "supporting": good,
                  "support": dict(supp), "contradict": dict(contra)}
            events.append(ev)
            if db is not None:
                db.audit(kind, c.id, ev, mirror)
    return events


def validate_mechanism(edit: dict, *, index: dict, claims, ontology) -> dict:
    data = edit.get("data") or {}
    if set(data) == {MECHANISM} and isinstance(data[MECHANISM], dict):
        data = data[MECHANISM]
    metric, obs, verdict = data.get("metric"), data.get("observed"), data.get("verdict")
    if metric not in CANONICAL:
        raise EditError(f"mechanism.metric {metric!r} is not a canonical metric")
    if obs not in ("up", "down", "flat") or verdict not in VERDICTS:
        raise EditError("mechanism needs observed up|down|flat and verdict confirmed|contradicted|side_effect")
    scope = edit.get("scope") or {}
    tech = ontology.get(scope.get("technique"))
    ev = edit.get("evidence") or []
    if not isinstance(ev, list) or not ev or any(i not in index for i in ev):
        raise EditError("mechanism must cite outcome ids from the log")
    recs = [index[i] for i in ev]
    if tech is None:
        techs = {r.technique for r in recs}
        tech = ontology.get(techs.pop()) if len(techs) == 1 else None
    if tech is None:
        raise EditError("mechanism needs a known technique")
    ds = [r.metrics_delta.get(metric) for r in recs if isinstance(r.metrics_delta.get(metric), (int, float))]
    if len(ds) < len(recs):
        raise EditError(f"not every cited record has a measured {metric} delta")
    band = flat_band(metric)
    dirs = Counter("up" if d > band else "down" if d < -band else "flat" for d in ds)
    major = dirs.most_common(1)[0][0]
    if major != obs:
        raise EditError(f"observed {obs!r} but the cited deltas are mostly {major!r} ({dict(dirs)})")
    pred = tech.predicts.get(metric)
    if pred is None and verdict != "side_effect":
        raise EditError(f"{tech.id} predicts nothing for {metric}: verdict must be side_effect")
    if pred is not None:
        agrees = direction_agrees(pred, {"up": 1.0, "down": -1.0, "flat": 0.0}[obs] * 2 * band, band)
        want = "confirmed" if agrees else "contradicted"
        if verdict != want and verdict != "side_effect":
            raise EditError(f"{tech.id} predicts {metric} {pred}; observed {obs} means {want}, not {verdict}")
    base = validate_edit({"kind": "revise_description", "text": edit.get("text"), "evidence": ev,
                          "scope": {k: v for k, v in scope.items() if k != "technique"} | {"technique": recs[0].technique}},
                         index=index, claims=claims)
    base["scope"]["technique"] = tech.id if tech.id in {r.technique for r in recs} else recs[0].technique
    return {"kind": MECHANISM, "text": base["text"], "scope": base["scope"],
            "data": {"metric": metric, "observed": obs, "verdict": verdict, "predicted": pred,
                     "technique_id": tech.id},
            "evidence": base["evidence"], "counter_evidence": []}


class SemanticUpdaterV2(SemanticUpdater):
    def __init__(self, kv, *, model: str, **kw):
        self.kv = kv
        kw.setdefault("known_states", set(kv.hierarchy.states))
        super().__init__(kv.outcomes, kv.claims, model=model, **kw)

    def validate_and_apply(self, raw_edits: Any, *, index: dict) -> tuple[list[dict], list[dict]]:
        if isinstance(raw_edits, dict):
            raw_edits = raw_edits.get("edits", [])
        if not isinstance(raw_edits, list):
            return [], [{"edit": raw_edits, "reason": "edits is not a list"}]
        applied, rejected = [], []
        known_t = (self.known_techniques or set()) | set(self.kv.ontology.techniques)
        for raw in raw_edits:
            if len(applied) >= self.max_edits:
                rejected.append({"edit": raw, "reason": "over max_edits"})
                continue
            try:
                kind = raw.get("kind") if isinstance(raw, dict) else None
                if kind == LIFT:
                    e = validate_lift(raw, index=index, claims=self.claims, registry=self.kv.registry)
                    applied.append(apply_lift(e, self.claims))
                    self.kv.db.audit("claim_lifted", e["claim_id"], applied[-1], self.kv.review_path)
                    continue
                if kind == MECHANISM:
                    e = validate_mechanism(raw, index=index, claims=self.claims, ontology=self.kv.ontology)
                else:
                    e = validate_edit(raw, index=index, claims=self.claims, known_techniques=known_t,
                                      known_states=self.known_states | set(self.kv.hierarchy.states))
            except EditError as exc:
                rejected.append({"edit": raw, "reason": str(exc)})
                continue
            except (AttributeError, TypeError, KeyError) as exc:
                rejected.append({"edit": raw, "reason": f"malformed: {exc}"})
                continue
            applied.append(apply_edit(e, self.claims))
        return applied, rejected

    def _context_claims(self, records):
        """v1 context + target-level claims of *other* targets of the same family on the
        batch's techniques / op classes (what a lift would generalise)."""
        out = {c.id: c for c in super()._context_claims(records)}
        reg = self.kv.registry
        keys = {(r.technique, r.op_class) for r in records}
        for c in self.claims.active():                    # claims on other targets / lower scopes
            if (c.technique, c.op_class) in keys and claim_scope(c, reg) != UNIVERSAL:
                out.setdefault(c.id, c)
        return sorted(out.values(), key=lambda c: -c.rank(self.claims.epoch))[: self.max_prompt_claims + 10]

    def lift_candidates(self, records, existing) -> list[str]:
        """Deterministic hints: a target-level claim and batch records from another target of
        its family that agree with it (the model still decides; validate_lift checks)."""
        lines = []
        for c in existing:
            if c.kind not in (APPLICABILITY, PITFALL, PARAMETER, MECHANISM):
                continue
            reg = self.kv.registry
            home = _claim_home(c, reg)
            if home == UNIVERSAL:
                continue
            ev = [r.id for r in records if r.technique == c.technique and (not c.op_class or r.op_class == c.op_class)
                  and home not in reg.chain(target_key(r.target, r.arch)) and consistent(c, r.status)]
            if ev:
                lines.append(f"- claim {c.id} ({home}) is supported by {ev[:6]} on targets outside its scope")
        return lines[:8]

    def build_messages(self, records, existing) -> list:
        """v1 prompt + the v2 instructions + mechanism checks + lift candidates."""
        msgs = super().build_messages(records, existing)
        msgs[0]["content"] += V2_INSTRUCTIONS
        extra = ""
        checks = self.kv.mechanism_lines([r.id for r in records])
        if checks:
            extra += "MECHANISM CHECKS (measured deltas vs predicts):\n" + "\n".join(checks) + "\n\n"
        lifts = self.lift_candidates(records, existing)
        if lifts:
            extra += ("LIFT CANDIDATES (lift only if the lesson really holds on both targets; evidence must "
                      "include these ids):\n" + "\n".join(lifts) + "\n\n")
        if extra:
            msgs[1]["content"] = msgs[1]["content"].replace('Return {"edits": [...]} now.',
                                                            extra + 'Return {"edits": [...]} now.')
        return msgs

    async def maybe_reflect(self, *, force: bool = False) -> list[dict]:
        applied = await super().maybe_reflect(force=force)
        try:
            await self.kv.maybe_upkeep(force=force)
        except Exception as exc:          # upkeep must never break optimisation
            self._log("warning", f"v2 upkeep failed: {exc!r}")
        return applied

    @property
    def total_usage_all(self) -> Counter:
        return self.total_usage + self.kv.usage
