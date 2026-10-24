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

from ..claims import (ACTIVE, APPLICABILITY, INTERACTION, MECHANISM, PARAMETER, PITFALL, Claim)
from ..outcomes import COMPILE_ERROR, IMPROVED, NEUTRAL, REGRESSED, RUNTIME_ERROR, WRONG_OUTPUT
from ..reflect import EditError, SemanticUpdater, apply_edit, validate_edit
from .metrics import CANONICAL, flat_band
from .ontology import direction_agrees
from .targets import FAMILY, TARGET, UNIVERSAL, target_key

LIFT = "lift"
FAILURES = (COMPILE_ERROR, WRONG_OUTPUT, RUNTIME_ERROR)
VERDICTS = ("confirmed", "contradicted", "side_effect")
LEVEL_ORDER = {TARGET: 0, FAMILY: 1, UNIVERSAL: 2}

V2_INSTRUCTIONS = """
Knowledge v2 additions:
- Claims have a level: target (one GPU/arch), family (e.g. every SIMT GPU), universal. New claims
  start at target level. States are hierarchical (e.g. memory_bound > cache_thrashing); scope a claim
  to the most general state it holds for.
- lift: {"kind": "lift", "claim_id": "...", "to": "family|universal", "evidence": [ids]} when a claim
  holds on SEVERAL targets: family needs consistent support on >= 2 targets of one family (e.g.
  cuda:sm_75 and cuda:sm_89), universal on >= 2 families. Checked against the cited records' targets.
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


def validate_lift(edit: dict, *, index: dict, claims, registry) -> dict:
    c = claims.get(edit.get("claim_id")) if isinstance(edit.get("claim_id"), str) else None
    if c is None or c.status != ACTIVE:
        raise EditError(f"unknown or retired claim {edit.get('claim_id')!r}")
    to = edit.get("to")
    cur = getattr(c, "level", TARGET) or TARGET
    if to not in (FAMILY, UNIVERSAL) or LEVEL_ORDER[to] <= LEVEL_ORDER[cur]:
        raise EditError(f"lift.to={to!r} must be above the claim's level {cur!r}")
    ev = edit.get("evidence") or []
    if not isinstance(ev, list) or any(i not in index for i in ev):
        raise EditError("lift evidence cites ids not in the outcome log")
    ids = list(dict.fromkeys(list(c.support_ids) + list(ev)))
    supp, contra, used = Counter(), Counter(), []
    for i in ids:
        r = index.get(i)
        if r is None:
            continue
        if c.technique and r.technique != c.technique:
            continue
        if c.op_class and r.op_class != c.op_class:
            continue
        if consistent(c, r.status):
            supp[target_key(r.target, r.arch)] += 1
            used.append(i)
    for i in c.contradict_ids:
        r = index.get(i)
        if r is not None:
            contra[target_key(r.target, r.arch)] += 1
    good = [t for t in supp if supp[t] > contra[t]]
    fam_of = {t: registry.family_of(t) for t in good}
    if to == FAMILY:
        home = c.family or registry.family_of(target_key(c.target, c.arch or "unknown"))
        good_home = [t for t in good if fam_of[t] == home]
        if len(good_home) < 2:
            raise EditError(f"lift to family {home} needs consistent support on >= 2 of its targets, "
                            f"has {sorted(good_home) or 'none'} (support {dict(supp)}, contradict {dict(contra)})")
        return {"kind": LIFT, "claim_id": c.id, "to": FAMILY, "family": home, "evidence": used,
                "targets": sorted(good_home)}
    fams = sorted(set(fam_of.values()))
    if len(fams) < 2:
        raise EditError(f"lift to universal needs support on >= 2 families, has {fams or 'none'}")
    return {"kind": LIFT, "claim_id": c.id, "to": UNIVERSAL, "family": None, "evidence": used, "families": fams}


def apply_lift(e: dict, claims) -> dict:
    with claims._lock:
        c = claims.get(e["claim_id"])
        c.add_support(e["evidence"])
        frm = c.level
        c.level, c.family = e["to"], e["family"]
        c.arch = None
        claims._touch(c)
    return {"kind": LIFT, "claim_id": c.id, "result": f"lifted {frm}->{e['to']}", "support": c.support,
            "contradict": c.contradict, "evidence": e["evidence"],
            "targets": e.get("targets") or e.get("families")}


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
                    self.kv.db.review("claim_lifted", e["claim_id"], applied[-1], self.kv.review_path)
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

    def build_messages(self, records, existing) -> list:
        """v1 prompt + the v2 instructions + the mechanism checks of these records."""
        msgs = super().build_messages(records, existing)
        msgs[0]["content"] += V2_INSTRUCTIONS
        checks = self.kv.mechanism_lines([r.id for r in records])
        if checks:
            msgs[1]["content"] = msgs[1]["content"].replace(
                'Return {"edits": [...]} now.', "MECHANISM CHECKS (measured deltas vs predicts):\n"
                + "\n".join(checks) + '\n\nReturn {"edits": [...]} now.')
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
