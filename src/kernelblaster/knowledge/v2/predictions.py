# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refining technique predictions from evidence (closes the loop of design section 5).

A technique's ``predicts`` is a prior about which way each canonical metric
moves. Measured outcomes and validated ``mechanism`` claims can say otherwise
-- e.g. coarsening or unrolling a naive kernel wins while ``compute_util``
*falls* (the naive kernel was busy issuing redundant instructions). This pass
turns that evidence into **prediction overrides**, scoped to the most general
state where it holds, and rolls them back when later evidence disagrees:

* for (technique, metric, state s) take the outcomes whose chain contains s
  (s = None: all) with the metric measured before and after; when >= 2/3 of
  them disagree with the technique's prediction there, the majority observed
  direction becomes the prediction for s. It needs ``MIN_N`` outcomes plus an
  eligible mechanism claim on that technique and metric pointing the same way
  (the reflection pass's validated explanation), or ``2 * MIN_N`` outcomes on
  their own;
* the most general such state wins (unconditional > root state > child), and
  a state below an accepted one is not overridden again;
* an override is reverted when >= ``MIN_N`` later outcomes under its state
  disagree with it by the same 2/3 rule.

Overrides live in the DB (``meta.prediction_overrides``) next to the seed --
seed refreshes never touch them -- and every change is in the audit log. The
mechanism check and the prompts use the effective prediction for the slot's
state chain.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from .metrics import MEASURED, flat_band
from .ontology import direction_agrees

MIN_N = 4
SHARE = 2 / 3
META_KEY = "prediction_overrides"


def observed_dir(d: float, band: float) -> str:
    return "up" if d > band else "down" if d < -band else "flat"


def summarize_dirs(dirs: list) -> str:
    """Majority direction, widened to ``x_or_flat`` when the rest is flat."""
    c = Counter(dirs)
    top, _ = c.most_common(1)[0]
    if top == "flat":
        return "flat"
    other = "down" if top == "up" else "up"
    return f"{top}_or_flat" if c.get("flat") and not c.get(other) else top


class PredictionBook:
    def __init__(self, db, hierarchy, mirror=None):
        self.db, self.hierarchy, self.mirror = db, hierarchy, mirror
        self.overrides: dict = db.get_meta(META_KEY, {}) or {}       # tech -> [override]

    def save(self) -> None:
        self.db.set_meta(META_KEY, self.overrides)

    # -- use -----------------------------------------------------------------
    def effective(self, tech: str, base: dict, chain: Optional[list]) -> dict:
        """``base`` predicts with the overrides for this chain (deeper state wins)."""
        out = dict(base or {})
        ovs = self.overrides.get(tech) or []
        order = [None] + list(chain or [])
        for s in order:
            for o in ovs:
                if o["state"] == s:
                    out[o["metric"]] = o["dir"]
        return out

    # -- refine / revert ---------------------------------------------------------
    @staticmethod
    def _measured(r: dict, metric: str) -> Optional[float]:
        b = (r.get("v2") or {}).get("metrics_before", {}).get(metric)
        a = (r.get("v2") or {}).get("metrics_after", {}).get(metric)
        if not b or not a or b.get("src") != MEASURED or a.get("src") != MEASURED:
            return None
        return a["v"] - b["v"]

    def _claim_support(self, claims, tech: str, metric: str, direction: str) -> list[str]:
        out = []
        for c in claims.active() if claims is not None else []:
            d = c.data or {}
            if c.kind == "mechanism" and claims.eligible(c) and d.get("metric") == metric \
                    and (d.get("technique_id") == tech or c.technique == tech) \
                    and d.get("verdict") in ("contradicted", "side_effect") \
                    and direction.startswith(d.get("observed", "?")):
                out.append(c.id)
        return out

    def refine(self, rows: list[dict], ontology, claims=None) -> list[dict]:
        events = []
        by_tech: dict = {}
        for r in rows:
            item = (r.get("v2") or {}).get("item_id") or r["technique"]
            if item in ontology.techniques:
                by_tech.setdefault(item, []).append(r)
        for tech, rs in by_tech.items():
            t = ontology.techniques[tech]
            metrics = set(t.predicts) | {o["metric"] for o in self.overrides.get(tech, [])}
            for metric in sorted(metrics):
                band = flat_band(metric)
                obs = [(r.get("v2", {}).get("chain") or [], d, r.get("seq", 0)) for r in rs
                       if (d := self._measured(r, metric)) is not None]
                if not obs:
                    continue
                events += self._revert(tech, metric, t.predicts, obs, band)
                events += self._propose(tech, metric, t.predicts, obs, band, claims)
        if events:
            self.save()
        return events

    def _subset(self, obs, s):
        return [(c, d, q) for c, d, q in obs if s is None or s in c]

    def _propose(self, tech, metric, base, obs, band, claims) -> list[dict]:
        events = []
        states = [None] + sorted({s for c, _, _ in obs for s in c},
                                 key=lambda s: (len(self.hierarchy.ancestors(s)), s))
        taken = [o["state"] for o in self.overrides.get(tech, []) if o["metric"] == metric]
        for s in states:
            if any(x is None or (s is not None and x in self.hierarchy.ancestors(s)) for x in taken):
                continue                                        # covered by a more general override
            sub = self._subset(obs, s)
            chain = self.hierarchy.ancestors(s)[::-1] if s else []
            pred = self.effective(tech, base, chain).get(metric)
            if len(sub) < MIN_N or pred is None and len(sub) < 2 * MIN_N:
                continue
            dirs = [observed_dir(d, band) for _, d, _ in sub]
            agree = sum(1 for _, d, _ in sub if pred is not None and direction_agrees(pred, d, band))
            if pred is not None and len(sub) - agree < SHARE * len(sub):
                continue
            new = summarize_dirs(dirs)
            if pred is not None and new == pred:
                continue
            if pred is None and Counter(dirs).most_common(1)[0][1] < SHARE * len(sub):
                continue                                        # no consistent direction to learn
            backing = self._claim_support(claims, tech, metric, new)
            if not backing and len(sub) < 2 * MIN_N:
                continue
            o = {"state": s, "metric": metric, "dir": new, "from": pred, "n": len(sub),
                 "disagree_frac": round((len(sub) - agree) / len(sub), 3), "claims": backing[:6],
                 "since_seq": max(q for _, _, q in sub)}
            self.overrides.setdefault(tech, []).append(o)
            taken.append(s)
            ev = {"event": "prediction_refined", "technique": tech, **o}
            self.db.audit("prediction_refined", tech, ev, self.mirror)
            events.append(ev)
        return events

    def _revert(self, tech, metric, base, obs, band) -> list[dict]:
        events, keep = [], []
        for o in self.overrides.get(tech, []):
            if o["metric"] != metric:
                keep.append(o)
                continue
            later = [(c, d, q) for c, d, q in self._subset(obs, o["state"]) if q > o["since_seq"]]
            bad = sum(1 for _, d, _ in later if not direction_agrees(o["dir"], d, band))
            if len(later) >= MIN_N and bad >= SHARE * len(later):
                ev = {"event": "prediction_reverted", "technique": tech, "state": o["state"], "metric": metric,
                      "dir": o["dir"], "back_to": o.get("from"), "later_n": len(later), "disagree": bad}
                self.db.audit("prediction_reverted", tech, ev, self.mirror)
                events.append(ev)
            else:
                keep.append(o)
        if events:
            self.overrides[tech] = keep
        return events
