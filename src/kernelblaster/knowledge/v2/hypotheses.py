# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hypotheses: proposed techniques with a test budget (design section 6).

A hypothesis has a technique's shape (mechanism, preconditions, predicts,
realization for the target) plus a lifecycle::

    proposed -> testing -> promoted | retired

*Triggers* are deterministic (no LLM): ``gap`` (a state where every
applicable technique's estimate is <= 0 after enough tries), ``port`` (a
realization exists for another family/backend but not this one), ``combine``
(two techniques with positive estimates that were never stacked on one
slot), ``profiler`` (NCU rule hints with a high estimated speedup),
``code_mining`` (a verified win whose mechanism check contradicted its
technique) and ``graph`` (ModelBlaster proposals, ids shared with MB's
``modelblaster.graph_hints/v1`` records). One LLM call turns the triggers
into specs; :func:`validate_spec` checks them and dedupes against known
techniques and hypotheses by name and mechanism similarity.

Promotion (:meth:`HypothesisBook.review`) is automatic -- nobody reviews it,
so the bar carries the safeguards: wins (reward > 0, status improved) on >= 2
problems, mean reward >= ``MIN_EFFECT`` (1% of partition time: below that a
"win" is indistinguishable from timing noise on a shared GPU, where the
agent's own win threshold is 3% of the slot), lower confidence bound of the
mean > 0, mechanism confirmed in >= 50% of the measured attempts. The new
technique is scoped to the lowest common ancestor of the targets it won on
(one target -> that target) and is used at once. Retired once the budget is
spent without meeting the bar (the reason is kept).

Promoted techniques keep being checked (:meth:`HypothesisBook.revalidate`):
when later outcomes on some target under its scope average <= 0 over >= 3
attempts, the technique is rescoped to the common ancestor of the targets
where it still pays (or to one target), and retired when none is left.
"""
from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ..claims import normalise, similar
from . import predicates as P
from .ontology import CONFIRMED, MECH_UNKNOWN, Technique, validate_predicts
from .targets import ROOT, TARGET, UNIVERSAL

PROPOSED, TESTING, PROMOTED, RETIRED = "proposed", "testing", "promoted", "retired"
SOURCES = ("gap", "port", "combine", "profiler", "code_mining", "graph")
DEFAULT_BUDGET = 4
EXPLORE_FRAC = 0.15
MIN_WIN_PROBLEMS = 2
MIN_EFFECT = 0.01            # mean reward (fraction of partition time) a promotion needs
DEMOTE_MIN_N = 3             # outcomes on one target before its evidence can demote a technique
NAME_RE = re.compile(r"[a-z][a-z0-9_]{2,63}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Hypothesis:
    name: str
    mechanism: str
    source: str
    how: str                                  # realization text for the target it was proposed for
    target_key: str
    preconditions: Optional[dict] = None
    predicts: dict = field(default_factory=dict)
    states: list = field(default_factory=list)
    op_classes: list = field(default_factory=list)
    risks: list = field(default_factory=list)
    based_on: list = field(default_factory=list)      # technique ids it ports / combines
    trigger: dict = field(default_factory=dict)
    status: str = PROPOSED
    budget: int = DEFAULT_BUDGET
    attempts: list = field(default_factory=list)      # [{outcome_id, problem_id, target_key, reward, status, mechanism}]
    reason: str = ""
    promoted_to: Optional[str] = None
    kind: str = "slot"                                # slot | graph
    id: str = field(default_factory=lambda: "h_" + uuid.uuid4().hex[:10])
    created: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Hypothesis":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})

    @property
    def open(self) -> bool:
        return self.status in (PROPOSED, TESTING)

    def as_technique_desc(self) -> str:
        return f"[hypothesis under test] {self.mechanism}. How: {self.how}"


def _jacc(a: str, b: str) -> float:
    ta, tb = set(normalise(a).split()), set(normalise(b).split())
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def similar_mechanism(a: str, b: str) -> bool:
    return similar(normalise(a), normalise(b)) or _jacc(a, b) >= 0.6


def validate_spec(raw: dict, *, ontology, existing: list, target_key: str, known_states: set) -> Hypothesis:
    """Raw LLM spec -> Hypothesis; raises ValueError with the reason."""
    if not isinstance(raw, dict):
        raise ValueError("spec is not an object")
    name = str(raw.get("name", "")).strip().lower()
    if not NAME_RE.fullmatch(name):
        raise ValueError(f"name {name!r} must be snake_case")
    mech = str(raw.get("mechanism", "")).strip()
    how = str(raw.get("how", "")).strip()
    if len(mech) < 15 or len(how) < 15:
        raise ValueError("mechanism and how must be real sentences")
    src = raw.get("source")
    if src not in SOURCES:
        raise ValueError(f"source {src!r} not in {SOURCES}")
    pre = raw.get("preconditions") or None
    if pre is not None:
        P.validate(pre)
        bad = P.unit_problems(pre)
        if bad:
            raise ValueError("preconditions in the wrong unit: " + "; ".join(bad))
    pred = raw.get("predicts") or {}
    validate_predicts(pred)
    states = [s for s in (raw.get("states") or []) if s in known_states]
    if raw.get("states") and not states:
        raise ValueError(f"states {raw.get('states')} are not known states")
    if ontology.resolve(name):
        raise ValueError(f"{name!r} is already technique {ontology.resolve(name)!r}")
    for t in ontology.techniques.values():
        if t.slot and similar_mechanism(mech, t.mechanism):
            raise ValueError(f"mechanism duplicates technique {t.id!r}")
    for h in existing:
        if h.name == name or similar_mechanism(mech, h.mechanism):
            raise ValueError(f"duplicates hypothesis {h.id} ({h.name})")
    based = [ontology.resolve(b) for b in (raw.get("based_on") or []) if ontology.resolve(b)]
    return Hypothesis(name=name, mechanism=mech[:400], source=src, how=how[:800], target_key=target_key,
                      preconditions=pre, predicts=pred, states=states,
                      op_classes=[c for c in (raw.get("op_classes") or []) if isinstance(c, str)],
                      risks=[str(r)[:200] for r in (raw.get("risks") or [])][:4], based_on=based,
                      trigger=raw.get("trigger") if isinstance(raw.get("trigger"), dict) else {})


class HypothesisBook:
    """Hypotheses in the knowledge DB; selection eligibility, attempts and the lifecycle."""

    def __init__(self, db, registry, ontology):
        self.db, self.registry, self.ontology = db, registry, ontology
        self.items: dict[str, Hypothesis] = {}
        self.reload()

    def reload(self) -> None:
        self.items = {d["id"]: Hypothesis.from_dict(d) for d in self.db.rows("hypotheses")}

    def save(self, h: Hypothesis) -> None:
        self.items[h.id] = h
        self.db.put("hypotheses", h.id, h.to_dict(), status=h.status)

    def add(self, h: Hypothesis, mirror=None) -> Hypothesis:
        self.save(h)
        self.db.audit("hypothesis_proposed", h.id, {"name": h.name, "source": h.source,
                                                    "mechanism": h.mechanism, "target": h.target_key}, mirror)
        return h

    def open(self) -> list[Hypothesis]:
        return [h for h in self.items.values() if h.open]

    def testable(self, *, target_key: str, op_class: str, chain: list, values: dict, ctx: dict) -> list[Hypothesis]:
        out = []
        for h in self.open():
            if h.kind != "slot" or len(h.attempts) >= h.budget:
                continue
            if self.registry.lca([h.target_key, target_key]) == ROOT:   # only below a shared node
                continue
            if h.op_classes and op_class not in h.op_classes:
                continue
            if h.states and not set(h.states) & set(chain):
                continue
            if P.evaluate(h.preconditions, values, ctx) is False:
                continue
            out.append(h)
        return sorted(out, key=lambda h: (len(h.attempts), h.created))

    def record_attempt(self, hid: str, *, outcome_id: str, problem_id: str, target_key: str,
                       reward: float, status: str, mechanism: str, measured: bool) -> Optional[Hypothesis]:
        h = self.items.get(hid)
        if h is None:
            return None
        h.attempts.append({"outcome_id": outcome_id, "problem_id": problem_id, "target_key": target_key,
                           "reward": round(float(reward), 5), "status": status, "mechanism": mechanism,
                           "measured": measured})
        if h.status == PROPOSED:
            h.status = TESTING
        self.save(h)
        return h

    # -- lifecycle ------------------------------------------------------------
    @staticmethod
    def verdict(h: Hypothesis) -> tuple[bool, str]:
        """(promotable, reason) from the attempts so far."""
        rs = [a["reward"] for a in h.attempts]
        wins = [a for a in h.attempts if a["reward"] > 0 and a["status"] == "improved"]
        win_probs = {a["problem_id"] for a in wins}
        if len(win_probs) < MIN_WIN_PROBLEMS:
            return False, f"wins on {len(win_probs)} problem(s) < {MIN_WIN_PROBLEMS}"
        n = len(rs)
        mean = sum(rs) / n
        sd = math.sqrt(sum((r - mean) ** 2 for r in rs) / (n - 1)) if n > 1 else float("inf")
        lcb = mean - sd / math.sqrt(n)
        if mean < MIN_EFFECT:
            return False, f"mean reward {mean:.4f} < minimum effect {MIN_EFFECT}"
        if lcb <= 0:
            return False, f"mean reward {mean:.3f}, LCB {lcb:.3f} <= 0"
        meas = [a for a in h.attempts if a.get("measured") and a["mechanism"] != MECH_UNKNOWN]
        if meas:
            conf = sum(1 for a in meas if a["mechanism"] == CONFIRMED) / len(meas)
            if conf < 0.5:
                return False, f"mechanism confirmed in {conf:.0%} of {len(meas)} measured attempts"
        return True, (f"wins on {len(win_probs)} problems, mean {mean:.3f}, LCB {lcb:.3f}, "
                      + (f"mechanism confirmed {conf:.0%} of {len(meas)} measured" if meas
                         else "mechanism unmeasured"))

    def evidence_scope(self, h: Hypothesis) -> str:
        """Lowest common ancestor of the targets the hypothesis won on (one target -> itself)."""
        tks = sorted({a["target_key"] for a in h.attempts if a["reward"] > 0}) or [h.target_key]
        return self.registry.lca(tks)

    def promote(self, h: Hypothesis, reason: str, mirror=None) -> Technique:
        scope = self.evidence_scope(h)
        level = self.registry.level_of(scope)
        rscope = (f"target:{scope}" if level == TARGET else
                  f"node:{scope}" if scope != ROOT else f"node:{self.registry.family_of(h.target_key)}")
        tid = h.name if not self.ontology.resolve(h.name) else f"{h.name}_{h.id[2:6]}"
        t = Technique(id=tid, mechanism=h.mechanism, level=level, scope=None if scope == ROOT else scope,
                      preconditions=h.preconditions,
                      predicts=h.predicts, risks=h.risks, states=h.states, op_classes=h.op_classes, slot=True,
                      realizations=[{"scope": rscope, "how": h.how}], source=f"hypothesis:{h.id}",
                      relations={"based_on": h.based_on} if h.based_on else {})
        self.ontology.add(t)
        self.db.put("techniques", t.id, t.to_dict())
        h.status, h.reason, h.promoted_to = PROMOTED, reason, t.id
        self.save(h)
        self.db.audit("hypothesis_promoted", h.id, {"technique": t.id, "level": level, "scope": scope,
                                                   "reason": reason, "attempts": len(h.attempts)}, mirror)
        return t

    def revalidate(self, rows: list[dict], mirror=None) -> list[dict]:
        """Demote / retire promoted techniques that later outcomes contradict (automatic rollback)."""
        events = []
        for t in list(self.ontology.techniques.values()):
            if not t.source.startswith("hypothesis:") or t.status != "active":
                continue
            sid = self.registry.scope_id(t.level, t.scope)
            by: dict[str, list] = {}
            for r in rows:
                if ((r.get("v2") or {}).get("item_id") or r["technique"]) == t.id and \
                        (r.get("v2") or {}).get("reward") is not None:
                    by.setdefault(self.registry.get(f"{r['target']}:{r['arch']}").key, []).append(
                        float(r["v2"]["reward"]))
            under = lambda k: sid in self.registry.chain(k)                       # noqa: E731
            neg = {k for k, xs in by.items() if len(xs) >= DEMOTE_MIN_N and sum(xs) / len(xs) <= 0}
            bad = sorted(k for k in neg if under(k))
            if not bad:
                continue
            h = self.items.get(t.source.split(":", 1)[1])
            won = {a["target_key"] for a in (h.attempts if h else []) if a["reward"] > 0}
            good = sorted(k for k in set(by) | won if k not in neg and under(k)
                          and (k not in by or sum(by[k]) / len(by[k]) >= MIN_EFFECT))
            dest = self.registry.lca(good) if good else None
            stats = {k: [len(xs), round(sum(xs) / len(xs), 4)] for k, xs in by.items()}
            if dest is None:
                t.status, t.slot = "retired", False
                kind = "technique_retired"
            elif dest != sid:
                t.level, t.scope = self.registry.level_of(dest), (None if dest == ROOT else dest)
                if self.registry.level_of(dest) == TARGET:
                    t.realizations = [dict(r, scope=f"target:{dest}") for r in t.realizations[:1]]
                kind = "technique_demoted"
            else:
                continue
            self.db.put("techniques", t.id, t.to_dict())
            ev = {"event": kind, "technique": t.id, "from": sid, "to": dest, "bad_targets": bad, "per_target": stats}
            self.db.audit(kind, t.id, ev, mirror)
            events.append(ev)
        return events

    def retire(self, h: Hypothesis, reason: str, mirror=None) -> None:
        h.status, h.reason = RETIRED, reason
        self.save(h)
        self.db.audit("hypothesis_retired", h.id, {"name": h.name, "reason": reason,
                                                   "attempts": len(h.attempts)}, mirror)

    def review(self, mirror=None) -> list[dict]:
        """Promote or retire every open hypothesis whose evidence decides it."""
        events = []
        for h in list(self.open()):
            if h.kind != "slot":
                continue
            bad = P.unit_problems(h.preconditions)
            if bad:                          # untestable as written (stored before the unit check)
                self.retire(h, "preconditions in the wrong unit: " + "; ".join(bad), mirror)
                events.append({"event": "retired", "id": h.id, "reason": h.reason})
                continue
            ok, why = self.verdict(h)
            if ok:
                t = self.promote(h, why, mirror)
                events.append({"event": "promoted", "id": h.id, "technique": t.id, "level": t.level, "reason": why})
            elif len(h.attempts) >= h.budget:
                self.retire(h, f"budget of {h.budget} attempts spent: {why}", mirror)
                events.append({"event": "retired", "id": h.id, "reason": h.reason})
        return events

    # -- graph-level hypotheses (ModelBlaster proposals) ------------------------
    def from_proposals(self, props: dict, target_key: str, mirror=None) -> list[Hypothesis]:
        """``kb-proposal/1`` -> graph hypotheses keyed like MB's graph_hints ids (fuse:<kind>, split:<slot>)."""
        out = []
        for p in props.get("proposals", []):
            if p.get("kind") == "fuse":
                hid = "fuse:fused__" + "__".join(p["pattern"])
                name = "fuse_" + "_".join(p["pattern"])
            elif p.get("kind") == "split":
                hid, name = f"split:{p['slot']}", f"split_{p['slot']}"
            else:
                continue
            h = self.items.get(hid) or Hypothesis(
                id=hid, name=re.sub(r"[^a-z0-9_]", "_", name.lower())[:64], kind="graph", source="graph",
                mechanism=p.get("rationale", "")[:400] or hid, how=json.dumps({k: p[k] for k in p if k != "rationale"})[:800],
                target_key=target_key, budget=0)
            h.trigger = {"evidence": p.get("evidence", {})}
            if hid not in self.items:
                self.add(h, mirror)
            else:
                self.save(h)
            out.append(h)
        return out

    def sync_graph_hints(self, doc: dict) -> int:
        """Fold ModelBlaster's decisions (graph_hints/v1 accepted|rejected) into graph hypotheses."""
        n = 0
        for hint in doc.get("hints", []):
            h = self.items.get(hint.get("id"))
            if h is None or h.kind != "graph":
                continue
            st = {"accepted": PROMOTED, "rejected": RETIRED}.get(hint.get("status"))
            if st and h.status != st:
                h.status, h.reason = st, f"ModelBlaster graph_hints: {hint.get('status')} " \
                                         f"({len(hint.get('evidence', []))} decisions)"
                self.save(h)
                n += 1
        return n


# ---------------------------------------------------------------- generation prompt

GEN_SYSTEM = """You propose NEW kernel-optimisation techniques as testable hypotheses.
Each trigger below says why the current technique catalogue is not enough (a state where
nothing known helps, a technique that exists for another target but not this one, two
winners never stacked, a profiler hint, a win nobody can explain). For each trigger worth
it, propose ONE hypothesis: a concrete, implementable change to a single kernel, with its
mechanism (why it should be faster), preconditions (predicate DSL over canonical metrics),
predicted metric directions, and how to realise it on the given target.

Do not restate a known technique under a new name; build on it (based_on) instead.
Predicate DSL: {"all": [...]}, {"any": [...]}, {"not": p}, {"<metric>": "<|<=|>|>=", "v": number},
{"op_class": "in", "v": ["gemm", ...]}. Canonical metrics (with units): {metrics}.
Utilisations, hit rates, occupancy and reg_pressure are PERCENT (0-100, e.g. {"bw_util": "<", "v": 60});
parallel_fill and launch_frac are fractions (0-1); arith_intensity is FLOP/byte.
predicts: {"<metric>": "up|down|flat|up_or_flat|down_or_flat"}.

Return ONLY JSON: {"hypotheses": [{"trigger": <trigger index>, "source": "gap|port|combine|profiler|code_mining",
 "name": "snake_case", "mechanism": "...", "how": "target-specific realization, 1-3 sentences",
 "preconditions": {...} or null, "predicts": {...}, "states": ["state ids it targets"],
 "op_classes": ["..."], "based_on": ["technique ids"], "risks": ["..."]}]}
At most {k} hypotheses. Return {"hypotheses": []} if no trigger justifies one."""


def build_generation_messages(triggers: list[dict], *, techniques: list[dict], hypotheses: list[dict],
                              metrics: list[str], k: int) -> list[dict]:
    system = GEN_SYSTEM.replace("{metrics}", ", ".join(metrics)).replace("{k}", str(k))
    parts = ["TRIGGERS:"] + [json.dumps({"index": i, **t}, default=str) for i, t in enumerate(triggers)]
    parts += ["", "KNOWN TECHNIQUES (id: mechanism):"] + [f"- {t['id']}: {t['mechanism']}" for t in techniques]
    if hypotheses:
        parts += ["", "EXISTING HYPOTHESES (do not duplicate; retired ones failed):"]
        parts += [f"- {h['name']} [{h['status']}]: {h['mechanism']}" + (f" ({h['reason']})" if h.get("reason") else "")
                  for h in hypotheses]
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(parts)}]
