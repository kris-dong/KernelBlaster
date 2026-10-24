# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evidence-driven state discovery with validation (design section 4).

Replaces v1's "NEW_STATE_NEEDED -> copy the nearest state's techniques x0.8".

1. Triggers (deterministic): within one leaf state, one technique's rewards
   split into two clearly different groups (>= 6 outcomes, best two-group
   split with >= 3 per side and a mean difference >= 0.1); or >= 5
   ``unexplained`` profiles (a new root).
2. The LLM gets the cluster (metric vectors with their sources, the two
   groups, diff summaries, the parent's predicate chain, earlier rejected
   proposals) minus a deterministic held-out third, and returns a predicate
   in the DSL, the parent it refines, a mechanism and predicted effects.
3. :func:`validate_proposal` (no LLM): the predicate evaluates (no unknown)
   on >= 80% of the cluster, both sides have >= 3 outcomes, and it reduces
   the within-side reward variance by >= 30% vs the parent on the held-out
   outcomes (on the whole cluster when fewer than 4 are held out; recorded).
   Otherwise the proposal is stored as rejected with the reason.
4. Admitted states are ``probationary``; :func:`review_probation` makes them
   ``core`` after >= 10 more decidable outcomes keep the separation, and
   retires them when it collapses. Every change goes to the review queue.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Optional

from . import predicates as P
from .metrics import CANONICAL
from .states import CORE, PROBATIONARY, RETIRED, STATE_ID, UNEXPLAINED, State

MIN_GROUP_N = 6
MIN_SIDE = 3
MIN_DIFF = 0.1
MIN_UNEXPLAINED = 5
COVERAGE = 0.8
VAR_REDUCTION = 0.3
PROBATION_N = 10
COLLAPSE = 0.1
RETRIGGER_AFTER = 4


def _var(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def held_out(oid: str) -> bool:
    return int(hashlib.sha1(oid.encode()).hexdigest(), 16) % 3 == 0


def best_split(rewards: list[float], min_side: int = MIN_SIDE) -> Optional[tuple[float, float]]:
    """(threshold, mean difference) of the best two-group split of sorted rewards."""
    xs = sorted(rewards)
    best = None
    for i in range(min_side, len(xs) - min_side + 1):
        lo, hi = xs[:i], xs[i:]
        within = (len(lo) * _var(lo) + len(hi) * _var(hi)) / len(xs)
        diff = sum(hi) / len(hi) - sum(lo) / len(lo)
        if best is None or within < best[2]:
            best = ((xs[i - 1] + xs[i]) / 2, diff, within)
    return (best[0], best[1]) if best else None


def _row_reward(r: dict) -> Optional[float]:
    v = (r.get("v2") or {}).get("reward")
    return float(v) if v is not None else None


def _row_vector(r: dict) -> dict:
    return (r.get("v2") or {}).get("metrics_before") or {}


def _row_chain(r: dict) -> list:
    return (r.get("v2") or {}).get("chain") or []


def find_triggers(rows: list[dict], hierarchy, *, seen: Optional[dict] = None) -> list[dict]:
    seen = seen or {}
    groups: dict[tuple, list] = {}
    unexplained = []
    for r in rows:
        rw, chain = _row_reward(r), _row_chain(r)
        if rw is None:
            continue
        if not chain:
            if (r.get("v2") or {}).get("classification") == UNEXPLAINED:
                unexplained.append(r)
            continue
        groups.setdefault((chain[-1], (r.get("v2") or {}).get("item_id") or r["technique"]), []).append(r)
    out = []
    for (state, item), rs in sorted(groups.items()):
        key = f"split|{state}|{item}"
        if len(rs) < MIN_GROUP_N or len(rs) - seen.get(key, -RETRIGGER_AFTER) < RETRIGGER_AFTER:
            continue
        sp = best_split([_row_reward(r) for r in rs])
        if sp and sp[1] >= MIN_DIFF:
            out.append({"kind": "split", "key": key, "state": state, "technique": item,
                        "threshold": sp[0], "diff": sp[1], "rows": rs})
    key = "unexplained"
    if len(unexplained) >= MIN_UNEXPLAINED and len(unexplained) - seen.get(key, -RETRIGGER_AFTER) >= RETRIGGER_AFTER:
        out.append({"kind": "unexplained", "key": key, "state": None, "technique": None, "rows": unexplained})
    return out


# ---------------------------------------------------------------- prompt

DISCOVERY_SYSTEM = """You refine a hierarchy of kernel performance STATES. A state is a predicate over
canonical profiler metrics; techniques' effects are learned per state. Inside one state a technique's
outcomes split into two groups (HIGH reward vs LOW reward). Find a predicate over the metrics of the
kernels BEFORE the attempt that separates the groups, so that the state can be split into a child
state where the technique behaves differently. Use only metrics that are present in most rows
(measured ones are better evidence than estimated ones). If the groups are not separable by the
metrics, say so with {"proposal": null, "why": "..."}.

Predicate DSL: {"all": [...]}, {"any": [...]}, {"not": p}, {"<metric>": "<|<=|>|>=", "v": number},
{"<metric>": "<", "ref": "<other metric>"}, {"op_class": "in", "v": [...]}. Canonical metrics: {metrics}.

Return ONLY JSON: {"proposal": {"id": "snake_case_state_id", "parent": "<state id>",
 "predicate": {...}, "mechanism": "why kernels in this sub-state respond differently",
 "predicted_effects": {"<technique id>": "helps|hurts|no_effect"}}}"""


def build_discovery_messages(trig: dict, hierarchy, rejected: list[dict]) -> list[dict]:
    rows = [r for r in trig["rows"] if not held_out(r["id"])]
    thr = trig.get("threshold")
    lines = []
    for r in rows:
        rw = _row_reward(r)
        grp = ("HIGH" if rw > thr else "LOW") if thr is not None else "-"
        vec = {k: (m["v"], "m" if m["src"] == "measured" else "e") for k, m in _row_vector(r).items()}
        lines.append(json.dumps({"id": r["id"], "group": grp, "reward": round(rw, 4), "status": r["status"],
                                 "problem": r.get("problem_id"), "op_class": r.get("op_class"),
                                 "target": f"{r.get('target')}:{r.get('arch')}", "diff": r.get("diff_summary"),
                                 "metrics(value,m=measured|e=estimated)": vec}))
    parent = trig.get("state")
    pchain = hierarchy.predicate_chain(parent) if parent else []
    parts = [f"STATE: {parent or '(none: these profiles match no root state)'}; technique: {trig.get('technique')}",
             "PARENT PREDICATE CHAIN: " + json.dumps(pchain),
             f"ROWS ({len(rows)}; reward = partition time saved / partition time):"] + lines
    if rejected:
        parts += ["", "EARLIER PROPOSALS FOR THIS CLUSTER WERE REJECTED:"]
        parts += [f"- {json.dumps(p.get('proposal'))}: {p.get('reason')}" for p in rejected[-3:]]
    system = DISCOVERY_SYSTEM.replace("{metrics}", ", ".join(CANONICAL))
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(parts)}]


# ---------------------------------------------------------------- validation (no LLM)

def separation(pred: dict, rows: list[dict]) -> dict:
    sides: dict[bool, list] = {True: [], False: []}
    unknown = 0
    for r in rows:
        v = P.evaluate(pred, {k: m["v"] for k, m in _row_vector(r).items()},
                       {"op_class": r.get("op_class"), "dtype": r.get("dtype")})
        if v is None:
            unknown += 1
        else:
            sides[v].append(_row_reward(r))
    allr = sides[True] + sides[False]
    n = len(allr)
    within = (len(sides[True]) * _var(sides[True]) + len(sides[False]) * _var(sides[False])) / n if n else 0.0
    base = _var(allr)
    return {"n": len(rows), "decidable": n, "coverage": n / len(rows) if rows else 0.0,
            "n_true": len(sides[True]), "n_false": len(sides[False]),
            "mean_true": sum(sides[True]) / len(sides[True]) if sides[True] else None,
            "mean_false": sum(sides[False]) / len(sides[False]) if sides[False] else None,
            "var_parent": base, "var_within": within,
            "reduction": (1 - within / base) if base > 0 else 0.0}


def validate_proposal(raw: dict, trig: dict, hierarchy) -> tuple[bool, str, dict]:
    """-> (admit, reason, numbers). Never raises on bad LLM output."""
    if not isinstance(raw, dict):
        return False, "no proposal", {}
    sid = str(raw.get("id", ""))
    if not STATE_ID.fullmatch(sid):
        return False, f"id {sid!r} is not snake_case", {}
    if sid in hierarchy.states:
        return False, f"state {sid!r} already exists", {}
    parent = raw.get("parent")
    if trig["kind"] == "split" and parent not in hierarchy.ancestors(trig["state"]):
        return False, f"parent {parent!r} is not {trig['state']} or one of its ancestors", {}
    if trig["kind"] == "unexplained" and parent is not None:
        return False, "an unexplained cluster needs a new root (parent null)", {}
    try:
        P.validate(raw.get("predicate"))
    except P.PredicateError as e:
        return False, f"bad predicate: {e}", {}
    pred = raw["predicate"]
    if pred is True or pred is None:
        return False, "predicate is trivially true", {}
    rows = trig["rows"]
    full = separation(pred, rows)
    if full["coverage"] < COVERAGE:
        return False, f"predicate decidable on {full['coverage']:.0%} of the cluster (< {COVERAGE:.0%})", full
    if trig["kind"] == "unexplained":
        if full["n_true"] < COVERAGE * full["decidable"]:
            return False, f"covers {full['n_true']}/{full['decidable']} unexplained profiles", full
        return True, "covers the unexplained cluster", full
    if min(full["n_true"], full["n_false"]) < MIN_SIDE:
        return False, f"sides {full['n_true']}/{full['n_false']} (need >= {MIN_SIDE} each)", full
    ho = [r for r in rows if held_out(r["id"])]
    hsep = separation(pred, ho) if len(ho) >= 4 else None
    use = hsep if hsep and hsep["n_true"] and hsep["n_false"] else full
    numbers = {"full": full, "held_out": hsep, "judged_on": "held_out" if use is hsep else "full_cluster"}
    if use["reduction"] < VAR_REDUCTION:
        return False, (f"reward variance reduced {use['reduction']:.0%} on the "
                       f"{numbers['judged_on'].replace('_', ' ')} (< {VAR_REDUCTION:.0%})"), numbers
    return True, f"variance reduced {use['reduction']:.0%} on the {numbers['judged_on'].replace('_', ' ')}", numbers


def admit(raw: dict, trig: dict, hierarchy, db, numbers: dict, *, seq: int, mirror=None) -> State:
    pid = "sp_" + uuid.uuid4().hex[:8]
    s = State(id=raw["id"], parent=raw.get("parent"), predicate=raw["predicate"],
              mechanism=str(raw.get("mechanism", ""))[:400], status=PROBATIONARY, source=f"discovered:{pid}",
              order=-1, evidence={"technique": trig.get("technique"), "trigger_state": trig.get("state"),
                                  "admitted_seq": seq, "numbers": numbers,
                                  "predicted_effects": raw.get("predicted_effects") or {}})
    hierarchy.add(s)
    db.put("states", s.id, s.to_dict(), status=s.status)
    db.review("state_admitted", s.id, {"parent": s.parent, "predicate": P.render(s.predicate),
                                       "mechanism": s.mechanism, "numbers": numbers}, mirror)
    return s


def review_probation(hierarchy, rows: list[dict], db, mirror=None) -> list[dict]:
    events = []
    for s in list(hierarchy.states.values()):
        if s.status != PROBATIONARY:
            continue
        since = s.evidence.get("admitted_seq", 0)
        tech = s.evidence.get("technique")
        pop = [r for r in rows if r.get("seq", 0) > since and _row_reward(r) is not None
               and s.parent in _row_chain(r) and ((r.get("v2") or {}).get("item_id") or r["technique"]) == tech]
        sep = separation(s.predicate, pop)
        if sep["decidable"] < PROBATION_N:
            continue
        if sep["reduction"] >= VAR_REDUCTION and min(sep["n_true"], sep["n_false"]) >= MIN_SIDE:
            s.status = CORE
        elif sep["reduction"] < COLLAPSE:
            s.status = RETIRED
        else:
            continue
        db.put("states", s.id, s.to_dict(), status=s.status)
        db.review(f"state_{s.status}", s.id, {"probation": sep}, mirror)
        events.append({"state": s.id, "status": s.status, "reduction": sep["reduction"]})
    return events
