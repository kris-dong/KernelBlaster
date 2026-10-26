# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Semantic claims distilled from outcome records.

A :class:`Claim` is one short, scoped statement ("tensor_core_utilization
helps gemm on sm_89 when K is a multiple of 16"), the outcome-record ids
that support it, and the ids that contradict it. Claims are produced by the
reflection pass (:mod:`kernelblaster.knowledge.reflect`) as validated,
structured edits, never as free text pasted into the store.

Rules (see ``docs/semantic_update_pass.md`` for rationale):

* support / contradiction are counts of *distinct* outcome records; citing
  the same record twice does not double count, and a record cannot both
  support and contradict one claim;
* contradicting claims live side by side with their counts; nothing is
  overwritten;
* a claim reaches prompts only when ``support >= min_support(kind)`` and
  ``support > contradict``;
* time is measured in reflection *epochs*; claims that are not eligible
  and have not been touched for ``AGE_OUT_EPOCHS`` are retired, and ranking
  decays with a half-life of ``HALF_LIFE_EPOCHS``;
* claims are scoped by target (always), arch, op_class, technique, state,
  and (impl notes) problem/slot/impl; retrieval never crosses targets and a
  scoped field that is set on both sides must match;
* near-duplicates (same kind and scope, similar normalised text, same
  numbers) are merged on insert and by :meth:`ClaimStore.dedupe`.

The store also keeps a deterministic per-(problem, slot) tally of attempts
(:meth:`ClaimStore.observe`), so "already tried" knowledge is available to
parallel rollouts immediately, without waiting for a reflection pass.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Optional

from .outcomes import IMPROVED, OutcomeRecord

# ---------------------------------------------------------------------------
# Kinds and constants
# ---------------------------------------------------------------------------

APPLICABILITY = "applicability"
PITFALL = "pitfall"
INTERACTION = "interaction"
PARAMETER = "parameter"
REVISE_DESCRIPTION = "revise_description"
STATE_SPLIT = "state_split"
IMPL_NOTE = "impl_note"
MECHANISM = "mechanism"          # v2: a technique's predicted vs observed metric effects
CLAIM_KINDS = (APPLICABILITY, PITFALL, INTERACTION, PARAMETER, REVISE_DESCRIPTION,
               STATE_SPLIT, IMPL_NOTE, MECHANISM)
#: X-145: a fact measured by a recorded probe program (knowledge/v2/experiments.py). Deliberately NOT in CLAIM_KINDS
#: (the reflection pass's edit schema / prompt is built from it and must not change); written only by the experiment
#: ingest, hidden from ``ClaimStore.relevant`` unless asked for by kind.
EXPERIMENT = "experiment"
VALID_KINDS = CLAIM_KINDS + (EXPERIMENT,)

ACTIVE = "active"
RETIRED = "retired"

#: scope fields, in the order used for keys and rendering
SCOPE_FIELDS = ("target", "arch", "op_class", "state", "technique", "problem_id", "slot", "impl_id")
#: fields that, when set on a claim, require an exact match from the query
#: (problem-local fields: an unset query value does NOT match a set claim value)
LOCAL_FIELDS = ("problem_id", "slot")

MIN_SUPPORT = 2                 # default prompt threshold
# one observed dead end on a slot is the fact; a pitfall must cite a real failure
# (validated) and its fix is cheap to try, so one is enough for those two kinds
MIN_SUPPORT_BY_KIND = {IMPL_NOTE: 1, PITFALL: 1, EXPERIMENT: 1}
AGE_OUT_EPOCHS = 12             # retire ineligible claims untouched this long
HALF_LIFE_EPOCHS = 30           # ranking decay of eligible claims
PURGE_RETIRED_EPOCHS = 100      # drop retired claims from the file after this
MAX_ACTIVE_CLAIMS = 2000
MAX_CITED_IDS = 64              # ids kept per claim (counts keep growing)
MAX_TEXT_CHARS = 400
SIM_THRESHOLD = 0.85            # SequenceMatcher ratio on normalised text
JACCARD_THRESHOLD = 0.8

_STOP = frozenset("a an the of to in on for and or is are be when with this that it its by as at "
                  "from than then use using".split())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalise(text: str) -> str:
    t = re.sub(r"[^a-z0-9_.x]+", " ", (text or "").lower())
    toks = [w.strip(".") for w in t.split()]
    return " ".join(w for w in toks if w and w not in _STOP)


def _numbers(norm: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", norm))


def similar(a: str, b: str) -> bool:
    """Cheap near-duplicate test on *normalised* strings."""
    if a == b:
        return True
    if _numbers(a) != _numbers(b):   # "tile 64" vs "tile 128" are different claims
        return False
    ta, tb = set(a.split()), set(b.split())
    if ta and tb and len(ta & tb) / len(ta | tb) >= JACCARD_THRESHOLD:
        return True
    return SequenceMatcher(None, a, b).ratio() >= SIM_THRESHOLD


# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    kind: str
    text: str
    target: str = "cuda"
    arch: Optional[str] = None
    op_class: Optional[str] = None
    state: Optional[str] = None
    technique: Optional[str] = None
    problem_id: Optional[str] = None
    slot: Optional[str] = None
    impl_id: Optional[str] = None
    #: knowledge v2 (docs/knowledge_architecture.md section 8): abstraction level
    #: (target | family | universal), the family for family-level claims, and the
    #: hypothesis a claim is about. v1 claims are all target-level.
    level: str = "target"
    family: Optional[str] = None
    hypothesis_id: Optional[str] = None
    #: kind-specific structured payload (see ``reflect.EDIT_SCHEMA``):
    #: applicability {effect}, pitfall {symptom, fix}, interaction {other, relation},
    #: parameter {params}, state_split {proposed_state, criterion}, impl_note {note_type}
    data: dict = field(default_factory=dict)
    support: int = 0
    contradict: int = 0
    support_ids: list = field(default_factory=list)
    contradict_ids: list = field(default_factory=list)
    status: str = ACTIVE
    created_epoch: int = 0
    last_epoch: int = 0
    source: str = "reflect"          # reflect | consolidate | manual
    id: str = field(default_factory=lambda: "c_" + uuid.uuid4().hex[:10])
    created_ts: str = field(default_factory=_now)
    updated_ts: str = field(default_factory=_now)

    def __post_init__(self):
        if self.kind not in VALID_KINDS:
            raise ValueError(f"bad claim kind {self.kind!r}")
        if not self.text or not self.text.strip():
            raise ValueError("empty claim text")
        self.text = self.text.strip()[:MAX_TEXT_CHARS]
        if not self.target:
            raise ValueError("claim needs a target")

    # -- identity ---------------------------------------------------------
    @property
    def norm(self) -> str:
        return normalise(self.text)

    def scope(self) -> tuple:
        return tuple(getattr(self, f) for f in SCOPE_FIELDS)

    def dedupe_key(self) -> tuple:
        """Claims can only merge when kind, scope and polarity all agree."""
        d = self.data or {}
        if self.kind == APPLICABILITY:
            pol = (d.get("effect"),)
        elif self.kind == INTERACTION:
            pol = (d.get("relation"), d.get("other"))
        elif self.kind == PARAMETER:
            pol = (json.dumps(d.get("params", {}), sort_keys=True),)
        elif self.kind == STATE_SPLIT:
            pol = (d.get("proposed_state"),)
        elif self.kind == IMPL_NOTE:
            pol = (d.get("note_type"),)
        elif self.kind == MECHANISM:
            pol = (d.get("metric"), d.get("verdict"))
        elif self.kind == EXPERIMENT:
            pol = (d.get("fact_type"), d.get("subject"))
        else:
            pol = ()
        return (self.kind, self.level, self.family) + self.scope() + pol

    # -- evidence ---------------------------------------------------------
    def add_support(self, ids: Iterable[str]) -> int:
        n = 0
        for i in ids:
            if i in self.support_ids or i in self.contradict_ids:
                continue
            self.support_ids.append(i)
            self.support += 1
            n += 1
        del self.support_ids[:-MAX_CITED_IDS]
        return n

    def add_contradiction(self, ids: Iterable[str]) -> int:
        n = 0
        for i in ids:
            if i in self.support_ids or i in self.contradict_ids:
                continue
            self.contradict_ids.append(i)
            self.contradict += 1
            n += 1
        del self.contradict_ids[:-MAX_CITED_IDS]
        return n

    def min_support(self) -> int:
        return MIN_SUPPORT_BY_KIND.get(self.kind, MIN_SUPPORT)

    def eligible(self, min_support: Optional[int] = None) -> bool:
        ms = self.min_support() if min_support is None else max(min_support, 1)
        return self.status == ACTIVE and self.support >= ms and self.support > self.contradict

    def rank(self, epoch: int) -> float:
        age = max(0, epoch - self.last_epoch)
        return (self.support - self.contradict) * 0.5 ** (age / HALF_LIFE_EPOCHS)

    @classmethod
    def from_dict(cls, d: dict) -> "Claim":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class ClaimStore:
    """JSON-persisted set of claims plus per-(problem, slot) attempt tallies."""

    VERSION = 1

    OBSERVED_KEEP = 4096         # outcome ids remembered by observe() (resume idempotency)

    def __init__(self, path: Path, *, min_support: Optional[int] = None):
        self.path = Path(path)
        self.min_support = min_support   # None -> per-kind defaults
        self.claims: dict[str, Claim] = {}
        self.meta: dict = {"epoch": 0, "reflect_cursor": 0, "reflections": 0}
        self.tried: dict[str, dict] = {}
        self._lock = threading.RLock()
        self.load()

    # -- persistence ------------------------------------------------------
    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8") or "{}")
        with self._lock:
            self.meta.update(data.get("meta", {}))
            self.claims = {}
            for d in data.get("claims", []):
                try:
                    c = Claim.from_dict(d)
                except (TypeError, ValueError):
                    continue
                self.claims[c.id] = c
            self.tried = data.get("tried", {})

    def save(self) -> None:
        with self._lock:
            data = {
                "version": self.VERSION,
                "meta": self.meta,
                "claims": [asdict(c) for c in self.claims.values()],
                "tried": self.tried,
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.path)

    # -- basic access -----------------------------------------------------
    @property
    def epoch(self) -> int:
        return int(self.meta.get("epoch", 0))

    def get(self, claim_id: str) -> Optional[Claim]:
        return self.claims.get(claim_id)

    def active(self) -> list[Claim]:
        return [c for c in self.claims.values() if c.status == ACTIVE]

    def eligible(self, claim: Claim) -> bool:
        return claim.eligible(self.min_support)

    # -- edits ------------------------------------------------------------
    def find_duplicate(self, claim: Claim) -> Optional[Claim]:
        key, norm = claim.dedupe_key(), claim.norm
        best = None
        for c in self.claims.values():
            if c.id == claim.id or c.status != ACTIVE or c.dedupe_key() != key:
                continue
            if similar(norm, c.norm) and (best is None or c.support > best.support):
                best = c
        return best

    def add(self, claim: Claim, evidence: Iterable[str] = (),
            contra: Iterable[str] = ()) -> tuple[Claim, str]:
        """Insert *claim* (or fold it into a near-duplicate). Returns (claim, "added"|"merged")."""
        evidence, contra = list(evidence), list(contra)
        with self._lock:
            dup = self.find_duplicate(claim)
            if dup is not None:
                dup.add_support(evidence)
                dup.add_contradiction(contra)
                if claim.kind == REVISE_DESCRIPTION and len(claim.text) > len(dup.text):
                    dup.text = claim.text
                self._touch(dup)
                return dup, "merged"
            claim.created_epoch = claim.last_epoch = self.epoch
            claim.add_support(evidence)
            claim.add_contradiction(contra)
            self.claims[claim.id] = claim
            self._touch(claim)
            return claim, "added"

    def support(self, claim_id: str, ids: Iterable[str]) -> int:
        with self._lock:
            c = self.claims[claim_id]
            n = c.add_support(ids)
            self._touch(c)
            return n

    def contradict(self, claim_id: str, ids: Iterable[str]) -> int:
        with self._lock:
            c = self.claims[claim_id]
            n = c.add_contradiction(ids)
            self._touch(c)
            return n

    def merge(self, keep_id: str, absorb_ids: Iterable[str], text: Optional[str] = None) -> Claim:
        """Fold *absorb_ids* into *keep_id* (union of evidence); absorbed claims are retired."""
        with self._lock:
            keep = self.claims[keep_id]
            for aid in absorb_ids:
                a = self.claims.get(aid)
                if a is None or a.id == keep.id:
                    continue
                keep.add_support(a.support_ids)
                keep.add_contradiction(a.contradict_ids)
                # counts beyond the cited-id window are carried over approximately
                keep.support = max(keep.support, a.support)
                keep.contradict = max(keep.contradict, a.contradict)
                a.status = RETIRED
                a.data = dict(a.data, merged_into=keep.id)
                a.updated_ts = _now()
            if text:
                keep.text = text.strip()[:MAX_TEXT_CHARS]
            self._touch(keep)
            return keep

    def dedupe(self) -> int:
        """Deterministic near-duplicate merge over all active claims. Returns #absorbed."""
        merged = 0
        with self._lock:
            groups: dict[tuple, list[Claim]] = {}
            for c in self.active():
                groups.setdefault(c.dedupe_key(), []).append(c)
            for group in groups.values():
                group.sort(key=lambda c: (-c.support, c.created_ts))
                for i, keep in enumerate(group):
                    if keep.status != ACTIVE:
                        continue
                    absorb = [o.id for o in group[i + 1:]
                              if o.status == ACTIVE and similar(keep.norm, o.norm)]
                    if absorb:
                        self.merge(keep.id, absorb)
                        merged += len(absorb)
        return merged

    def _touch(self, c: Claim) -> None:
        c.last_epoch = self.epoch
        c.updated_ts = _now()

    def tick(self) -> dict:
        """Advance one reflection epoch; age out weak claims. Returns counters."""
        with self._lock:
            self.meta["epoch"] = self.epoch + 1
            ep = self.epoch
            retired = purged = 0
            for c in list(self.claims.values()):
                age = ep - c.last_epoch
                if c.status == ACTIVE and not self.eligible(c) and age >= AGE_OUT_EPOCHS:
                    c.status = RETIRED
                    c.updated_ts = _now()
                    retired += 1
                elif c.status == RETIRED and age >= PURGE_RETIRED_EPOCHS:
                    del self.claims[c.id]
                    purged += 1
            act = self.active()
            if len(act) > MAX_ACTIVE_CLAIMS:
                act.sort(key=lambda c: c.rank(ep))
                for c in act[: len(act) - MAX_ACTIVE_CLAIMS]:
                    c.status = RETIRED
                    retired += 1
            return {"epoch": ep, "retired": retired, "purged": purged}

    # -- retrieval --------------------------------------------------------
    @staticmethod
    def _matches(c: Claim, q: dict) -> Optional[int]:
        """None if *c* is out of scope for query *q*, else #specific fields that matched."""
        if c.target != q["target"]:
            return None
        spec = 0
        for f in SCOPE_FIELDS[1:]:
            cv, qv = getattr(c, f), q.get(f)
            if cv is None:
                continue
            if f in LOCAL_FIELDS:
                if qv != cv:
                    return None
            elif f == "impl_id":
                if qv is not None and qv != cv:
                    return None
            elif qv is not None and qv != cv:
                return None
            spec += 1 if qv is not None else 0
        return spec

    def relevant(self, *, target: str, arch: Optional[str] = None, op_class: Optional[str] = None,
                 state: Optional[str] = None, technique: Optional[str] = None,
                 problem_id: Optional[str] = None, slot: Optional[str] = None,
                 k: int = 12, impl_id: Optional[str] = None, kinds: Optional[Iterable[str]] = None,
                 include_ineligible: bool = False) -> list[Claim]:
        """Eligible claims in scope, most specific and best supported first.

        A claim field left unset matches anything; a set field must equal the
        query value when the query gives one (problem_id / slot must always
        match, so impl notes stay local to their problem).
        """
        q = dict(target=target, arch=arch, op_class=op_class, state=state, technique=technique,
                 problem_id=problem_id, slot=slot, impl_id=impl_id)
        kinds = set(kinds) if kinds else None
        ep = self.epoch
        scored = []
        with self._lock:
            for c in self.claims.values():
                if kinds and c.kind not in kinds:
                    continue
                if c.kind == EXPERIMENT and not (kinds and EXPERIMENT in kinds):
                    continue                          # X-145: experiment facts reach prompts only through their own opt-in
                if c.status != ACTIVE or (not include_ineligible and not self.eligible(c)):
                    continue
                spec = self._matches(c, q)
                if spec is None:
                    continue
                scored.append((spec, c.rank(ep), c.support, c))
        scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        return [t[3] for t in scored[:k]]

    @staticmethod
    def _label(c: Claim) -> str:
        bits = [c.kind]
        if c.kind == APPLICABILITY and c.data.get("effect"):
            bits[0] = c.data["effect"].replace("_", " ")
        elif c.kind == INTERACTION and c.data.get("relation"):
            bits[0] = f"{c.data['relation'].replace('_', ' ')} {c.data.get('other', '')}".strip()
        elif c.kind == REVISE_DESCRIPTION:
            bits[0] = "description"
        elif c.kind == STATE_SPLIT:
            bits[0] = f"state split -> {c.data.get('proposed_state', '?')}"
        for f in ("technique", "op_class", "arch", "state", "slot"):
            v = getattr(c, f)
            if v:
                bits.append(v)
        return " | ".join(bits)

    def render(self, claims: Iterable[Claim]) -> str:
        """Compact one-line-per-claim rendering with (+support/-contradict) counts."""
        lines = []
        for c in claims:
            extra = ""
            if c.kind == PITFALL and c.data.get("fix"):
                extra = f" FIX: {c.data['fix']}"
            elif c.kind == PARAMETER and c.data.get("params"):
                extra = " " + json.dumps(c.data["params"], sort_keys=True)
            lines.append(f"- (+{c.support}/-{c.contradict}) [{self._label(c)}] {c.text}{extra}")
        return "\n".join(lines)

    # -- deterministic "already tried" tallies ----------------------------
    @staticmethod
    def _tkey(problem_id: Optional[str], slot: Optional[str]) -> str:
        return f"{problem_id}|{slot or ''}"

    def observe(self, rec: OutcomeRecord) -> None:
        """Fold one outcome into the per-(problem, slot, technique) tally (no LLM)."""
        if not rec.problem_id:
            return
        with self._lock:
            # idempotent by outcome id: a step replayed after a kill is not tallied twice
            seen = self.meta.setdefault("observed", [])
            if rec.id in seen:
                return
            seen.append(rec.id)
            del seen[:-self.OBSERVED_KEEP]
            bucket = self.tried.setdefault(self._tkey(rec.problem_id, rec.slot), {})
            t = bucket.setdefault(rec.technique, {"n": 0, "statuses": {}, "best_speedup": None,
                                                  "last_error": None, "target": rec.target})
            t["n"] += 1
            t["statuses"][rec.status] = t["statuses"].get(rec.status, 0) + 1
            if rec.speedup is not None and (t["best_speedup"] is None or rec.speedup > t["best_speedup"]):
                t["best_speedup"] = round(float(rec.speedup), 4)
            if rec.error:
                t["last_error"] = rec.error.strip().splitlines()[0][:160] if rec.error.strip() else None

    def already_tried(self, problem_id: Optional[str], slot: Optional[str] = None,
                      *, max_items: int = 12) -> str:
        """Rendered tallies + impl notes for this problem/slot ("" when nothing known)."""
        lines = []
        bucket = self.tried.get(self._tkey(problem_id, slot), {})
        items = sorted(bucket.items(), key=lambda kv: -kv[1]["n"])[:max_items]
        for tech, t in items:
            st = ", ".join(f"{k} {v}" for k, v in sorted(t["statuses"].items(), key=lambda kv: -kv[1]))
            best = f"; best {t['best_speedup']:.2f}x" if t.get("best_speedup") else ""
            no_win = "" if t["statuses"].get(IMPROVED) else " -- no improvement yet"
            err = f'; last error: "{t["last_error"]}"' if t.get("last_error") else ""
            lines.append(f"- {tech}: {t['n']} tries ({st}){best}{no_win}{err}")
        notes = [c for c in self.claims.values()
                 if c.kind == IMPL_NOTE and c.status == ACTIVE and self.eligible(c)
                 and c.problem_id == problem_id and (c.slot or None) == (slot or None)]
        notes.sort(key=lambda c: -c.rank(self.epoch))
        if notes:
            lines.append(self.render(notes[:max_items]))
        return "\n".join(lines)
