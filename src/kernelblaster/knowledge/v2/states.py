# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""States as predicates over canonical metrics, in a hierarchy (design section 4).

Classification is deterministic: roots are tried in order, the first whose
predicate is True is entered, then its children the same way, and the
deepest matching chain is returned (``memory_bound > cache_thrashing``). A
vector that fits no child stays at the parent; one that fits no root is
``unexplained`` (all roots False: state discovery's input) or ``unknown``
(no root True and some root could not be evaluated: missing metrics).

A state never owns technique values; those come from the hierarchical
statistics (``stats.py``), shrunk toward the parent state.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from . import predicates as P
from .metrics import values as metric_values
from .targets import DATA_DIR, UNIVERSAL, TargetRegistry, empty_seed

CORE, PROBATIONARY, RETIRED = "core", "probationary", "retired"
CLASSIFIED, UNKNOWN, UNEXPLAINED = "classified", "unknown", "unexplained"
STATE_ID = re.compile(r"[a-z][a-z0-9_]{2,63}")


@dataclass
class State:
    id: str
    parent: Optional[str]
    predicate: dict
    mechanism: str = ""
    level: str = UNIVERSAL
    scope: Optional[str] = None          # family name / target key for non-universal states
    status: str = CORE
    source: str = "seed"                 # seed | discovered:<proposal id>
    order: int = 0                       # sibling priority (lower first)
    evidence: dict = field(default_factory=dict)   # discovery: trigger technique, admitted cursor, ...

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class Classification:
    chain: list                      # root .. leaf state ids ([] when not classified)
    status: str                      # classified | unknown | unexplained
    decided_by: dict = field(default_factory=dict)   # metric -> value that the chain used

    @property
    def leaf(self) -> str:
        return self.chain[-1] if self.chain else self.status

    def label(self) -> str:
        return " > ".join(self.chain) if self.chain else self.status

    def to_dict(self) -> dict:
        return {"chain": list(self.chain), "status": self.status, "decided_by": self.decided_by}


def load_seed(path: Optional[Path] = None) -> tuple[list[State], dict]:
    if empty_seed() and not path:                   # KB_EMPTY_SEED=1: no states; the classifier answers "unexplained"
        return [], {}
    d = json.loads((Path(path) if path else DATA_DIR / "states.json").read_text())
    states = []
    for i, s in enumerate(d["states"]):
        st = State.from_dict({**s, "order": s.get("order", i)})
        P.validate(st.predicate)
        states.append(st)
    legacy = {k: v for k, v in d.get("legacy_map", {}).items() if not k.startswith("_")}
    return states, legacy


class StateHierarchy:
    def __init__(self, states: Iterable[State], registry: Optional[TargetRegistry] = None,
                 legacy_map: Optional[dict] = None):
        self.registry = registry or TargetRegistry()
        self.states: dict[str, State] = {}
        self.legacy_map = dict(legacy_map or {})
        for s in states:
            self.add(s)

    @classmethod
    def seed(cls, registry: Optional[TargetRegistry] = None) -> "StateHierarchy":
        states, legacy = load_seed()
        return cls(states, registry, legacy)

    # -- structure --------------------------------------------------------
    def add(self, s: State) -> None:
        if not STATE_ID.fullmatch(s.id):
            raise ValueError(f"bad state id {s.id!r}")
        if s.parent is not None and s.parent not in self.states:
            raise ValueError(f"state {s.id}: unknown parent {s.parent!r}")
        P.validate(s.predicate)
        self.states[s.id] = s

    def get(self, sid: Optional[str]) -> Optional[State]:
        return self.states.get(sid) if sid else None

    def live(self, s: State) -> bool:
        return s.status in (CORE, PROBATIONARY)

    def children(self, sid: Optional[str], target: Optional[str] = None) -> list[State]:
        out = [s for s in self.states.values() if s.parent == sid and self.live(s)
               and (target is None or self.registry.covers(s.level, s.scope, target))]
        return sorted(out, key=lambda s: (s.order, s.id))

    def ancestors(self, sid: str) -> list[str]:
        """``sid`` and its parents, leaf first."""
        out, cur = [], self.get(sid)
        while cur is not None:
            out.append(cur.id)
            cur = self.get(cur.parent)
        return out

    def map_legacy(self, name: Optional[str]) -> Optional[str]:
        """v1 / technique-DB state name -> state id (None when unmapped)."""
        if not name:
            return None
        if name in self.states:
            return name
        return self.legacy_map.get(name)

    # -- classification ---------------------------------------------------
    def classify(self, vec: dict, *, target: str, op_class: Optional[str] = None,
                 dtype: Optional[str] = None) -> Classification:
        vals = metric_values(vec)
        ctx = {"op_class": op_class, "dtype": dtype, "target": self.registry.get(target).key,
               "family": self.registry.family_of(target)}
        chain: list[str] = []
        level_unknown = False
        parent = None
        while True:
            hit = None
            for s in self.children(parent, target):
                r = P.evaluate(s.predicate, vals, ctx)
                if r is True:
                    hit = s
                    break
                if r is None and parent is None:
                    level_unknown = True
            if hit is None:
                break
            chain.append(hit.id)
            parent = hit.id
        used = set().union(*(P.metrics_used(self.states[s].predicate) for s in chain)) if chain else set()
        decided = {k: round(vals[k], 3) for k in sorted(used) if k in vals}
        if chain:
            return Classification(chain, CLASSIFIED, decided)
        return Classification([], UNKNOWN if level_unknown else UNEXPLAINED, {})

    def predicate_chain(self, sid: str) -> list[dict]:
        return [{"id": a, "predicate": self.states[a].predicate} for a in reversed(self.ancestors(sid))]
