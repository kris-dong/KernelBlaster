# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Technique ontology: mechanisms with predictions and per-scope realizations (design section 5).

Seeded from ``data/kernelblaster/knowledge/techniques/*.json`` (curated: the
multi-slot catalogue and the technique-DB names folded onto mechanisms) and,
once, from the legacy JSON technique DB and the CUDA / OpenCL / RISC-V
footers (:func:`seed_legacy`): every name not folded into a curated entry
becomes a technique with its DB description as mechanism, no predictions
and ``slot=False``.

:func:`mechanism_check` labels an attempt ``confirmed`` / ``contradicted`` /
``unknown`` by sign agreement of the *measured* canonical metric deltas with
the technique's ``predicts``.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from . import predicates as P
from .metrics import CANONICAL, MEASURED, flat_band
from .targets import DATA_DIR, NODE, REPO_DATA_DIR, TARGET, UNIVERSAL, TargetRegistry

CONFIRMED, CONTRADICTED, MECH_UNKNOWN = "confirmed", "contradicted", "unknown"
DIRECTIONS = ("up", "down", "flat", "up_or_flat", "down_or_flat")
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{1,63}")


@dataclass
class Technique:
    id: str
    mechanism: str
    level: str = UNIVERSAL
    scope: Optional[str] = None
    aliases: list = field(default_factory=list)
    preconditions: Optional[dict] = None
    predicts: dict = field(default_factory=dict)
    risks: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    relations: dict = field(default_factory=dict)
    realizations: list = field(default_factory=list)   # [{scope, how, params_prior?, exemplars?}]
    states: list = field(default_factory=list)         # gate to these state chains ([] = any)
    op_classes: list = field(default_factory=list)     # gate to these op classes ([] = any)
    op_kinds: list = field(default_factory=list)       # gate to these ModelBlaster op kinds ([] = any)
    prior: dict = field(default_factory=dict)          # seed prior {mean, n}: low pseudo-count, evidence overrides
    slot: bool = True                                  # offered by the multi-slot agent
    source: str = "seed"                               # seed | legacy_db | footer:<x> | hypothesis:<id>
    status: str = "active"                             # active | retired (automatic rollback)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Technique":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})


def validate_predicts(pred: dict) -> None:
    if not isinstance(pred, dict):
        raise ValueError("predicts must be an object")
    for k, v in pred.items():
        if k not in CANONICAL:
            raise ValueError(f"predicts: unknown metric {k!r}")
        if v not in DIRECTIONS:
            raise ValueError(f"predicts.{k}: {v!r} not in {DIRECTIONS}")


# ---------------------------------------------------------------- realization scopes

def _arch_num(key: str) -> Optional[int]:
    m = re.search(r"sm_(\d+)", key)
    return int(m.group(1)) if m else None


def scope_rank(scope: str, target_key: str, registry: TargetRegistry) -> Optional[float]:
    """How specifically realization ``scope`` matches ``target_key`` (lower = more specific; None = no).

    Scopes: ``target:<key>`` (0), ``target:<backend>:sm_NN+`` (1), ``node:<id>`` /
    ``family:<id>`` (v1 spelling) / ``universal`` by their distance up the target's chain
    (1 + index), ``target:<backend>`` just below the node that backend maps to."""
    t = registry.get(target_key)
    chain = registry.chain(t.key)
    if scope == "universal":
        return 1.0 + chain.index("universal")
    kind, _, rest = scope.partition(":")
    if kind in ("family", "node"):
        return 1.0 + chain.index(rest) if rest in chain else None
    if kind != "target":
        return None
    if rest == t.key:
        return 0.0
    m = re.fullmatch(r"([a-z0-9_]+):sm_(\d+)\+", rest)
    if m:
        n = _arch_num(t.key)
        lo = int(m.group(2))                    # sm_90+ is more specific than sm_80+ for sm_90
        return 1.0 - lo / 1000.0 if m.group(1) == t.backend and n is not None and n >= lo else None
    if rest == t.backend:
        bn = registry.node_for_backend(t.key)
        return 0.5 + chain.index(bn) if bn in chain else 1.5
    return None


def scope_level(scope: str) -> str:
    if scope == "universal":
        return UNIVERSAL
    return NODE if scope.split(":", 1)[0] in ("family", "node") else TARGET


def scope_of(scope: str) -> str:
    """Realization scope string -> scope id (``family:X`` / ``node:X`` -> X)."""
    if scope == "universal":
        return "universal"
    kind, _, rest = scope.partition(":")
    return rest


# ---------------------------------------------------------------- mechanism check

def direction_agrees(pred: str, d: float, band: float) -> Optional[bool]:
    """Does an observed delta ``d`` agree with predicted direction ``pred``? (band = flat tolerance)"""
    up, down, flat = d > band, d < -band, abs(d) <= band
    return {"up": up, "down": down, "flat": flat, "up_or_flat": up or flat,
            "down_or_flat": down or flat}.get(pred)


def mechanism_check(predicts: dict, before: Optional[dict], after: Optional[dict]) -> dict:
    """Sign agreement of measured metric deltas with ``predicts``.

    ``confirmed`` when agreeing metrics outnumber disagreeing ones,
    ``contradicted`` when the reverse, ``unknown`` on a tie or with no
    measured metric on both sides.
    """
    agree, disagree, detail = 0, 0, {}
    for k, pred in (predicts or {}).items():
        b, a = (before or {}).get(k), (after or {}).get(k)
        if not b or not a or b.get("src") != MEASURED or a.get("src") != MEASURED:
            continue
        d = a["v"] - b["v"]
        ok = direction_agrees(pred, d, flat_band(k))
        detail[k] = {"pred": pred, "delta": round(d, 3), "agrees": ok}
        if ok:
            agree += 1
        else:
            disagree += 1
    label = CONFIRMED if agree > disagree else CONTRADICTED if disagree > agree else MECH_UNKNOWN
    return {"label": label, "agree": agree, "disagree": disagree, "metrics": detail}


def mechanism_from_delta(predicts: dict, measured_delta: dict) -> dict:
    """:func:`mechanism_check` when only measured canonical deltas are known (v1 records)."""
    zero = {k: {"v": 0.0, "src": MEASURED} for k in measured_delta}
    return mechanism_check(predicts, zero, {k: {"v": d, "src": MEASURED} for k, d in measured_delta.items()})


def _kind_match(op_kind: str, kinds: list) -> bool:
    """A fused slot (``fused__conv2d__relu``) matches when any of its stages is listed.

    ``$KERNELBLASTER_KIND_MATCH=base`` (opt-in, A2): a stage also matches by its base kind without the dtype
    suffix (``conv2d_f16`` matches a technique gated on ``conv2d``). By default the match is exact, so kind-gated
    seed techniques (implicit_gemm, rnn_input_projection_hoisting, ...) never apply to ``_f16`` slots."""
    stages = op_kind.split("__")[1:] if op_kind.startswith("fused__") else [op_kind]
    stages = [s for s in stages if s and s != "par"] or stages
    if os.environ.get("KERNELBLASTER_KIND_MATCH", "exact").strip().lower() == "base":
        stages = stages + [_base_kind(s) for s in stages]
    return any(k in kinds for k in stages)


def _base_kind(kind: str) -> str:
    for suffix in ("_f16", "_s8_pc", "_s8", "_i8"):
        if kind.endswith(suffix):
            return kind[: -len(suffix)]
    return kind


def load_seed_techniques(data_dir: Optional[Path] = None) -> list[dict]:
    """Curated seeds: every ``techniques/*.json`` (sorted), or the single v1 ``techniques.json``."""
    base = Path(data_dir) if data_dir else DATA_DIR
    files = sorted((base / "techniques").glob("*.json")) if (base / "techniques").is_dir() else [base / "techniques.json"]
    out, seen = [], {}
    for f in files:
        for t in json.loads(f.read_text())["techniques"]:
            if t["id"] in seen:
                raise ValueError(f"technique {t['id']!r} defined in {seen[t['id']]} and {f.name}")
            seen[t["id"]] = f.name
            out.append(t)
    return out


# ---------------------------------------------------------------- ontology

class Ontology:
    def __init__(self, techniques: Iterable[Technique] = (), registry: Optional[TargetRegistry] = None):
        self.registry = registry or TargetRegistry()
        self.techniques: dict[str, Technique] = {}
        self._alias: dict[str, str] = {}
        for t in techniques:
            self.add(t)

    @classmethod
    def seed(cls, registry: Optional[TargetRegistry] = None, *, legacy: bool = True) -> "Ontology":
        o = cls([Technique.from_dict(x) for x in load_seed_techniques()], registry)
        if legacy:
            seed_legacy(o)
        return o

    def add(self, t: Technique, *, replace: bool = True) -> Technique:
        if not ID_RE.fullmatch(t.id):
            raise ValueError(f"bad technique id {t.id!r}")
        if t.preconditions is not None:
            P.validate(t.preconditions)
        validate_predicts(t.predicts)
        if t.id in self.techniques and not replace:
            return self.techniques[t.id]
        self.techniques[t.id] = t
        self._alias[t.id.lower()] = t.id
        for a in t.aliases:
            self._alias.setdefault(a.lower(), t.id)
        return t

    def resolve(self, name: Optional[str]) -> Optional[str]:
        """Technique id for a name or alias (case-insensitive); None if unknown."""
        if not name:
            return None
        return self._alias.get(name.lower())

    def get(self, name: Optional[str]) -> Optional[Technique]:
        tid = self.resolve(name)
        return self.techniques.get(tid) if tid else None

    def realization(self, t: Technique, target_key: str) -> Optional[dict]:
        """Most specific realization for the target (target > arch range > backend > family > universal)."""
        best, best_r = None, None
        for r in t.realizations:
            rank = scope_rank(r.get("scope", "universal"), target_key, self.registry)
            if rank is not None and (best_r is None or rank < best_r):
                best, best_r = r, rank
        return best

    def applies(self, t: Technique, *, target_key: str, op_class: Optional[str], chain: list,
                values: dict, ctx: dict, slot_only: bool = True, op_kind: Optional[str] = None) -> bool:
        if (slot_only and not t.slot) or t.status != "active":
            return False
        if not self.registry.covers(t.level, t.scope, target_key):
            return False
        if self.realization(t, target_key) is None:
            return False
        if t.op_classes and op_class not in t.op_classes:
            return False
        if t.op_kinds and op_kind is not None and not _kind_match(op_kind, t.op_kinds):
            return False
        if t.states and not set(t.states) & set(chain):
            return False
        # unknown preconditions do not exclude (missing metrics are common); False does
        return P.evaluate(t.preconditions, values, ctx) is not False

    def candidates(self, **kw) -> list[Technique]:
        return [t for t in self.techniques.values() if self.applies(t, **kw)]


# ---------------------------------------------------------------- one-time legacy seeding

_FOOTERS = {"footer:cuda": ("optimization_database_footer.md", "target:cuda"),
            "footer:opencl": ("optimization_database_footer_opencl.md", "target:opencl"),
            "footer:riscv": ("optimization_database_footer_riscv.md", "family:vector_cpu")}


def seed_legacy(o: Ontology, db_path: Optional[Path] = None, footer_dir: Optional[Path] = None) -> dict:
    """Read the old technique DB + footers once; add every unfolded name. Returns counters."""
    base = REPO_DATA_DIR.parent                     # the v1 DB + footers live next to the seeds
    db_path = Path(db_path) if db_path else base / "optimization_database.json"
    footer_dir = Path(footer_dir) if footer_dir else base
    added, folded = [], []
    try:
        from .states import load_seed
        legacy_map = load_seed()[1]
    except Exception:
        legacy_map = {}
    if db_path.exists():
        db = json.loads(db_path.read_text())
        for sname, sd in db.get("optimization_strategies", {}).items():
            for opt in sd.get("optimizations", []):
                name = opt.get("technique")
                if not name:
                    continue
                if o.resolve(name):
                    folded.append(name)
                    continue
                st = legacy_map.get(sname)
                o.add(Technique(id=re.sub(r"[^A-Za-z0-9_.+-]", "_", name)[:64],
                                mechanism=(opt.get("description") or name)[:300], slot=False,
                                states=[st] if st else [], source="legacy_db",
                                realizations=[{"scope": "target:cuda", "how": opt.get("description") or name}]))
                added.append(name)
    for src, (fname, scope) in _FOOTERS.items():
        fp = footer_dir / fname
        if not fp.exists():
            continue
        text = fp.read_text()
        names = re.findall(r"^#{3,4} Expert Technique: (\S+)", text, re.M)
        names += re.findall(r"^- `([0-9.]*_?[a-z][a-z0-9_]+)` \(\d+%\)", text, re.M)
        for name in dict.fromkeys(names):
            tid = o.resolve(name)
            if tid:
                t = o.techniques[tid]
                if not any(r.get("scope") == scope for r in t.realizations) and scope_level(scope) != UNIVERSAL:
                    m = re.search(rf"`{re.escape(name)}` \(\d+%\): (.+?)(?=\n- `|\n\n)", text, re.S)
                    if m:
                        t.realizations.append({"scope": scope, "how": " ".join(m.group(1).split())[:400],
                                               "source": src})
                folded.append(name)
                continue
            m = re.search(rf"`{re.escape(name)}` \(\d+%\): (.+?)(?=\n- `|\n\n)", text, re.S)
            how = " ".join(m.group(1).split())[:400] if m else name.replace("_", " ")
            o.add(Technique(id=re.sub(r"[^A-Za-z0-9_.+-]", "_", name)[:64], mechanism=how[:300],
                            level=NODE, scope=(scope_of(scope) if scope_level(scope) == NODE
                                               else o.registry.node_for_backend(scope_of(scope) + ":x")),
                            slot=False, source=src, realizations=[{"scope": scope, "how": how}]))
            added.append(name)
    return {"added": len(added), "folded": len(set(folded)), "added_names": added}
