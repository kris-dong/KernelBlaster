# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tiny JSON predicate DSL over canonical metrics, with three-valued logic.

    {"all": [p, ...]}  {"any": [p, ...]}  {"not": p}  true
    {"<metric>": "<"|"<="|">"|">="|"=="|"!=", "v": number}
    {"op_class"|"dtype"|"family"|"target": "in"|"not_in", "v": [str, ...]}
    {"<metric>": "<"|..., "ref": "<metric>"}        (compare two metrics)

Evaluation returns True / False / None (unknown). A comparison on an absent
metric is unknown, never true; ``all`` is False if any part is False, else
unknown if any part is unknown; ``any`` is True if any part is True, else
unknown if any part is unknown; ``not`` keeps unknown.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from .metrics import CANONICAL

CMP = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
       ">=": lambda a, b: a >= b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b}
SET_OPS = ("in", "not_in")
CONTEXT_FIELDS = ("op_class", "dtype", "family", "target")
MAX_DEPTH = 6


class PredicateError(ValueError):
    pass


def _leaf(p: dict) -> tuple[str, str, Any, Optional[str]]:
    keys = [k for k in p if k not in ("v", "ref")]
    if len(keys) != 1:
        raise PredicateError(f"comparison needs exactly one field, got {sorted(p)}")
    name = keys[0]
    return name, p[name], p.get("v"), p.get("ref")


def validate(p: Any, depth: int = 0) -> None:
    """Raise :class:`PredicateError` unless ``p`` is a well-formed predicate."""
    if depth > MAX_DEPTH:
        raise PredicateError("predicate nested too deeply")
    if p is True or p is None:
        return
    if not isinstance(p, dict) or not p:
        raise PredicateError(f"predicate must be an object, got {p!r}")
    if "all" in p or "any" in p:
        k = "all" if "all" in p else "any"
        if len(p) != 1 or not isinstance(p[k], list) or not p[k]:
            raise PredicateError(f"'{k}' takes a non-empty list and nothing else")
        for q in p[k]:
            validate(q, depth + 1)
        return
    if "not" in p:
        if len(p) != 1:
            raise PredicateError("'not' takes one predicate")
        validate(p["not"], depth + 1)
        return
    name, op, v, ref = _leaf(p)
    if name in CONTEXT_FIELDS:
        if op not in SET_OPS or not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise PredicateError(f"{name} takes 'in'/'not_in' and a list of strings")
        return
    if name not in CANONICAL:
        raise PredicateError(f"unknown metric {name!r} (canonical: {', '.join(CANONICAL)})")
    if op not in CMP:
        raise PredicateError(f"bad comparison {op!r} for {name}")
    if ref is not None:
        if ref not in CANONICAL:
            raise PredicateError(f"unknown ref metric {ref!r}")
    elif not isinstance(v, (int, float)) or isinstance(v, bool):
        raise PredicateError(f"{name} {op} needs a numeric 'v'")


def evaluate(p: Any, values: dict, ctx: Optional[dict] = None) -> Optional[bool]:
    """Three-valued evaluation. ``values``: metric -> float; ``ctx``: op_class/dtype/family/target."""
    if p is True or p is None:
        return True
    ctx = ctx or {}
    if "all" in p:
        rs = [evaluate(q, values, ctx) for q in p["all"]]
        if any(r is False for r in rs):
            return False
        return None if any(r is None for r in rs) else True
    if "any" in p:
        rs = [evaluate(q, values, ctx) for q in p["any"]]
        if any(r is True for r in rs):
            return True
        return None if any(r is None for r in rs) else False
    if "not" in p:
        r = evaluate(p["not"], values, ctx)
        return None if r is None else not r
    name, op, v, ref = _leaf(p)
    if name in CONTEXT_FIELDS:
        cur = ctx.get(name)
        if cur is None:
            return None
        return (cur in v) if op == "in" else (cur not in v)
    a = values.get(name)
    b = values.get(ref) if ref is not None else v
    if a is None or b is None:
        return None
    return CMP[op](a, b)


def metrics_used(p: Any) -> set[str]:
    if p is True or p is None or not isinstance(p, dict):
        return set()
    for k in ("all", "any"):
        if k in p:
            return set().union(*(metrics_used(q) for q in p[k]))
    if "not" in p:
        return metrics_used(p["not"])
    name, _, _, ref = _leaf(p)
    out = {name} if name in CANONICAL else set()
    return out | ({ref} if ref else set())


def canonical(p: Any) -> str:
    """Stable text form (store / diff / dedupe)."""
    return json.dumps(p, sort_keys=True, separators=(",", ":"))


def render(p: Any) -> str:
    """Human form: ``bw_util >= 50 and compute_util < 40``."""
    if p is True or p is None:
        return "always"
    for k, j in (("all", " and "), ("any", " or ")):
        if k in p:
            return "(" + j.join(render(q) for q in p[k]) + ")"
    if "not" in p:
        return "not " + render(p["not"])
    name, op, v, ref = _leaf(p)
    return f"{name} {op} {ref if ref is not None else v}"
