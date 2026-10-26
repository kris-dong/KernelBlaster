# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exposed tile / launch parameters of a kernel source, so an autotuner can sweep them without an LLM.

Two kinds of tunables are recognised:

* **contract** -- the shape-family convention of ModelBlaster's ``kb_family_tune`` (``shape_family.patch``),
  reused unchanged: ``#define MB_NUM_CONFIGS N`` compile-time configurations and one host chooser
  ``mb_choose_config(const int* s)`` honouring ``MB_FORCE_CONFIG`` (forced sweep) and, when ``MB_TUNED_TABLE`` is
  defined, a generated ``mb_tuned_lookup(s)`` (per-shape table). A variant is ``#define MB_FORCE_CONFIG c``
  prepended to the source.
* **params** -- integer constants named like tile / block / thread / stage constants (``#define BM 64``,
  ``constexpr int TILE_K = 32``; same name filter as ``family_data/family_analysis.py``), plus
  ``__launch_bounds__(T[, B])``. A variant rewrites the defining line in place (``variant_source``); every
  variant must still pass the driver's checks (nothing here guarantees a rewritten constant is legal).

``extract_tunables(src)`` -> ``{"format", "contract": {...}|None, "params": {name: value}, "launch_bounds": [..]|None,
"sweep": [{name: [values]}]}``; ``variant_source(src, settings)``; ``sweep_plan(t, max_variants)``.
Pure text processing (no compiler), deterministic.
"""
from __future__ import annotations

import itertools
import re
from typing import Optional

FORMAT = "kb-tunables/1"

_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_DEF = re.compile(r"^(?P<lead>[ \t]*#[ \t]*define[ \t]+)(?P<name>[A-Z_][A-Z0-9_]*)(?P<sep>[ \t]+\(?[ \t]*)(?P<val>\d+)"
                  r"(?P<tail>[ \t]*\)?[ \t]*(?://[^\n]*)?)$", re.M)
_CONSTEXPR = re.compile(r"(?P<lead>(?:static[ \t]+)?(?:constexpr|const)[ \t]+(?:int|unsigned|size_t|uint32_t|int32_t)"
                        r"[ \t]+)(?P<name>[A-Z_][A-Z0-9_]*)(?P<sep>[ \t]*=[ \t]*\(?[ \t]*)(?P<val>\d+)")
_LB = re.compile(r"__launch_bounds__\s*\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)")
# the same name filter the shape-family analysis used for tile constants
_TILE_NAME = re.compile(r"(TILE|BLOCK|BM|BN|BK|^TM|^TN|THREAD|WARP|STAGE|PIX|OCB|ICB|CO|CI|K_|M_|N_|SPLIT|VEC|UNROLL)")
_CONTRACT_N = re.compile(r"^\s*#\s*define\s+MB_NUM_CONFIGS\s+\(?\s*(\d+)", re.M)


def contract(src: str) -> Optional[dict]:
    """The family contract of ``src`` (``kb_family_tune.contract`` semantics) or None."""
    m = _CONTRACT_N.search(src or "")
    if not m:
        return None
    n = int(m.group(1))
    out = {"configs": n, "force_hook": "MB_FORCE_CONFIG" in src and "mb_choose_config" in src,
           "table_hook": "MB_TUNED_TABLE" in src and "mb_tuned_lookup" in src,
           "tuned_table": bool(re.search(r"^\s*#\s*define\s+MB_TUNED_TABLE\b", src, re.M))}
    out["sweepable"] = n >= 2 and out["force_hook"]
    return out


def _pow2_neighbours(v: int) -> list:
    """Values around ``v`` a tuner tries: half / double (powers of two stay powers of two), bounded."""
    if v <= 0:
        return [v]
    cand = {v, max(1, v // 2), v * 2}
    return sorted(x for x in cand if 1 <= x <= 4096)


def extract_tunables(src: str) -> dict:
    """See the module doc. ``params`` keeps the first definition of each name (the one ``variant_source`` edits)."""
    src = src or ""
    clean = _COMMENT.sub(lambda m: " " * len(m.group(0)), src)   # same offsets as src
    params: dict = {}
    for rx in (_DEF, _CONSTEXPR):
        for m in rx.finditer(clean):
            name, val = m.group("name"), int(m.group("val"))
            if name.startswith("MB_") or not _TILE_NAME.search(name):
                continue
            params.setdefault(name, val)
    lb = None
    m = _LB.search(clean)
    if m:
        lb = [int(m.group(1))] + ([int(m.group(2))] if m.group(2) else [])
    con = contract(src)
    sweep = []
    if con and con["sweepable"]:
        sweep.append({"MB_FORCE_CONFIG": list(range(min(con["configs"], 32)))})
    for name, v in sorted(params.items()):
        sweep.append({name: _pow2_neighbours(v)})
    return {"format": FORMAT, "contract": con, "params": params, "launch_bounds": lb, "sweep": sweep,
            "tunable": bool((con and con["sweepable"]) or params)}


def variant_source(src: str, settings: dict) -> str:
    """``src`` with ``settings`` applied: ``MB_FORCE_CONFIG`` is prepended as a define (the contract's own
    mechanism); any other name rewrites the value of its first ``#define`` / ``constexpr`` definition in place.
    Unknown names raise KeyError (a sweep must not silently time the unchanged kernel)."""
    out = src
    pre = []
    for name, val in settings.items():
        if name == "MB_FORCE_CONFIG":
            pre.append(f"#define MB_FORCE_CONFIG {int(val)}\n")
            continue
        done = False
        clean = _COMMENT.sub(lambda m: " " * len(m.group(0)), out)   # same offsets; commented code is ignored
        for rx in (_DEF, _CONSTEXPR):
            for m in rx.finditer(clean):
                if m.group("name") != name:
                    continue
                out = out[:m.start("val")] + str(int(val)) + out[m.end("val"):]
                done = True
                break
            if done:
                break
        if not done:
            raise KeyError(f"tunable {name!r} not defined in the source")
    return "".join(pre) + out


def sweep_plan(t: dict, max_variants: int = 8) -> list:
    """A bounded list of settings dicts to try: the contract's configs first (each is a complete, author-written
    configuration), then one-at-a-time neighbours of each param (never the cross product)."""
    out = []
    for d in t.get("sweep", []):
        for name, vals in d.items():
            base = (t.get("params") or {}).get(name)
            for v in vals:
                if base is not None and v == base:
                    continue
                out.append({name: v})
    return out[:max(0, int(max_variants))]


def best_settings(observations: list) -> dict:
    """Per shape key the fastest passing setting: observations [{"skey", "config"|None, "us", "ok"}] ->
    {skey: {"config": c, "us": t}} (what an autotuner table / ``Pool.tuned_table`` stores)."""
    best: dict = {}
    for o in observations:
        if not o.get("ok") or o.get("us") is None:
            continue
        k = o["skey"]
        if k not in best or o["us"] < best[k]["us"]:
            best[k] = {"config": o.get("config"), "us": o["us"]}
    return best


def _grid(t: dict, limit: int = 64) -> list:          # small helper for tests / tools
    names = [n for d in t.get("sweep", []) for n in d]
    vals = [v for d in t.get("sweep", []) for v in d.values()]
    return [dict(zip(names, c)) for c in itertools.islice(itertools.product(*vals), limit)]
