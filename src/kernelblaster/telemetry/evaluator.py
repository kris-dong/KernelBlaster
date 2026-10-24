# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluator instrumentation at the interface boundary.

``knowledge.slots.Evaluator.__init_subclass__`` calls
:func:`instrument_class` on every evaluator class, which wraps the methods
of :data:`STAGES` that the class itself defines in spans:

    evaluate / profile          -> eval   (purpose from the tag, see purpose_of)
    _build / _compile / ...     -> build
    _nvcc                       -> build.nvcc
    _timed_run                  -> run    (lease + retries on infra failures)
    _run / _exec_staged         -> run.exec  (ncu.exec under profile());
                                   MB_PHASE_MS lines in its output become
                                   driver.<phase> child spans
    _stage / _send / ...        -> run.stage (ssh upload)
    _workdir                    -> io.workdir

so an evaluator added later (e.g. one talking to a persistent driver host)
is covered at ``evaluate`` / ``profile`` with no glue, and gets the finer
stages by using these method names or by adding its own to ``STAGES``
(or ``TELEMETRY_STAGES = {...}`` on the class). Queue waits are recorded where
they happen: ``build.queue`` (servers/utils/resources.AdaptiveLimiter),
``gpu.queue`` (GPUPool.lease), build-cache hits/misses as counters
(servers/utils/build_cache.BuildCache).
"""
from __future__ import annotations

import functools
import inspect
import re
from typing import Optional

from . import core

STAGES = {
    "evaluate": "eval",
    "profile": "eval",
    "_build": "build",
    "_build_units": "build",
    "_compile": "build",
    "_compile_units": "build",
    "_nvcc": "build.nvcc",
    "_timed_run": "run",
    "_run": "run.exec",
    "_exec_staged": "run.exec",
    "_stage": "run.stage",
    "_send": "run.stage",
    "_remote_data": "run.stage",
    "_workdir": "io.workdir",
}

_PHASE_RE = re.compile(r"^MB_PHASE_MS (\S+) (\S+)", re.M)


def purpose_of(method: str, tag) -> str:
    """What an evaluate()/profile() call is for, from the agent's tag."""
    if method == "profile":
        return "profile"
    t = str(tag or "")
    if t == "baseline":
        return "baseline"
    if t == "store_seeds":
        return "seed"
    if t.startswith("final"):
        return "final"
    if t.endswith("_confirm"):
        return "confirm"
    m = re.search(r"_f(\d+)$", t)
    if m:
        return "candidate" if m.group(1) == "0" else "fix"
    return "other"


def driver_phases(text) -> list[tuple[str, float]]:
    """[(phase, seconds)] from a driver's MB_PHASE_MS lines (``total`` dropped)."""
    if not isinstance(text, str) or "MB_PHASE_MS" not in text:
        return []
    out = []
    for m in _PHASE_RE.finditer(text):
        if m.group(1) == "total":
            continue
        try:
            out.append((m.group(1), float(m.group(2)) / 1000.0))
        except ValueError:
            pass
    return out


def add_driver_phases(sp, phases: list[tuple[str, float]]) -> None:
    """Lay the phases end to end, ending when ``sp`` ends, as its children
    (scaled down if they exceed it); the rest of ``sp`` is process start-up,
    the GPU lock wait and teardown."""
    if not phases or not isinstance(sp, core.Span):
        return
    t1 = core._now()
    dur = t1 - sp.t0
    tot = sum(s for _, s in phases)
    scale = min(1.0, dur / tot) if tot > 0 else 1.0
    t = t1 - tot * scale
    pfx = "ncu.driver." if sp.name == "ncu.exec" else "driver."
    for name, s in phases:
        core.record_span(pfx + name, t, t + s * scale, parent=sp)
        t += s * scale
    for c in sp.collectors:
        for name, s in phases:
            c.add_count(f"driver_ms.{name}", round(s * 1000.0, 3))


def _device_label(dev) -> Optional[str]:
    if dev is None:
        return None
    lab = getattr(dev, "label", None)
    try:
        return lab() if callable(lab) else str(dev)
    except Exception:  # noqa: BLE001
        return str(dev)


def _wrap(cls_name: str, meth: str, stage: str, fn):
    is_eval = stage == "eval"
    is_exec = stage == "run.exec"

    def open_span(a, kw):
        cur = core.current()
        if cur is not None and (cur.name == stage or (is_exec and cur.name == "ncu.exec")):
            return None                       # nested call of the same stage (super(), helpers): one span
        attrs = {}
        if is_eval:
            tag = kw.get("tag", a[3] if len(a) > 3 else None)
            # an explicit attrs(purpose=...) above wins (eval spans never nest)
            purpose = "profile" if meth == "profile" else (
                (cur.attrs.get("purpose") if cur is not None else None) or purpose_of(meth, tag))
            attrs.update(evaluator=cls_name, tag=str(tag) if tag is not None else None, purpose=purpose)
        name = stage
        if is_exec:
            if (cur.attrs.get("purpose") if cur is not None else None) == "profile":
                name = "ncu.exec"
            dev = _device_label(kw.get("device"))
            if dev:
                attrs["device"] = dev
        return core.span(name, **attrs)

    def finish(sp, res):
        if is_eval:
            ok = getattr(res, "ok", None)
            if ok is None and isinstance(res, tuple) and len(res) == 2:      # profile(): (by_slot, err)
                ok = res[1] is None
            sp.set(ok=ok, stage=getattr(res, "stage", None))
            core.count(f"eval.{sp.attrs.get('purpose') or 'other'}")
        elif is_exec:
            add_driver_phases(sp, driver_phases(res))

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def aw(*a, **kw):
            if not core.enabled():
                return await fn(*a, **kw)
            cm = open_span(a, kw)
            if cm is None:
                return await fn(*a, **kw)
            with cm as sp:
                res = await fn(*a, **kw)
                finish(sp, res)
                return res
        aw.__kb_traced__ = stage
        return aw

    @functools.wraps(fn)
    def w(*a, **kw):
        if not core.enabled():
            return fn(*a, **kw)
        cm = open_span(a, kw)
        if cm is None:
            return fn(*a, **kw)
        with cm as sp:
            res = fn(*a, **kw)
            finish(sp, res)
            return res
    w.__kb_traced__ = stage
    return w


def instrument_class(cls) -> None:
    """Wrap the stage methods ``cls`` defines (not inherited ones: those are
    wrapped on the class that defines them)."""
    stages = {**STAGES, **(getattr(cls, "TELEMETRY_STAGES", None) or {})}
    for meth, stage in stages.items():
        fn = cls.__dict__.get(meth)
        if fn is None or not callable(fn) or getattr(fn, "__kb_traced__", None):
            continue
        setattr(cls, meth, _wrap(cls.__name__, meth, stage, fn))
