# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the agent loop records (docs/trace.md, "Records").

Hook points (each a no-op without a configured store):

* ``llm_recorded`` wraps ``query.generate_code`` -- every LLM call of every
  role (codegen, fix, reflect, upkeep, legacy agents), relayed or direct;
* ``on_extract`` -- ``opt_multislot.extract_code``: the candidate source taken
  from a response (or that there was none);
* ``step_input`` / ``step_done`` -- the multi-slot agent's step: the slot's
  source before the step, and the decision with its result.json link;
* ``instrument_evaluator`` wraps ``evaluate`` / ``profile`` of every evaluator
  class (``Evaluator.__init_subclass__``), ``tool`` / ``output`` are called at
  the compiler and the driver / NCU output boundaries in ``knowledge/slots.py``.
"""
from __future__ import annotations

import contextvars
import functools
import hashlib
import inspect
import re
import time
import uuid
from typing import Any, Optional

from . import core

# the request id of the LLM call in flight (llm_relay sends it to the server)
_REQ: contextvars.ContextVar = contextvars.ContextVar("kb_trace_request", default=None)
# inside an evaluate()/profile() already recorded (nested super() calls record nothing more)
_EVAL: contextvars.ContextVar = contextvars.ContextVar("kb_trace_eval", default=None)


def request_id() -> Optional[str]:
    """The trace's id for the LLM call in flight (None outside one / when tracing is off)."""
    return _REQ.get()


def sha(text: Optional[str]) -> Optional[str]:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest() if text is not None else None


def _msg_text(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):            # content blocks
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    return "" if c is None else str(c)


def _usage(resp, model) -> dict:
    u = getattr(resp, "usage", None)
    try:
        from ..telemetry.llm import normalize_usage
        n = normalize_usage(u, model)
        return {k: n[k] for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                  "cache_write_input_tokens", "stop_reason", "max_tokens", "truncated",
                                  "cost_usd", "est_usd") if n.get(k) not in (None, 0, 0.0) or k == "output_tokens"}
    except Exception:  # noqa: BLE001
        return {k: v for k, v in (u or {}).items() if isinstance(v, (int, float, str))} if isinstance(u, dict) else {}


def _role() -> str:
    return core.telemetry_attr("role") or core.current().get("role") or "other"


def _llm_key(ctx: dict, role: str, rid: str) -> str:
    """Stable id of an LLM call: per (step, attempt) inside a step, else by request id."""
    if ctx.get("step") is not None and ctx.get("attempt") is not None:
        return core.key(f"a{ctx['attempt']}", "llm", ctx=ctx)
    return core.key("llm", role, rid, ctx=ctx)


def llm_recorded(fn):
    """Decorator for ``generate_code(messages, n_tasks=1, model=None)`` (inside the
    telemetry wrapper, so the telemetry span is the call's ``llm.<role>`` span)."""
    @functools.wraps(fn)
    async def wrapper(*a, **kw):
        st = core.store()
        if st is None:
            return await fn(*a, **kw)
        messages = kw.get("messages", a[0] if a else None) or []
        n = kw.get("n_tasks", a[1] if len(a) > 1 else 1) or 1
        model = kw.get("model", a[2] if len(a) > 2 else None)
        ctx = core.current()
        role = _role()
        rid = uuid.uuid4().hex
        tok = _REQ.set(rid)
        t0 = time.time()
        msgs = [{"role": m.get("role", "?") if isinstance(m, dict) else "?",
                 "content": st.put(_msg_text(m.get("content") if isinstance(m, dict) else m))} for m in messages]
        base = {"request_id": rid, "role": role, "model": model, "n": n, "messages": msgs,
                "prompt_sha": sha("\x00".join(_msg_text(m.get("content")) if isinstance(m, dict) else str(m)
                                              for m in messages)),
                "prompt_chars": sum(len(_msg_text(m.get("content"))) for m in messages if isinstance(m, dict)),
                "tel_span": core.telemetry_span_id()}
        lk = _llm_key(ctx, role, rid)
        try:
            resp = await fn(*a, **kw)
        except BaseException as e:
            core.record("llm", rid=lk, ctx=ctx, **base, ok=False, error=f"{type(e).__name__}: {e}"[:2000],
                        seconds=round(time.time() - t0, 3), call=st.seq(lk))
            raise
        finally:
            _REQ.reset(tok)
        gens = list(getattr(resp, "generations", None) or [])
        u = getattr(resp, "usage", None)
        tel = (u.get("telemetry") if isinstance(u, dict) else None) or {}
        core.record("llm", rid=lk, ctx=ctx, **base, ok=True, seconds=round(time.time() - t0, 3),
                    call=st.seq(lk), response=st.put(gens[0]) if gens else None,
                    responses=[st.put(g) for g in gens[1:]] or None,
                    usage=_usage(resp, model), served_model=getattr(resp, "model", None),
                    relay={k: tel[k] for k in ("relay_queue_s", "relay_network_s", "relay_attempts",
                                               "relay_replayed") if k in tel} or None,
                    stop_reasons=tel.get("stop_reasons"))
        st.flush()                        # on disk with the resume journal's copy of the response
        if ctx is not None:
            ctx["last_llm"] = rid
        return resp
    wrapper.__kb_trace__ = True
    return wrapper


def llm_replayed(journal_key: str, hit: dict, *, role: str = None, model: str = None) -> None:
    """A response the resume journal returned instead of a new call: a ``llm_replay``
    record pointing at the original ``llm`` record (same stable id) -- the response
    blob deduplicates against the original's, so nothing is stored twice."""
    st = core.store()
    if st is None:
        return
    ctx = core.current()
    a = ctx.get("attempt")
    orig = core.key(f"a{a}", "llm", ctx=ctx) if ctx.get("step") is not None and a is not None else None
    text = hit.get("text") or ""
    core.record("llm_replay", rid=core.key(f"a{a}" if a is not None else None, "llm_replay", journal_key, ctx=ctx),
                ctx=ctx, journal_key=journal_key, replay_of=orig, role=hit.get("role", role),
                model=hit.get("model", model), response=st.put(text), response_sha=sha(text))
    ctx["last_llm"] = None


def in_llm() -> bool:
    return _REQ.get() is not None


# ---------------------------------------------------------------- the agent's step

def step_input(*, slot: str, technique: str, source: str, **extra) -> None:
    """A step picked ``slot`` / ``technique``; ``source`` is the slot's code it starts from."""
    if core.store() is None:
        return
    core.set_attrs(slot=slot, technique=technique, attempt=0, code_sha=None)
    core.record("step_input", rid=core.key("input"), source=core.put(source), source_sha=sha(source),
                **{k: v for k, v in extra.items() if v is not None})


def attempt(n) -> None:
    """The step's fix attempt ``n`` starts (its LLM call, candidate and evaluation)."""
    core.set_attrs(attempt=n)


def on_extract(text: Optional[str], code: Optional[str]) -> None:
    """``extract_code(text) -> code`` inside a step: the candidate of this attempt."""
    st = core.store()
    if st is None or in_llm():          # a stub LLM calling extract_code is not a candidate
        return
    ctx = core.current()
    if ctx.get("step") is None:
        return
    a = ctx.get("attempt", 0)
    fields: dict = {"request_id": ctx.get("last_llm")}
    if code is None:
        fields.update(no_code=True, reason="no fenced code block in the response",
                      response_chars=len(text or ""))
    else:
        fields.update(code=st.put(code), code_sha=sha(code), lines=code.count("\n"))
        ctx["code_sha"] = fields["code_sha"]
    core.record("candidate", rid=core.key(f"a{a}", "candidate", ctx=ctx), ctx=ctx, **fields)


def link(**kw) -> None:
    """Attach ids (outcome id, impl id, ...) to the current step's record."""
    core.set_attrs(**{f"link_{k}": v for k, v in kw.items()})


def step_done(agent, ridx: int, step: int, n_before: int, span=None, ctx: Optional[dict] = None) -> None:
    """The step's decision record: what was decided and why, the StepLog and its index in
    result.json ``steps[]``, the outcome id and the telemetry ``step`` span id."""
    if core.store() is None:
        return
    ctx = ctx if ctx is not None else core.current()
    idx, sl = None, None
    for i in range(len(agent.steps_log) - 1, n_before - 1, -1):
        s = agent.steps_log[i]
        if s.rollout == ridx and s.step == step:
            idx, sl = i, s
            break
    if sl is None:
        return
    from dataclasses import asdict
    d = asdict(sl)
    e = (sl.context or {}).get("e2e") or {}
    best = getattr(agent, "best", {}).get(sl.slot)
    final_sha = ctx.get("code_sha")
    in_best = bool(best is not None and final_sha and sha(best.source) == final_sha)
    if sl.status in ("compile_error", "runtime_error", "wrong_output"):
        decision, why = "rejected", f"{sl.status} after {sl.fix_attempts} fix attempt(s)"
    elif e.get("accept") or (not e and in_best and sl.status == "improved"):
        decision, why = "merged", "joined the composite best"
        if e:
            why += f" (gain {e.get('gain')}, confirmed {e.get('confirm_gain')})"
    else:
        decision = "not_merged"
        acc = getattr(agent, "e2e_accept_pct", None)
        if e:
            why = (f"end-to-end gain {e.get('gain')} ({e.get('pairing')})"
                   + (f", confirm gain {e.get('confirm_gain')}" if "confirm_gain" in e else "")
                   + f" vs accept {acc}%")
        else:
            why = f"{sl.status}: slot {sl.slot_us_before} -> {sl.slot_us_after} us (confirm {agent.confirm_pct}%)"
    core.record("step", rid=core.key("step", ctx=ctx), ctx=ctx, decision=decision, why=why, status=sl.status,
                reward=sl.reward, fix_attempts=sl.fix_attempts, seconds=sl.seconds, steps_index=idx,
                result_json=str(agent.out / "result.json"), outcome_id=ctx.get("link_outcome_id"),
                tel_span=getattr(span, "id", None) or None, in_best=in_best, final_code_sha=final_sha,
                steplog=d)


# ---------------------------------------------------------------- evaluators

def _attempt_of(tag: str) -> Any:
    m = re.search(r"_f(\d+)$", tag or "")
    if m:
        return int(m.group(1))
    if (tag or "").endswith("_confirm"):
        return "confirm"
    return None


def _purpose(meth: str, tag: str) -> str:
    try:
        from ..telemetry.evaluator import purpose_of
        return purpose_of(meth, tag)
    except Exception:  # noqa: BLE001
        return "profile" if meth == "profile" else "other"


def _result_summary(res) -> dict:
    if isinstance(res, tuple) and len(res) == 2:            # profile(): (by slot, error)
        by, err = res
        return {"ok": err is None, "error": err, "slots": by or None}
    out = {}
    for k in ("ok", "stage", "partition_us", "partition_graph_us", "e2e_us", "e2e_basis", "e2e_ratio",
              "graph_skip", "slot_us", "attr_slot_us", "attr_slot_launches", "slot_checks", "outputs", "errors",
              "runtime", "eval_wall_s", "gpu", "attr_skip"):
        v = getattr(res, k, None)
        if v not in (None, {}, []):
            out[k] = v
    return out


def _wrap_eval(cls_name: str, meth: str, fn):
    @functools.wraps(fn)
    async def w(self, problem, sources, tag, *a, **kw):
        st = core.store()
        if st is None or _EVAL.get() is not None:
            return await fn(self, problem, sources, tag, *a, **kw)
        ctx = core.current()
        tag = str(tag)
        purpose = _purpose(meth, tag)
        att = _attempt_of(tag)
        if ctx.get("problem") is None:
            ctx = {**ctx, "problem": getattr(problem, "name", None)}
        eid = core.key(meth, tag, ctx=ctx)
        occ = uuid.uuid4().hex[:12]
        try:
            composed = sha(problem.assemble(sources))
        except Exception:  # noqa: BLE001
            composed = None
        srcs = {s: st.put(c) for s, c in dict(sources).items()}
        inc = kw.get("incumbent")
        opts = {k: kw.get(k) for k in ("only_slots", "screen", "slots") if kw.get(k) is not None}
        if a:
            opts["args"] = [x for x in a if isinstance(x, (str, int, float, list, bool))]
        if inc is not None:
            opts["incumbent"] = {s: sha(c) for s, c in dict(inc).items()}
        t0 = time.time()
        tok = _EVAL.set(eid)
        err = res = span = None
        try:
            with core.scope(problem=ctx.get("problem"), eval=eid, eval_occ=occ, purpose=purpose,
                            **({"attempt": att} if att is not None else {})):
                span = core.telemetry_span_id()
                res = await fn(self, problem, sources, tag, *a, **kw)
            return res
        except BaseException as e:
            err = f"{type(e).__name__}: {e}"[:2000]
            raise
        finally:
            _EVAL.reset(tok)
            rec_ctx = {**ctx, **({"attempt": att} if att is not None else {})}
            core.record("eval", rid=eid, ctx=rec_ctx, occ=occ, method=meth, tag=tag, purpose=purpose,
                        evaluator=cls_name, options=opts or None, sources=srcs,
                        source_shas={s: sha(c) for s, c in dict(sources).items()}, composed_sha=composed,
                        result=_result_summary(res) if res is not None else None, error=err,
                        log=st.put(getattr(res, "log", None)) if getattr(res, "log", None) else None,
                        seconds=round(time.time() - t0, 3), tel_span=span)
    w.__kb_trace__ = True
    return w


def instrument_evaluator(cls) -> None:
    """Wrap ``evaluate`` / ``profile`` that ``cls`` itself defines (call before the
    telemetry instrumentation, so its span encloses these records)."""
    for meth in ("evaluate", "profile"):
        fn = cls.__dict__.get(meth)
        if fn is None or not inspect.iscoroutinefunction(fn) or getattr(fn, "__kb_trace__", False) \
                or getattr(fn, "__kb_traced__", None):
            continue
        setattr(cls, meth, _wrap_eval(cls.__name__, meth, fn))


def _child_key(kind: str) -> tuple[str, dict]:
    ctx = core.current()
    st = core.store()
    parent = ctx.get("eval") or core.key(ctx=ctx)
    n = st.seq((ctx.get("eval_occ") or parent) + "/" + kind)
    return f"{parent}/{kind}{n}", ctx


def tool(name: str, cmd, rc: Optional[int], output: Optional[str], seconds: Optional[float] = None,
         **extra) -> None:
    """A compiler (or other tool) invocation: command, exit code, stdout+stderr."""
    st = core.store()
    if st is None:
        return
    rid, ctx = _child_key("build")
    if isinstance(cmd, (list, tuple)):
        cmd = [st.scrub(str(c)) for c in cmd]
    core.record("build", rid=rid, ctx=ctx, tool=name, cmd=cmd, rc=rc, output=st.put(output or ""),
                seconds=round(seconds, 3) if seconds is not None else None,
                tel_span=core.telemetry_span_id(), **extra)


def output(kind: str, text: Optional[str], **extra) -> None:
    """Raw output of a run: ``driver`` (the MB_* lines; process, host or server) or ``ncu`` (CSV)."""
    st = core.store()
    if st is None or text is None:
        return
    rid, ctx = _child_key(kind)
    if "env" in extra:
        extra["env"] = st.scrub.env(extra["env"])
    core.record(kind, rid=rid, ctx=ctx, output=st.put(text), chars=len(text),
                tel_span=core.telemetry_span_id(), **extra)
