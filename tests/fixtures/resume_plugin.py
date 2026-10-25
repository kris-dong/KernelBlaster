"""run_multislot.py --plugin for the resume / fault-injection tests (tests/test_resume.py).

* ``stub:resume`` -- a deterministic LLM: the reply depends only on the slot, the
  technique and the current source (a ``// SPEED <factor>`` marker, or ``// BROKEN``
  that the fix prompt repairs). Every call is appended to ``$KB_TEST_LLM_LOG``.
* ``kb_evaluator`` (``--evaluator plugin``) -- a GPU-free evaluator: a slot's time is
  its base time times the SPEED factors in its source; ``// BROKEN`` fails to compile.
* ``$KB_FAULT`` -- SIGKILL this process at a chosen point:
  ``llm:N`` (inside the Nth LLM call, after it was paid), ``llmfix:N`` (the Nth fix call),
  ``eval:<regex>:N`` (inside the Nth evaluation whose tag matches), ``save:N`` (right
  after the Nth journal checkpoint, i.e. between steps). ``$KB_TEST_SLOW`` (seconds)
  makes each call and evaluation sleep, for kills at random times.
"""
import asyncio
import hashlib
import json
import os
import re
import signal

from kernelblaster.agents import journal as _journal
from kernelblaster.agents.opt_multislot import extract_code
from kernelblaster.agents.utils import agentic_llm
from kernelblaster.knowledge.slots import EvalResult, Evaluator

FAULT = os.getenv("KB_FAULT", "")
SLOW = float(os.getenv("KB_TEST_SLOW", "0") or 0)
_n = {"llm": 0, "llmfix": 0, "eval": 0, "save": 0}


def _die():
    os.kill(os.getpid(), signal.SIGKILL)


def _fault(kind, tag=""):
    if not FAULT:
        return
    parts = FAULT.split(":")
    if parts[0] != kind:
        return
    if kind == "eval":
        if not re.search(parts[1], tag):
            return
        _n["eval"] += 1
        if _n["eval"] == int(parts[2]):
            _die()
        return
    _n[kind] += 1
    if _n[kind] == int(parts[1]):
        _die()


def _log_call(kind):
    p = os.getenv("KB_TEST_LLM_LOG")
    if p:
        with open(p, "a") as f:
            f.write(json.dumps({"kind": kind, "pid": os.getpid()}) + "\n")
            f.flush()
            os.fsync(f.fileno())


async def _llm(msgs):
    if SLOW:
        await asyncio.sleep(SLOW)
    user = msgs[-1]["content"]
    fix = "failed:" in user
    cur = extract_code(user.split("Current source:")[-1] if fix else user.split("Current slot source:")[-1]) or ""
    _log_call("fix" if fix else "codegen")
    _fault("llm")
    if fix:
        _fault("llmfix")
        code = cur.replace("// BROKEN\n", "") + "// SPEED 0.9 fixed\n"
    else:
        sid = re.search(r"## Slot `([^`]+)`", user).group(1)
        tech = re.search(r"## Technique to apply: (\S+)", user).group(1)
        h = int(hashlib.sha1(f"{sid}|{tech}|{cur.count('SPEED')}".encode()).hexdigest(), 16) % 5
        code = cur + ("// BROKEN\n" if h == 0 else f"// SPEED {[0.6, 0.85, 1.2, 0.97][h - 1]} {tech}\n")
    return {"text": f"```cuda\n{code}```", "usage": {"prompt_tokens": 1000 + len(user) // 100,
                                                     "completion_tokens": 200 + len(code) // 50}}


agentic_llm.register_stub("resume", _llm)


class FakeEvaluator(Evaluator):
    def __init__(self):
        self.calls = []

    @staticmethod
    def _us(problem, sid, src):
        base = 10.0 * (1 + sorted(problem.slots).index(sid))
        for f in re.findall(r"// SPEED ([0-9.]+)", src):
            base *= float(f)
        return round(base, 6)

    async def evaluate(self, problem, sources, tag, *, only_slots=None, screen=False, incumbent=None):
        self.calls.append(tag)
        if os.getenv("KB_TEST_EVAL_LOG"):
            with open(os.getenv("KB_TEST_EVAL_LOG"), "a") as f:
                f.write(f"{problem.name} {tag}\n")
        if SLOW:
            await asyncio.sleep(SLOW)
        else:
            await asyncio.sleep(0)
        _fault("eval", tag)
        bad = [s for s, c in sources.items() if "// BROKEN" in c]
        if bad:
            return EvalResult(ok=False, stage="compile", errors=[f"{bad[0]}: error: BROKEN marker"])
        slot_us = {s: self._us(problem, s, c) for s, c in sources.items()}
        return EvalResult(ok=True, stage="ok", partition_us=round(sum(slot_us.values()), 6), slot_us=slot_us,
                          slot_checks={s: {"ok": True, "rel_err": 0.0} for s in sources},
                          gpu={"name": "FAKE GPU", "host": "test"})


def kb_evaluator(target, args, out):
    return FakeEvaluator()


_save = _journal.ProblemJournal.save


def _save_and_maybe_die(self, state):
    _save(self, state)
    _fault("save")


_journal.ProblemJournal.save = _save_and_maybe_die
