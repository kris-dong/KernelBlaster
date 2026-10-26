"""run_multislot.py --plugin for tests/test_efficiency_rules.py (GPU-free).

* ``stub:flat`` -- an LLM whose candidates never beat the incumbent (``// SPEED 1.0``) except the calls listed
  in ``$KB_TEST_WIN_CALLS`` (comma list of 1-based codegen call numbers: ``// SPEED 0.5``). Each call sleeps
  ``$KB_TEST_SLOW`` x (1 + rollout) seconds, so the parallel rollouts interleave and steps are in flight when
  another commits. Every codegen call is appended to ``$KB_TEST_LLM_LOG``.
* ``kb_evaluator`` -- the resume tests' fake evaluator (slot time = base x the SPEED factors in its source).
"""
import asyncio
import json
import os
import sys

from kernelblaster.agents.opt_multislot import extract_code
from kernelblaster.agents.utils import agentic_llm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resume_plugin import FakeEvaluator  # noqa: E402

_n = {"codegen": 0}


async def _llm(msgs):
    user = msgs[-1]["content"]
    fix = "failed:" in user
    cur = extract_code(user.split("Current source:")[-1] if fix else user.split("Current slot source:")[-1]) or ""
    _n["codegen"] += 1
    k = _n["codegen"]
    wins = {int(x) for x in os.getenv("KB_TEST_WIN_CALLS", "").split(",") if x.strip()}
    slow = float(os.getenv("KB_TEST_SLOW", "0") or 0)
    if slow:
        await asyncio.sleep(slow * (1 + (k % 3)))
    p = os.getenv("KB_TEST_LLM_LOG")
    if p:
        with open(p, "a") as f:
            f.write(json.dumps({"kind": "codegen", "n": k}) + "\n")
    code = cur + (f"// SPEED 0.5 win{k}\n" if k in wins else f"// SPEED 1.0 flat{k}\n")
    return {"text": f"```cuda\n{code}```", "usage": {"prompt_tokens": 1000, "completion_tokens": 200}}


agentic_llm.register_stub("flat", _llm)


def kb_evaluator(target, args, out):
    return FakeEvaluator()
