"""run_multislot.py --plugin for tests/test_rollout_lineage.py (GPU-free; search fixes item 4).

``stub:lineage`` -- codegen call k (1-based, in call order) appends ``// SPEED <f> c<k>`` to the current source, with
f from ``$KB_TEST_SPEEDS`` (comma list, default 1.0) and sleeps ``$KB_TEST_SLEEPS`` seconds (comma list, default
0.02), so two rollouts' first steps overlap and commit in a scripted order. Every call is logged to
``$KB_TEST_LLM_LOG`` with the markers of the source it started from. ``kb_evaluator``: the resume tests' evaluator.
"""
import asyncio
import json
import os
import re
import sys

from kernelblaster.agents.opt_multislot import extract_code
from kernelblaster.agents.utils import agentic_llm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resume_plugin import FakeEvaluator  # noqa: E402

_n = {"codegen": 0}


def _lst(name, default):
    return [float(x) for x in os.getenv(name, "").split(",") if x.strip()] or [default]


async def _llm(msgs):
    user = msgs[-1]["content"]
    fix = "failed:" in user
    cur = extract_code(user.split("Current source:")[-1] if fix else user.split("Current slot source:")[-1]) or ""
    _n["codegen"] += 1
    k = _n["codegen"]
    sp, sl = _lst("KB_TEST_SPEEDS", 1.0), _lst("KB_TEST_SLEEPS", 0.02)
    f = sp[k - 1] if k <= len(sp) else 1.0
    await asyncio.sleep(sl[k - 1] if k <= len(sl) else sl[-1])
    p = os.getenv("KB_TEST_LLM_LOG")
    if p:
        with open(p, "a") as fh:
            fh.write(json.dumps({"n": k, "cur": re.findall(r"c\d+", " ".join(re.findall(r"// SPEED \S+ c\d+", cur)))})
                     + "\n")
    return {"text": f"```cuda\n{cur}// SPEED {f} c{k}\n```", "usage": {"prompt_tokens": 1000, "completion_tokens": 200}}


agentic_llm.register_stub("lineage", _llm)


def kb_evaluator(target, args, out):
    return FakeEvaluator()
