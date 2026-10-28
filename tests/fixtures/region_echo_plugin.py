"""run_multislot.py --plugin: the scripted stub LLM ``stub:region_echo`` ($0, no relay, no key) for the GPU SMOKE of the region loop.

For a region-loop conversation (the system prompt starts with the region rules) it replies with a fixed script of actions, one per turn:
show region, profile, edit (a no-op comment on the entry function: a new candidate that computes exactly what the committed program does),
verify, time, commit, status, finish. With ``KERNELBLASTER_REGION_LOOP_COMMIT=-1,-1`` the commit is accepted whatever the timing says, so the
whole path runs on a GPU: the evaluator backend (full checks), the paired timing, the final re-verification, the adoption into the problem's
composite, outcome records. Any other request (a generic slot step, reflect, upkeep) gets the slot's current source back, like ``stub:echo``.
Used by tests/test_region_loop.py (fake backend) and by the RUNBOOK's smoke (real evaluator).
"""
import json
import re

from kernelblaster.agents.opt_multislot import extract_code
from kernelblaster.agents.utils import agentic_llm

MARK = "You write the CUDA source of one REGION slot"


def _act(tool, **args):
    return "```action\n" + json.dumps({"tool": tool, "args": args}) + "\n```"


def script(msgs):
    system = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else ""
    user0 = msgs[1]["content"] if len(msgs) > 1 else ""
    if MARK not in system:                                  # not a region loop turn: the slot's current source, unchanged
        user = msgs[-1]["content"]
        cur = extract_code(user.split("Current source:")[-1] if "Current source:" in user else user.split("Current slot source:")[-1])
        return "```cuda\n" + (cur or "") + "```"
    turn = sum(1 for m in msgs if m["role"] == "assistant")
    m = re.search(r"Signature \(define EXACTLY this\):\n```c\n(.*?)```", user0, re.S)
    sig = m.group(1) if m else ""
    ent = re.search(r"\b(kernel_\w+)\s*\(", sig)
    entry = ent.group(1) if ent else "kernel_region"
    steps = [lambda: _act("show", what="region"), lambda: _act("profile"),
             lambda: _act("edit") + "\n```edit\n<<<<<<< SEARCH\nvoid %s(\n=======\n/* region_echo */ void %s(\n>>>>>>> REPLACE\n```" % (entry, entry),
             lambda: _act("verify"), lambda: _act("time"), lambda: _act("commit"), lambda: _act("status"),
             lambda: _act("finish", summary="region_echo smoke: no-op edit")]
    return steps[min(turn, len(steps) - 1)]()


agentic_llm.register_stub("region_echo", script)
