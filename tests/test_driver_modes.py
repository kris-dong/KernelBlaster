"""ModelBlaster driver run modes from the evaluators (MB docs/gpu_harness_perf.md):
MB_ONLY_SLOTS / MB_SCREEN for the agent's one-slot candidates, MB_GRAPH timing.

The option and parsing tests run everywhere; the runs need nvcc + a CUDA GPU.
"""
import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from kernelblaster.agents.opt_multislot import MultiSlotAgent
from kernelblaster.agents.utils import agentic_llm
from kernelblaster.knowledge import slots as slots_mod
from kernelblaster.knowledge.slots import LocalEvaluator, ServerEvaluator, SlotProblem, driver_env, \
    parse_driver_output, step_eval_options

FIX = Path(__file__).parent / "fixtures" / "mb_lenet"


def _gpu_arch():
    if shutil.which("nvcc") is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.split()
        return out[0].replace(".", "") if out else None
    except Exception:
        return None


ARCH = _gpu_arch()
needs_gpu = pytest.mark.skipif(ARCH is None, reason="needs nvcc and a CUDA GPU")


def test_driver_env_and_step_options(monkeypatch):
    assert driver_env() == {}
    assert driver_env(["relu", "conv2d"], screen=True) == {"MB_ONLY_SLOTS": "relu,conv2d", "MB_SCREEN": "1"}
    monkeypatch.delenv("KERNELBLASTER_MULTISLOT_ONLY_SLOTS", raising=False)
    monkeypatch.delenv("KERNELBLASTER_MULTISLOT_SCREEN", raising=False)
    assert step_eval_options("relu") == {"only_slots": ["relu"], "screen": True}
    monkeypatch.setenv("KERNELBLASTER_MULTISLOT_ONLY_SLOTS", "0")
    monkeypatch.setenv("KERNELBLASTER_MULTISLOT_SCREEN", "off")
    assert step_eval_options("relu") == {"only_slots": None, "screen": False}


def test_parse_graph_time():
    r = parse_driver_output("MB_PARTITION_US 10.0 reps=200\nMB_PARTITION_GRAPH_US 8.5 reps=200 nodes=4\npassed\n")
    assert r.ok and r.partition_us == 10.0 and r.partition_graph_us == 8.5
    assert parse_driver_output("MB_PARTITION_US 10.0 reps=3\nMB_GRAPH_SKIP x\npassed\n").partition_graph_us is None


def test_server_evaluator_passes_driver_env(tmp_path, monkeypatch):
    seen = {}

    async def fake_compile(self, problem, sources, tag):
        return tmp_path / "k.cu", tmp_path / "exe", None

    async def fake_run(exe, gpu, timeout, job_name, prefix_command=None, n_runs=1):
        seen["prefix"] = prefix_command
        return "MB_PARTITION_US 1.0 reps=3\npassed\n", ""

    from kernelblaster.agents.utils import commands
    monkeypatch.setattr(ServerEvaluator, "_compile", fake_compile)
    monkeypatch.setattr(commands, "run_gpu_executable", fake_run)
    ev = ServerEvaluator("gpu", tmp_path)
    p = SlotProblem(FIX)
    assert asyncio.run(ev.evaluate(p, p.seed_sources(), "t", only_slots=["relu"], screen=True)).ok
    assert seen["prefix"] == "MB_ONLY_SLOTS=relu MB_SCREEN=1 "
    asyncio.run(ev.evaluate(p, p.seed_sources(), "t"))
    assert seen["prefix"] is None


@needs_gpu
def test_local_only_slots_screen_and_graph(tmp_path):
    p = SlotProblem(FIX)
    ev = LocalEvaluator(ARCH, work_root=tmp_path / "w")
    full = asyncio.run(ev.evaluate(p, p.seed_sources(), "full"))
    part = asyncio.run(ev.evaluate(p, p.seed_sources(), "part", only_slots=["relu"], screen=True))
    assert full.ok and part.ok, part.log
    assert set(full.slot_checks) == set(p.slots) and set(part.slot_checks) == {"relu"}
    assert set(part.slot_us) == set(p.slots) and part.outputs and part.partition_us
    # a broken slot that is not screened is still caught by the partition output check
    bad = dict(p.seed_sources())
    bad["relu"] = bad["relu"].replace("fmaxf(x[i], 0.0f)", "fmaxf(x[i], 1.0f)")
    assert bad["relu"] != p.seed_sources()["relu"]
    res = asyncio.run(ev.evaluate(p, bad, "bad", only_slots=["linear"], screen=True))
    assert not res.ok and not all(o["ok"] for o in res.outputs.values())
    g = asyncio.run(LocalEvaluator(ARCH, work_root=tmp_path / "g", graph=True).evaluate(p, p.seed_sources(), "g"))
    assert g.ok and g.partition_graph_us is not None and g.partition_us is not None


@needs_gpu
def test_agent_step_screens_only_the_changed_slot(tmp_path):
    """Steps run with only_slots=[sid] + screen; baseline, confirm and final in full."""
    calls = []

    class Recording(LocalEvaluator):
        async def evaluate(self, problem, sources, tag, *, only_slots=None, screen=False):
            calls.append((tag, only_slots, screen))
            return await super().evaluate(problem, sources, tag, only_slots=only_slots, screen=screen)

    prob_dir = tmp_path / "lenet"
    shutil.copytree(FIX, prob_dir)
    p = SlotProblem(prob_dir)
    agentic_llm.register_stub("same", lambda msgs: "```cuda\n" + msgs[-1]["content"].split("```cuda")[-1]
                              .split("```")[0] + "\n```")
    try:
        agent = MultiSlotAgent(p, Recording(ARCH, work_root=tmp_path / "work"), model="stub:same",
                               out_dir=tmp_path / "out", arch=ARCH, rollouts=1, steps=2, min_share=0.0, seed=1)
        res = asyncio.run(agent.run())
    finally:
        agentic_llm.unregister_stub("same")
    assert res["final_ok"]
    by_tag = {t: (o, s) for t, o, s in calls}
    assert by_tag["baseline"] == (None, False) and by_tag["final_composite"] == (None, False)
    steps = [(t, o, s) for t, o, s in calls if t.startswith("r0_s") and not t.endswith("_confirm")]
    assert steps and all(o == [t.split("_")[2]] for t, o, s in steps)
    assert all(s for _, _, s in steps)
    assert all(o is None and not s for t, o, s in calls if t.endswith("_confirm"))
