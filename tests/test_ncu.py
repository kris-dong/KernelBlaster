"""NCU per-slot profiles: parsing (real L40S capture), fallback, agent integration."""
import asyncio
import shutil
from pathlib import Path

import pytest

from kernelblaster.knowledge import ncu
from kernelblaster.knowledge.outcomes import OutcomeLog

FIX = Path(__file__).parent / "fixtures"
ALL = (FIX / "ncu" / "lenet_sm89_all_slots.csv").read_text()


def test_parse_real_capture_by_slot():
    by = ncu.profile_by_slot(ALL)
    assert set(by) == {"conv2d", "relu", "maxpool2d", "linear"}
    lin = by["linear"]
    assert lin["n_kernels"] == 3 and lin["duration_us"] > 0
    assert 0 <= lin["achieved_occupancy"] <= 100 and lin["theoretical_occupancy"] == 100.0
    assert lin["state"] == "latency_bound"                    # 1-block grids on 142 SMs
    assert lin["rules"] and lin["rules"][0]["rule"] == "LaunchConfiguration"
    txt = ncu.render(lin)
    assert txt.startswith("NCU (measured, 3 kernel launches") and "LaunchConfiguration" in txt


def test_errors_and_command():
    assert ncu.ncu_error(ALL) is None
    err = ncu.ncu_error((FIX / "ncu" / "no_counter_permission.txt").read_text())
    assert "ERR_NVGPUCTRPERM" in err
    assert ncu.ncu_error("==WARNING== No kernels were profiled.") == "no kernels matched the slot filter"
    cmd = ncu.ncu_command(slot="conv2d")
    assert "regex:'mbslot_conv2d::'" in cmd and "--print-kernel-base demangled" in cmd and "--csv" in cmd


def test_state_and_delta():
    assert ncu.ncu_state({"compute_throughput": 80, "memory_throughput": 30}) == "compute_bound"
    assert ncu.ncu_state({"compute_throughput": 20, "memory_throughput": 70}) == "memory_bound"
    assert ncu.ncu_state({"compute_throughput": 5, "memory_throughput": 9, "achieved_occupancy": 60}) \
        == "underutilized_memory"
    d = ncu.delta({"achieved_occupancy": 10.0, "duration_us": 5.0}, {"achieved_occupancy": 55.5, "duration_us": 2.0})
    assert d == {"achieved_occupancy": 45.5, "duration_us": -3.0}
    assert ncu.delta(None, {"x": 1}) == {}


from test_multislot_agent import ARCH, FAST_RELU, needs_gpu  # noqa: E402


@needs_gpu
@pytest.mark.parametrize("fail", [False, True])
def test_agent_uses_ncu_or_falls_back(tmp_path, fail):
    from kernelblaster.agents.opt_multislot import MultiSlotAgent
    from kernelblaster.agents.utils import agentic_llm
    from kernelblaster.knowledge.slots import LocalEvaluator, SlotProblem

    class Replay(LocalEvaluator):                      # real builds/runs, replayed L40S profile
        calls = 0

        async def profile(self, problem, sources, tag):
            Replay.calls += 1
            if fail:
                return {}, "no GPU performance-counter permission (ERR_NVGPUCTRPERM)"
            by = ncu.profile_by_slot(ALL)
            if "k_relu4" in sources["relu"]:           # the improved relu: pretend occupancy rose
                by["relu"] = {**by["relu"], "achieved_occupancy": by["relu"]["achieved_occupancy"] + 40}
            return by, None

    seen = []
    agentic_llm.register_stub("ncu", lambda m: (seen.append(m[-1]["content"]),
                                                FAST_RELU if "## Slot `relu`" in m[-1]["content"] else "x")[1])
    p = SlotProblem(shutil.copytree(FIX / "mb_lenet", tmp_path / "p"))
    olog = OutcomeLog(tmp_path / "o.jsonl")
    try:
        a = MultiSlotAgent(p, Replay(ARCH, work_root=tmp_path / "w"), model="stub:ncu", out_dir=tmp_path / "o",
                           arch=ARCH, outcome_log=olog, rollouts=1, steps=6, min_share=0.0, max_fix_attempts=0,
                           ncu="best", seed=3)
        res = asyncio.run(a.run())
    finally:
        agentic_llm.unregister_stub("ncu")
    assert res["final_ok"]
    if fail:
        assert Replay.calls == 1 and res["ncu"]["disabled"] and not any("NCU (measured" in t for t in seen)
        assert all(s["context"]["ncu"] is False for s in res["steps"])
    else:
        assert res["ncu"]["disabled"] is None and res["ncu"]["runs"] >= 1
        assert any("NCU (measured" in t for t in seen)
        assert all(s["context"]["state"] == "latency_bound" for s in res["steps"] if s["context"]["ncu"])
        relu = [o for o in olog if o.slot == "relu" and o.metrics_delta]
        assert not relu or relu[0].metrics_delta["achieved_occupancy"] == pytest.approx(40.0)
