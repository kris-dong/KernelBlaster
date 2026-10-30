"""X-149: ncu clock-control flag (command identity) and the GPU clock sanity guard.

The nvidia-smi texts are the ones read on an L40S (max 2520 MHz) in the reproduction of a killed ncu
(l3_farm/analysis/ncu_guard_data/repro_run*.log): locked = idle GPU held at 1065 MHz in P0.
"""
import asyncio

import pytest

from kernelblaster.knowledge import clock_guard as cg
from kernelblaster.knowledge import ncu
from kernelblaster.servers.utils.resources import Device, GPUPool

UUID = "GPU-9a0558d3-9be2-0718-c2a2-28ca22bc12b9"
ROW = "0, {u}, {sm} MHz, 2520 MHz, {mem} MHz, 9001 MHz, [Requested functionality has been deprecated], {r}, {ut} %, {ps}"


def smi(sm, mem=9001, r="0x0000000000000000", ut=0, ps="P0", u=UUID):
    return ROW.format(u=u, sm=sm, mem=mem, r=r, ut=ut, ps=ps)


IDLE = smi(210, 405, "0x0000000000000001", 0, "P8")          # healthy idle (a few seconds after the last kernel)
BOOST_IDLE = smi(2520, 9001, "0x0000000000000001", 0, "P0")   # right after a run
LOCKED = smi(1065, 9001, "0x0000000000000001", 0, "P0")       # reproduced: ncu killed with its target hung
LOCKED_BUSY = smi(1065, 9001, "0x0000000000000000", 100, "P0")
BUSY = smi(2520, 9001, "0x0000000000000000", 100, "P0")
RAMP = smi(270, 405, "0x0000000000000001", 0, "P8")
THERMAL = smi(1800, 9001, "0x0000000000000040", 100, "P0")


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- ncu command identity

GOLDEN = ("ncu --kernel-name-base demangled --print-kernel-base mangled -k regex:'mbslot_relu::' --launch-count 256 "
          "--page details --section=SpeedOfLight --section=Occupancy --section=LaunchStats "
          "--section=MemoryWorkloadAnalysis --metrics " + ",".join(ncu.ACCESS_METRICS) + " --csv")


def test_ncu_command_default_is_unchanged(monkeypatch):
    monkeypatch.delenv("KERNELBLASTER_NCU_CLOCK_CONTROL", raising=False)
    monkeypatch.delenv("KB_NCU_ACCESS_METRICS", raising=False)
    assert ncu.DEFAULT_CLOCK_CONTROL == "base"
    assert ncu.ncu_command(slot="relu") == GOLDEN               # byte-identical to the command before X-149
    monkeypatch.setenv("KERNELBLASTER_NCU_CLOCK_CONTROL", "base")
    assert ncu.ncu_command(slot="relu") == GOLDEN                # explicit base adds nothing either
    monkeypatch.setenv("KERNELBLASTER_NCU_CLOCK_CONTROL", "bogus")
    assert ncu.ncu_command(slot="relu") == GOLDEN                # unknown value: the default


def test_ncu_command_none_adds_clock_control(monkeypatch):
    monkeypatch.setenv("KERNELBLASTER_NCU_CLOCK_CONTROL", "none")
    monkeypatch.delenv("KB_NCU_ACCESS_METRICS", raising=False)
    cmd = ncu.ncu_command(slot="relu")
    assert cmd == GOLDEN + " --clock-control none"
    assert ncu.ncu_command(slots=["a", "b"]).endswith("--csv --clock-control none")


# ---------------------------------------------------------------- parsing / classification

def test_parse_documented_query_and_extended():
    short = f"0, {UUID}, 1065 MHz, 2520 MHz, 9001 MHz, 9001 MHz, 2520 MHz, 0x0000000000000001"
    s = cg.parse_query(short)[0]
    assert (s.index, s.uuid, s.sm, s.max_sm, s.mem, s.app_gfx, s.reasons, s.util) == (0, UUID, 1065, 2520, 9001, 2520, 1, None)
    s = cg.parse_query(LOCKED)[0]
    assert s.app_gfx is None and s.util == 0 and s.pstate == "P0" and s.reason_names() == ["gpu_idle"]
    assert cg.parse_query("garbage\n\n") == [] and cg.parse_query(None) == []
    assert "clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,clocks.applications.graphics,clocks_throttle_reasons.active" \
        in cg.query_command(UUID, 0) and f"-i {UUID}" in cg.query_command(UUID, 0) and "-i 3" in cg.query_command(None, 3)


@pytest.mark.parametrize("text,state", [
    (IDLE, cg.OK), (BOOST_IDLE, cg.OK), (BUSY, cg.OK), (RAMP, cg.OK),          # unlocked, unthrottled
    (LOCKED, cg.LOCKED), (smi(1065, 9001, "0x0", 0, "P0"), cg.LOCKED),          # locked, idle (the reproduced state)
    (LOCKED_BUSY, cg.THROTTLED),                                                  # low while busy
    (smi(1065, 9001, "0x0", 38, "P0"), cg.THROTTLED),                              # seen right after the flow's ncu timeout
    (smi(2520, 9001, "0x0", 38, "P0"), cg.OK),
    (THERMAL, cg.THROTTLED),                                                      # hw thermal slowdown bit
    (smi(2300, ut=100, r="0x0000000000000004"), cg.OK),                           # sw power cap alone is normal
    (smi(2520, r="0x0000000000000008", ut=100), cg.THROTTLED),                    # hw slowdown even at max
])
def test_classify(text, state):
    assert cg.classify(cg.parse_query(text)[0])[0] == state


def test_classify_application_clocks_and_unknown():
    s = cg.parse_query(f"0, {UUID}, 2520 MHz, 2520 MHz, 9001 MHz, 9001 MHz, 1000 MHz, 0x0, 0 %, P0")[0]
    assert cg.classify(s)[0] == cg.LOCKED and "application clocks" in cg.classify(s)[1][0]
    assert cg.classify(cg.ClockSample())[0] == cg.UNKNOWN


class Seq:
    """A scripted nvidia-smi: successive answers (the last repeats)."""

    def __init__(self, *texts):
        self.t, self.n, self.cmds = list(texts), 0, []

    async def __call__(self, cmd):
        self.cmds.append(cmd)
        r = self.t[min(self.n, len(self.t) - 1)]
        self.n += 1
        return r


def _guard(**kw):
    async def nosleep(_):
        return None
    return cg.ClockGuard(quarantine=60, confirm=0.0, sleep=nosleep, **kw)


def test_guard_confirms_a_lock_with_a_second_sample():
    g, r = _guard(), Seq(LOCKED, LOCKED)
    v = run(g.check(r, ident=UUID, index=0, key="k", label="0"))
    assert v.state == cg.LOCKED and v.suspect and v.confirmed and r.n == 2
    assert g.suspects == 1 and g.events[-1]["state"] == "locked" and g.events[-1]["sm_mhz"] == 1065
    d = v.to_dict()
    assert d["suspect"] and d["uuid"] == UUID and d["max_sm_mhz"] == 2520 and d["confirmed"]


def test_guard_ramp_down_is_not_a_lock():
    g = _guard()
    v = run(g.check(Seq(smi(1065, ut=0), IDLE), ident=UUID))      # a clock on its way down to the idle floor
    assert v.state == cg.OK and not v.suspect and g.suspects == 0
    v = run(g.check(Seq(BOOST_IDLE), ident=UUID))                   # still at boost after the last run
    assert v.state == cg.OK


def test_guard_confirms_a_low_busy_clock():
    g = _guard()
    v = run(g.check(Seq(smi(1065, ut=38), smi(1065, ut=0, r="0x1")), ident=UUID))   # still busy, then idle: held at 1065
    assert v.state == cg.LOCKED and v.confirmed
    v = run(g.check(Seq(smi(500, ut=90), smi(2520, ut=100)), ident=UUID))            # a load ramping up: not suspect
    assert v.state == cg.OK


def test_guard_query_failure_is_unknown_not_suspect():
    g = _guard()

    async def boom(cmd):
        return None
    v = run(g.check(boom, ident=UUID))
    assert v.state == cg.UNKNOWN and not v.suspect and g.failed_queries == 1


def test_clock_suspect_helper():
    class R:
        gpu = {"clock": {"suspect": True}}
    class C:
        gpu = {"clock": {"suspect": False}}
    assert cg.clock_suspect(R()) and not cg.clock_suspect(C()) and not cg.clock_suspect(None)
    assert cg.clock_suspect({"gpu": {"clock": {"suspect": True}}})


# ---------------------------------------------------------------- the pool

def test_pool_quarantine_alternatives_and_soft_fallback():
    a, b = Device(None, "GPU-a", "L40S", 0), Device(None, "GPU-b", "L40S", 1)
    pool = GPUPool([a, b], quarantine_s=300)

    async def go():
        async with pool.lease("g") as l:
            assert pool.healthy_others(l.device, "g") != []
            l.clock_suspect, l.quarantine_s = "locked", 600
            first = l.device
        assert first.clock_suspect == "locked" and first.quarantined_until > pool.clock() and first.quarantines == 1
        for _ in range(3):                                          # the next leases avoid the quarantined GPU
            async with pool.lease("g") as l:
                assert l.device is not first
        async with pool.lease("g") as l:
            second = l.device
            assert pool.healthy_others(second, "g") == []           # the other one is quarantined
            l.clock_suspect, l.quarantine_s = "locked", 600
        # both quarantined by the guard: a lease still returns (does not wait 600 s); its results get flagged
        async def again():
            async with pool.lease("g") as l:
                return l.device
        assert await asyncio.wait_for(again(), 5) in (a, b)
        pool.unquarantine(a)
        assert a.clock_suspect is None and a.quarantined_until == 0.0
    run(go())


def test_pool_infra_quarantine_still_waits(monkeypatch):
    a = Device(None, "GPU-a", "L40S", 0)
    t = [0.0]
    pool = GPUPool([a], quarantine_after=1, quarantine_s=5, clock=lambda: t[0])
    pool.quarantine  # exists
    async def go():
        async with pool.lease("g") as l:
            l.infra_error = True
        dev, wait = pool._pick("g")
        assert dev is None and wait > 0                              # unchanged behaviour for infrastructure failures
    run(go())


# ---------------------------------------------------------------- the evaluator's timing path

from types import SimpleNamespace  # noqa: E402

from kernelblaster.knowledge import slots  # noqa: E402


def _ev(monkeypatch, tmp_path, texts, *, guard=True, ndev=2):
    """A LocalEvaluator on a fake pool; ``texts``: device index -> scripted nvidia-smi answers."""
    if guard:
        monkeypatch.setenv("KERNELBLASTER_CLOCK_GUARD", "1")
    else:
        monkeypatch.delenv("KERNELBLASTER_CLOCK_GUARD", raising=False)
    monkeypatch.setenv("KERNELBLASTER_CLOCK_CONFIRM_S", "0")
    devs = [Device(None, f"GPU-{i}", "L40S", i) for i in range(ndev)]
    ev = slots.LocalEvaluator("89", work_root=tmp_path, pool=GPUPool(devs), runtime="process")
    seq = {i: Seq(*t) for i, t in texts.items()}
    ev.queried, ev.ran = [], []

    async def clock_run(device, cmd):
        ev.queried.append(device.index)
        return await seq[device.index](cmd)

    async def _run(exe, env=None, prefix="", device=None, problem=None):
        ev.ran.append((device.index, prefix))
        return None if prefix == "TIMEOUT" else "passed\n"
    ev._clock_run, ev._run = clock_run, _run
    return ev, devs


PROB = SimpleNamespace(root="/x/prob", name="prob")


def test_timed_run_retimes_on_another_gpu_and_quarantines(monkeypatch, tmp_path):
    ev, devs = _ev(monkeypatch, tmp_path, {0: [LOCKED], 1: [IDLE]})

    async def go():
        cvd = {}
        slots._CLOCKV.set(cvd)
        text = await ev._timed_run(PROB, tmp_path / "exe")
        return text, cvd
    text, cvd = run(go())
    assert text == "passed\n" and [d for d, _ in ev.ran] == [1]            # timed on the clean GPU only
    assert devs[0].clock_suspect and devs[0].quarantined_until > 0 and not devs[1].clock_suspect
    assert cvd["verdict"].state == cg.OK                                       # the verdict of the GPU it was timed on
    assert ev.clock_stats()["suspect"] == 1


def test_timed_run_single_locked_gpu_runs_flagged(monkeypatch, tmp_path):
    ev, devs = _ev(monkeypatch, tmp_path, {0: [LOCKED]}, ndev=1)

    async def go():
        cvd = {}
        slots._CLOCKV.set(cvd)
        await ev._timed_run(PROB, tmp_path / "exe")
        res = SimpleNamespace(gpu={"index": 0, "clock": cvd["verdict"].to_dict()})
        return res
    res = run(go())
    assert [d for d, _ in ev.ran] == [0]                                       # nowhere else to go: runs, flagged
    assert cg.clock_suspect(res) and devs[0].clock_suspect


def test_ncu_timeout_triggers_a_check_and_quarantine(monkeypatch, tmp_path):
    ev, devs = _ev(monkeypatch, tmp_path, {0: [LOCKED, LOCKED], 1: [IDLE]})
    monkeypatch.setattr(ev, "_clock_gate", lambda *a, **k: asyncio.sleep(0, True))   # isolate the post-ncu check
    text = run(ev._timed_run(PROB, tmp_path / "exe", prefix="TIMEOUT"))
    assert text is None and ev.queried[:2] == [0, 0]                           # queried right after the timeout (+ confirm)
    assert devs[0].clock_suspect and "locked" in devs[0].clock_suspect


def test_ncu_success_is_not_checked_after_and_guard_off_does_nothing(monkeypatch, tmp_path):
    ev, devs = _ev(monkeypatch, tmp_path, {0: [IDLE], 1: [IDLE]})
    run(ev._timed_run(PROB, tmp_path / "exe", prefix="ncu --csv"))              # pre-check only: one query
    assert len(ev.queried) == 1 and not any(d.clock_suspect for d in devs)
    ev2, _ = _ev(monkeypatch, tmp_path, {0: [LOCKED], 1: [LOCKED]}, guard=False)
    text = run(ev2._timed_run(PROB, tmp_path / "exe"))
    assert text == "passed\n" and ev2.queried == [] and ev2.clock_stats() is None   # flag off: no query at all
