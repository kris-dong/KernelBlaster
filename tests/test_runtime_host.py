"""The persistent driver host (knowledge/runtime_host.py, MB driver.cpp -DMB_HOST) and the
end-to-end reward (agents/opt_multislot.py).

Protocol, crash recovery and the reward decision run everywhere (a scripted fake host, a fake
evaluator); the real-host tests build the LeNet fixture's host and need nvcc + a CUDA GPU.
"""
import asyncio
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from kernelblaster.agents.opt_multislot import MultiSlotAgent, e2e_gain, partition_runtime, runtime_text, \
    slot_weights
from kernelblaster.agents.utils import agentic_llm
from kernelblaster.knowledge.outcomes import OutcomeLog, OutcomeRecord
from kernelblaster.knowledge.runtime_host import HostClient, HostPool, supports_host
from kernelblaster.knowledge.slots import EvalResult, Evaluator, LocalEvaluator, SlotProblem, parse_driver_output, \
    step_eval_options

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

# A stand-in for the host: speaks the protocol; what a request does is named by its so= file.
FAKE_HOST = r'''#!%s
import os, sys, time
print("MB_HOST_READY abi=1 nb=1 nops=1 nslots=1 init_ms=1.0", flush=True)
n = 0
for line in sys.stdin:
    f = line.rstrip("\n").split("\t")
    if f[0] == "QUIT":
        break
    nonce = f[1]
    kv = dict(x.split("=", 1) for x in f[2:] if "=" in x)
    what = os.path.basename(kv["so"]).split(".")[0]
    n += 1
    print("MB_CHECK output y max_abs_err=0 bound=1 ok=1", flush=True)
    if what == "crash":
        os._exit(11)
    if what == "hang":
        time.sleep(3600)
    print("MB_E2E_US 10.0 basis=graph mean=10.0 reps=5", flush=True)
    if "ref" in kv:
        print("MB_E2E_RATIO med=0.90000 mean=0.9 min=0.8 max=1.0 n=5", flush=True)
    print("MB_ENV " + ",".join(sorted(k for k in kv if k.startswith("MB_"))), flush=True)
    print("MB_PHASE_MS graph_build 1.0\nMB_PHASE_MS time_partition 2.0\nMB_PHASE_MS total 3.0", flush=True)
    print("passed", flush=True)
    poisoned = int(what == "poison")
    recycle = int(what == "recycle")
    print(f"MB_HOST_DONE {nonce} rc=0 poisoned={poisoned} recycle={recycle}", flush=True)
    if poisoned or recycle:
        break
''' % sys.executable


def _fake_host(tmp_path) -> Path:
    p = tmp_path / "fake_host.py"
    p.write_text(FAKE_HOST)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def test_host_client_protocol_and_recovery(tmp_path):
    exe = _fake_host(tmp_path)

    async def go():
        h = HostClient([str(exe)], label="fake", start_timeout=30)
        r1 = await h.request("ok.so", env={"MB_SCREEN": "1"}, timeout=30)
        assert r1.status == "done" and r1.rc == 0 and r1.started and "MB_ENV MB_SCREEN" in r1.text
        assert "MB_HOST_DONE" not in r1.text
        r2 = await h.request("ok.so", ref="inc.so", timeout=30)
        assert not r2.started and parse_driver_output(r2.text).e2e_ratio["med"] == 0.9
        # a crash ends the stream: reported, and the next request starts a new host
        r3 = await h.request("crash.so", timeout=30)
        assert r3.status == "crash" and r3.crashed and r3.rc == 11 and not h.alive
        r4 = await h.request("ok.so", timeout=30)
        assert r4.status == "done" and r4.started and h.starts == 2
        # a hang is killed at the timeout
        r5 = await h.request("hang.so", timeout=1.5)
        assert r5.status == "timeout" and not h.alive
        # a poisoned context and a recycle end the host after the reply
        r6 = await h.request("poison.so", timeout=30)
        assert r6.status == "done" and r6.poisoned and r6.crashed and not h.alive
        r7 = await h.request("recycle.so", timeout=30)
        assert r7.recycle and not r7.crashed and not h.alive
        r8 = await h.request("ok.so", timeout=30)
        assert r8.status == "done" and h.starts == 5
        await h.close()
        assert not h.alive
    asyncio.run(go())


def test_host_pool_per_device_lru(tmp_path):
    exe = _fake_host(tmp_path)

    async def go():
        pool = HostPool(per_device=1)
        a = await pool.get(("p1", "d0"), "d0", lambda: HostClient([str(exe)]))
        await a.request("ok.so", timeout=30)
        b = await pool.get(("p2", "d0"), "d0", lambda: HostClient([str(exe)]))
        assert ("p1", "d0") not in pool.hosts and not a.alive        # LRU closed on the same device
        c = await pool.get(("p1", "d1"), "d1", lambda: HostClient([str(exe)]))
        assert set(pool.hosts) == {("p2", "d0"), ("p1", "d1")} and b is not c
        await pool.close()
    asyncio.run(go())


class _FakeHostEvaluator(LocalEvaluator):
    """LocalEvaluator on the fake host: the 'shared object' is named after the slot source."""

    def __init__(self, exe, **kw):
        super().__init__("75", runtime="host", **kw)
        self.exe = exe

    async def _host_exe(self, problem):
        return self.exe

    async def _build_so(self, problem, sources, work):
        name = sources["relu"].strip()
        return work / f"{name if name.isalpha() else 'incumbent'}.so", None

    async def pool(self):
        from kernelblaster.servers.utils.resources import Device, GPUPool
        if self._pool is None:
            self._pool = GPUPool([Device(None, None, "fake")])
        return self._pool


def test_evaluator_maps_host_failures(tmp_path):
    p = SlotProblem(FIX)
    assert supports_host(p.driver_path.read_text())
    ev = _FakeHostEvaluator(_fake_host(tmp_path), work_root=tmp_path / "w", timeout_s=2)
    base = p.seed_sources()

    from kernelblaster import telemetry as tel

    async def go():
        out = {}
        with tel.span("step", collect=True) as tsp:
            for what in ("ok", "crash", "ok", "hang", "poison", "ok"):
                r = await ev.evaluate(p, {**base, "relu": what}, f"r0_s0_relu_{what}_f0", **step_eval_options("relu"),
                                      incumbent=base)
                out.setdefault(what, []).append(r)
        await ev.close()
        return out, tsp
    tel.set_enabled(True)
    try:
        out, tsp = asyncio.run(go())
    finally:
        tel.close()
    assert all(r.ok and r.runtime == "host" and r.e2e_us == 10.0 for r in out["ok"])
    assert out["ok"][0].e2e_ratio["med"] == 0.9                 # paired with the incumbent
    for what in ("crash", "hang", "poison"):
        r = out[what][0]
        assert not r.ok and r.stage == "run" and "restarted" in r.errors[-1]
    st = ev.host_stats()
    assert st["crashes"] == 1 and st["timeouts"] == 1 and st["poisoned"] == 1 and st["starts"] == 4
    # telemetry: host starts / restarts, the request as run.exec with the driver's phases
    s = tel.summary_of(tsp)
    assert s["counters"]["host.start"] == 4 and s["counters"]["host.restart"] == 3
    assert s["counters"]["host.crash"] == 1 and s["counters"]["host.timeout"] == 1 and s["counters"]["host.poisoned"] == 1
    for k in ("eval", "run", "run.exec", "driver.host_start", "driver.graph_build", "driver.time_partition", "build"):
        assert k in s["time_s"], k
    assert s["evals"]["candidate"]["n"] == 6


# ---------------------------------------------------------------- real host (LeNet fixture)

def _own_fixture(tmp_path) -> SlotProblem:
    import re
    d = tmp_path / "lenet"
    shutil.copytree(FIX, d)
    drv = d / "driver.cpp"
    drv.write_text(re.sub(r'(\.incbin \\")[^"\\]+', lambda m: m.group(1) + str(d / "data.bin"), drv.read_text()))
    return SlotProblem(d)


RELU_SEGV = '''#include <cuda_runtime.h>
void kernel_relu(const float *input, float *output, int n) { *(volatile int *)nullptr = n; }
'''
RELU_ILLEGAL = '''#include <cuda_runtime.h>
__global__ void k_bad(float *y, int n) { long i = blockIdx.x * (long)blockDim.x + threadIdx.x; if (i < n) ((float *)0x10)[i * 1024] = 1.0f; }
void kernel_relu(const float *input, float *output, int n) { k_bad<<<(n + 255) / 256, 256>>>(output, n); }
'''
RELU_HANG = '''#include <cuda_runtime.h>
__global__ void k_spin(volatile int *flag) { while (*flag == 0) {} }
void kernel_relu(const float *input, float *output, int n) { k_spin<<<1, 1>>>((volatile int *)output); }
'''
RELU_WRONG = '''#include <cuda_runtime.h>
__global__ void k_r(const float *x, float *y, int n) { long i = blockIdx.x * (long)blockDim.x + threadIdx.x; if (i < n) y[i] = x[i] > 0.f ? x[i] + 1.f : 0.f; }
void kernel_relu(const float *input, float *output, int n) { k_r<<<(n + 255) / 256, 256>>>(input, output, n); }
'''


@needs_gpu
def test_host_matches_process_path(tmp_path):
    """Same checks, same verdicts and errors as one driver process per run (graph build)."""
    p = _own_fixture(tmp_path)
    host = LocalEvaluator(ARCH, work_root=tmp_path / "h", runtime="host")
    proc = LocalEvaluator(ARCH, work_root=tmp_path / "p", runtime="process", graph=True)
    seeds = p.seed_sources()

    async def go():
        out = {}
        for name, src, kw in [("full", seeds, {}), ("step", seeds, step_eval_options("relu")),
                              ("wrong", {**seeds, "relu": RELU_WRONG}, step_eval_options("relu"))]:
            out[name] = (await host.evaluate(p, src, name, **kw), await proc.evaluate(p, src, name, **kw))
        await host.close()
        return out
    out = asyncio.run(go())
    for name, (h, q) in out.items():
        assert h.runtime == "host" and q.runtime == "process"
        assert h.ok == q.ok == (name != "wrong")
        assert h.slot_checks == q.slot_checks                # same isolated checks, identical errors
        assert {k: v["max_abs_err"] for k, v in h.outputs.items()} == {k: v["max_abs_err"] for k, v in q.outputs.items()}
    h, q = out["full"]
    assert h.e2e_basis == q.e2e_basis == "graph" and h.e2e_us and q.e2e_us
    assert set(h.slot_us) == set(q.slot_us) == set(p.order)
    if h.attr_skip is None:                                  # CUPTI available
        assert set(h.attr_slot_us) == set(p.order) and h.attr_launches >= len(p.manifest["ops"])
    assert out["wrong"][0].failing_slots() == out["wrong"][1].failing_slots() == ["relu"]


@needs_gpu
def test_host_crash_recovery(tmp_path):
    """A segfault, a sticky CUDA error and a hang are runtime errors; the next candidate gets a fresh host."""
    p = _own_fixture(tmp_path)
    ev = LocalEvaluator(ARCH, work_root=tmp_path / "w", runtime="host", timeout_s=20)
    seeds = p.seed_sources()

    async def go():
        rows = []
        for what, src in [("good", seeds["relu"] + "\n// 0\n"), ("segv", RELU_SEGV), ("good", seeds["relu"] + "\n// a\n"),
                          ("illegal", RELU_ILLEGAL), ("good", seeds["relu"] + "\n// b\n"), ("hang", RELU_HANG),
                          ("good", seeds["relu"] + "\n// c\n")]:
            r = await ev.evaluate(p, {**seeds, "relu": src}, what, **step_eval_options("relu"), incumbent=seeds)
            rows.append((what, r))
        st = ev.host_stats()
        await ev.close()
        return rows, st
    rows, st = asyncio.run(go())
    for what, r in rows:
        if what == "good":
            assert r.ok and r.runtime == "host", r.error_text()
            assert r.e2e_ratio and r.e2e_ratio["n"] >= 1
        else:
            assert not r.ok and r.stage == "run", (what, r.error_text())
    assert st["crashes"] == 1 and st["poisoned"] == 1 and st["timeouts"] == 1 and st["starts"] == 4


# ---------------------------------------------------------------- the end-to-end reward

def _res(slot_us, e2e, ratio=None, attr=None, launches=None, ok=True):
    r = EvalResult(ok=ok, stage="ok" if ok else "check", partition_us=e2e * 1.05, slot_us=dict(slot_us),
                   slot_checks={s: {"ok": True, "rel_err": 0.0} for s in slot_us}, e2e_us=e2e, e2e_basis="graph",
                   runtime="host")
    if ratio is not None:
        r.e2e_ratio = {"med": ratio, "mean": ratio, "min": ratio, "max": ratio, "n": 9}
    if attr:
        r.attr_slot_us = dict(attr)
        r.attr_sum_us = sum(attr.values())
        r.attr_slot_launches = dict(launches or {s: 1.0 for s in attr})
        r.attr_launches = sum(r.attr_slot_launches.values())
    return r


class _ScriptedEvaluator(Evaluator):
    """Results by the candidate's relu source: 'fused' is faster end to end (fewer launches) but
    its event-timed slot is not; 'tiny' halves the slot's event time and changes nothing end to end."""

    def __init__(self, p):
        self.p, self.calls = p, []
        self.base = {s: 10.0 for s in p.order}

    def pairs_incumbent(self, problem):
        return True

    async def evaluate(self, problem, sources, tag, *, only_slots=None, screen=False, incumbent=None):
        self.calls.append((tag, incumbent is not None))
        relu = sources["relu"]
        slots, e2e, ratio = dict(self.base), 100.0, 1.0 if incumbent is not None else None
        if "NOCAP" in relu and "FUSED" not in relu:   # correct, but the partition no longer captures
            r = _res(slots, 105.0, 1.0 if incumbent is not None else None, attr={s: v / 2 for s, v in slots.items()})
            r.e2e_basis, r.graph_skip = "eager", "operation not permitted when stream is capturing"
            return r
        if "FUSED" in relu:
            slots["relu"], e2e, ratio = 10.5, 90.0, 0.90 if incumbent is not None else None
        elif "TINY" in relu:
            slots["relu"], ratio = 5.0, 1.0 if incumbent is not None else None
        if incumbent is not None and "FUSED" in incumbent.get("relu", ""):
            ratio = 1.0 if "FUSED" in relu else ratio
        return _res(slots, e2e, ratio, attr={s: v / 2 for s, v in slots.items()})


def _run_agent(tmp_path, marker, reward):
    p = SlotProblem(FIX)
    name = f"rw_{marker}_{reward}"
    agentic_llm.register_stub(name, lambda msgs: f"```cuda\n// {marker}\n" + msgs[-1]["content"].split("```cuda")[-1]
                              .split("```")[0] + "\n```")
    ev = _ScriptedEvaluator(p)
    olog = OutcomeLog(tmp_path / f"{name}.jsonl")
    try:
        agent = MultiSlotAgent(p, ev, model=f"stub:{name}", out_dir=tmp_path / name, arch="75", rollouts=1, steps=1,
                               min_share=0.0, seed=0, reward=reward, outcome_log=olog, propose=False)
        agent._pick_slot = lambda: "relu"
        res = asyncio.run(agent.run())
    finally:
        agentic_llm.unregister_stub(name)
    return res, agent, ev, list(olog)


def test_e2e_reward_accepts_launch_win_the_slot_rule_misses(tmp_path):
    res, agent, ev, recs = _run_agent(tmp_path, "FUSED", "e2e")
    st = res["steps"][0]
    assert st["reward_basis"] == "e2e" and abs(st["reward"] - 0.10) < 1e-9
    e = st["context"]["e2e"]
    assert e["accept"] and e["pairing"] == "paired" and e["confirm_ok"]
    assert e["slot_rule"]["accept"] is False and e["slot_rule"]["reward"] < 0   # the old rule: a 5% slower slot
    assert "FUSED" in agent.best["relu"].source
    assert ("r0_s0_relu_confirm", True) in ev.calls                      # re-measured against the composite
    assert recs[-1].timing_basis == "e2e_graph" and abs(recs[-1].partition_effect - 0.10) < 1e-9
    assert recs[-1].status == "improved"
    assert res["reward"] == "e2e" and res["baseline_e2e_us"] == 100.0


def test_e2e_reward_ignores_slot_only_gain(tmp_path):
    res, agent, ev, recs = _run_agent(tmp_path, "TINY", "e2e")
    e = res["steps"][0]["context"]["e2e"]
    assert not e["accept"] and e["slot_rule"]["accept"] is True                # the old rule would take it
    assert res["steps"][0]["reward"] == 0.0 and "TINY" not in agent.best["relu"].source
    assert recs[-1].status == "neutral"


def test_slot_reward_keeps_the_previous_behaviour(tmp_path):
    res, agent, ev, recs = _run_agent(tmp_path, "TINY", "slot")
    st = res["steps"][0]
    assert st["reward_basis"] == "slot" and st["reward"] > 0 and "TINY" in agent.best["relu"].source
    assert all(not paired for _, paired in ev.calls)                         # no incumbent pairing
    assert recs[-1].timing_basis == "slot_events"


def test_capture_defect_goes_to_the_fix_loop(tmp_path):
    p = SlotProblem(FIX)
    seen = []

    def llm(msgs):
        txt = msgs[-1]["content"]
        seen.append("can no longer be captured" in txt)
        marker = "FUSED" if seen[-1] else "NOCAP"
        return f"```cuda\n// {marker}\n" + txt.split("```cuda")[-1].split("```")[0] + "\n```"
    agentic_llm.register_stub("rw_cap", llm)
    try:
        agent = MultiSlotAgent(p, _ScriptedEvaluator(p), model="stub:rw_cap", out_dir=tmp_path / "cap", arch="75",
                               rollouts=1, steps=1, min_share=0.0, seed=0, reward="e2e", propose=False)
        agent._pick_slot = lambda: "relu"
        res = asyncio.run(agent.run())
    finally:
        agentic_llm.unregister_stub("rw_cap")
    st = res["steps"][0]
    assert seen == [False, True]                        # codegen, then a fix call with the capture error
    assert st["fix_attempts"] == 1 and st["context"]["graph_capture"]["defects"]
    assert st["context"]["e2e"]["accept"] and "FUSED" in agent.best["relu"].source


def test_runtime_signals_and_prompt():
    r = _res({"conv2d": 50.0, "relu": 20.0}, 100.0, attr={"conv2d": 45.0, "relu": 5.0},
             launches={"conv2d": 2.0, "relu": 8.0})
    r.launch_overhead_us, r.launch_overhead_frac = 55.0, 0.52
    w = slot_weights(r)                                # kernels + their launches' share of 50 us overhead
    assert abs(w["conv2d"] - (45.0 + 10.0)) < 1e-9 and abs(w["relu"] - (5.0 + 40.0)) < 1e-9
    rt = partition_runtime(r)
    assert rt["launch_bound"] and rt["e2e_basis"] == "graph" and rt["kernel_us"] == 50.0
    p = SlotProblem(FIX)
    txt = runtime_text(r, "relu", p)
    assert "END-TO-END" in txt and "Launch-bound" in txt and "8 launches" in txt
    assert e2e_gain(_res({"relu": 1.0}, 90.0), 100.0) == (pytest.approx(0.1), "unpaired")
    assert e2e_gain(_res({"relu": 1.0}, 90.0, ratio=0.95), 100.0) == (pytest.approx(0.05), "paired")


def test_outcome_records_backward_compatible(tmp_path):
    log = tmp_path / "o.jsonl"
    old = {"technique": "t", "status": "improved", "metric_before": 2.0, "metric_after": 1.0, "id": "a", "ts": "x"}
    newer = {**old, "id": "b", "timing_basis": "e2e_graph", "partition_effect": 0.1, "some_future_field": 1}
    log.write_text(json.dumps(old) + "\n" + json.dumps(newer) + "\n")
    recs = list(OutcomeLog(log))
    assert recs[0].timing_basis is None and recs[1].timing_basis == "e2e_graph" and recs[1].partition_effect == 0.1
    assert OutcomeRecord(technique="t", status="neutral").timing_basis is None
