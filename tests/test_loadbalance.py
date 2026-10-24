"""Load balancing: CPU / cgroup / memory math, the adaptive limiter, GPU
inventory parsing and selection, the GPU pool, multi-host ssh dispatch (fake
ssh), the servers' defaults, and LocalEvaluator / run_multislot on the local
GPU (skipped without one)."""
import asyncio
import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelblaster.servers.utils import resources as R
from kernelblaster.servers.utils.resources import (AdaptiveLimiter, Device, GPUInfo, GPUPool, auto_jobs,
                                                   build_limiter, cgroup_cpu_limit, free_cpus, is_infra_failure,
                                                   mem_jobs, parse_cfs, parse_cpu_max, parse_gpu_spec,
                                                   parse_inventory, select_gpus, usable_cpus)

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "mb_lenet"
GiB = R.GiB


# ---------------------------------------------------------------- CPUs / memory

def test_cpu_max_and_cfs_parsing():
    assert parse_cpu_max("max 100000") is None
    assert parse_cpu_max("200000 100000") == 2.0
    assert parse_cpu_max("50000 100000\n") == 0.5
    assert parse_cfs("-1", "100000") is None
    assert parse_cfs("400000\n", "100000\n") == 4.0


def test_cgroup_v2_limit_is_tightest_ancestor(tmp_path):
    cg = tmp_path / "proc_cgroup"
    cg.write_text("0::/user.slice/user-1.slice/session-2.scope\n")
    root = tmp_path / "cg"
    (root / "user.slice" / "user-1.slice" / "session-2.scope").mkdir(parents=True)
    (root / "cpu.max").write_text("max 100000\n")
    (root / "user.slice" / "cpu.max").write_text("800000 100000\n")              # 8 CPUs
    (root / "user.slice" / "user-1.slice" / "cpu.max").write_text("300000 100000\n")   # 3 CPUs
    (root / "user.slice" / "user-1.slice" / "session-2.scope" / "cpu.max").write_text("max 100000\n")
    assert cgroup_cpu_limit(str(cg), str(root)) == 3.0
    (root / "user.slice" / "user-1.slice" / "cpu.max").write_text("max 100000\n")
    assert cgroup_cpu_limit(str(cg), str(root)) == 8.0


def test_cgroup_v1_limit(tmp_path):
    cg = tmp_path / "proc_cgroup"
    cg.write_text("4:cpu,cpuacct:/docker/abc\n3:memory:/docker/abc\n")
    d = tmp_path / "cg" / "cpu,cpuacct" / "docker" / "abc"
    d.mkdir(parents=True)
    (d / "cpu.cfs_quota_us").write_text("150000\n")
    (d / "cpu.cfs_period_us").write_text("100000\n")
    assert cgroup_cpu_limit(str(cg), str(tmp_path / "cg")) == 1.5
    (tmp_path / "none").write_text("0::/\n")
    assert cgroup_cpu_limit(str(tmp_path / "none"), str(tmp_path / "missing")) is None


def test_usable_free_and_memory_math():
    assert usable_cpus(affinity=48, quota=None) == 48
    assert usable_cpus(affinity=48, quota=6.0) == 6.0
    assert usable_cpus(affinity=4, quota=16.0) == 4
    # load 20, 8 of it ours -> others use 12 of 48
    assert free_cpus(usable=48, load=20, own=8) == 36
    assert free_cpus(usable=8, load=30, own=0) == 0
    assert free_cpus(usable=48, load=3, own=8) == 48         # our jobs not yet in the load average
    assert mem_jobs(2 * GiB, available=10 * GiB, reserve=2 * GiB) == 4
    assert mem_jobs(2 * GiB, available=10 * GiB, reserve=2 * GiB, own=3) == 7
    assert mem_jobs(2 * GiB, available=1 * GiB) == 0


def test_auto_jobs_backs_off_and_scales_up():
    kw = dict(share=0.5, usable=48, available=100 * GiB, per_job_mem=1 * GiB, own=0, max_jobs=16)
    assert auto_jobs(load=0, **kw) == 16                     # idle box: the cap
    assert auto_jobs(load=30, **kw) == 9                     # (48 - 30) * 0.5
    assert auto_jobs(load=60, **kw) == 1                     # saturated: never below 1
    assert auto_jobs(load=0, **{**kw, "available": 6 * GiB}) == 4      # memory-bound
    assert auto_jobs(load=0, **{**kw, "usable": 2}) == 1


def test_build_limiter_explicit_and_env_win(monkeypatch):
    assert build_limiter(3).fixed == 3
    monkeypatch.setenv("KB_TEST_JOBS", "5")
    lim = build_limiter(None, env="KB_TEST_JOBS")
    assert lim.fixed == 5 and lim.capacity() == 5
    assert build_limiter(2, env="KB_TEST_JOBS").fixed == 2
    monkeypatch.setenv("KB_TEST_JOBS", "auto")
    assert build_limiter(None, env="KB_TEST_JOBS").adaptive


def test_adaptive_limiter_follows_capacity():
    t = [0.0]
    cap = [2]
    lim = AdaptiveLimiter(lambda n: cap[0], interval=1.0, clock=lambda: t[0], name="t")
    running, peak = [0], []

    async def job(hold):
        async with lim:
            running[0] += 1
            peak.append(running[0])
            await asyncio.sleep(hold)
            running[0] -= 1

    async def go():
        own0 = R.own_running()
        tasks = [asyncio.create_task(job(0.05)) for _ in range(6)]
        await asyncio.sleep(0.01)
        assert lim.in_flight == 2 and R.own_running() == own0 + 2
        await asyncio.gather(*tasks)
        assert max(peak) == 2
        peak.clear()
        cap[0] = 4                    # host got idle: capacity re-probed after `interval`
        t[0] += 2.0
        tasks = [asyncio.create_task(job(0.05)) for _ in range(8)]
        await asyncio.gather(*tasks)
        assert max(peak) == 4
        peak.clear()
        cap[0] = 1                    # host got busy
        t[0] += 2.0
        await asyncio.gather(*[job(0.01) for _ in range(4)])
        assert max(peak) == 1
        assert R.own_running() == own0
    asyncio.run(go())
    assert [c for _, c in lim.history] == [2, 4, 1]


def test_adaptive_limiter_waiters_reprobe_without_releases():
    cap = [1]
    lim = AdaptiveLimiter(lambda n: cap[0], interval=0.05, name="t")

    async def go():
        await lim.acquire()                       # holds the only slot
        waiter = asyncio.create_task(lim.acquire())
        await asyncio.sleep(0.1)
        assert not waiter.done()
        cap[0] = 2                                # nothing released, yet the waiter gets in
        await asyncio.wait_for(waiter, 1.0)
        await lim.release()
        await lim.release()
    asyncio.run(go())


# ---------------------------------------------------------------- GPU inventory

FIRESIM_GPUS = """\
0, GPU-a2a44f60-c7d1-4450-93df-ead875ea76e8, NVIDIA L40S, 34564, 46068, 0
1, GPU-6a6a9d5a-07da-01f7-4acb-594afc813213, NVIDIA L40S, 38181, 46068, 0
2, GPU-22350958-ee15-b07b-0e89-77c5157f1d3b, NVIDIA L40S, 0, 46068, 0
3, GPU-019a51ce-7238-4d15-f03d-e9194d1c4e79, NVIDIA L40S, 3, 46068, 37
"""
FIRESIM_APPS = """\
GPU-a2a44f60-c7d1-4450-93df-ead875ea76e8, 2340766, ./.venv/bin/python, 34550
GPU-6a6a9d5a-07da-01f7-4acb-594afc813213, 3191359, VLLM::EngineCore, 38172
"""


def _inv(host=None, own=()):
    return parse_inventory(FIRESIM_GPUS + R._SEP + "\n" + FIRESIM_APPS, host=host, own_pids=own)


def test_parse_inventory():
    inv = _inv("u@h")
    assert [g.index for g in inv] == [0, 1, 2, 3]
    assert inv[0].procs == [(2340766, "./.venv/bin/python", 34550.0)] and inv[1].procs[0][1] == "VLLM::EngineCore"
    assert inv[2].procs == [] and inv[2].mem_total_mb == 46068 and inv[3].util == 37
    assert all(g.host == "u@h" and g.name == "NVIDIA L40S" for g in inv)
    # our own processes are not foreign
    assert _inv(own={2340766})[0].procs == []
    # [N/A] fields, no compute apps, garbage lines
    inv = parse_inventory("0, GPU-x, Tesla T4, [N/A], 15360, [N/A]\nNo devices were found\n" + R._SEP + "\n")
    assert len(inv) == 1 and inv[0].util == 0 and inv[0].procs == []
    assert parse_inventory("") == []


def test_select_gpus(monkeypatch):
    monkeypatch.delenv("KERNELBLASTER_ALLOW_SHARED_GPU", raising=False)
    inv = _inv()
    idx = lambda gs: [g.index for g in gs]   # noqa: E731
    assert idx(select_gpus("auto", inv, visible=None)[0]) == [2]           # 3 is at 37% util
    assert idx(select_gpus(None, inv, visible=None, util_max=50)[0]) == [2, 3]
    assert idx(select_gpus("auto", inv, visible=None, allow_shared=True)[0]) == [0, 1, 2, 3]
    monkeypatch.setenv("KERNELBLASTER_ALLOW_SHARED_GPU", "1")
    assert idx(select_gpus("auto", inv, visible=None)[0]) == [0, 1, 2, 3]
    monkeypatch.delenv("KERNELBLASTER_ALLOW_SHARED_GPU")
    # explicit always wins, busy or not; UUID prefixes work
    assert idx(select_gpus("1,0", inv, visible=None)[0]) == [1, 0]
    assert idx(select_gpus(["GPU-019a51ce"], inv, visible=None)[0]) == [3]
    chosen, why = select_gpus("7", inv, visible=None)
    assert chosen == [] and "not found" in why
    # an external CUDA_VISIBLE_DEVICES bounds auto
    assert idx(select_gpus("auto", inv, visible="0,1,3", util_max=50)[0]) == [3]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a2a44f60-c7d1-4450-93df-ead875ea76e8,2")
    assert idx(select_gpus("auto", inv)[0]) == [2]
    assert idx(select_gpus("all", inv)[0]) == [0, 2]
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    # nothing idle: the least busy one, flagged
    busy = [g for g in inv if g.index in (0, 1)]
    chosen, why = select_gpus("auto", busy, visible=None)
    assert idx(chosen) == [0] and "least busy" in why
    assert parse_gpu_spec(" auto ") is None and parse_gpu_spec([0, 2]) == ["0", "2"]


def test_local_devices_without_nvidia_smi():
    assert [d.ident for d in R.local_devices(None, inventory=[])] == [None]
    assert [d.ident for d in R.local_devices("0,1", inventory=[])] == ["0", "1"]
    devs = R.local_devices("auto", inventory=_inv()[2:3])
    assert devs[0].ident.startswith("GPU-2235") and devs[0].env() == {"CUDA_VISIBLE_DEVICES": devs[0].ident}


def test_infra_failure_classification():
    assert is_infra_failure("CUDA error: no CUDA-capable device is detected")
    assert is_infra_failure("MB_ERROR [infra] scp to h failed")
    assert is_infra_failure("ssh: connect to host x port 22: Connection refused")
    assert not is_infra_failure("MB_ERROR CUDA error: an illegal memory access was encountered")
    assert not is_infra_failure("MB_CHECK output y max_abs_err=1 bound=0.1 ok=0\nfailed")
    assert not is_infra_failure(None)


# ---------------------------------------------------------------- GPU pool

def _devs(n, hosts=("a", "b"), name="L40S"):
    return [Device(h, f"GPU-{h}{i}", name, i) for h in hosts for i in range(n)]


def test_pool_exclusive_and_spread():
    pool = GPUPool(_devs(2))                     # 2 hosts x 2 GPUs
    active: dict = {}
    peak: dict = {}
    order = []

    async def run(i):
        d = await pool.acquire(group="p")
        active[d.key] = active.get(d.key, 0) + 1
        peak[d.key] = max(peak.get(d.key, 0), active[d.key])
        order.append(d.key)
        await asyncio.sleep(0.005 * (1 + i % 3))
        active[d.key] -= 1
        await pool.release(d)

    async def go():
        await asyncio.gather(*[run(i) for i in range(40)])
    asyncio.run(go())
    assert set(peak) == {d.key for d in pool.devices} and max(peak.values()) == 1
    assert pool.peak_busy == 4
    runs = [d.runs for d in pool.devices]
    assert sum(runs) == 40 and min(runs) >= 6          # LRU keeps every device busy
    assert len(set(order[:4])) == 4                    # first wave: one run per device


def test_pool_lru_choice():
    pool = GPUPool(_devs(3, hosts=("a",)))

    async def go():
        seq = []
        for _ in range(6):
            d = await pool.acquire()
            seq.append(d.index)
            await pool.release(d)
        return seq
    assert asyncio.run(go()) == [0, 1, 2, 0, 1, 2]


def test_pool_pins_group_to_gpu_model():
    devs = [Device(None, "GPU-t", "TITAN RTX", 0), Device("h", "GPU-l0", "L40S", 0),
            Device("h", "GPU-l1", "L40S", 1)]
    pool = GPUPool(devs)
    assert not pool.homogeneous()

    async def go():
        d1 = await pool.acquire(group="p1")            # LRU tie -> first device
        d2 = await pool.acquire(group="p2")            # p2 gets an L40S
        d3 = await pool.acquire(group="p2")
        assert d1.name == "TITAN RTX" and d2.name == d3.name == "L40S"
        # p1 is pinned to the TITAN: it waits for it even though nothing else is busy-free
        t = asyncio.create_task(pool.acquire(group="p1"))
        await pool.release(d2)                         # an L40S frees up: not for p1
        await asyncio.sleep(0.02)
        assert not t.done()
        await pool.release(d1)
        d4 = await asyncio.wait_for(t, 1)
        assert d4.name == "TITAN RTX" and pool.pinned_model("p1") == "TITAN RTX"
        # ungrouped leases may use anything
        d5 = await pool.acquire()
        assert d5.name == "L40S"
    asyncio.run(go())


def test_pool_quarantine_after_infra_failures():
    t = [100.0]
    pool = GPUPool(_devs(1), quarantine_after=3, quarantine_s=60, clock=lambda: t[0])
    a, b = pool.devices

    async def go():
        # candidate failures never quarantine
        for _ in range(5):
            d = await pool.acquire()
            await pool.release(d, infra_error=False)
        assert a.quarantines == b.quarantines == 0
        # host a keeps failing: 2 strikes, a success resets, then 3 strikes
        for errs in ([True, True, False], [True, True, True]):
            for e in errs:
                await pool.release(await _acquire_dev(pool, a), infra_error=e)
        assert a.quarantines == 1 and a.quarantined_until == 160
        for _ in range(4):                              # only b is handed out now
            d = await pool.acquire()
            assert d is b
            await pool.release(d)
        t[0] = 161                                      # quarantine over
        seen = set()
        for _ in range(2):
            d = await pool.acquire()
            seen.add(d.host)
            await pool.release(d)
        assert seen == {"a", "b"}
    asyncio.run(go())


async def _acquire_dev(pool, want):
    """Acquire until ``want`` comes up (others are released straight back)."""
    held = []
    while True:
        d = await pool.acquire()
        if d is want:
            for h in held:
                await pool.release(h)
            return d
        held.append(d)


def test_pool_waits_when_all_quarantined():
    pool = GPUPool([Device(None, "0", "g", 0)], quarantine_after=1, quarantine_s=0.1)

    async def go():
        d = await pool.acquire()
        await pool.release(d, infra_error=True)
        t0 = time.monotonic()
        d = await asyncio.wait_for(pool.acquire(), 2)
        assert time.monotonic() - t0 >= 0.05
        await pool.release(d)
    asyncio.run(go())


# ---------------------------------------------------------------- multi-host ssh dispatch (fake ssh)

FAKE_SSH = r'''#!/usr/bin/env python3
import os, subprocess, sys
a = sys.argv[1:]
while a and a[0] in ("-o", "-i"):
    a = a[2:]
host, cmd = a[0], " ".join(a[1:])
state = os.environ["FAKE_SSH_STATE"]
if os.path.exists(os.path.join(state, "down_" + host)):
    sys.stderr.write(f"ssh: connect to host {host} port 22: Connection refused\n"); sys.exit(255)
if "nvidia-smi" in cmd:
    sys.stdout.write(open(os.path.join(state, "inv_" + host)).read()); sys.exit(0)
sys.exit(subprocess.call(["bash", "-c", cmd], env={**os.environ, "FAKE_HOST": host}))
'''
FAKE_SCP = r'''#!/usr/bin/env python3
import os, shutil, sys
a = [x for x in sys.argv[1:] if x != "-q"]
while a and a[0] in ("-o", "-i"):
    a = a[2:]
src, dst = a
host, path = dst.split(":", 1)
if os.path.exists(os.path.join(os.environ["FAKE_SSH_STATE"], "down_" + host)):
    sys.stderr.write(f"ssh: connect to host {host} port 22: Connection refused\n"); sys.exit(1)
shutil.copy(src, path)
'''
FAKE_BINARY = r'''#!/bin/bash
echo "start $(date +%s.%N) $FAKE_HOST $CUDA_VISIBLE_DEVICES" >> "$FAKE_RUN_LOG"
sleep 0.15
echo "end $(date +%s.%N) $FAKE_HOST $CUDA_VISIBLE_DEVICES" >> "$FAKE_RUN_LOG"
echo "MB_SLOT_US a 1.0"
echo "MB_PARTITION_US 1.0"
echo "MB_DEVICE $FAKE_HOST $CUDA_VISIBLE_DEVICES"
echo passed
'''


def _exe(p: Path, text: str):
    p.write_text(text)
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _fake_ssh_env(tmp_path, monkeypatch, hosts):
    state = tmp_path / "ssh_state"
    state.mkdir()
    _exe(tmp_path / "ssh", FAKE_SSH)
    _exe(tmp_path / "scp", FAKE_SCP)
    for h, n_busy in hosts.items():
        lines = [f"{i}, GPU-{h}-{i}, NVIDIA L40S, 0, 46068, 0" for i in range(4)]
        apps = [f"GPU-{h}-{i}, {1000 + i}, someone_else, 30000" for i in range(n_busy)]
        (state / f"inv_{h}").write_text("\n".join(lines) + f"\n{R._SEP}\n" + "\n".join(apps) + "\n")
        (tmp_path / f"remote_{h}").mkdir()
    monkeypatch.setenv("FAKE_SSH_STATE", str(state))
    monkeypatch.setenv("FAKE_RUN_LOG", str(tmp_path / "runs.log"))
    monkeypatch.delenv("KERNELBLASTER_ALLOW_SHARED_GPU", raising=False)
    return state


def _fake_ssh_evaluator(tmp_path, hosts, **kw):
    from kernelblaster.knowledge.slots import SSHEvaluator

    class FakeBuild(SSHEvaluator):
        async def _build(self, problem, sources, work):
            _exe(work / "problem", FAKE_BINARY)
            return None

    return FakeBuild(list(hosts), arch="89", remote_dir={h: str(tmp_path / f"remote_{h}") for h in hosts},
                     ssh=str(tmp_path / "ssh"), scp=str(tmp_path / "scp"), work_root=tmp_path / "work", **kw)


def _max_overlap(log: Path) -> dict:
    ev = []
    for line in log.read_text().splitlines():
        kind, ts, host, dev = line.split()
        ev.append((float(ts), 0 if kind == "end" else 1, (host, dev)))
    cur, peak = {}, {}
    for _, k, dev in sorted(ev):
        cur[dev] = cur.get(dev, 0) + (1 if k else -1)
        peak[dev] = max(peak.get(dev, 0), cur[dev])
    return peak


def test_ssh_multi_host_dispatch(tmp_path, monkeypatch):
    _fake_ssh_env(tmp_path, monkeypatch, {"u@h1": 1, "u@h2": 2})     # h1: GPUs 1-3 idle, h2: 2-3 idle
    ev = _fake_ssh_evaluator(tmp_path, ["u@h1", "u@h2"], cuda_visible_devices={"u@h1": "auto", "u@h2": None})
    prob = SimpleNamespace(root=tmp_path, name="p")

    async def go():
        return await asyncio.gather(*[ev.evaluate(prob, {}, f"c{i}") for i in range(15)])
    t0 = time.time()
    res = asyncio.run(go())
    wall = time.time() - t0
    assert all(r.ok and r.partition_us == 1.0 for r in res)
    pool = ev._pool
    assert sorted(d.label() for d in pool.devices) == ["u@h1:1", "u@h1:2", "u@h1:3", "u@h2:2", "u@h2:3"]
    peak = _max_overlap(tmp_path / "runs.log")
    assert set(peak) == {(d.host, d.ident) for d in pool.devices}           # every idle GPU used
    assert max(peak.values()) == 1                                         # never two runs on a GPU
    assert {r.device for r in res} == {f"{d.host} {d.ident}" for d in pool.devices}
    assert pool.peak_busy == 5
    assert wall < 15 * 0.15                                                # ran concurrently
    # binaries are removed from the remote dirs
    assert not list((tmp_path / "remote_u@h1").iterdir()) and not list((tmp_path / "remote_u@h2").iterdir())


def test_ssh_explicit_gpus_and_infra_retry(tmp_path, monkeypatch):
    state = _fake_ssh_env(tmp_path, monkeypatch, {"u@h1": 0, "u@h2": 0})
    ev = _fake_ssh_evaluator(tmp_path, ["u@h1", "u@h2"], cuda_visible_devices="0,1")
    prob = SimpleNamespace(root=tmp_path, name="p")

    async def go():
        await ev.pool()
        (state / "down_u@h2").write_text("")          # h2 drops off after the inventory
        return await asyncio.gather(*[ev.evaluate(prob, {}, f"c{i}") for i in range(8)])
    res = asyncio.run(go())
    assert all(r.ok for r in res)                     # infra failures retried on h1, not the candidate's fault
    assert {r.device.split()[0] for r in res} == {"u@h1"}
    by = {d.label(): d for d in ev._pool.devices}
    assert set(by) == {"u@h1:0", "u@h1:1", "u@h2:0", "u@h2:1"}
    assert by["u@h2:0"].quarantines + by["u@h2:1"].quarantines >= 1


def test_ssh_unreachable_host_is_skipped(tmp_path, monkeypatch):
    state = _fake_ssh_env(tmp_path, monkeypatch, {"u@h1": 0, "u@h2": 0})
    (state / "down_u@h2").write_text("")
    ev = _fake_ssh_evaluator(tmp_path, ["u@h1", "u@h2"], cuda_visible_devices="all")
    pool = asyncio.run(ev.pool())
    assert {d.host for d in pool.devices} == {"u@h1"} and len(pool) == 4


# ---------------------------------------------------------------- servers

def test_exec_server_gpu_ids(monkeypatch):
    pytest.importorskip("uvicorn")
    from kernelblaster.servers import exec_server

    monkeypatch.setattr(R, "gpu_inventory", lambda *a, **k: _inv())
    monkeypatch.delenv("KERNELBLASTER_ALLOW_SHARED_GPU", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("KERNELBLASTER_GPU_SERVER_GPU_IDS", raising=False)
    monkeypatch.delenv("KERNELBLASTER_GPU_SERVER_NUM_WORKERS", raising=False)
    assert exec_server.resolve_local_gpu_ids() == ["GPU-22350958-ee15-b07b-0e89-77c5157f1d3b"]
    monkeypatch.setenv("KERNELBLASTER_ALLOW_SHARED_GPU", "1")
    assert len(exec_server.resolve_local_gpu_ids()) == 4
    monkeypatch.setenv("KERNELBLASTER_GPU_SERVER_NUM_WORKERS", "2")
    assert exec_server.resolve_local_gpu_ids() == ["0", "1"]
    monkeypatch.setenv("KERNELBLASTER_GPU_SERVER_GPU_IDS", "3,1")
    assert exec_server.resolve_local_gpu_ids() == ["3", "1"]
    monkeypatch.setattr(R, "gpu_inventory", lambda *a, **k: [])
    monkeypatch.delenv("KERNELBLASTER_GPU_SERVER_GPU_IDS")
    monkeypatch.delenv("KERNELBLASTER_GPU_SERVER_NUM_WORKERS")
    assert exec_server.resolve_local_gpu_ids() == ["0"]


def test_compile_server_workers(monkeypatch):
    pytest.importorskip("uvicorn")
    from kernelblaster.servers import compile_server

    assert compile_server.resolve_workers("3") == (3, None)
    assert compile_server.resolve_workers(5) == (5, None)
    monkeypatch.setattr(R, "usable_cpus", lambda **k: 8.0)
    monkeypatch.setattr(R, "mem_available", lambda: 100 * GiB)
    n, lim = compile_server.resolve_workers("auto")
    assert n == 4 and lim.adaptive and not lim.counts
    monkeypatch.setattr(R, "load_average", lambda: 0.0)
    assert lim.capacity(force=True) == 3                    # 8 free x 0.75 = 6 procs = 3 jobs
    monkeypatch.setattr(R, "load_average", lambda: 7.0)
    assert lim.capacity(force=True) == 1
    monkeypatch.setattr(R, "mem_available", lambda: 4 * GiB)
    assert compile_server.resolve_workers("auto")[0] == 1


def test_fast_builder_jobs(monkeypatch, tmp_path):
    from kernelblaster.servers.cuda_fast import FastCudaBuilder

    monkeypatch.setenv("KERNELBLASTER_FAST_COMPILE_JOBS", "8")
    assert FastCudaBuilder(tmp_path)._jobs.fixed == 8
    monkeypatch.delenv("KERNELBLASTER_FAST_COMPILE_JOBS")
    assert FastCudaBuilder(tmp_path)._jobs.adaptive


def test_gpu_preflight_scopes_uuids(monkeypatch):
    pytest.importorskip("fastapi")
    from kernelblaster.servers import gpu

    async def fake(cmd, *a, **k):
        return "0, GPU-aaa\n1, GPU-bbb\n", ""
    monkeypatch.setattr(gpu, "exec_command", fake)
    monkeypatch.setattr(gpu, "GPU_IDS", ["GPU-bbb", "0"])
    assert asyncio.run(gpu._resolve_assigned_gpu_uuids()) == {"GPU-aaa", "GPU-bbb"}


# ---------------------------------------------------------------- shared knowledge with parallel problems

def test_forced_reflect_waits_for_a_running_pass(tmp_path):
    from kernelblaster.agents.utils import agentic_llm
    from kernelblaster.knowledge.claims import ClaimStore
    from kernelblaster.knowledge.outcomes import OutcomeLog, OutcomeRecord
    from kernelblaster.knowledge.reflect import SemanticUpdater

    log = OutcomeLog(tmp_path / "o.jsonl")
    store = ClaimStore(tmp_path / "c.json")
    calls = []

    async def slow(m):
        calls.append(len(m[-1]["content"]))
        await asyncio.sleep(0.05)
        return '{"edits": []}'
    agentic_llm.register_stub("lb_slow", slow)
    try:
        up = SemanticUpdater(log, store, model="stub:lb_slow", every_n=4)
        for _ in range(4):
            log.append(OutcomeRecord(technique="t", status="neutral"))

        async def go():
            first = asyncio.create_task(up.maybe_reflect())      # problem A's periodic pass
            await asyncio.sleep(0.01)
            log.append(OutcomeRecord(technique="t", status="neutral"))   # problem B finishes meanwhile
            assert await up.maybe_reflect() == []                       # non-forced: skipped
            await up.maybe_reflect(force=True)                          # forced: waits, then reflects B's record
            await first
        asyncio.run(go())
    finally:
        agentic_llm.unregister_stub("lb_slow")
    assert len(calls) == 2 and store.meta["reflect_cursor"] == 5


# ---------------------------------------------------------------- local GPU integration

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


@needs_gpu
def test_local_evaluator_concurrent_on_pool(tmp_path, monkeypatch):
    from kernelblaster.knowledge.slots import LocalEvaluator, SlotProblem

    monkeypatch.setenv("KERNELBLASTER_ALLOW_SHARED_GPU", "1")     # shared dev box: use the GPU anyway
    p = SlotProblem(FIX)
    seeds = p.seed_sources()
    ev = LocalEvaluator(ARCH, work_root=tmp_path / "w", build_jobs=4)
    seen = []
    orig = ev._run

    async def spy(exe, env=None, prefix="", device=None):
        seen.append(device.env().get("CUDA_VISIBLE_DEVICES"))
        return await orig(exe, env, prefix, device=device)
    ev._run = spy
    cands = [{**seeds, "relu": seeds["relu"] + f"\n// candidate {i}\n"} for i in range(4)]

    async def go():
        return await asyncio.gather(*[ev.evaluate(p, c, f"c{i}") for i, c in enumerate(cands)])
    res = asyncio.run(go())
    assert all(r.ok and r.partition_us for r in res), [r.error_text(300) for r in res]
    pool = ev._pool
    assert sum(d.runs for d in pool.devices) == 4 and pool.peak_busy <= len(pool)
    assert ev._jobs.fixed == 4 and 1 <= ev._jobs.peak <= 4
    uuids = {d.ident for d in pool.devices}
    assert set(seen) <= uuids and all(s and s.startswith("GPU-") for s in seen)


@needs_gpu
def test_run_multislot_parallel_problems(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELBLASTER_ALLOW_SHARED_GPU", "1")
    for n in ("a", "b"):
        shutil.copytree(FIX, tmp_path / "probs" / n)
        m = json.loads((tmp_path / "probs" / n / "manifest.json").read_text())
        m["name"] = f"lenet_{n}"
        (tmp_path / "probs" / n / "manifest.json").write_text(json.dumps(m))
    plugin = tmp_path / "plugin.py"
    plugin.write_text(textwrap.dedent('''
        from kernelblaster.agents.opt_multislot import extract_code
        from kernelblaster.agents.utils import agentic_llm

        def same(msgs):
            user = msgs[-1]["content"]
            cur = extract_code(user.split("Current source:")[-1] if "Current source:" in user
                               else user.split("Current slot source:")[-1])
            return "```cuda\\n" + cur + "\\n// same\\n```"
        agentic_llm.register_stub("lb_same", same)
    '''))
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_multislot.py"), str(tmp_path / "probs"),
                        "--out", str(out), "--model", "stub:lb_same", "--plugin", str(plugin), "--arch", ARCH,
                        "--rollouts", "1", "--steps", "1", "--parallel-problems", "2", "--compile-jobs", "4"],
                       capture_output=True, text=True, timeout=900, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert r.returncode == 0, r.stderr[-3000:]
    summary = json.loads((out / "summary.json").read_text())
    probs = {s["problem"] for s in summary if "problem" in s}
    assert probs == {"lenet_a", "lenet_b"} and all(s.get("final_ok") for s in summary if "problem" in s)
    assert (out / "lenet_a" / "result.json").exists() and (out / "lenet_b" / "result.json").exists()
    lb = next(s["load_balance"] for s in summary if "load_balance" in s)
    assert lb["parallel_problems"] == 2 and lb["cuda"]["compile_jobs"] == "4"
    assert sum(g["runs"] for g in lb["cuda"]["gpus"]) >= 2 * 3      # baseline + step + final per problem
