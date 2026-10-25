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
a, user = sys.argv[1:], None
while a and a[0] in ("-o", "-i", "-l"):
    if a[0] == "-l":
        user = a[1]
    a = a[2:]
host, cmd = (f"{user}@{a[0]}" if user else a[0]), " ".join(a[1:])
state = os.environ["FAKE_SSH_STATE"]
if os.path.exists(os.path.join(state, "down_" + host)):
    sys.stderr.write(f"ssh: connect to host {host} port 22: Connection refused\n"); sys.exit(255)
if "nvidia-smi" in cmd and os.path.exists(os.path.join(state, "inv_" + host)):
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
    monkeypatch.setenv("KERNELBLASTER_GPU_LOCK_DIR", str(state))
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
    # rsync deltas against each host's last binary of the problem; only that basis stays behind
    assert ev.transfer == {"rsync": 15, "scp": 0}
    for h in ("u@h1", "u@h2"):
        assert [f.name for f in (tmp_path / f"remote_{h}").iterdir()] == [ev._own]       # its own stage dir
        assert [f.name.startswith(".kb_basis_") for f in (tmp_path / f"remote_{h}" / ev._own).iterdir()] == [True]
    asyncio.run(ev.close())
    assert not list((tmp_path / "remote_u@h1").iterdir()) and not list((tmp_path / "remote_u@h2").iterdir())


def test_ssh_scp_without_rsync(tmp_path, monkeypatch):
    _fake_ssh_env(tmp_path, monkeypatch, {"u@h1": 0})
    monkeypatch.setenv("KERNELBLASTER_SSH_RSYNC", "0")
    ev = _fake_ssh_evaluator(tmp_path, ["u@h1"], cuda_visible_devices="2,3")
    prob = SimpleNamespace(root=tmp_path, name="p")

    async def go():
        return await asyncio.gather(*[ev.evaluate(prob, {}, f"c{i}") for i in range(4)])
    assert all(r.ok for r in asyncio.run(go()))
    assert ev.transfer == {"rsync": 0, "scp": 4}
    assert not list((tmp_path / "remote_u@h1" / ev._own).iterdir())        # each binary removed after its run
    asyncio.run(ev.close())
    assert not list((tmp_path / "remote_u@h1").iterdir())
    assert {d.label() for d in ev._pool.devices} == {"u@h1:2", "u@h1:3"}


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


def _running_copy(path: Path):
    """``path`` (a copy of ``sleep``) running, as a persistent host binary would be."""
    p = subprocess.Popen([str(path), "60"])
    time.sleep(0.2)
    assert p.poll() is None
    return p


def test_ssh_shared_remote_dir_never_copies_over_a_running_host(tmp_path, monkeypatch):
    """Two processes on one remote dir (a flow's KernelBlaster run and ModelBlaster's
    re-measurement of a hint): the second one stages the same content-named host
    binary while the first one's host runs it. It used to be copied over the running
    executable -- ``scp: dest open ...: Failure`` (ETXTBSY), three infrastructure
    failures, the GPU quarantined and the re-measurement lost -- and either's close()
    removed the other's binaries and shared objects."""
    _fake_ssh_env(tmp_path, monkeypatch, {"u@h1": 0})
    host_bin = tmp_path / "kbhost_src"
    shutil.copy(shutil.which("sleep"), host_bin)
    so = tmp_path / "cand.so"
    so.write_text("shared object")
    a = _fake_ssh_evaluator(tmp_path, ["u@h1"], cuda_visible_devices="2")
    b = _fake_ssh_evaluator(tmp_path, ["u@h1"], cuda_visible_devices="2")
    rdir = tmp_path / "remote_u@h1"
    assert a._stage_dir("u@h1") != b._stage_dir("u@h1")
    assert asyncio.run(a._stage_file("u@h1", host_bin, "kbhost_1d16bff26c55e4fd", set())) is None
    assert asyncio.run(a._stage_file("u@h1", so, "kbso_x.so", set())) is None
    running = _running_copy(Path(a._stage_dir("u@h1")) / "kbhost_1d16bff26c55e4fd")
    try:
        # the same content-named binary while a's host runs it: b's own copy
        assert asyncio.run(b._stage_file("u@h1", host_bin, "kbhost_1d16bff26c55e4fd", set())) is None
        assert asyncio.run(b._stage_file("u@h1", so, "kbso_x.so", set())) is None
        assert (Path(b._stage_dir("u@h1")) / "kbhost_1d16bff26c55e4fd").exists()
        assert not [f for f in Path(b._stage_dir("u@h1")).iterdir() if ".part." in f.name]   # renamed into place
        asyncio.run(b.close())                         # b done: a's files and host untouched
        assert not Path(b._stage_dir("u@h1")).exists()
        assert {f.name for f in Path(a._stage_dir("u@h1")).iterdir()} == {"kbhost_1d16bff26c55e4fd", "kbso_x.so"}
        assert running.poll() is None
    finally:
        running.kill()
        running.wait()
    asyncio.run(a.close())
    assert not list(rdir.iterdir())


def test_ssh_copy_error_text_keeps_the_cause(tmp_path, monkeypatch):
    """A failed copy's error names its cause (scp's first line), on one line."""
    from kernelblaster.knowledge.slots import _copy_err
    out = ('scp: dest open "/r/kbhost_1d16bff26c55e4fd": Failure\r\n'
           "scp: failed to upload file /very/long/local/path/" + "x" * 150 + ".host to /r/kbhost_1d16bff26c55e4fd\r\n")
    txt = _copy_err(out)
    assert txt.startswith('scp: dest open "/r/kbhost_1d16bff26c55e4fd": Failure | scp: failed to upload')
    assert "\r" not in txt and "\n" not in txt


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


# ---------------------------------------------------------------- run-time data.bin over ssh (> 1 GiB problems)

# rsync / scp stand-ins: log every data.bin transfer (start, end, host), fail while a
# datafail_<host> counter is positive, sleep $FAKE_DATA_SLEEP, then copy for real
FAKE_DATA_COPY = r'''#!/usr/bin/env python3
import os, shutil, subprocess, sys, time
state, tool = os.environ["FAKE_SSH_STATE"], os.path.basename(sys.argv[0])
dst = sys.argv[-1]
host = dst.split(":", 1)[0]
data = ".data.bin.part" in dst
if os.path.exists(os.path.join(state, "down_" + host)):
    sys.stderr.write(f"ssh: connect to host {host} port 22: Connection refused\n"); sys.exit(255)
if data:
    f = os.path.join(state, "datafail_" + host)
    n = int(open(f).read() or 0) if os.path.exists(f) else 0
    if n > 0:
        open(f, "w").write(str(n - 1))
        sys.stderr.write(f"{tool}: connection to {host} reset\n"); sys.exit(12)
    t0 = time.time()
    time.sleep(float(os.environ.get("FAKE_DATA_SLEEP", "0")))
if tool.startswith("rsync"):
    rc = subprocess.call([os.environ.get("REAL_RSYNC", "rsync"), *sys.argv[1:]])
else:
    a = [x for x in sys.argv[1:] if x != "-q"]
    while a and a[0] in ("-o", "-i"):
        a = a[2:]
    shutil.copy(a[0], a[1].split(":", 1)[1]); rc = 0
if data:
    with open(os.environ["FAKE_COPY_LOG"], "a") as f:
        f.write(f"{tool} {t0} {time.time()} {host}\n")
sys.exit(rc)
'''
# the driver's run-time data read: fails like the real one without MB_DATA_PATH
FAKE_DATA_BINARY = r'''#!/bin/bash
[ -n "$MB_DATA_PATH" ] && [ -r "$MB_DATA_PATH" ] || { echo "MB_ERROR cannot open ${MB_DATA_PATH:-/nonexistent/data.bin}"; echo failed; exit 1; }
echo "start $(date +%s.%N) $FAKE_HOST $CUDA_VISIBLE_DEVICES $(sha1sum < "$MB_DATA_PATH" | cut -c1-40) $MB_DATA_PATH" >> "$FAKE_RUN_LOG"
echo "MB_SLOT_US a 1.0"
echo "MB_PARTITION_US 1.0"
echo passed
'''
FAKE_DATA_HOST = r'''#!%s
import hashlib, os, sys
p = os.environ.get("MB_DATA_PATH", "/nonexistent/data.bin")
if not os.path.exists(p):
    print(f"MB_ERROR cannot open {p}\nfailed", flush=True); sys.exit(1)
sha = hashlib.sha1(open(p, "rb").read()).hexdigest()
print("MB_HOST_READY abi=1 nb=1 nops=1 nslots=1 init_ms=1.0", flush=True)
for line in sys.stdin:
    f = line.rstrip("\n").split("\t")
    if f[0] == "QUIT":
        break
    with open(os.environ["FAKE_RUN_LOG"], "a") as log:
        log.write(f"host {os.environ.get('FAKE_HOST')} {sha} {p}\n")
    print("MB_CHECK output y max_abs_err=0 bound=1 ok=1", flush=True)
    print("MB_E2E_US 10.0 basis=graph mean=10.0 reps=5", flush=True)
    print("passed", flush=True)
    print(f"MB_HOST_DONE {f[1]} rc=0 poisoned=0 recycle=0", flush=True)
''' % sys.executable


def _data_problem(tmp_path, name, data: bytes, host_marker=False):
    d = tmp_path / name
    d.mkdir()
    (d / "data.bin").write_bytes(data)
    drv = d / "driver.cpp"
    drv.write_text('const char *path = getenv("MB_DATA_PATH") ? getenv("MB_DATA_PATH") : "/nonexistent/data.bin";\n'
                   + ("// MB_HOST_READY\n" if host_marker else ""))
    return SimpleNamespace(root=d, name=name, driver_path=drv, manifest={"data": "data.bin"}, target="cuda")


def _data_env(tmp_path, monkeypatch, hosts, sleep=0.0):
    state = _fake_ssh_env(tmp_path, monkeypatch, hosts)
    for tool in ("rsync", "scp"):
        _exe(tmp_path / f"{tool}_data", FAKE_DATA_COPY)
        (tmp_path / tool).unlink(missing_ok=True)
        (tmp_path / tool).symlink_to(tmp_path / f"{tool}_data")
    monkeypatch.setenv("REAL_RSYNC", shutil.which("rsync") or "rsync")
    monkeypatch.setenv("FAKE_COPY_LOG", str(tmp_path / "copies.log"))
    monkeypatch.setenv("FAKE_DATA_SLEEP", str(sleep))
    monkeypatch.delenv("KERNELBLASTER_SSH_KEEP_DATA", raising=False)
    return state


def _data_evaluator(tmp_path, hosts, binary=FAKE_DATA_BINARY, **kw):
    from kernelblaster.knowledge.slots import SSHEvaluator

    class FakeBuild(SSHEvaluator):
        async def _build(self, problem, sources, work):
            _exe(work / "problem", binary)
            return None

    kw.setdefault("runtime", "process")
    return FakeBuild(list(hosts), arch="89", remote_dir={h: str(tmp_path / f"remote_{h}") for h in hosts},
                     ssh=str(tmp_path / "ssh"), scp=str(tmp_path / "scp"), rsync=str(tmp_path / "rsync"),
                     work_root=tmp_path / "work", **kw)


def _copies(tmp_path):
    f = tmp_path / "copies.log"
    return [ln.split() for ln in f.read_text().splitlines()] if f.exists() else []


def _runs(tmp_path):
    return [ln.split() for ln in (tmp_path / "runs.log").read_text().splitlines()]


needs_rsync = pytest.mark.skipif(shutil.which("rsync") is None, reason="needs rsync")


@needs_rsync
def test_ssh_data_one_transfer_shared_by_racing_runs(tmp_path, monkeypatch):
    """12 runs of two problems with the same data race for one host: one rsync, before any GPU is leased."""
    import hashlib
    _data_env(tmp_path, monkeypatch, {"u@h1": 0}, sleep=0.5)
    blob = os.urandom(300_000)
    sha = hashlib.sha1(blob).hexdigest()
    from kernelblaster import telemetry as tel
    ev = _data_evaluator(tmp_path, ["u@h1"], cuda_visible_devices="0,1")
    probs = [_data_problem(tmp_path, n, blob) for n in ("p1", "p2")]
    tel.set_enabled(True)
    tel.configure(tmp_path / "tel.jsonl")

    async def go():
        with tel.span("problem", collect=True) as sp:
            return await asyncio.gather(*[ev.evaluate(probs[i % 2], {}, f"c{i}") for i in range(12)]), sp
    try:
        res, sp = asyncio.run(go())
    finally:
        tel.close()
    evs = [json.loads(ln) for ln in (tmp_path / "tel.jsonl").read_text().splitlines()]
    stage = [e for e in evs if e.get("name") == "run.stage.data"]
    assert len(stage) == 1 and stage[0]["attrs"]["bytes"] == len(blob) and stage[0]["attrs"]["method"] == "rsync"
    assert stage[0]["dur"] >= 0.5 and tel.summary_of(sp)["counters"]["ssh_data_bytes"] == len(blob)
    assert all(r.ok for r in res), [r.error_text(200) for r in res]
    copies = _copies(tmp_path)
    assert [c[0] for c in copies] == ["rsync"]
    final = tmp_path / "remote_u@h1" / f"data_{sha[:12]}" / "data.bin"
    assert final.read_bytes() == blob and not list(final.parent.glob(".data.bin.part*"))
    runs = _runs(tmp_path)
    assert len(runs) == 12 and {r[4] for r in runs} == {sha} and {r[5] for r in runs} == {str(final)}
    assert min(float(r[1]) for r in runs) >= float(copies[0][2])           # no GPU held while it was in flight
    assert ev.data_transfer["copies"] == 1 and ev.data_transfer["bytes"] == len(blob)
    assert ev.host_stats()["data"]["copies"] == 1
    asyncio.run(ev.close())
    assert not list((tmp_path / "remote_u@h1").iterdir())                 # the copy it made is gone


@needs_rsync
def test_ssh_data_reused_across_runs_and_kept_on_request(tmp_path, monkeypatch):
    _data_env(tmp_path, monkeypatch, {"u@h1": 0})
    blob = os.urandom(50_000)
    prob = _data_problem(tmp_path, "p", blob)
    monkeypatch.setenv("KERNELBLASTER_SSH_KEEP_DATA", "1")
    ev = _data_evaluator(tmp_path, ["u@h1"], cuda_visible_devices="0")
    assert ev.keep_data
    assert asyncio.run(ev.evaluate(prob, {}, "a")).ok
    asyncio.run(ev.close())
    kept = list((tmp_path / "remote_u@h1").glob("data_*/data.bin"))
    assert len(kept) == 1 and kept[0].read_bytes() == blob                 # kept for the next run
    ev2 = _data_evaluator(tmp_path, ["u@h1"], cuda_visible_devices="0", keep_data=False)
    assert asyncio.run(ev2.evaluate(prob, {}, "b")).ok
    assert ev2.data_transfer == {**ev2.data_transfer, "copies": 0, "reused": 1} and len(_copies(tmp_path)) == 1
    asyncio.run(ev2.close())
    assert kept[0].exists()                                                # not its copy: left alone
    # a copy with the right size but other content is not reused
    kept[0].write_bytes(bytes(len(blob)))
    ev3 = _data_evaluator(tmp_path, ["u@h1"], cuda_visible_devices="0", keep_data=False)
    assert asyncio.run(ev3.evaluate(prob, {}, "c")).ok
    assert ev3.data_transfer["copies"] == 1 and kept[0].read_bytes() == blob
    asyncio.run(ev3.close())
    assert not list((tmp_path / "remote_u@h1").iterdir())


@needs_rsync
def test_ssh_data_copy_failure_is_an_infra_retry(tmp_path, monkeypatch):
    state = _data_env(tmp_path, monkeypatch, {"u@h1": 0, "u@h2": 0})
    blob = os.urandom(40_000)
    prob = _data_problem(tmp_path, "p", blob)
    (state / "datafail_u@h2").write_text("1000")                          # h2 never takes the data
    ev = _data_evaluator(tmp_path, ["u@h1", "u@h2"], cuda_visible_devices="0,1")

    async def go():
        return await asyncio.gather(*[ev.evaluate(prob, {}, f"c{i}") for i in range(6)])
    res = asyncio.run(go())
    assert all(r.ok for r in res)                                         # retried on h1, not the candidate's fault
    assert {r[2] for r in _runs(tmp_path)} == {"u@h1"}
    by = {d.label(): d for d in ev._pool.devices}
    assert by["u@h2:0"].infra_fail + by["u@h2:1"].infra_fail + by["u@h2:0"].quarantines >= 1
    assert ev.data_transfer["failures"] >= 1 and ev.data_transfer["copies"] == 1
    asyncio.run(ev.close())
    assert not list((tmp_path / "remote_u@h2").rglob("*.part*"))          # its temporary file is cleaned up

    # one host whose link drops twice (rsync, then the scp fallback): the retry resumes and succeeds
    (state / "datafail_u@h1").write_text("2")
    prob2 = _data_problem(tmp_path, "p2", os.urandom(40_000))
    ev2 = _data_evaluator(tmp_path, ["u@h1"], cuda_visible_devices="0")
    r = asyncio.run(ev2.evaluate(prob2, {}, "x"))
    assert r.ok and ev2.data_transfer["failures"] == 1 and ev2.data_transfer["copies"] == 1
    asyncio.run(ev2.close())


@needs_rsync
def test_ssh_data_prefers_hosts_that_have_it(tmp_path, monkeypatch):
    """Runs stay on the host holding the data while it has a GPU spare; a busy pool copies to a second host."""
    _data_env(tmp_path, monkeypatch, {"u@h1": 0, "u@h2": 0})
    prob = _data_problem(tmp_path, "p", os.urandom(30_000))
    ev = _data_evaluator(tmp_path, ["u@h1", "u@h2"], cuda_visible_devices="0,1")

    async def one_at_a_time():
        for i in range(5):
            assert (await ev.evaluate(prob, {}, f"s{i}")).ok
    asyncio.run(one_at_a_time())
    assert len(_copies(tmp_path)) == 1 and len({r[2] for r in _runs(tmp_path)}) == 1
    monkeypatch.setenv("FAKE_DATA_SLEEP", "0.2")

    async def burst():
        return await asyncio.gather(*[ev.evaluate(prob, {}, f"b{i}") for i in range(8)])
    assert all(r.ok for r in asyncio.run(burst()))
    assert len(_copies(tmp_path)) == 2 and {r[2] for r in _runs(tmp_path)} == {"u@h1", "u@h2"}
    asyncio.run(ev.close())
    assert not list((tmp_path / "remote_u@h1").iterdir()) and not list((tmp_path / "remote_u@h2").iterdir())


@needs_rsync
def test_ssh_data_removed_under_a_run_is_copied_again(tmp_path, monkeypatch):
    _data_env(tmp_path, monkeypatch, {"u@h1": 0})
    prob = _data_problem(tmp_path, "p", os.urandom(20_000))
    ev = _data_evaluator(tmp_path, ["u@h1"], cuda_visible_devices="0")
    assert asyncio.run(ev.evaluate(prob, {}, "a")).ok
    for f in (tmp_path / "remote_u@h1").glob("data_*/data.bin"):
        f.unlink()                                                         # e.g. another run's close()
    assert asyncio.run(ev.evaluate(prob, {}, "b")).ok
    assert ev.data_transfer["copies"] == 2
    asyncio.run(ev.close())


@needs_rsync
def test_ssh_data_persistent_host_env(tmp_path, monkeypatch):
    """The persistent host over ssh is started with MB_DATA_PATH at the remote copy."""
    import hashlib
    _data_env(tmp_path, monkeypatch, {"u@h1": 0})
    blob = os.urandom(60_000)
    prob = _data_problem(tmp_path, "p", blob, host_marker=True)
    host = tmp_path / "fake_host.py"
    _exe(host, FAKE_DATA_HOST)
    ev = _data_evaluator(tmp_path, ["u@h1"], cuda_visible_devices="0", runtime="host")

    async def host_exe(problem):
        return host

    async def build_so(problem, sources, work):
        work.mkdir(parents=True, exist_ok=True)
        so = work / f"{hashlib.sha1(repr(sorted(sources.items())).encode()).hexdigest()}.so"
        so.write_bytes(b"so")
        return so, None
    ev._host_exe, ev._build_so = host_exe, build_so

    async def go():
        return await asyncio.gather(*[ev.evaluate(prob, {"a": str(i)}, f"c{i}") for i in range(4)])
    res = asyncio.run(go())
    assert all(r.ok and r.runtime == "host" for r in res), [r.error_text(200) for r in res]
    sha = hashlib.sha1(blob).hexdigest()
    lines = _runs(tmp_path)
    assert len(lines) == 4 and {ln[2] for ln in lines} == {sha}
    assert {ln[3] for ln in lines} == {str(tmp_path / "remote_u@h1" / f"data_{sha[:12]}" / "data.bin")}
    assert len(_copies(tmp_path)) == 1
    asyncio.run(ev.close())
    assert not list((tmp_path / "remote_u@h1").iterdir())


def _lenet_runtime_data(tmp_path):
    """The LeNet fixture as kb_export writes a > 1 GiB problem: data.bin mmapped from
    $MB_DATA_PATH, the baked-in fallback path does not exist."""
    import re
    from kernelblaster.knowledge.slots import SlotProblem
    d = tmp_path / "lenet_bigdata"
    shutil.copytree(FIX, d)
    drv = d / "driver.cpp"
    mmap = ('#include <fcntl.h>\n#include <sys/mman.h>\n#include <sys/stat.h>\n#include <unistd.h>\n'
            'static const unsigned char *mb_blob = [] {\n'
            '    const char *path = getenv("MB_DATA_PATH") ? getenv("MB_DATA_PATH") : "/nonexistent/kb_test/data.bin";\n'
            '    int fd = open(path, O_RDONLY); struct stat st;\n'
            '    if (fd < 0 || fstat(fd, &st) != 0) { printf("MB_ERROR cannot open %s\\n", path); printf("failed\\n"); exit(1); }\n'
            '    void *p = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);\n'
            '    close(fd);\n'
            '    if (p == MAP_FAILED) { printf("MB_ERROR cannot map %s\\n", path); printf("failed\\n"); exit(1); }\n'
            '    return (const unsigned char *)p;\n}();')
    txt, n = re.subn(r'__asm__\("\.section \.rodata.*\n.*mb_blob\[\];', lambda m: mmap, drv.read_text())
    assert n == 1
    drv.write_text(txt)
    return SlotProblem(d)


@needs_gpu
@needs_rsync
@pytest.mark.parametrize("runtime", ["process", "host"])
def test_ssh_runtime_data_driver_runs(tmp_path, monkeypatch, runtime):
    """A real run-time-data driver (nvcc, this GPU) over the evaluator through a local fake ssh."""
    from kernelblaster.knowledge.slots import LocalEvaluator, SSHEvaluator
    _data_env(tmp_path, monkeypatch, {})
    (tmp_path / "remote_u@h1").mkdir()
    monkeypatch.setenv("KERNELBLASTER_ALLOW_SHARED_GPU", "1")
    p = _lenet_runtime_data(tmp_path)
    seeds = p.seed_sources()
    ev = SSHEvaluator(["u@h1"], arch=ARCH, remote_dir=str(tmp_path / "remote_u@h1"), ssh=str(tmp_path / "ssh"),
                      scp=str(tmp_path / "scp"), rsync=str(tmp_path / "rsync"), work_root=tmp_path / "w",
                      runtime=runtime, cuda_visible_devices="all")
    local = LocalEvaluator(ARCH, work_root=tmp_path / "l", runtime="process")

    async def go():
        r = [await ev.evaluate(p, seeds, "seed"),
             await ev.evaluate(p, {**seeds, "relu": seeds["relu"] + "\n// cand\n"}, "cand")]
        bad = await local.evaluate(p, seeds, "local")                     # no MB_DATA_PATH: the fallback path
        await ev.close()
        return r, bad
    (seed, cand), bad = asyncio.run(go())
    assert seed.ok and cand.ok, (seed.error_text(300), cand.error_text(300))
    assert seed.runtime == cand.runtime == runtime and seed.partition_us and set(seed.slot_us) == set(p.order)
    assert not bad.ok and "cannot open /nonexistent/kb_test/data.bin" in bad.error_text(2000)
    assert ev.data_transfer["copies"] == 1 and len(_copies(tmp_path)) == 1
    assert not list((tmp_path / "remote_u@h1").iterdir())


@needs_gpu
@needs_rsync
def test_ssh_two_processes_share_a_remote_dir_with_live_hosts(tmp_path, monkeypatch):
    """The re-measurement failure, on this GPU through the local fake ssh: a second
    evaluator (ModelBlaster's re-measurement) times the problem a first one (the flow's
    KernelBlaster run) has a persistent host running for, in the same remote dir. Both pass,
    and the second one's close() leaves the first one's host and files alone."""
    from kernelblaster.knowledge.slots import SSHEvaluator
    _data_env(tmp_path, monkeypatch, {})
    rdir = tmp_path / "remote_u@h1"
    rdir.mkdir()
    monkeypatch.setenv("KERNELBLASTER_ALLOW_SHARED_GPU", "1")
    p = _lenet_runtime_data(tmp_path)
    seeds = p.seed_sources()

    def ev(name):
        return SSHEvaluator(["u@h1"], arch=ARCH, remote_dir=str(rdir), ssh=str(tmp_path / "ssh"),
                            scp=str(tmp_path / "scp"), rsync=str(tmp_path / "rsync"), work_root=tmp_path / name,
                            runtime="host", cuda_visible_devices="all")
    a, b = ev("a"), ev("b")

    async def go():
        ra = await a.evaluate(p, seeds, "kb_run")                # a's host stays up until a.close()
        rb = await b.evaluate(p, seeds, "remeasure")
        await b.close()
        ra2 = await a.evaluate(p, {**seeds, "relu": seeds["relu"] + "\n// cand\n"}, "kb_run2")
        await a.close()
        return ra, rb, ra2
    ra, rb, ra2 = asyncio.run(go())
    for r in (ra, rb, ra2):
        assert r.ok and r.runtime == "host", r.error_text(500)
    assert not list(rdir.iterdir())


def test_staged_missing_names_files_removed_under_us():
    from kernelblaster.knowledge.slots import SSHEvaluator
    names = {"kbhost_1d16bff26c55e4fd", "kbso_b378842e59fe1919.so", "kbso_aaaa.so"}
    so = ("MB_ERROR cannot load incumbent ./kbso_b378842e59fe1919.so: ./kbso_b378842e59fe1919.so: "
          "cannot open shared object file: No such file or directory\nfailed\n")
    assert SSHEvaluator._staged_missing(so, names) == {"kbso_b378842e59fe1919.so"}
    host = "KB_HOST_PID 1488285\nenv: \u2018./kbhost_1d16bff26c55e4fd\u2019: No such file or directory\n"
    assert SSHEvaluator._staged_missing(host, names) == {"kbhost_1d16bff26c55e4fd"}
    assert SSHEvaluator._staged_missing("MB_ERROR cannot load ./kbso_aaaa.so: undefined symbol: foo\n", names) == set()
    assert SSHEvaluator._staged_missing(None, names) == set()


@needs_gpu
@needs_rsync
def test_ssh_staged_files_removed_under_us_are_restaged(tmp_path, monkeypatch):
    """A host binary / shared object this evaluator staged and that went missing on the
    host (removed by another process): an infrastructure failure, staged again and
    retried -- never the candidate's runtime error (the FP16 run's
    ``cannot load incumbent ./kbso_<sha>.so: No such file``, which failed a whole
    KernelBlaster run)."""
    from kernelblaster.knowledge.slots import SSHEvaluator
    _data_env(tmp_path, monkeypatch, {})
    rdir = tmp_path / "remote_u@h1"
    rdir.mkdir()
    monkeypatch.setenv("KERNELBLASTER_ALLOW_SHARED_GPU", "1")
    p = _lenet_runtime_data(tmp_path)
    seeds = p.seed_sources()
    ev = SSHEvaluator(["u@h1"], arch=ARCH, remote_dir=str(rdir), ssh=str(tmp_path / "ssh"),
                      scp=str(tmp_path / "scp"), rsync=str(tmp_path / "rsync"), work_root=tmp_path / "w",
                      runtime="host", cuda_visible_devices="all")
    seen = []
    orig = SSHEvaluator._staged_missing

    def spy(text, names):
        seen.append(orig(text, names))
        return seen[-1]
    monkeypatch.setattr(SSHEvaluator, "_staged_missing", staticmethod(spy))

    async def go():
        r1 = await ev.evaluate(p, seeds, "seed")
        await ev._hosts.close()                                   # the next request starts a host
        gone = [f for f in Path(ev._stage_dir("u@h1")).iterdir() if f.name.startswith(("kbso_", "kbhost_"))]
        assert gone
        for f in gone:                                            # removed under us
            f.unlink()
        r2 = await ev.evaluate(p, seeds, "again")                 # the same (still "staged") names
        await ev.close()
        return r1, r2
    r1, r2 = asyncio.run(go())
    assert r1.ok and r2.ok, (r1.error_text(500), r2.error_text(500))
    assert any(seen), seen                                         # detected, restaged, retried
    assert not list(rdir.iterdir())


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



@needs_gpu
def test_local_evaluator_concurrent_on_pool(tmp_path, monkeypatch):
    from kernelblaster.knowledge.slots import LocalEvaluator, SlotProblem

    monkeypatch.setenv("KERNELBLASTER_ALLOW_SHARED_GPU", "1")     # shared dev box: use the GPU anyway
    p = SlotProblem(FIX)
    seeds = p.seed_sources()
    ev = LocalEvaluator(ARCH, work_root=tmp_path / "w", build_jobs=4)
    seen = []
    orig = ev._run

    async def spy(exe, env=None, prefix="", device=None, **kw):
        seen.append(device.env().get("CUDA_VISIBLE_DEVICES"))
        return await orig(exe, env, prefix, device=device, **kw)
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


def test_gpu_lock_serialises_processes(tmp_path, monkeypatch):
    if shutil.which("flock") is None:
        pytest.skip("no flock")
    monkeypatch.setenv("KERNELBLASTER_GPU_LOCK_DIR", str(tmp_path))

    def wall(idents):
        t0 = time.time()
        ps = [subprocess.Popen(["bash", "-c", R.gpu_lock_wrap("sleep 0.4", i)]) for i in idents]
        for p in ps:
            assert p.wait() == 0
        return time.time() - t0
    assert wall(["GPU-x", "GPU-x"]) >= 0.8              # same GPU: one after the other
    assert wall(["GPU-x", "GPU-y"]) < 0.75             # different GPUs: together
    assert (tmp_path / "kernelblaster_gpu_GPU-x.lock").exists()
    monkeypatch.setenv("KERNELBLASTER_GPU_FLOCK", "0")
    assert R.gpu_lock_wrap("true", "GPU-x") == "true"
    assert R.gpu_lock_wrap("true", None) == "true"
    # an unopenable lock file: run unlocked rather than fail
    monkeypatch.delenv("KERNELBLASTER_GPU_FLOCK")
    monkeypatch.setenv("KERNELBLASTER_GPU_LOCK_DIR", str(tmp_path / "missing_dir"))
    assert subprocess.run(["bash", "-c", R.gpu_lock_wrap("echo ran", "GPU-z")], capture_output=True,
                          text=True).stdout.strip() == "ran"


def test_pool_and_limiter_survive_several_event_loops():
    pool = GPUPool([Device(None, "0", "g", 0)])
    lim = AdaptiveLimiter(fixed=1)

    async def go():
        async def one():
            async with lim:
                d = await pool.acquire()
                await asyncio.sleep(0.01)
                await pool.release(d)
        await asyncio.gather(*[one() for _ in range(3)])     # contended: primitives bind to this loop
    asyncio.run(go())
    asyncio.run(go())                                         # a new loop: fresh primitives, no error
    assert pool.devices[0].runs == 6


def test_pick_host_prefers_hosts_with_spare_devices():
    pool = GPUPool(_devs(2, hosts=("a", "b")))
    assert pool.pick_host("p", prefer_hosts={"b"}) == "b"                 # the data is there
    assert pool.pick_host("p", prefer_hosts={"b"}) == "b"
    assert [pool.pick_host("p", prefer_hosts={"b"}) for _ in range(2)] == ["a", "a"]   # b is full: spill over
    assert pool.pick_host("p", prefer_hosts={"b"}) == "b"                 # nothing spare: prefer it again
    assert pool.pick_host("p") == "a"                                     # no preference: the old order


def test_pick_host_spreads_staged_runs():
    pool = GPUPool(_devs(2, hosts=("a", "b")) + [Device("c", "GPU-c0", "L40S", 0)])
    picks = [pool.pick_host("p") for _ in range(5)]          # nothing leased yet: reservations spread them
    assert sorted(picks) == ["a", "a", "b", "b", "c"]
    for h in picks:
        pool.unreserve(h)
    pool.host_failure("a")
    assert pool.devices[0].infra_fail == pool.devices[1].infra_fail == 1
    assert pool.pick_host("p", avoid_hosts={"a", "b"}) == "c"
