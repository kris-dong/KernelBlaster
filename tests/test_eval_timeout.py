"""Configurable / adaptive evaluation timeouts (knowledge/slots.py, runtime_host.py,
servers/utils/resources.py, scripts/run_multislot.py).

Everything runs without a GPU: the fake host of test_runtime_host.py, stub executables that sleep,
and real flock files in a temp dir.
"""
import asyncio
import fcntl
import importlib.util
import os
import stat
import sys
import time
from pathlib import Path

import pytest

from kernelblaster import telemetry as tel
from kernelblaster.knowledge.runtime_host import HostClient
from kernelblaster.knowledge.slots import LocalEvaluator, SlotProblem
from kernelblaster.servers.utils.resources import (Device, GpuLockTimeout, acquire_gpu_lock, gpu_lock_file,
                                                    gpu_lock_wrap)

sys.path.insert(0, str(Path(__file__).parent))
from test_runtime_host import FIX, _FakeHostEvaluator, _fake_host  # noqa: E402


def _free(path) -> bool:
    """The flock on ``path`` can be taken right now."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


@pytest.fixture
def lockdir(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELBLASTER_GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.delenv("KERNELBLASTER_GPU_FLOCK", raising=False)
    return tmp_path


# ---- the limit

class _P:
    root = "/p/one"


def test_fixed_mode_is_unchanged():
    ev = LocalEvaluator("75", timeout_s=900)
    assert ev.eval_timeout_mode == "fixed"
    assert ev.run_timeout(_P) == 900
    ev.note_incumbent_wall(_P, 2.0)                      # recorded, but not used
    assert ev.run_timeout(_P) == 900
    # nvcc and the lock wait default to the one timeout that used to cover them
    assert ev.build_timeout_s == 900 and ev.gpu_lock_wait_s == 900
    ev = LocalEvaluator("75", timeout_s=120, build_timeout_s=900, gpu_lock_wait_s=30)
    assert (ev.run_timeout(_P), ev.build_timeout_s, ev.gpu_lock_wait_s) == (120, 900, 30)
    with pytest.raises(ValueError):
        LocalEvaluator("75", eval_timeout_mode="sometimes")


def test_adaptive_limit_clamps_and_fallback():
    ev = LocalEvaluator("75", timeout_s=900, eval_timeout_mode="adaptive", eval_timeout_min_s=60,
                        eval_timeout_factor=20)
    assert ev.run_timeout(_P) == 300                      # nothing measured yet: 5 x min
    ev.note_incumbent_wall(_P, 2.0)
    assert ev.run_timeout(_P) == 60                       # 40 s -> raised to the minimum
    ev.note_incumbent_wall(_P, 10.0)
    assert ev.run_timeout(_P) == 200                      # factor x wall
    ev.note_incumbent_wall(_P, 100.0)
    assert ev.run_timeout(_P) == 900                      # 2000 s -> the cap
    ev.note_incumbent_wall(_P, None)                      # no measurement: the last one stays
    ev.note_incumbent_wall(_P, 0)
    assert ev.run_timeout(_P) == 900

    class Q:
        root = "/p/two"
    assert ev.run_timeout(Q) == 300                       # per problem
    assert ev.run_timeout(_P, profiling=True) == 900      # ncu: always the cap
    small = LocalEvaluator("75", timeout_s=30, eval_timeout_mode="adaptive", eval_timeout_min_s=60)
    assert small.run_timeout(_P) == 30                    # the cap wins over the minimum


# ---- a timed-out candidate

def test_adaptive_timeout_result_and_next_request_not_blocked(tmp_path):
    p = SlotProblem(FIX)
    ev = _FakeHostEvaluator(_fake_host(tmp_path), work_root=tmp_path / "w", timeout_s=30,
                            eval_timeout_mode="adaptive", eval_timeout_min_s=1, eval_timeout_factor=20)
    base = p.seed_sources()

    async def go():
        a = await ev.evaluate(p, {**base, "relu": "ok"}, "baseline")
        assert a.ok and a.eval_wall_s is not None
        ev.note_incumbent_wall(p, 0.01)                                      # a fast incumbent: limit = the minimum
        assert ev.run_timeout(p) == 1
        t0 = time.monotonic()
        hung = await ev.evaluate(p, {**base, "relu": "hang"}, "r0_hang")
        took = time.monotonic() - t0
        t1 = time.monotonic()
        nxt = await ev.evaluate(p, {**base, "relu": "ok"}, "r0_next")        # the runtime restarts, no stall
        after = time.monotonic() - t1
        await ev.close()
        return hung, took, nxt, after
    tel.set_enabled(True)
    try:
        with tel.span("step", collect=True) as sp:
            hung, took, nxt, after = asyncio.run(go())
    finally:
        tel.close()
    assert not hung.ok and hung.stage == "run"
    assert hung.errors == ["timeout after 1s (kernel presumed hung; adaptive limit)"]
    assert took < 10 and nxt.ok and after < 10
    assert ev.host_stats()["timeouts"] == 1
    assert tel.summary_of(sp)["counters"]["eval.timeout.adaptive"] == 1


def test_fixed_timeout_text_unchanged(tmp_path):
    p = SlotProblem(FIX)
    ev = _FakeHostEvaluator(_fake_host(tmp_path), work_root=tmp_path / "w", timeout_s=1)
    r = asyncio.run(ev.evaluate(p, {**p.seed_sources(), "relu": "hang"}, "t"))
    assert not r.ok and r.stage == "run" and r.errors[-1] == "timeout after 1s (the runtime was restarted)"


def test_timeout_is_a_runtime_error_outcome():
    from kernelblaster.knowledge.outcomes import RUNTIME_ERROR
    # opt_multislot maps a failed EvalResult by stage; "run" is the runtime-error status
    src = (Path(__file__).parent.parent / "src/kernelblaster/agents/opt_multislot.py").read_text()
    assert '"run": RUNTIME_ERROR' in src


# ---- the lock

def test_host_timeout_releases_the_gpu_lock_and_lock_wait_is_not_charged(tmp_path, lockdir):
    exe = _fake_host(tmp_path)
    lock = gpu_lock_file("GPU-test")

    async def go():
        h = HostClient([str(exe)], label="fake", start_timeout=30, client_lock=True)
        # a hang under the lock: killed at the timeout, the lock is free at once
        r = await h.request("hang.so", lock=lock, lock_wait=5, timeout=1.0)
        assert r.status == "timeout" and not h.alive and _free(lock)
        # another process holds the lock for 2 s: the wait is not part of the 1 s run timeout
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT)
        fcntl.flock(fd, fcntl.LOCK_EX)
        asyncio.get_running_loop().call_later(2.0, os.close, fd)
        t0 = time.monotonic()
        r = await h.request("ok.so", lock=lock, lock_wait=10, timeout=1.0)
        assert r.status == "done" and time.monotonic() - t0 >= 1.8 and _free(lock)
        # a lock that stays taken past lock_wait: reported, the host survives, nothing is held
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT)
        fcntl.flock(fd, fcntl.LOCK_EX)
        r = await h.request("ok.so", lock=lock, lock_wait=0.5, timeout=1.0)
        os.close(fd)
        assert r.status == "lock_timeout" and h.alive and _free(lock)
        await h.close()
    asyncio.run(go())


def test_host_lock_marker_restarts_the_clock(tmp_path):
    """A host that says MB_HOST_LOCKED (it took the lock itself) is timed from then on."""
    script = tmp_path / "h.py"
    script.write_text(f'''#!{sys.executable}
import sys, time
print("MB_HOST_READY", flush=True)
for line in sys.stdin:
    f = line.split("\\t")
    time.sleep(1.5)
    print("MB_HOST_LOCKED", flush=True)
    time.sleep(0.5)
    print("MB_HOST_DONE " + f[1] + " rc=0 poisoned=0 recycle=0", flush=True)
''')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)

    async def go():
        h = HostClient([str(script)], start_timeout=30)
        r = await h.request("a.so", lock="/nonexistent-lock", lock_wait=10, timeout=1.0)   # 1.5 s wait + 0.5 s run
        await h.close()
        return r
    assert asyncio.run(go()).status == "done"


def test_process_run_timeout_kills_the_group_and_frees_the_lock(tmp_path, lockdir):
    exe = tmp_path / "stub.sh"
    pidf = tmp_path / "pid"
    exe.write_text(f"#!/bin/sh\nsleep 30 &\necho $! > {pidf}\nwait\n")
    exe.chmod(0o755)
    dev = Device(None, "GPU-test")
    lock = gpu_lock_file(dev.ident)
    ev = LocalEvaluator("75", timeout_s=1, gpu_lock_wait_s=5)

    async def go():
        t0 = time.monotonic()
        text = await ev._run(exe, device=dev, problem=_P)
        return text, time.monotonic() - t0
    text, took = asyncio.run(go())
    assert text is None and took < 8 and _free(lock)
    time.sleep(0.3)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidf.read_text()), 0)                 # the child sleep did not outlive the timeout


def test_process_run_lock_wait_not_charged(tmp_path, lockdir):
    exe = tmp_path / "ok.sh"
    exe.write_text("#!/bin/sh\nsleep 0.3\necho passed\n")
    exe.chmod(0o755)
    dev = Device(None, "GPU-test")
    fd = os.open(gpu_lock_file(dev.ident), os.O_WRONLY | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    ev = LocalEvaluator("75", timeout_s=1, gpu_lock_wait_s=10)

    async def go():
        asyncio.get_running_loop().call_later(2.0, os.close, fd)
        return await ev._run(exe, device=dev, problem=_P)
    assert "passed" in asyncio.run(go())                   # waited 2 s under a 1 s run timeout
    ev2 = LocalEvaluator("75", timeout_s=1, gpu_lock_wait_s=0.4)
    fd = os.open(gpu_lock_file(dev.ident), os.O_WRONLY | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    out = asyncio.run(ev2._run(exe, device=dev, problem=_P))
    os.close(fd)
    assert out.startswith("MB_ERROR [lock]")


def test_acquire_gpu_lock(lockdir):
    async def go():
        a = await acquire_gpu_lock("GPU-x", 1)
        with pytest.raises(GpuLockTimeout):
            await acquire_gpu_lock("GPU-x", 0.3)
        a.release()
        a.release()
        b = await acquire_gpu_lock("GPU-x", 1)
        b.release()
        return await acquire_gpu_lock(None, 1)             # nothing to lock
    assert asyncio.run(go()).fd is None


def test_gpu_lock_wrap_bounds_the_run_remotely():
    w = gpu_lock_wrap("./x", "GPU-1", wait_s=7, run_timeout_s=42)
    assert "flock -w 7" in w and "timeout -k 5 42" in w
    assert "timeout" not in gpu_lock_wrap("./x", "GPU-1", wait_s=7)


# ---- the command line

def _parser():
    spec = importlib.util.spec_from_file_location("run_multislot_cli",
                                                  Path(__file__).parent.parent / "scripts" / "run_multislot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_flags():
    mod = _parser()
    ap = mod.build_parser()
    a = ap.parse_args(["p", "--out", "o"])
    assert (a.eval_timeout_s, a.eval_timeout_mode, a.eval_timeout_min_s, a.eval_timeout_factor, a.build_timeout_s,
            a.gpu_lock_wait_s) == (900, "fixed", 60.0, 20.0, 900, 900)
    a = ap.parse_args(["p", "--out", "o", "--eval-timeout-s", "300", "--eval-timeout-mode", "adaptive",
                       "--eval-timeout-min-s", "30", "--eval-timeout-factor", "10", "--build-timeout-s", "1200",
                       "--gpu-lock-wait-s", "60"])
    kw = mod.eval_timeout_kwargs(a)
    assert kw == dict(timeout_s=300, eval_timeout_mode="adaptive", eval_timeout_min_s=30.0,
                      eval_timeout_factor=10.0, build_timeout_s=1200, gpu_lock_wait_s=60)
    ev = LocalEvaluator("75", **kw)
    assert ev.run_timeout(_P) == 150 and ev.build_timeout_s == 1200      # 5 x min, before an incumbent
    with pytest.raises(SystemExit):
        ap.parse_args(["p", "--out", "o", "--eval-timeout-mode", "bogus"])
