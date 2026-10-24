"""GPU-model pinning, the device each result records, and scripts/time_problems.py (the timed
runs ModelBlaster's model result and hint re-measurement send through these evaluators).

The pool / recording tests run everywhere (fake devices, a stubbed build and run); the script
tests build the LeNet fixture and need nvcc + a CUDA GPU.
"""
import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kernelblaster.knowledge.slots import LocalEvaluator, SlotProblem
from kernelblaster.servers.utils.resources import (Device, NoDevice, gpu_model_key, restrict_devices,
                                                    same_gpu_model)

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "mb_lenet"


def _gpu():
    if shutil.which("nvcc") is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.splitlines()
        if not out:
            return None
        name, cc = (x.strip() for x in out[0].split(","))
        return name, cc.replace(".", "")
    except Exception:
        return None


GPU = _gpu()
needs_gpu = pytest.mark.skipif(GPU is None, reason="needs nvcc and a CUDA GPU")


def _devs():
    return [Device(None, "GPU-t0", "NVIDIA TITAN RTX", 0), Device("u@h", "GPU-l2", "NVIDIA L40S", 2),
            Device("u@h", "GPU-l3", "NVIDIA L40S", 3)]


def test_model_names_and_restriction():
    assert gpu_model_key("NVIDIA  L40S") == "l40s" == gpu_model_key("l40s")
    assert gpu_model_key("default") is None and gpu_model_key("gpu") is None and gpu_model_key(None) is None
    assert same_gpu_model("NVIDIA L40S", "L40S") and not same_gpu_model("default", "default")
    assert [d.ident for d in restrict_devices(_devs(), gpu_model="NVIDIA L40S")] == ["GPU-l2", "GPU-l3"]
    assert [d.ident for d in restrict_devices(_devs(), gpu_model="l40s", max_devices=1)] == ["GPU-l2"]
    assert len(restrict_devices(_devs())) == 3
    with pytest.raises(NoDevice, match="H100"):
        restrict_devices(_devs(), gpu_model="NVIDIA H100")
    with pytest.raises(NoDevice):                     # an unknown model never stands in for the required one
        restrict_devices([Device(None, None, "default")], gpu_model="NVIDIA L40S")


class _Stub(LocalEvaluator):
    """No nvcc, no GPU: the build succeeds, the run prints a passing driver output."""

    def __init__(self, devs, **kw):
        super().__init__("89", runtime="process", **kw)
        self._fake = devs

    async def _devices(self):
        return list(self._fake)

    async def _build(self, problem, sources, work):
        return None

    async def _run(self, exe, env=None, prefix="", device=None, problem=None):
        return "MB_PARTITION_US 10.0 reps=5\nMB_E2E_US 9.0 basis=graph\npassed\n"


def test_pool_pins_the_model_and_results_record_the_device(tmp_path):
    ev = _Stub(_devs(), gpu_model="NVIDIA L40S", max_devices=1, work_root=tmp_path)
    r = asyncio.run(ev.evaluate(SlotProblem(FIX), SlotProblem(FIX).seed_sources(), "t"))
    assert r.ok and r.e2e_us == 9.0
    assert r.gpu == {"host": "u@h", "label": "u@h:2", "index": 2, "uuid": "GPU-l2", "name": "NVIDIA L40S",
                     "arch": "sm_89"}
    assert [d.ident for d in asyncio.run(ev.pool()).devices] == ["GPU-l2"]
    with pytest.raises(NoDevice):
        asyncio.run(_Stub(_devs(), gpu_model="NVIDIA H100", work_root=tmp_path).evaluate(
            SlotProblem(FIX), SlotProblem(FIX).seed_sources(), "t"))


def _script(tmp_path, jobs, *args):
    jf, out = tmp_path / "jobs.json", tmp_path / "out.json"
    jf.write_text(json.dumps(jobs))
    cp = subprocess.run([sys.executable, str(ROOT / "scripts" / "time_problems.py"), str(jf), "--out", str(out),
                         "--arch", GPU[1], "--work", str(tmp_path / "work"), *args],
                        capture_output=True, text=True, timeout=1200, cwd=str(ROOT))
    return cp, json.loads(out.read_text())


def _jobs(order="rounds"):
    p = SlotProblem(FIX)
    seeds = p.seed_sources()
    cand = {**seeds, p.order[0]: seeds[p.order[0]] + "\n// candidate\n"}
    return {"order": order, "jobs": [{"id": "paired", "problem": str(FIX), "sources": cand, "incumbent": seeds,
                                      "reps": 2},
                                     {"id": "plain", "problem": str(FIX), "sources": None, "reps": 1}]}


@needs_gpu
def test_time_problems_paired_on_the_host(tmp_path):
    cp, res = _script(tmp_path, _jobs(), "--gpu-model", GPU[0], "--telemetry", str(tmp_path / "tel.jsonl"))
    assert cp.returncode == 0, cp.stdout[-2000:] + cp.stderr[-2000:]
    assert res["format"] == "kernelblaster.time_problems/v1" and res["error"] is None
    assert len(res["pool"]) == 1 and same_gpu_model(res["models"][0], GPU[0])
    j = res["jobs"]["paired"]
    assert j["pairing"] == "host" and len(j["runs"]) == 2
    assert all(r["ok"] and r["runtime"] == "host" and r["ref_e2e_us"] and r["e2e_ratio"] for r in j["runs"])
    assert same_gpu_model(j["gpu"]["name"], GPU[0]) and j["gpu"]["uuid"].startswith("GPU-")
    assert j["gpu"]["arch"] == f"sm_{GPU[1]}"
    assert res["jobs"]["plain"]["paired"] is False and res["jobs"]["plain"]["runs"][0]["ref_e2e_us"] is None
    assert res["telemetry"]["evals"] and (tmp_path / "tel.jsonl").exists()


@needs_gpu
def test_time_problems_one_process_per_run_interleaves_the_incumbent(tmp_path):
    cp, res = _script(tmp_path, _jobs("jobs"), "--runtime", "process")
    assert cp.returncode == 0, cp.stdout[-2000:] + cp.stderr[-2000:]
    j = res["jobs"]["paired"]
    assert j["pairing"] == "interleaved"
    assert len(j["runs"]) == len(j["ref_runs"]) == 2
    assert all(r["ok"] and r["runtime"] == "process" and r["e2e_us"] for r in j["runs"] + j["ref_runs"])


@needs_gpu
def test_time_problems_refuses_another_gpu_model(tmp_path):
    cp, res = _script(tmp_path, _jobs(), "--gpu-model", "NVIDIA H100 80GB HBM3")
    assert cp.returncode == 1
    assert res["error"].startswith("NoDevice") and "H100" in res["error"]
    assert res["jobs"] == {}
