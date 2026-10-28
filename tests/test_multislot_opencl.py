"""Multi-slot agent on OpenCL (Qualcomm Adreno) problems.

Pure-python units run everywhere. The device tests build with g++ and run on
this host's OpenCL GPU (any OpenCL 2.0 device with coarse-grained SVM; the
Adreno board's role) and are skipped without one. The board evaluator test
runs only when KB_OPENCL_BOARD_HOST names an ssh host with g++ and an OpenCL
GPU (``KB_OPENCL_BOARD_DIR``: its scratch directory).
"""
import asyncio
import functools
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from kernelblaster.agents.opt_multislot import (SYSTEM_PROMPT, MultiSlotAgent, build_codegen_messages,
                                                build_fix_messages, extract_code, slot_state, target_profile,
                                                techniques_for)
from kernelblaster.agents.utils import agentic_llm
from kernelblaster.knowledge import opencl_profile
from kernelblaster.knowledge.impl_store import ImplStore
from kernelblaster.knowledge.outcomes import OutcomeLog
from kernelblaster.knowledge.slots import (AdrenoEvaluator, OpenCLLocalEvaluator, SlotProblem, opencl_main,
                                           opencl_source_errors, parse_driver_output)

FIX = Path(__file__).parent / "fixtures" / "mb_lenet_opencl"
ARCH = "titanrtx"          # store key of the local stand-in device

FAST_RELU = """```cpp
static mb_program cl_relu4 = MB_CL(R"CL(
__kernel void k_relu4(__global const float *x, __global float *y, int n) {
    long i = get_global_id(0), j = 4 * i;
    if (j + 3 < n) vstore4(fmax(vload4(i, x), (float4)(0.0f)), i, y);
    else for (long k = j; k < n; k++) y[k] = fmax(x[k], 0.0f);
}
)CL");
void kernel_relu(const float *input, float *output, int n) {
    mb_launch(MB_KERNEL(cl_relu4, k_relu4), mb_grid((n + 3) / 4), input, output, n);
}
```"""

DRIVER_OUT = """MB_DEVICE QUALCOMM Adreno(TM) 650 | OpenCL 2.0 Adreno(TM) 650
MB_DEVICE_INFO max_wg=1024 compute_units=3 local_mem=32768 global_mem=8000000000 fp16=1
MB_CHECK output fc3 max_abs_err=1e-08 bound=0.0001 ok=1
MB_SLOT_CHECK linear rel_err=1e-07 ok=1
MB_SLOT_CHECK relu rel_err=0 ok=1
MB_PARTITION_US 120.5 reps=10
MB_SLOT_US linear 80.0
MB_SLOT_US relu 40.5
MB_OP_US fc1 80.0
MB_OP_US relu 40.5
MB_KERNEL_US linear k_linear 79.0 n=3
MB_KERNEL_US relu k_relu 12.0 n=4
MB_KINFO linear k_linear wg_max=512 pref_mult=64 local_mem=0 private_mem=64
MB_KINFO relu k_relu wg_max=1024 pref_mult=64 local_mem=0 private_mem=0
passed
"""


# ---------------------------------------------------------------- units

def test_target_profiles():
    cu, cl = target_profile("cuda"), target_profile("opencl")
    assert cu.system_prompt is SYSTEM_PROMPT and cu.arch_label("75") == "sm_75" and cu.fence == "cuda"
    assert cl.arch_label("adreno650") == "adreno650" and cl.fence == "cpp" and cl.slot_ext == ".cpp"
    assert "mb_launch" in cl.system_prompt and "no double" in cl.system_prompt
    names = [t for t, _ in techniques_for("gemm", cl.techniques)]
    assert "local_memory_tiling" in names and "vectorized_memory_access" in names
    assert "warp_shuffle_reduction" not in [t for t, _ in techniques_for("reduction", cl.techniques)]
    assert cl.peaks_for("adreno650") == (1228.0, 44.0)
    with pytest.raises(ValueError):
        target_profile("vulkan")


def test_prompts_follow_the_problem_target():
    p = SlotProblem(FIX)
    assert p.target == "opencl"
    src = p.seed_sources()["relu"]
    msgs = build_codegen_messages(problem=p, sid="relu", source=src, technique="t", technique_desc="d",
                                  state="latency_bound", state_numbers={}, slot_us=10.0, partition_us=100.0,
                                  neighbours=[], examples=[], adapt=None, knowledge_text="", already_tried="",
                                  best_note="")
    assert msgs[0]["content"] == target_profile("opencl").system_prompt
    assert "```cpp\n" + src in msgs[1]["content"] and "```cuda" not in msgs[1]["content"]
    fix = build_fix_messages(problem=p, sid="relu", source=src, error="boom")
    assert "```cpp" in fix[1]["content"]
    assert extract_code(FAST_RELU).startswith("static mb_program cl_relu4")


def test_slot_state_uses_target_peaks():
    p = SlotProblem(FIX)
    st_adreno, n1 = slot_state(p, "linear", 50.0, "adreno650", target_profile("opencl").peaks_for("adreno650"))
    st_cuda, n2 = slot_state(p, "linear", 50.0, "75")
    assert n1["pct_peak_bw"] > n2["pct_peak_bw"]          # same traffic is a bigger share of 44 GB/s


def test_direct_enqueue_is_rejected():
    p = SlotProblem(FIX)
    src = dict(p.seed_sources())
    src["relu"] = src["relu"].replace("mb_launch(", "clEnqueueNDRangeKernel(")
    bad = opencl_source_errors(p, src)
    assert bad is not None and bad.stage == "compile" and "relu" in bad.errors[0]
    assert opencl_source_errors(p, p.seed_sources()) is None
    main = opencl_main(p, p.seed_sources())
    assert "namespace mbslot_relu" in main and "#define MB_SINGLE_TU 1" in main and "int main(" in main


def test_parse_opencl_driver_output_and_profile():
    r = parse_driver_output(DRIVER_OUT)
    assert r.ok and r.partition_us == 120.5 and r.kernel_us == {"linear/k_linear": 79.0, "relu/k_relu": 12.0}
    assert r.device.startswith("QUALCOMM Adreno")
    by = opencl_profile.profile_by_slot(DRIVER_OUT, {"linear": 3, "relu": 2})
    lin, relu = by["linear"], by["relu"]
    assert lin["kernel_wg_max_min"] == 512 and lin["private_mem_max"] == 64
    assert any("register pressure" in h for h in lin["hints"]) and any("private memory" in h for h in lin["hints"])
    assert relu.get("state") == "latency_bound"            # 2 launches of 3 us per op instance
    assert "state" not in lin                               # no measured bound: roofline state stays
    txt = opencl_profile.render(lin)
    assert "k_linear 79.0 us x3" in txt and "wg_max 512" in txt
    assert opencl_profile.delta(lin, {**lin, "duration_us": 40.0})["duration_us"] == -39.0


def test_store_never_mixes_targets(tmp_path):
    store = ImplStore(tmp_path)
    p = SlotProblem(FIX)
    s = p.slots["relu"]
    kw = dict(op=s.op, op_class=s.op_class, signature_sha=s.signature_sha, shapes=s.shapes(),
              buckets=["n:x"], slot_us=1.0, speedup_vs_seed=2.0, problem_id="other")
    store.add("cuda code", target="cuda", arch="sm_75", **kw)
    r = store.retrieve(target="opencl", arch=ARCH, op=s.op, op_class=s.op_class, signature=s.signature,
                       shapes=s.shapes())
    assert r["seed"] is None and r["adapt"] is None and not r["examples"]
    store.add("opencl code", target="opencl", arch=ARCH, **kw)
    assert [w.target for w in store.winners(target="opencl")] == ["opencl"]


# ---------------------------------------------------------------- device

@functools.lru_cache(None)
def _opencl_gpu() -> bool:
    if shutil.which("g++") is None:
        return False
    d = Path(tempfile.mkdtemp())
    try:
        (d / "q.cpp").write_text(
            "#define CL_TARGET_OPENCL_VERSION 200\n#include <CL/cl.h>\nint main(){cl_platform_id p[8];cl_uint n=0;"
            "if(clGetPlatformIDs(8,p,&n)||!n)return 1;for(cl_uint i=0;i<n;i++){cl_uint d=0;"
            "if(!clGetDeviceIDs(p[i],CL_DEVICE_TYPE_GPU,0,0,&d)&&d)return 0;}return 1;}\n")
        if subprocess.run(["g++", "-o", str(d / "q"), str(d / "q.cpp"), "-lOpenCL"], capture_output=True).returncode:
            return False
        return subprocess.run([str(d / "q")], capture_output=True, timeout=60).returncode == 0
    except Exception:
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


needs_opencl = pytest.mark.skipif(not _opencl_gpu(), reason="needs g++, OpenCL headers and an OpenCL GPU")


def scripted(state):
    """relu -> a float4 kernel; linear -> broken first, fixed on the fix prompt; else unchanged."""
    def fn(msgs):
        user = msgs[-1]["content"]
        state["prompts"].append(user)
        cur = extract_code(user.split("Current source:")[-1] if "Current source:" in user
                           else user.split("Current slot source:")[-1])
        if "failed:" in user:
            state["fix_calls"] += 1
            return f"```cpp\n{state['linear_ok']}\n```"
        if "## Slot `relu`" in user:
            return FAST_RELU
        if "## Slot `linear`" in user and not state["broke"]:
            state["broke"] = True
            state["linear_ok"] = cur
            return "```cpp\nvoid kernel_linear(const float *input) { this is not valid }\n```"
        return f"```cpp\n{cur}```"
    return fn


@needs_opencl
def test_baseline_passes_and_wrong_slot_is_attributed(tmp_path):
    p = SlotProblem(shutil.copytree(FIX, tmp_path / "lenet"))
    ev = OpenCLLocalEvaluator(work_root=tmp_path / "work")
    base = asyncio.run(ev.evaluate(p, p.seed_sources(), "base"))
    assert base.ok, base.error_text()
    assert set(base.slot_us) == {"conv2d", "relu", "maxpool2d", "linear"} and base.partition_us > 0
    assert "relu/k_relu" in base.kernel_us and base.device
    bad = dict(p.seed_sources())
    bad["relu"] = bad["relu"].replace("fmax(v, 0.0f)", "fmax(v, 0.0f) + 0.5f")
    r = asyncio.run(ev.evaluate(p, bad, "wrong"))
    assert not r.ok and r.failing_slots() == ["relu"]
    prof, err = asyncio.run(ev.profile(p, p.seed_sources(), "prof"))
    assert err is None and set(prof) == {"conv2d", "relu", "maxpool2d", "linear"}
    assert prof["relu"]["launches"] == 4                    # four relu instances, once each


@needs_opencl
def test_agent_end_to_end_opencl(tmp_path):
    p = SlotProblem(shutil.copytree(FIX, tmp_path / "lenet"))
    state = {"broke": False, "fix_calls": 0, "linear_ok": None, "prompts": []}
    agentic_llm.register_stub("scripted_cl", scripted(state))
    store, olog = ImplStore(tmp_path / "impl"), OutcomeLog(tmp_path / "outcomes.jsonl")
    try:
        agent = MultiSlotAgent(p, OpenCLLocalEvaluator(work_root=tmp_path / "work"), model="stub:scripted_cl",
                               out_dir=tmp_path / "out", arch=ARCH, impl_store=store, outcome_log=olog,
                               rollouts=2, steps=4, min_share=0.0, seed=1, ncu="best")
        res = asyncio.run(agent.run())
    finally:
        agentic_llm.unregister_stub("scripted_cl")
    assert res["final_ok"], res
    assert res["target"] == "opencl" and res["arch"] == ARCH
    outcomes = list(olog)
    assert len(outcomes) == 8 and all(o.target == "opencl" and o.arch == ARCH for o in outcomes)
    if "linear" in {o.slot for o in outcomes}:
        assert state["fix_calls"] >= 1
    recs = store.records()
    assert recs and all(r.target == "opencl" and r.arch == ARCH for r in recs)
    assert (tmp_path / "out" / "best" / "slots" / "relu.cpp").exists()
    # OpenCL event profiles (the NCU analogue) reached the prompts
    assert res["ncu"]["runs"] >= 1 and not res["ncu"]["disabled"]
    assert any("OpenCL profile (cl_event times" in u for u in state["prompts"])
    assert all("```cpp" in u for u in state["prompts"] if "Current slot source" in u)


@pytest.mark.skipif(not os.environ.get("KB_OPENCL_BOARD_HOST"), reason="set KB_OPENCL_BOARD_HOST=user@board")
def test_board_evaluator(tmp_path):
    p = SlotProblem(shutil.copytree(FIX, tmp_path / "lenet"))
    ev = AdrenoEvaluator(os.environ["KB_OPENCL_BOARD_HOST"],
                         remote_dir=os.environ.get("KB_OPENCL_BOARD_DIR", "/tmp/kernelblaster_multislot_test"),
                         work_root=tmp_path / "work")
    try:
        r = asyncio.run(ev.evaluate(p, p.seed_sources(), "board"))
        assert r.ok, r.error_text()
        broken = {**p.seed_sources(), "relu": "void kernel_relu(const float *input) { nope }\n"}
        r2 = asyncio.run(ev.evaluate(p, broken, "board_broken"))
        assert not r2.ok and r2.stage == "compile"
    finally:
        asyncio.run(ev.close())


def test_board_evaluator_lock(tmp_path):
    """A shared board: every build + run holds flock on the board's lock file and has a board-side timeout."""
    p = SlotProblem(FIX)
    ev = AdrenoEvaluator("u@board", remote_dir="/r/kb", work_root=tmp_path / "work", lock="/r/board.lock",
                         timeout_s=60)
    calls = []

    async def fake_cmd(*argv, timeout=None, stdin=None):
        calls.append(argv)
        return 0, "MB_PARTITION_US 1.0 reps=3\npassed\n" if argv[0] == "ssh" and "flock" in argv[-1] else ""

    ev._cmd = fake_cmd
    asyncio.run(ev.evaluate(p, p.seed_sources(), "t"))
    run = [a[-1] for a in calls if a[0] == "ssh" and "main.cpp" in a[-1]]
    assert len(run) == 1 and "flock /r/board.lock sh -c" in run[0] and "timeout -k 10 60 env" in run[0]
    assert run[0].index("flock") < run[0].index("rm -rf /r/kb/")     # the candidate dir goes after the lock


def test_server_evaluator_plumbing(tmp_path, monkeypatch):
    """OpenCLServerEvaluator: one main.cpp through compile_opencl, data.bin as a kernel file,
    the driver pointed at ./data.bin (and `profile` as argv[2]); the servers are faked here."""
    from kernelblaster.agents.utils import commands
    from kernelblaster.knowledge.slots import OpenCLServerEvaluator

    p = SlotProblem(FIX)
    seen = {}

    async def fake_compile(main, kernel, gpu, timeout=0, job_name="", remote=True):
        seen["main"], seen["remote"] = Path(main).read_text(), remote
        return str(tmp_path / "exe")

    async def fake_run(exe, gpu, timeout, job_name, kernel_files=None, extra_files=None, n_runs=1,
                       profile=False, extra_args=""):
        seen.setdefault("runs", []).append((kernel_files, extra_args))
        return (DRIVER_OUT if "profile" not in extra_args else
                DRIVER_OUT.replace("passed\n", "MB_PROFILE_DONE\n")), ""

    monkeypatch.setattr(commands, "compile_opencl", fake_compile)
    monkeypatch.setattr(commands, "run_adreno_executable", fake_run)
    ev = OpenCLServerEvaluator("adreno650", tmp_path / "work")
    r = asyncio.run(ev.evaluate(p, p.seed_sources(), "srv"))
    assert r.ok and r.partition_us == 120.5
    assert "#define MB_SINGLE_TU 1" in seen["main"] and seen["remote"]
    files, args = seen["runs"][0]
    assert files == [str(p.root / "data.bin")] and args == "data.bin"
    prof, err = asyncio.run(ev.profile(p, p.seed_sources(), "srv"))
    assert err is None and "linear" in prof and seen["runs"][1][1] == "data.bin profile"
    bad = asyncio.run(ev.evaluate(p, {**p.seed_sources(), "relu": "void f() { clFinish(q); }"}, "srv"))
    assert not bad.ok and bad.stage == "compile"
