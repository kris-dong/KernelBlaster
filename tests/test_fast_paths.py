"""Fast compile / exec / profile paths (docs/fast_paths.md).

The build tests need nvcc + a CUDA GPU (CMake + LibTorch for the native
comparison) and are skipped otherwise.
"""
import asyncio
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from kernelblaster.knowledge.slots import LocalEvaluator, SlotProblem
from kernelblaster.servers import compile as compile_mod
from kernelblaster.servers import cuda_fast
from kernelblaster.servers.utils.build_cache import BuildCache, cache_key, external_inputs

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "mb_lenet"
RELU = ROOT / "data" / "kernelbench-cuda" / "level1" / "019_ReLU"


def _gpu_arch():
    if shutil.which("nvcc") is None:
        return None
    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):    # GPUs hidden (nvidia-smi ignores it)
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.split()
        return out[0].replace(".", "") if out else None
    except Exception:
        return None


ARCH = _gpu_arch()
needs_gpu = pytest.mark.skipif(ARCH is None, reason="needs nvcc and a CUDA GPU")
needs_cmake = pytest.mark.skipif(ARCH is None or shutil.which("cmake") is None,
                                 reason="needs nvcc, cmake, LibTorch and a CUDA GPU")


def _stable(text: str) -> list[str]:
    """Driver output without the timing lines."""
    return [l for l in text.splitlines()
            if not re.match(r"MB_(SLOT|OP|PARTITION|E2E|ATTR_SLOT|ATTR_OP|ATTR_SUM|LAUNCH_OVERHEAD)_US|MB_PARTITION_DIST|MB_PHASE_MS", l)]


# ---------------------------------------------------------------- cache


def test_build_cache_builds_once_under_concurrency(tmp_path):
    cache = BuildCache(tmp_path / "c")
    calls = []

    async def build(tmp: Path):
        calls.append(tmp)
        await asyncio.sleep(0.05)
        tmp.write_text("obj")

    async def main():
        return await asyncio.gather(*[cache.get_or_build("k" * 64, ".o", build) for _ in range(8)])

    res = asyncio.run(main())
    assert len(calls) == 1
    assert len({p for p, _ in res}) == 1 and sum(hit for _, hit in res) == 7
    assert not list((tmp_path / "c").rglob("*.tmp"))


def test_build_cache_failure_is_not_cached(tmp_path):
    cache = BuildCache(tmp_path / "c")

    async def bad(tmp: Path):
        tmp.write_text("partial")
        raise RuntimeError("compile error")

    async def good(tmp: Path):
        tmp.write_text("ok")

    async def main():
        with pytest.raises(RuntimeError):
            await cache.get_or_build("a" * 64, ".o", bad)
        return await cache.get_or_build("a" * 64, ".o", good)

    path, hit = asyncio.run(main())
    assert not hit and path.read_text() == "ok"


def test_cache_key_tracks_incbin_data(tmp_path):
    data = tmp_path / "data.bin"
    data.write_bytes(b"1234")
    src = f'__asm__(".incbin \\"{data}\\"\\n");'
    k1 = cache_key(src, *external_inputs(src))
    data.write_bytes(b"12345")
    k2 = cache_key(src, *external_inputs(src))
    assert str(data) in external_inputs(src)[0] and k1 != k2


# ---------------------------------------------------------------- toolchain


def test_parse_toolchain_templates_commands(tmp_path):
    src, bd = tmp_path / "proj", tmp_path / "proj" / "build"
    (bd / "CMakeFiles" / "main.dir").mkdir(parents=True)
    (bd / "compile_commands.json").write_text(json.dumps([
        {"directory": str(bd), "file": str(src / "main.cpp"), "output": "CMakeFiles/main.dir/main.cpp.o",
         "command": f"/usr/bin/c++ -isystem /t/include -O3 -o CMakeFiles/main.dir/main.cpp.o -c {src}/main.cpp"},
        {"directory": str(bd), "file": str(src / "cuda_model.cu"), "output": "CMakeFiles/main.dir/cuda_model.cu.o",
         "command": "/usr/local/cuda/bin/nvcc --options-file CMakeFiles/main.dir/includes_CUDA.rsp -O3 "
                    f"-x cu -c {src}/cuda_model.cu -o CMakeFiles/main.dir/cuda_model.cu.o"},
    ]))
    (bd / "CMakeFiles" / "main.dir" / "link.txt").write_text(
        "/usr/bin/c++ -O3 CMakeFiles/main.dir/main.cpp.o CMakeFiles/main.dir/cuda_model.cu.o -o main "
        "-L/usr/local/cuda/lib/stubs -L/usr/local/cuda/lib -Wl,-rpath,/t/lib /t/lib/libtorch.so -lcudart_static\n")
    tc = cuda_fast.parse_toolchain("sm_89", src, bd)
    assert tc.driver_cmd.endswith("-o {obj} -c {src}")
    assert f"--options-file {bd}/CMakeFiles/main.dir/includes_CUDA.rsp" in tc.kernel_cmd
    assert "{src}" in tc.kernel_cmd and "{obj}" in tc.kernel_cmd
    assert "{objs} -o {exe}" in tc.link_torch_cmd and "libtorch.so" in tc.link_torch_cmd
    assert "libtorch" not in tc.link_plain_cmd and "-L/usr/local/cuda/lib" in tc.link_plain_cmd
    assert "stubs" not in tc.link_plain_cmd


def test_torch_detection():
    assert cuda_fast.needs_torch("#include <torch/torch.h>\nint main(){}")
    assert cuda_fast.needs_torch("auto t = at::empty({1});")
    assert not cuda_fast.needs_torch(FIX.joinpath("driver.cpp").read_text())
    assert not cuda_fast.needs_torch("#include <cuda_runtime.h>\n// torch is not used here\n")


def test_split_without_torch_for_plain_drivers(tmp_path):
    drv = FIX / "driver.cpp"
    cu = tmp_path / "k.cu"
    cu.write_text("void launch_gpu_implementation(void **b) {}\n")
    _, legacy_header, _ = compile_mod.split_files_for_compilation(str(drv), str(cu))
    main, header, kernel = compile_mod.split_files_for_compilation(str(drv), str(cu), driver_torch=False)
    assert "torch/torch.h" in legacy_header and "torch" not in header
    assert header.replace("#include <cstdint>\n", "").strip().startswith("void launch_gpu_implementation")


# ---------------------------------------------------------------- nsys kernel names


def test_kernel_base_name():
    from kernelblaster.agents.utils.parsing import kernel_base_name

    assert kernel_base_name("relu_fp16_kernel(const __half *, __half *, long)") == "relu_fp16_kernel"
    assert kernel_base_name("void at::native::vectorized_elementwise_kernel<(int)4, at::native::AbsFunctor"
                            "<c10::Half>, std::array<char *, (unsigned long)2>>(int, T2, T3)") == \
        "vectorized_elementwise_kernel"
    assert kernel_base_name("void at::native::<unnamed>::distribution_elementwise_grid_stride_kernel"
                            "<float, (int)4>(long)") == "distribution_elementwise_grid_stride_kernel"
    assert kernel_base_name("void mbslot_relu::k_relu<128>(const float *, float *, int)") == "k_relu"


def test_find_kernel_names_trace(tmp_path):
    from kernelblaster.agents.utils.parsing import find_kernel_names_trace

    src = tmp_path / "k.cu"
    src.write_text("__global__ void gemm_tile(float *a) {}\n__global__ void unused_kernel() {}\n"
                   "void launch_gpu_implementation() { gemm_tile<<<1, 1>>>(0); }\n")
    trace = ("nsys noise\n__GPUTRACE_CSV_START__\n"
             "100,20,void at::native::vectorized_elementwise_kernel<(int)4>(int)\n"
             "200,30,gemm_tile(float *)\n")
    assert find_kernel_names_trace(trace, src) == ["gemm_tile"]
    assert find_kernel_names_trace("no marker", src) == []


# ---------------------------------------------------------------- multi-slot units


def test_units_cover_every_slot_and_declare_entries():
    p = SlotProblem(FIX)
    units = dict(p.units(p.seed_sources()))
    assert set(units) == {f"slot_{s}.cu" for s in p.order} | {"walker_tail.cu"}
    tail = units["walker_tail.cu"]
    for sid, slot in p.slots.items():
        assert f"namespace mbslot_{sid} {{ {slot.signature}; }}" in tail
        assert f"namespace mbslot_{sid} {{" in units[f"slot_{sid}.cu"]
    # the assembled TU is unchanged (ModelBlaster's kb_export.assemble contract)
    assert len(re.findall(r"^namespace mbslot_\w+ \{", p.assemble(p.seed_sources()), re.M)) == len(p.order)


def _evaluate(ev, p, sources, tag="t"):
    return asyncio.run(ev.evaluate(p, sources, tag))


@needs_gpu
def test_incremental_build_matches_assembled(tmp_path):
    p = SlotProblem(FIX)
    mono = _evaluate(LocalEvaluator(ARCH, work_root=tmp_path / "m", runtime="process", incremental=False, keep=True), p,
                     p.seed_sources())
    inc_ev = LocalEvaluator(ARCH, work_root=tmp_path / "i", runtime="process", incremental=True, keep=True)
    inc = _evaluate(inc_ev, p, p.seed_sources())
    assert mono.ok and inc.ok
    assert _stable(mono.log) == _stable(inc.log)
    assert not list((tmp_path / "i").glob("*/kernels.cu"))          # built from units


def _own_fixture(tmp_path) -> SlotProblem:
    """A copy of the LeNet fixture whose driver links its own data.bin."""
    d = tmp_path / "lenet"
    shutil.copytree(FIX, d)
    drv = d / "driver.cpp"
    drv.write_text(re.sub(r'(\.incbin \\")[^"\\]+', lambda m: m.group(1) + str(d / "data.bin"),
                          drv.read_text()))
    p = SlotProblem(d)
    assert str(d / "data.bin") in p.driver_path.read_text()
    return p


@needs_gpu
def test_incremental_cache_invalidation(tmp_path):
    p = _own_fixture(tmp_path)
    ev = LocalEvaluator(ARCH, work_root=tmp_path / "w", runtime="process", incremental=True)
    seeds = p.seed_sources()
    assert _evaluate(ev, p, seeds).ok
    n_units = len(p.order) + 2                       # slots + tail + driver
    assert (ev._cache.misses, ev._cache.hits) == (n_units + 1, 0)
    assert _evaluate(ev, p, seeds).ok                # same candidate: link cached too
    assert ev._cache.misses == n_units + 1
    changed = {**seeds, "relu": seeds["relu"] + "\n// candidate 2\n"}
    assert _evaluate(ev, p, changed).ok
    assert ev._cache.misses == n_units + 3           # one slot object + one link
    # the driver's linked-in data changed -> driver object rebuilt
    data = p.root / "data.bin"
    st = data.stat()
    os.utime(data, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert _evaluate(ev, p, changed).ok
    assert ev._cache.misses == n_units + 5


@needs_gpu
def test_incremental_falls_back_to_assembled_tu(tmp_path):
    """A slot that leans on another slot's include only builds as one TU."""
    p = SlotProblem(FIX)
    hdr = tmp_path / "zero.h"
    hdr.write_text("#pragma once\n#define TEST_ZERO 0.0f\n")
    src = p.seed_sources()
    assert "fmaxf(x[i], 0.0f)" in src["relu"]
    src = {**src, "conv2d": f'#include "{hdr}"\n' + src["conv2d"],
           "relu": src["relu"].replace("fmaxf(x[i], 0.0f)", "fmaxf(x[i], TEST_ZERO)")}
    ev = LocalEvaluator(ARCH, work_root=tmp_path / "w", runtime="process", incremental=True, keep=True)
    r = _evaluate(ev, p, src)
    assert r.ok
    assert list((tmp_path / "w").glob("*/kernels.cu"))              # assembled build ran
    bad = {**src, "linear": "void kernel_linear(const float *input) { this is not valid }"}
    r = _evaluate(ev, p, bad)
    assert not r.ok and r.stage == "compile" and "error" in r.errors[0]


@needs_gpu
def test_concurrent_incremental_builds_share_cache(tmp_path):
    p = SlotProblem(FIX)
    ev = LocalEvaluator(ARCH, work_root=tmp_path / "w", runtime="process", incremental=True, max_parallel=4)
    seeds = p.seed_sources()
    cands = [{**seeds, sid: seeds[sid] + f"\n// c{i}\n"} for i, sid in enumerate(p.order)]

    async def main():
        return await asyncio.gather(*[ev.evaluate(p, c, f"c{i}") for i, c in enumerate(cands)])

    res = asyncio.run(main())
    assert all(r.ok for r in res)
    # driver + tail compiled once; each candidate adds its own slot object
    objs = list(ev._cache.root.rglob("*.o"))
    assert len(objs) == 2 + len(p.order) + len(p.order)


# ---------------------------------------------------------------- compile server fast path


def _compile(tmp_path, main, cuda, persistent: bool, name: str) -> Path:
    out = tmp_path / "out" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    return asyncio.run(compile_mod.exec_compilation(name, str(main), str(cuda), f"sm_{ARCH}", 0, out,
                                                    persistent))


def _run(exe: Path) -> str:
    return subprocess.run([str(exe)], capture_output=True, text=True, timeout=300).stdout


@pytest.fixture
def env_root(tmp_path, monkeypatch):
    root = tmp_path / "env"
    monkeypatch.setattr(compile_mod, "ENV_DIR", root)
    monkeypatch.setattr(cuda_fast, "_BUILDER", None)
    return root


@needs_cmake
def test_fast_and_cmake_builds_agree(tmp_path, env_root, monkeypatch):
    """Same candidate through the CMake path and the fast path: same driver
    output, for a LibTorch driver and for a torch-free multi-slot driver."""
    p = SlotProblem(FIX)
    kernels = tmp_path / "kernels.cu"
    kernels.write_text(p.assemble(p.seed_sources()))
    cases = [("relu", RELU / "driver.cpp", RELU / "init.cu"), ("lenet", p.driver_path, kernels)]
    for name, main, cuda in cases:
        monkeypatch.setenv("KERNELBLASTER_FAST_COMPILE", "0")
        legacy = _run(_compile(tmp_path, main, cuda, True, f"{name}_cmake"))
        monkeypatch.setenv("KERNELBLASTER_FAST_COMPILE", "1")
        fast = _run(_compile(tmp_path, main, cuda, True, f"{name}_fast"))
        assert "passed" in legacy
        assert _stable(legacy) == _stable(fast), name
    b = cuda_fast._BUILDER
    # second candidate of the same problem: only the kernel object is new
    cand = tmp_path / "cand.cu"
    cand.write_text((RELU / "init.cu").read_text() + "\n// candidate 2\n")
    misses = b.cache.misses
    exe = _compile(tmp_path, RELU / "driver.cpp", cand, False, "relu_cand2")
    assert "passed" in _run(exe) and b.cache.misses == misses + 2       # kernel + link
    lenet_exe = env_root / "persistent" / "lenet_fast" / f"build_sm_{ARCH}" / "main"
    libs = subprocess.run(["ldd", str(lenet_exe)], capture_output=True, text=True).stdout
    assert "libtorch" not in libs                                         # plain flavour


# ---------------------------------------------------------------- multi-slot NCU


def test_ncu_command_slot_filter():
    from kernelblaster.knowledge import ncu

    assert "regex:'mbslot_'" in ncu.ncu_command()
    assert "regex:'mbslot_relu::'" in ncu.ncu_command(slots=["relu"])
    assert "regex:'mbslot_(conv2d|relu)::'" in ncu.ncu_command(slots=["relu", "conv2d"])


def test_profile_skips_slots_with_cached_profiles():
    from kernelblaster.agents.opt_multislot import MultiSlotAgent

    calls = []

    class Ev:
        async def profile(self, problem, sources, tag, slots=None):
            calls.append(slots)
            return {s: {"duration_us": 1.0, "code": sources[s]} for s in (slots or sources)}, None

    a = object.__new__(MultiSlotAgent)
    a.ncu, a.ncu_disabled, a.ncu_cache, a.ncu_runs, a.ev, a.p = "best", None, {}, 0, Ev(), None
    src = {"a": "A0", "b": "B0", "c": "C0"}
    by = asyncio.run(a._profile(src, "baseline"))
    assert calls == [None] and set(by) == {"a", "b", "c"}
    by = asyncio.run(a._profile({**src, "b": "B1"}, "cand"))
    assert calls[-1] == ["b"] and by["b"]["code"] == "B1" and by["a"]["code"] == "A0"
    assert asyncio.run(a._profile({**src, "b": "B1"}, "again")) == by and len(calls) == 2
    assert a.ncu_runs == 2


# ---------------------------------------------------------------- exec by path


def test_exec_by_path_only_for_local_servers(monkeypatch):
    from kernelblaster.agents.utils import commands

    monkeypatch.delenv("KERNELBLASTER_EXEC_BY_PATH", raising=False)
    assert commands._exec_by_path("http://127.0.0.1:2002")
    assert commands._exec_by_path("http://localhost:2002")
    assert not commands._exec_by_path("https://api.nvcf.nvidia.com/x")
    monkeypatch.setenv("KERNELBLASTER_EXEC_BY_PATH", "0")
    assert not commands._exec_by_path("http://127.0.0.1:2002")


def test_local_exec_runs_binary_in_place(tmp_path):
    from kernelblaster.servers.strategies import LocalExecStrategy

    exe = tmp_path / "my prog"
    exe.write_text("#!/bin/sh\necho ran \"$0\"\n")
    exe.chmod(0o755)
    out, _ = asyncio.run(LocalExecStrategy(gpu_ids=["0"]).exec(
        worker_id=0, binary_data=b"", filename=exe.name, binary_path=str(exe)))
    assert out.strip() == f"ran {exe}" and exe.exists()
