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
    return [l for l in text.splitlines() if not re.match(r"MB_(SLOT|OP|PARTITION)_US", l)]


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
