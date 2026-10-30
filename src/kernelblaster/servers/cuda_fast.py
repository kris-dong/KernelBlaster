# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Fast CUDA builds for the compile server: no per-job CMake, cached objects.

The legacy path (:func:`compile.exec_compilation`) runs ``cmake`` + ``make``
in a CMake project per job; with ``persistent_artifacts`` (what the RL agents
use) every job gets a fresh project, so every candidate pays the CMake
configure (~3.5 s) and recompiles the unchanged driver, whose LibTorch
headers cost ~10 s, on top of the ~1.3 s kernel TU.

Here the CMake project is configured ONCE per SM version (no build), and the
exact compile / link commands it would run are read back from
``compile_commands.json`` and ``link.txt``. A job then runs those commands
directly:

  driver object  -- cached by (command, main.cpp, cuda_model.cuh, absolute
                    inputs such as an ``.incbin`` data file)
  kernel object  -- cached by (command, cuda_model.cu, cuda_model.cuh)
  binary         -- cached by (link command, the two object keys)

Same flags, same compilers, same sources as the CMake build, so the binary is
the one ``make`` would produce. Drivers that use no LibTorch at all (e.g.
ModelBlaster's multi-slot drivers) are compiled without the ``torch/torch.h``
include the legacy split forces into the host TU and linked against the CUDA
runtime only ("plain" flavour) -- smaller, faster to link and to start. If
the plain driver fails to compile, the driver is built with the LibTorch
flavour from then on; if the toolchain cannot be derived at all, the caller falls back to CMake.

``KERNELBLASTER_FAST_COMPILE=0`` disables this module (legacy path only).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..telemetry.core import traced
from .utils.build_cache import BuildCache, cache_key, external_inputs, link_or_copy
from .utils.subprocess import run_subprocess_shell

logger = logging.getLogger("uvicorn")

# Tokens that make a driver need LibTorch (headers in the host TU, libs at link).
_TORCH_RE = re.compile(
    r'#\s*include\s*[<"](?:torch|ATen|c10|caffe2|pybind11)/|#\s*include\s*[<"]Python\.h|\b(?:torch|c10|at)::'
)


def enabled() -> bool:
    return os.getenv("KERNELBLASTER_FAST_COMPILE", "1").lower() not in ("0", "false", "no", "off")


def needs_torch(text: str) -> bool:
    return bool(_TORCH_RE.search(text))


class ToolchainError(Exception):
    """The CMake project could not be turned into direct commands (caller
    falls back to the legacy CMake build)."""


@dataclass
class Toolchain:
    """Commands of the cuda_env CMake project for one SM version, with
    ``{src}`` / ``{obj}`` / ``{objs}`` / ``{exe}`` placeholders."""

    sm_version: str
    driver_cmd: str          # host C++ TU (main.cpp)
    kernel_cmd: str          # nvcc TU (cuda_model.cu)
    link_torch_cmd: str      # what make links: LibTorch + Python + CUDA
    link_plain_cmd: str      # CUDA runtime only, for torch-free drivers

    def fingerprint(self) -> str:
        return cache_key(self.driver_cmd, self.kernel_cmd, self.link_torch_cmd, self.link_plain_cmd)


def _swap(cmd: str, old: str, new: str) -> str:
    if old not in cmd:
        raise ToolchainError(f"expected {old!r} in {cmd!r}")
    return cmd.replace(old, new)


def parse_toolchain(sm_version: str, src_dir: Path, build_dir: Path) -> Toolchain:
    """Turn a configured (not built) cuda_env project into a :class:`Toolchain`."""
    entries = json.loads((build_dir / "compile_commands.json").read_text())
    by_file = {Path(e["file"]).name: e for e in entries}
    try:
        drv, ker = by_file["main.cpp"], by_file["cuda_model.cu"]
    except KeyError as e:
        raise ToolchainError(f"compile_commands.json lacks {e}")
    bd = str(build_dir)
    # relative paths in the commands are relative to the build dir
    driver_cmd = _swap(drv["command"], f"-o {drv['output']}", "-o {obj}")
    driver_cmd = _swap(driver_cmd, f"-c {src_dir / 'main.cpp'}", "-c {src}")
    kernel_cmd = _swap(ker["command"], f"-o {ker['output']}", "-o {obj}")
    kernel_cmd = _swap(kernel_cmd, f"-c {src_dir / 'cuda_model.cu'}", "-c {src}")
    kernel_cmd = re.sub(r"--options-file (\S+)",
                        lambda m: f"--options-file {m.group(1) if m.group(1).startswith('/') else bd + '/' + m.group(1)}",
                        kernel_cmd)
    link = (build_dir / "CMakeFiles" / "main.dir" / "link.txt").read_text().strip()
    link_torch_cmd = _swap(link, f"{drv['output']} {ker['output']} -o main", "{objs} -o {exe}")
    toks = shlex.split(link)
    lib_dirs = [t for t in toks if t.startswith("-L") and "stubs" not in t]
    # the same CUDA runtime make links (static cudart), without LibTorch/Python
    link_plain_cmd = " ".join([toks[0], "-O3", "-DNDEBUG", "{objs}", "-o", "{exe}", *lib_dirs,
                               "-lcudadevrt", "-lcudart_static", "-lrt", "-lpthread", "-ldl"])
    for c in (driver_cmd, kernel_cmd, link_torch_cmd):
        if "{src}" not in c and "{objs}" not in c:
            raise ToolchainError(f"could not template {c!r}")
    return Toolchain(sm_version, driver_cmd, kernel_cmd, link_torch_cmd, link_plain_cmd)


class FastCudaBuilder:
    """Per compile-server builder: one toolchain per SM version, one cache."""

    def __init__(self, root: Path, env: Optional[dict] = None):
        self.root = Path(root)
        self.env = env
        self.cache = BuildCache(Path(os.getenv("KERNELBLASTER_BUILD_CACHE_DIR") or self.root / "build_cache"))
        self._toolchains: dict[str, Toolchain] = {}
        self._tc_lock = asyncio.Lock()
        # concurrent compiler processes across all jobs (legacy: make -j8 per
        # job): KERNELBLASTER_FAST_COMPILE_JOBS pins it, else it follows the
        # free CPUs and memory (a LibTorch host TU needs ~2-3 GB)
        from .utils.resources import GiB, build_limiter
        self._jobs = build_limiter(env="KERNELBLASTER_FAST_COMPILE_JOBS", per_job_mem=int(2.5 * GiB),
                                   name="fast compile")
        # drivers whose plain (torch-free) build failed -> use LibTorch
        self._plain_failed: set[str] = set()

    async def toolchain(self, sm_version: str, timeout: float) -> Toolchain:
        async with self._tc_lock:
            tc = self._toolchains.get(sm_version)
            if tc is not None:
                return tc
            from .compile import (build_cmake_command, extract_arch_version,
                                  get_cuda_env_template_path)
            import shutil

            proj = self.root / f"toolchain_{sm_version}"
            if proj.exists():
                shutil.rmtree(proj)
            shutil.copytree(get_cuda_env_template_path(), proj)
            # placeholders: configure only needs the files to exist
            (proj / "main.cpp").write_text("int main() { return 0; }\n")
            (proj / "cuda_model.cu").write_text("\n")
            build_dir = proj / "build"
            cmd = build_cmake_command(build_dir, extract_arch_version(sm_version)) + \
                " -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
            await run_subprocess_shell(stage=f"cmake_config_once:{sm_version}", cmd=cmd, cwd=proj,
                                       timeout_s=timeout, env=self.env, error_factory=ToolchainError,
                                       logger=logger)
            # configure alone writes compile_commands.json and link.txt
            tc = parse_toolchain(sm_version, proj, build_dir)
            logger.info(f"[fast_compile] toolchain for {sm_version}: driver={tc.driver_cmd!r} "
                        f"kernel={tc.kernel_cmd!r} link={tc.link_torch_cmd!r} plain={tc.link_plain_cmd!r}")
            self._toolchains[sm_version] = tc
            return tc

    @traced("build.nvcc")                  # telemetry: compiler / linker process time
    async def _run(self, stage: str, template: str, cwd: Path, timeout: float, error_factory, **subst):
        cmd = template
        for k, v in subst.items():
            cmd = cmd.replace("{" + k + "}", v)
        await run_subprocess_shell(stage=stage, cmd=cmd, cwd=cwd, timeout_s=timeout, env=self.env,
                                   error_factory=error_factory, logger=logger)

    async def object(self, stage: str, template: str, src: Path, key_parts: list, timeout: float,
                     error_factory) -> tuple[str, Path]:
        """Compile ``src`` with ``template`` (cached); ``(key, object path)``."""
        key = cache_key(stage, template, *key_parts)

        async def build(tmp: Path):
            async with self._jobs:
                await self._run(stage, template, src.parent, timeout, error_factory,
                                src=shlex.quote(str(src)), obj=shlex.quote(str(tmp)))

        path, hit = await self.cache.get_or_build(key, ".o", build)
        if hit:
            logger.info(f"[{stage}] cache hit {key[:12]} for {src}")
        return key, path

    async def link(self, template: str, objs: list[tuple[str, Path]], exe: Path, timeout: float,
                   error_factory) -> Path:
        key = cache_key("link", template, *[k for k, _ in objs])

        async def build(tmp: Path):
            await self._run("fast_link", template, exe.parent, timeout, error_factory,
                            objs=" ".join(shlex.quote(str(p)) for _, p in objs), exe=shlex.quote(str(tmp)))

        path, hit = await self.cache.get_or_build(key, ".exe", build)
        if hit:
            logger.info(f"[fast_link] cache hit {key[:12]} -> {exe}")
        link_or_copy(path, exe)
        return exe

    async def build_driver_kernel(self, work_dir: Path, main_file: str, cuda_file: Optional[str],
                                  sm_version: str, timeout: float, error_factory) -> Path:
        """The fast equivalent of ``exec_compilation``'s write + cmake + make
        in ``work_dir``; returns ``work_dir/build_<sm>/main``."""
        from .compile import split_files_for_compilation

        tc = await self.toolchain(sm_version, timeout)
        raw_main = Path(main_file).read_text()
        raw_cuda = Path(cuda_file).read_text() if cuda_file else ""
        plain = not needs_torch(raw_main) and not needs_torch(raw_cuda)
        driver_id = cache_key(raw_main)
        if driver_id in self._plain_failed:
            plain = False
        try:
            return await self._build(tc, work_dir, main_file, cuda_file, sm_version, timeout,
                                     error_factory, plain, split_files_for_compilation)
        except error_factory as e:
            if not plain or not str(getattr(e, "message", e)).startswith("[fast_driver"):
                raise
            # A torch-free driver that only compiled with torch/torch.h's
            # transitive includes: build it exactly like the legacy split does
            # (kernel errors and link errors are the candidate's own).
            logger.info(f"[fast_compile] plain build of {main_file} failed; retrying with LibTorch")
            self._plain_failed.add(driver_id)
            return await self._build(tc, work_dir, main_file, cuda_file, sm_version, timeout,
                                     error_factory, False, split_files_for_compilation)

    async def _build(self, tc, work_dir, main_file, cuda_file, sm_version, timeout, error_factory,
                     plain, split):
        main_text, header_text, cuda_text = split(main_file, cuda_file, driver_torch=not plain)
        for name, text in (("main.cpp", main_text), ("cuda_model.cuh", header_text), ("cuda_model.cu", cuda_text)):
            (work_dir / name).write_text(text)
        build_dir = work_dir / f"build_{sm_version}"
        build_dir.mkdir(parents=True, exist_ok=True)
        jobs = [self.object("fast_driver", tc.driver_cmd, work_dir / "main.cpp",
                            [main_text, header_text, *external_inputs(main_text)], timeout, error_factory)]
        if cuda_file:
            jobs.append(self.object("fast_kernel", tc.kernel_cmd, work_dir / "cuda_model.cu",
                                    [cuda_text, header_text, *external_inputs(cuda_text)], timeout,
                                    error_factory))
        objs = await asyncio.gather(*jobs, return_exceptions=True)
        for o in objs:            # driver first: its failure decides the fallback
            if isinstance(o, BaseException):
                raise o
        exe = build_dir / "main"
        if exe.exists():
            exe.unlink()          # may be a hard link into the cache
        return await self.link(tc.link_plain_cmd if plain else tc.link_torch_cmd, list(objs), exe,
                               timeout, error_factory)


    async def build_units(self, main_file: str, units: list[str], sm_version: str, exe: Path,
                          timeout: float, error_factory) -> Path:
        """Multi-TU build: the driver as is (no launch-header split) plus each
        of ``units`` as its own nvcc TU, every object cached, then one link.

        For multi-slot problems (one TU per slot + the walker tail): a new
        candidate recompiles only the slot that changed.
        """
        tc = await self.toolchain(sm_version, timeout)
        main_text = Path(main_file).read_text()
        texts = [Path(u).read_text() for u in units]
        plain = not needs_torch(main_text) and not any(needs_torch(t) for t in texts)
        jobs = [self.object("fast_driver", tc.driver_cmd, Path(main_file),
                            [main_text, *external_inputs(main_text)], timeout, error_factory)]
        jobs += [self.object("fast_kernel", tc.kernel_cmd, Path(u), [t, *external_inputs(t)], timeout,
                             error_factory) for u, t in zip(units, texts)]
        objs = await asyncio.gather(*jobs, return_exceptions=True)
        for o in objs:
            if isinstance(o, BaseException):
                raise o
        return await self.link(tc.link_plain_cmd if plain else tc.link_torch_cmd, list(objs), exe,
                               timeout, error_factory)

_BUILDER: Optional[FastCudaBuilder] = None


def get_builder(root: Path, env: Optional[dict] = None) -> FastCudaBuilder:
    global _BUILDER
    if _BUILDER is None or _BUILDER.root != Path(root):
        _BUILDER = FastCudaBuilder(root, env)
    return _BUILDER
