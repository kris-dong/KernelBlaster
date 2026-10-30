"""compile_and_run_riscv_batched must stage against the real target."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelblaster.agents.utils import commands
from kernelblaster.config import GPUType


class _Timer:
    def start(self, _name):
        pass

    def stop(self, _name):
        return 0.0


class _Logger:
    def __init__(self):
        self.warnings = []

    def info(self, _msg):
        pass

    def warning(self, msg):
        self.warnings.append(msg)


class _FakeBatchClient:
    def __init__(self):
        self.kwargs = None

    async def submit_riscv(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(success=True, stdout="ok", stderr="", message="")


def _run(tmp_path, monkeypatch, **kw):
    io = tmp_path / "kb_19_ReLU" / "fp32" / "generated" / "io.npz"
    io.parent.mkdir(parents=True)
    io.write_bytes(b"")

    async def fake_compile(*_a, **_k):
        return str(tmp_path / "tmpabc.elf")

    monkeypatch.setattr(commands, "compile_riscv", fake_compile)
    client, log = _FakeBatchClient(), _Logger()
    asyncio.run(commands.compile_and_run_riscv_batched(
        tmp_path / "driver.c", tmp_path / "cand.c", GPUType.RISCV_SPIKE,
        _Timer(), log, client, io_npz_path=io, **kw,
    ))
    return client.kwargs, log, io


def test_explicit_target_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELBLASTER_MODELBLASTER_TARGET", "rvv")
    kw, log, io = _run(tmp_path, monkeypatch, target="scalar")
    assert kw["target"] == "scalar"
    assert kw["base_stage_dir"] == io.parent / "scalar"
    assert kw["mid"] == "kb_19_ReLU"
    assert not log.warnings


def test_env_target_used_when_not_passed(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELBLASTER_MODELBLASTER_TARGET", "scalar")
    kw, log, io = _run(tmp_path, monkeypatch)
    assert kw["target"] == "scalar"
    assert kw["base_stage_dir"] == io.parent / "scalar"
    assert not log.warnings


def test_board_guess_is_last_resort_and_logged(tmp_path, monkeypatch):
    monkeypatch.delenv("KERNELBLASTER_MODELBLASTER_TARGET", raising=False)
    kw, log, io = _run(tmp_path, monkeypatch)
    assert kw["target"] == "rvv"
    assert kw["base_stage_dir"] == io.parent / "rvv"
    assert len(log.warnings) == 1 and "guessed" in log.warnings[0]
