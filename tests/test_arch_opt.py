"""nvcc target argument: Blackwell uses -gencode=arch=compute_100a,code=sm_100a (nvcc 13.3's -arch=sm_100a drops the suffix and ptxas then
rejects tcgen05 PTX, verified on a B200); every other target keeps -arch=<flag> byte for byte."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernelblaster.agents import hw_profile  # noqa: E402
from kernelblaster.knowledge import slots  # noqa: E402

BW = "-gencode=arch=compute_100a,code=sm_100a"


def test_blackwell_uses_gencode():
    for a in ("100", "sm_100", "sm_100a"):
        assert slots._arch_opt(a) == BW
        assert hw_profile.arch_opt(a) == BW


def test_other_targets_unchanged():
    for a, flag in (("75", "sm_75"), ("80", "sm_80"), ("86", "sm_86"), ("89", "sm_89"), ("sm_89", "sm_89"), ("90", "sm_90a")):
        assert slots._arch_opt(a) == f"-arch={flag}" == f"-arch={slots._arch_flag(a)}"
        assert hw_profile.arch_opt(a) == f"-arch={hw_profile.arch_flag(a)}"


def test_prompt_shows_the_real_flag():
    assert BW in hw_profile.hardware_text("100", "bf16")
    assert "-arch=sm_89" in hw_profile.hardware_text("89", "fp16")
