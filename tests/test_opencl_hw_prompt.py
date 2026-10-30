"""Hardware-aware OpenCL prompt (Adreno 650): fp16-math rule, hardware block, technique catalogue, and the switches
MB_OPENCL_F16_MATH / MB_OPENCL_HW_INFO. Pure python, no device, no LLM."""
import os
from types import SimpleNamespace
from unittest import mock

import pytest

from kernelblaster.agents import multislot_opencl as mo
from kernelblaster.agents.opt_multislot import target_profile

TOL = {"slot_rel": 2e-3}


@pytest.fixture(autouse=True)
def _clean_env():
    with mock.patch.dict(os.environ, {}):
        for k in ("MB_OPENCL_F16_MATH", "MB_OPENCL_HW_INFO", "MB_OPENCL_ARCH"):
            os.environ.pop(k, None)
        yield


def test_default_on_for_adreno650():
    p = mo.opencl_system_prompt("fp16", TOL, "adreno650")
    assert p != mo.SYSTEM_PROMPT_OPENCL
    assert "cl_khr_fp16" in p and "HALF ARITHMETIC IS ALLOWED" in p
    assert "## Target hardware" in p and "QUALCOMM Adreno(TM) 650" in p
    assert "OpenCL C 2.0" in p and "1.2 kernels" not in p
    assert "2e-3" in p                                         # the manifest's per-slot bound reaches the prompt
    assert "fp32 in/out" not in p                              # the old rule said fp32 even for fp16 problems
    assert "MB_SCRATCH_IMAGE" in p                             # how to get an image from a slot
    assert "never call clCreate*/clEnqueue*/clFinish" in p.replace("Still never call", "never call") or \
        "Still never call clCreate*/clEnqueue*/clFinish yourself" in p
    assert "mb_launch" in p                                    # all launches still go through mb_launch


def test_hardware_block_facts():
    p = mo.opencl_system_prompt("fp16", TOL, "adreno650")
    for needle in ("cl_qcom_dot_product8", "cl_qcom_accelerated_image_ops", "cl_qcom_reqd_sub_group_size",
                   "cl_khr_image2d_from_buffer", "cl_qcom_ml_ops", "32 KB", "public specs", "HYBRID",
                   "330-900 GFLOP/s", "10-100x", "Conv2d", "BatchNorm", "not measured"):
        assert needle in p, needle


def test_switch_off_restores_the_legacy_prompt():
    os.environ["MB_OPENCL_F16_MATH"] = "0"
    os.environ["MB_OPENCL_HW_INFO"] = "0"
    assert mo.opencl_system_prompt("fp16", TOL, "adreno650") == mo.SYSTEM_PROMPT_OPENCL
    assert mo.opencl_system_prompt("fp32", None, "adreno650") == mo.SYSTEM_PROMPT_OPENCL


def test_switches_are_independent():
    os.environ["MB_OPENCL_F16_MATH"] = "0"
    p = mo.opencl_system_prompt("fp16", TOL, "adreno650")           # hardware block, storage-only fp16 rule
    assert "## Target hardware" in p and "HALF ARITHMETIC" not in p and "fp16 storage" in p
    os.environ["MB_OPENCL_F16_MATH"] = "auto"
    os.environ["MB_OPENCL_HW_INFO"] = "0"
    p = mo.opencl_system_prompt("fp16", TOL, "adreno650")           # math rule, no hardware block
    assert "HALF ARITHMETIC" in p and "## Target hardware" not in p


def test_standin_device_keeps_the_legacy_prompt():
    for arch in ("titanrtx", "l40s", None):
        assert mo.opencl_system_prompt("fp16", TOL, arch) == mo.SYSTEM_PROMPT_OPENCL
    os.environ["MB_OPENCL_ARCH"] = "adreno650"                      # arch from the environment (ModelBlaster sets it)
    assert "HALF ARITHMETIC" in mo.opencl_system_prompt("fp16", TOL, None)


def test_fp32_problem_on_adreno_gets_hardware_but_no_half_rule():
    p = mo.opencl_system_prompt("fp32", {"slot_rel": 1e-3}, "adreno650")
    assert "HALF ARITHMETIC" not in p and "fp32 in/out" in p and "1e-3" in p and "## Target hardware" in p


def test_target_profile_uses_it():
    tp = target_profile("opencl")
    prob = SimpleNamespace(dtype="fp16", tolerance=TOL)
    assert tp.prompt_for(prob, "adreno650") == mo.opencl_system_prompt("fp16", TOL, "adreno650")
    assert tp.prompt_for(prob, "titanrtx") == mo.SYSTEM_PROMPT_OPENCL
    assert tp.prompt_for(SimpleNamespace(dtype="mixed", tolerance=None), "adreno650") == \
        mo.opencl_system_prompt("fp32", None, "adreno650")


def test_technique_catalogue_additions():
    t = mo.OPENCL_TECHNIQUES
    names = lambda k: [n for n, _ in t[k]]
    for n in ("fp16_half_arithmetic", "half_vector_io", "fp32_accumulate_vs_fp16_accumulate", "build_options_fast_math",
              "kernel_work_group_limit"):
        assert n in names("*"), n
    assert {"oc_ow_register_tile_half8", "local_memory_weight_tile", "image2d_from_buffer_reads",
            "fuse_epilogue_into_conv"} <= set(names("conv"))
    assert "half8_b_tiles_fp32_acc" in names("gemm") and "subgroup_reduction" in names("reduction")
    assert "grid_stride_vectorized" in names("elementwise") and "half8_elementwise" in names("elementwise")
    assert "memory_coalescing" in names("*")                        # the old entries are still there


def test_knowledge_seed_has_the_new_adreno_techniques():
    import json
    from pathlib import Path
    d = json.loads((Path(__file__).resolve().parents[1] / "data/kernelblaster/knowledge/techniques/60_adreno.json").read_text())
    ids = {t["id"] for t in d["techniques"]}
    assert {"adreno_half8_vector_io", "fp32_accumulate_fp16_operands", "fp16_accumulate_with_flush",
            "oc_ow_register_tile_half8", "image2d_from_buffer_reads", "fast_math_build_options",
            "fuse_epilogue_into_conv", "adreno_half_compute"} <= ids
    for t in d["techniques"]:
        assert t["realizations"] and all(r["scope"].startswith("target:") for r in t["realizations"])
