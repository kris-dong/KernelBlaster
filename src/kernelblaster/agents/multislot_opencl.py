# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCL / Qualcomm Adreno assets for the multi-slot agent (``target: "opencl"``).

ModelBlaster's OpenCL slot (``pipeline/kb_export_opencl.py``) is C++11: its
OpenCL C programs as raw strings plus the host launcher with the op's exact
signature, on ModelBlaster's runtime (SVM pointers, ``mb_launch``, cl_event
timing). The technique catalogue folds in what the Adreno RL flow learned
(``backends/opencl.py`` technique map, ``docs/opencl_rl_techniques_summary.md``):
128-bit vector I/O, __local tiling and cooperative reductions, register
tiling, work-group size tuning, baked compile-time constants, masked tails.
"""
from __future__ import annotations

import os
from typing import Optional

# Peak fp32 GFLOP/s and DRAM GB/s per device, for the roofline state.
# Adreno numbers are approximate public figures (ALU count x clock; LPDDR bus).
OPENCL_PEAKS = {
    "adreno650": (1228.0, 44.0),      # QRB5165 / Snapdragon 865 (RB5), LPDDR5
    "adreno660": (1720.0, 51.0),      # Snapdragon 888
    "adreno730": (2200.0, 51.0),      # Snapdragon 8 Gen 1
    "adreno740": (3000.0, 67.0),      # Snapdragon 8 Gen 2
    "adreno750": (4600.0, 77.0),      # Snapdragon 8 Gen 3
    "titanrtx": (16300.0, 672.0),     # local NVIDIA OpenCL (functional stand-in; times are not Adreno's)
}
DEFAULT_ARCH = "adreno650"

OPENCL_TECHNIQUES: dict[str, list[tuple[str, str]]] = {
    "*": [
        ("fp16_half_arithmetic", "On a cl_khr_fp16 device (Adreno 650: yes, 2x the fp32 ALU rate on paper) use half / half2 / half4 / "
                                 "half8 for data and for exact cheap math (relu, clamp, add, mul); widen to float (convert_float8) for "
                                 "transcendentals, long sums and ops with float scalar parameters, round once on store (convert_half8). "
                                 "Measured on this board: half arithmetic was NOT faster than fp32 math on half data for convolutions "
                                 "(four sessions), and a half fma() is emulated and ~80x slower than an fp32 mad; the wins came from half "
                                 "VECTOR I/O, register tiling and layout."),
        ("half_vector_io", "Read and write half data through direct pointers with vload8 / vstore8 (one 128-bit transaction) and "
                           "vload4 / vstore4, not vload_half / vstore_half; keep a scalar tail for n % 8."),
        ("fp32_accumulate_vs_fp16_accumulate", "Default: fp16 operands, fp32 accumulators (the product of two halves is exact in fp32). "
                                               "Pure or per-ic half accumulation (half8 mad flushed to fp32 every 9-32 terms) is allowed when the "
                                               "per-slot bound still holds; QNN's FP16 mode does this, its HYBRID mode keeps fp32 accumulators. "
                                               "Check the rel_err the evaluator prints: 1.5e-3 against a 2e-3 bound for a 576-term conv with flush 16; "
                                               "6.4e-4 with a flush per input channel (cc_conv_min_add_mul: 6% faster); 2.4e-3 over 9 terms of one "
                                               "channel group failed (cc_conv_relu_bias); it was slower or equal in three other measurements."),
        ("build_options_fast_math", "MB_CL_OPTS(\"-cl-fast-relaxed-math\" / \"-cl-unsafe-math-optimizations\" / \"-cl-finite-math-only\" / "
                                    "\"-cl-denorms-are-zero\", src) for kernels with no compensated sums or NaN handling (they would "
                                    "reassociate mb_kadd); -cl-mad-enable is not forced; try one option at a time and keep the check green."),
        ("kernel_work_group_limit", "Read MB_KINFO (wg_max, pref_mult, private_mem) from the profile: a register-heavy kernel gets a smaller "
                                    "max work-group; size mb_range local sizes as multiples of pref_mult (64) within wg_max, and treat a "
                                    "falling wg_max or private_mem > 0 as spilling."),
        ("memory_coalescing", "Consecutive work-items touch consecutive addresses; reorder the work-item->element mapping."),
        ("vectorized_memory_access", "128-bit transactions: vload4/vstore4 (float4) on aligned runs, with a masked "
                                     "scalar tail for the remainder (the Adreno RL flow's most reliable win)."),
        ("work_group_size_tuning", "Choose the local size for Adreno (waves of 64/128 fibers): try 64/128/256 and 2-D "
                                   "shapes; keep it within the kernel's max work-group size."),
        ("thread_coarsening", "Give each work-item several outputs (registers hold the partial results) to amortise "
                              "index math and launch overhead."),
        ("compile_time_constants", "Bake loop bounds / shape scalars the listed instances share into the program "
                                   "(MB_CL_OPTS(\"-DK=...\", ...) or a specialised kernel chosen at launch) so the "
                                   "compiler unrolls; keep a generic path for other shapes."),
        ("native_math_and_mad", "Use mad()/fma(), native_exp/native_recip where the 1e-3 check allows, and "
                                "-cl-mad-enable via MB_CL_OPTS."),
    ],
    "gemm": [
        ("half8_b_tiles_fp32_acc", "Each work-item: 4-8 rows x 8 columns of C in float8 accumulators, half4 reads of A, half8 reads of "
                                   "a K-major (pre-transposed once per call) B / weight matrix; bias in the epilogue."),
        ("image_weight_reads", "Pre-pack the weight matrix into MB_SCRATCH_IMAGE(prog, idx, w, h, CL_RGBA, CL_HALF_FLOAT) (4 halves per "
                               "texel) and read it with read_imageh: the texture path has its own cache."),
        ("local_memory_tiling", "Tile A and B through __local memory (e.g. 16x16 or 32x8 tiles, barrier between "
                                "phases) so each element is read from DRAM once per tile."),
        ("register_tiling", "Each work-item computes a small TMxTN block of C in registers (outer products), "
                            "float4 loads of the A/B strips."),
        ("double_buffering", "Ping-pong two __local tiles: load the next K-tile while computing the current one."),
    ],
    "conv": [
        ("oc_ow_register_tile_half8", "Work-item = 4 output pixels x 8 output channels (float8 accumulators, weights re-laid out as "
                                      "[tap][OC] and read as half8, inputs as half4 / a sliding window); QNN-class kernels also tile IC."),
        ("local_memory_weight_tile", "Stage a [tap][OC-block] weight tile (and an input row strip) in __local once per work-group and "
                                     "reuse it across the work-group's pixels; barrier between phases, <= 32 KB."),
        ("image2d_from_buffer_reads", "Write the re-laid-out weights (or a staged input) into MB_SCRATCH_IMAGE(...).buf with a pack "
                                      "kernel, then read it as image2d_t (cl_khr_image2d_from_buffer) with read_imageh; "
                                      "cl_qcom_accelerated_image_ops / cl_qcom_vector_image_ops add fused image dot / filter ops."),
        ("fuse_epilogue_into_conv", "Where a fusion hint is accepted, apply bias / BatchNorm scale+shift / ReLU / residual add in the "
                                    "conv's store epilogue instead of separate kernels (QNN folds BN into the conv and fuses ReLU / Add)."),
        ("implicit_gemm", "Treat the convolution as a GEMM (M=N*OH*OW, N=OC, K=IC*KH*KW) without materialising "
                          "im2col; tile it through __local memory."),
        ("local_memory_tiling", "Stage an input patch and a filter tile in __local memory and reuse them across "
                                "output channels / pixels of the work-group."),
        ("register_blocking", "Each work-item computes several output channels and/or pixels, reusing loaded inputs "
                              "from registers."),
    ],
    "reduction": [
        ("subgroup_reduction", "With cl_khr_subgroups (OpenCL C 2.0): sub_group_reduce_add / _max (and cl_qcom_reqd_sub_group_size to pin the "
                               "wave to 64 or 128) replace the __local tree for the first 64 lanes; then one __local step across waves."),
        ("cooperative_local_reduction", "One work-group per reduced row; each work-item accumulates a strided "
                                        "partial (mb_kadd), then a __local tree reduction."),
        ("single_pass_fusion", "Compute statistics and the normalised output in as few passes over memory as "
                               "possible (e.g. online max/sum for softmax)."),
        ("two_stage_reduction", "Split long reductions across work-groups (partials in MB_SCRATCH), then a second "
                                "small kernel combines them."),
    ],
    "elementwise": [
        ("half8_elementwise", "8 elements per work-item with vload8 / vstore8 on half data and half8 arithmetic for relu / clamp / add / mul."),
        ("grid_stride_vectorized", "Fewer work-items, each looping over float4 chunks; minimal integer math per element."),
    ],
    "data_movement": [
        ("index_math_hoisting", "Precompute strides/divisions; fewer integer divides per element (mad24 for index math)."),
    ],
    "attention": [
        ("flash_style_tiling", "Tile Q/K/V through __local memory with an online softmax so the score matrix is "
                               "never materialised."),
    ],
    "scan": [
        ("work_group_scan", "Parallel scan: per-work-group __local scan of a chunk, then a pass that adds chunk "
                            "offsets (compensated sums for long axes)."),
    ],
}

SYSTEM_PROMPT_OPENCL = """You optimise one OpenCL *slot* inside a larger graph partition that ModelBlaster \
compiled for a Qualcomm Adreno GPU (OpenCL 2.0 host API, OpenCL C 1.2 kernels). The partition calls this \
slot's entry function once per op instance listed below; other slots are separate kernels you cannot change. \
Rewrite the slot to be faster while staying correct.

The slot is C++11. OpenCL C programs are raw strings built on first use:
  static mb_program cl_x = MB_CL(R"CL( ...OpenCL C... )CL");
  static mb_program cl_y = MB_CL_OPTS("-cl-mad-enable -DTILE=16", R"CL( ... )CL");
The device prelude defines MB_THREADS (256), MB_TID ((long)get_global_id(0)), mb_ksum / mb_ksum0() / \
mb_kadd(&s, v) (compensated fp32 sum), mb_idx6(i, d1..d5, long c[6]) and mb_atomic_addf.

Rules for the slot source you return:
- Define the entry function with EXACTLY the given signature (same name, parameter types and order). \
Pointers are device (SVM) pointers: pass them straight to kernels as __global arguments; pointer arithmetic \
is fine. Scalars are runtime shapes, so handle every listed instance shape (and stay correct for other sizes: \
guard bounds).
- Launch only through mb_launch(MB_KERNEL(cl_x, kernel_name), range, args...). range: mb_grid(n[, local]) \
(one work-item per element, rounded up to whole work-groups), mb_groups(groups, local) (like CUDA \
<<<groups, local>>>), mb_range2(gx, gy, lx, ly), mb_range3(...). Arguments must match the kernel's \
parameters in order and type (int vs long matters; float literals need an f suffix -- double does not \
compile); mb_local{bytes} passes a dynamic __local buffer. Device copies: mb_copy(dst, src, bytes), \
mb_zero(p, bytes). Temporary device memory: MB_SCRATCH(cl_x, bytes) (grow-only, per program). Never call \
clEnqueue*/clFinish/clCreate* yourself (the driver times every command through mb_launch), do not copy \
to/from the host, do not define main() or launch_gpu_implementation.
- OpenCL C 1.2: no double (Adreno has no fp64), no recursion, no variable-length arrays, pointer \
parameters of helper functions need __global/__local qualifiers. A work-group may not exceed the kernel's \
maximum work-group size (on Adreno often 256 or less for register-heavy kernels); global sizes must be \
multiples of the local size (mb_grid/mb_groups round up).
- The source is placed inside a C++ namespace: helpers are fine. #include lines are hoisted out; #undef \
anything you #define on the host side.
- fp32 in/out. Each op is checked alone against the reference: max error <= 1e-3 relative to the largest \
input/output magnitude. Accumulate long sums with mb_kadd or tree reductions.
- Adreno: prefer 128-bit vector loads/stores (vload4/vstore4) with a masked tail, __local tiling with \
barrier(CLK_LOCAL_MEM_FENCE), several outputs per work-item, mad()/native_* math where accuracy allows; \
__local memory is small (32 KB on Adreno 650).
- Return the COMPLETE slot source in a single ```cpp code block, nothing after it."""


# ---------------------------------------------------------------------------------------------------------
# Hardware-aware prompt (this change). Switches (environment, read per call):
#   MB_OPENCL_F16_MATH  0 / 1 / auto (default auto = on for the cl_khr_fp16 Adreno devices below): the fp16 rule
#                       of the prompt allows and encourages half arithmetic (the seeds are generated by ModelBlaster
#                       with the same switch: pipeline/opencl_seeds.py:f16_math_enabled);
#   MB_OPENCL_HW_INFO   1 (default) / 0: append the "Target hardware" block below (device facts, public peaks, what
#                       QNN's GPU backend does, calibration).
# With both off (or an arch without a table entry and no fp16 math) the prompt is SYSTEM_PROMPT_OPENCL, byte for byte.
# ---------------------------------------------------------------------------------------------------------
F16_MATH_ARCHS = {"adreno650", "adreno660", "adreno730", "adreno740", "adreno750"}


def _flag(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().lower()


def f16_math_enabled(arch: Optional[str] = None) -> bool:
    v = _flag("MB_OPENCL_F16_MATH", "auto")
    if v in ("0", "off", "false", "no"):
        return False
    if v in ("1", "on", "true", "yes"):
        return True
    return (arch or os.environ.get("MB_OPENCL_ARCH", "")).strip().lower() in F16_MATH_ARCHS


def hw_info_enabled() -> bool:
    return _flag("MB_OPENCL_HW_INFO", "1") not in ("0", "off", "false", "no")


# Measured on the board with cc_devinfo.py (l3_farm/qrb_a6/cc_runs/tools), 2026-10-29; peaks are PUBLIC SPECS, not measured.
OPENCL_HARDWARE = {
    "adreno650": """\
Device (measured on the board): QUALCOMM Adreno(TM) 650 in the QRB5165 (Snapdragon 865 class), OpenCL 2.0 / OpenCL C 2.0, \
1 GB global memory, 32 KB __local per work-group, max work-group 1024 (less for register-heavy kernels: read wg_max from the \
profile), global cache 128 KB with 64-byte lines, 64-wide waves, images supported (image2d width up to 16384), coarse-grained SVM.
Extensions the device exposes (all usable from a slot): cl_khr_fp16 (half arithmetic; the device reports native / preferred half vector width 1, \
yet vload4 / vload8 on half data measured faster than scalar and vload_half* loads), cl_khr_subgroups, cl_qcom_reqd_sub_group_size, \
cl_qcom_subgroup_shuffle, cl_khr_image2d_from_buffer, cl_khr_3d_image_writes, cl_qcom_accelerated_image_ops, \
cl_qcom_vector_image_ops, cl_qcom_dot_product8 (int8 dot products), cl_qcom_ml_ops, cl_khr_byte_addressable_store, \
cl_khr_global/local_int32_{base,extended}_atomics, cl_qcom_ext_host_ptr / cl_qcom_ion_host_ptr (zero-copy host buffers), \
cl_qcom_recordable_queues, cl_qcom_perf_hint. No fp64.
Peaks (public specs, not measured): ~1.2 TFLOP/s fp32, about 2x that for fp16 arithmetic (half ALU rate 2x), LPDDR5 ~44 GB/s. \
64-bit integer divide / modulo (mb_idx6) and a half fma() were measured very slow: use 32-bit index math and fp32 mad() for \
fused multiply-adds on half data.
How Qualcomm's own runtime (QNN GPU backend, QAIRT SDK) reaches the hardware on this device -- the design space you may use too: \
it runs the model as FP16 (precision modes FP16 = fp16 accumulators, HYBRID = fp16 data with fp32 accumulators, FP32), fuses \
BatchNorm, ReLU/Relu6, bias and residual Add into the convolution kernels (a conv + bn + relu + add block is one or two kernels), \
records the command queue (documented), and -- not documented in the SDK, inferred from the extensions the device exposes \
and from its speed -- most likely uses texture / image paths (image2d_from_buffer, accelerated and vector image ops) and subgroup operations, and its optional tuning mode benchmarks several kernel variants per op and keeps the best. Its GPU backend \
supports (SDK doc OpDef/SupportedOps, GpuOpDefSupplement; fp32 and fp16 tensors): Conv2d, DepthWiseConv2d, TransposeConv2d, \
FullyConnected, MatMul, Batchnorm, InstanceNorm, GroupNorm, LayerNorm, RmsNorm, Softmax, PoolAvg2d/PoolMax2d, Relu/Relu6/\
Gelu/Elu/HardSwish/Sigmoid/Tanh/Prelu, the ElementWise binary/unary family, Reduce{Sum,Mean,Max,Min,Prod}, Concat, Pad, Resize*, \
Transpose, Gather, TopK, Lstm; not Conv1d/Conv3d, Gru, pooling 3d, TransposeConv1d/3d. 
Calibration from measurements on this board: the previous seed-level kernels (direct fp32-math NCHW convolution, one work-item per \
output, scalar loads) reach 1-3 % of the peak (4-32 GFLOP/s); QNN's convolutions on the same shapes reach 330-900 GFLOP/s, i.e. \
there is 10-100x headroom, and it is in data layout, register tiling and fp16 vector loads, not in tweaking the scalar loop. \
Hand-directed sessions on this board (cc_runs/*/REPORT.md) reached 65-136 GFLOP/s on 3x3 convolutions with: weights re-laid out \
[tap][OC] by a small kernel in the same call and read as half8, a tile of 4 output pixels x 4 or 8 channels per work-item in fp32 \
accumulators written as named scalars / vectors (private float arrays: 3.4x slower; full unrolling: spills, 20x slower; 8 channels was \
1.8x slower than 4 at IC = 8 and 1.6x faster than 4 at IC = 64), native half4 / half8 loads (vload4 / vload8 on the half pointer with \
cl_khr_fp16: conv 47 -> 36 ms, elementwise 566 -> 239 us). Things that did NOT help or were harmful: vload_half8 (slower than scalar \
vload_half; vload_half4/8 + vstore_half4 made one driver run print nothing), half arithmetic accumulation (equal or slower, 2.4e-3 error \
over 9 terms), a half fma() (emulated: compile 12 s / 80x slower), 6-d broadcast index math with 64-bit div / mod and an opcode switch \
containing pow() (1 s for a 2 M-element bias add; per-opcode kernels with 32-bit math: 0.5 ms), work-group shape changes. \
First fp16-math seeds of this harness (half8 weight reads, 4 pixels x 8 channels, fp32 accumulators): 3x3 conv ~70 GFLOP/s, 1024^3 \
GEMM ~50 GFLOP/s, half8 maps ~33 GB/s: a starting point, still far below QNN. Untried so far: images / texture reads, __local weight \
tiles, subgroups, implicit GEMM, int8 dot products, fusing the elementwise epilogue into the convolution.""",
}

OPENCL_DTYPE_RULES = {
    "fp32": "- fp32 in/out. Each op is checked alone against the reference: max error <= {slot_rel} relative to the largest "
            "input/output magnitude. Accumulate long sums with mb_kadd or tree reductions.",
    "fp16_math": "- fp16 problem: every float pointer is `__half` in the host signature and `half` in kernels. The device has "
                 "cl_khr_fp16 and the runtime enables it (`#pragma OPENCL EXTENSION cl_khr_fp16 : enable` is prepended; "
                 "MB_HAS_FP16 is defined): HALF ARITHMETIC IS ALLOWED AND ENCOURAGED. Read and write half data through the pointer "
                 "with vload8/vload4/vstore8/vstore4 (128-bit transactions) and half2/half4/half8 operators; "
                 "convert_float8()/convert_half8() move between precisions (rounding to nearest even); vload_half/vstore_half still "
                 "work. Half ALUs run at twice the fp32 rate. Accumulation policy: the default is fp16 operands with fp32 "
                 "accumulators (exact products, no loss); accumulating in half is allowed if the check still passes -- flush half "
                 "partial sums into fp32 every 16-32 terms for long dot products (convs, GEMM K, reductions); fp16 transcendentals "
                 "(exp, tanh, erf) are less accurate: compute them in float. Round to half once on store.\n"
                 "- Each op is checked alone against a reference that rounds the exact fp32 result to fp16 once: max error <= "
                 "{slot_rel} relative to the largest input/output magnitude (about two fp16 ulps). Do not loosen it: a kernel that "
                 "fails the check is rejected.",
    "fp16_storage": "- fp16 storage: every float pointer is `__half` (binary16). Compute in fp32 (vload_half/vstore_half, one rounding "
                    "per stored value). Each op is checked alone against a reference that rounds the exact fp32 result to fp16 once: "
                    "max error <= {slot_rel} relative to the largest input/output magnitude.",
}

_OLD_STD = "(OpenCL 2.0 host API, OpenCL C 1.2 kernels)"
_OLD_C12 = ("- OpenCL C 1.2: no double (Adreno has no fp64), no recursion, no variable-length arrays, pointer "
            "parameters of helper functions need __global/__local qualifiers. ")
_OLD_DTYPE = ("- fp32 in/out. Each op is checked alone against the reference: max error <= 1e-3 relative to the largest "
              "input/output magnitude. Accumulate long sums with mb_kadd or tree reductions.")
_OLD_ADRENO = "- Adreno: prefer 128-bit vector loads/stores (vload4/vstore4) with a masked tail,"
_OLD_LAUNCH = "- Launch only through mb_launch"


def _sub(text: str, old: str, new: str) -> str:
    if old not in text:                      # the base prompt changed: fail loudly instead of silently dropping a rule
        raise ValueError(f"SYSTEM_PROMPT_OPENCL no longer contains {old[:50]!r}")
    return text.replace(old, new, 1)


def _bound(tolerance: Optional[dict]) -> str:
    rel = (tolerance or {}).get("slot_rel", 1e-3)
    s = f"{rel:.0e}" if float(f"{rel:.0e}") == rel else f"{rel:.1e}"
    return s.replace("e-0", "e-")


def opencl_system_prompt(dtype: str = "fp32", tolerance: Optional[dict] = None, arch: Optional[str] = None) -> str:
    """The OpenCL slot prompt for a problem of ``dtype`` on device ``arch`` (e.g. "adreno650"). The legacy prompt
    (``SYSTEM_PROMPT_OPENCL``, byte for byte) when fp16 math and the hardware block are both off for this arch."""
    arch_k = (arch or os.environ.get("MB_OPENCL_ARCH", "")).strip().lower()
    math = f16_math_enabled(arch_k)
    hw = hw_info_enabled() and arch_k in OPENCL_HARDWARE
    if not math and not hw:
        return SYSTEM_PROMPT_OPENCL
    text = SYSTEM_PROMPT_OPENCL
    if arch_k in F16_MATH_ARCHS or arch_k in OPENCL_HARDWARE:        # OpenCL C 2.0 devices
        text = _sub(text, _OLD_STD, "(OpenCL 2.0 host API; the kernels are built as OpenCL C 2.0 on this device, 1.2 where the "
                            "device has no 2.0; $MB_CL_STD overrides)")
        text = _sub(text, _OLD_C12, "- OpenCL C 2.0: no double (Adreno has no fp64), no recursion, no variable-length arrays; the generic "
                            "address space and cl_khr_subgroups (sub_group_reduce_add, sub_group_broadcast, ...) are available. Use "
                            "32-bit index math where sizes fit (64-bit divide / modulo is emulated and slow on Adreno). ")
        text = _sub(text, _OLD_LAUNCH, "- Build options: MB_CL_OPTS(opts, src) takes any option (-cl-fast-relaxed-math, "
                            "-cl-unsafe-math-optimizations, -cl-finite-math-only, -cl-denorms-are-zero, -DNAME=value); -cl-mad-enable is not "
                            "forced. Fast-math reassociates mb_kadd's compensated sum. Images: the slot's ABI pointers are SVM, but "
                            "`mb_image im = MB_SCRATCH_IMAGE(prog, idx, w, h, CL_RGBA, CL_HALF_FLOAT);` gives a slot-owned buffer "
                            "(`im.buf`, pass as a __global pointer argument to write it; rows are `im.pitch_bytes` apart) and an image2d "
                            "view of it (`im.img`, pass as an image2d_t argument and read it with read_imageh / read_imagef; "
                            "cl_khr_image2d_from_buffer): pack weights or staged inputs with one kernel, read them as an image in the "
                            "next. Still never call clCreate*/clEnqueue*/clFinish yourself.\n- Launch only through mb_launch")
    rule = (OPENCL_DTYPE_RULES["fp16_math"] if math and dtype == "fp16" else
            OPENCL_DTYPE_RULES["fp16_storage"] if dtype == "fp16" else OPENCL_DTYPE_RULES["fp32"])
    text = _sub(text, _OLD_DTYPE, rule.format(slot_rel=_bound(tolerance)))
    text = _sub(text, _OLD_ADRENO, "- Adreno: prefer 128-bit vector loads/stores (vload8/vstore8 on half, vload4/vstore4 on float) with a masked tail,")
    if hw:
        text += "\n\n## Target hardware\n" + OPENCL_HARDWARE[arch_k]
    return text
