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
        ("local_memory_tiling", "Tile A and B through __local memory (e.g. 16x16 or 32x8 tiles, barrier between "
                                "phases) so each element is read from DRAM once per tile."),
        ("register_tiling", "Each work-item computes a small TMxTN block of C in registers (outer products), "
                            "float4 loads of the A/B strips."),
        ("double_buffering", "Ping-pong two __local tiles: load the next K-tile while computing the current one."),
    ],
    "conv": [
        ("implicit_gemm", "Treat the convolution as a GEMM (M=N*OH*OW, N=OC, K=IC*KH*KW) without materialising "
                          "im2col; tile it through __local memory."),
        ("local_memory_tiling", "Stage an input patch and a filter tile in __local memory and reuse them across "
                                "output channels / pixels of the work-group."),
        ("register_blocking", "Each work-item computes several output channels and/or pixels, reusing loaded inputs "
                              "from registers."),
    ],
    "reduction": [
        ("cooperative_local_reduction", "One work-group per reduced row; each work-item accumulates a strided "
                                        "partial (mb_kadd), then a __local tree reduction."),
        ("single_pass_fusion", "Compute statistics and the normalised output in as few passes over memory as "
                               "possible (e.g. online max/sum for softmax)."),
        ("two_stage_reduction", "Split long reductions across work-groups (partials in MB_SCRATCH), then a second "
                                "small kernel combines them."),
    ],
    "elementwise": [
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
