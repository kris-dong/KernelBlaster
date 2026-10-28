"""Hardware description for the CUDA slot agent: what the target GPU is, which instructions it has, and
which GEMM-class techniques are worth offering on it.

The technique menu used to be scalar-SIMT only; the best kernels in practice are tensor-core pipelines
whose ISA differs per generation (mma.sync on sm_80/86/89, wgmma + TMA on sm_90, tcgen05 + TMEM on
sm_100). ``arch`` is the numeric sm string used everywhere else ("89", "100"; a leading ``sm_`` is accepted).
"""
from __future__ import annotations

from typing import Optional


def _sm(arch: str) -> int:
    s = str(arch).removeprefix("sm_").rstrip("af")
    try:
        return int(s)
    except ValueError:
        return 0


# Dense fp16 tensor-core peak with fp32 accumulate (GFLOP/s), fp32 non-tensor peak, DRAM GB/s.
HW = {
    80: dict(name="NVIDIA A100", family="Ampere", sms=108, smem_kb=164, l2_mb=40, dram_gbs=1555,
             fp32_gflops=19500, fp16_tc_gflops=312000),
    86: dict(name="NVIDIA RTX 3090 class", family="Ampere (GA10x)", sms=82, smem_kb=100, l2_mb=6, dram_gbs=936,
             fp32_gflops=35600, fp16_tc_gflops=142000),
    89: dict(name="NVIDIA L40S", family="Ada Lovelace", sms=142, smem_kb=100, l2_mb=48, dram_gbs=864,
             fp32_gflops=91600, fp16_tc_gflops=362000),
    90: dict(name="NVIDIA H100 SXM", family="Hopper", sms=132, smem_kb=228, l2_mb=50, dram_gbs=3350,
             fp32_gflops=67000, fp16_tc_gflops=989000),
    100: dict(name="NVIDIA B200", family="Blackwell", sms=148, smem_kb=228, l2_mb=126, dram_gbs=8000,
              fp32_gflops=80000, fp16_tc_gflops=2250000),
}

_ISA = {
    "mma": "Tensor cores through `wmma` or inline-PTX `mma.sync.aligned.m16n8k16` (fp16 in, fp32 accumulate) "
           "fed by `ldmatrix` from shared memory; `cp.async` (cp.async.cg / commit_group / wait_group) for "
           "global->shared copies.",
    "hopper": "Also: TMA (`cp.async.bulk.tensor`, tensor maps via cuTensorMapEncodeTiled), mbarrier "
              "(`mbarrier.try_wait`), thread-block clusters with distributed shared memory and multicast, and "
              "warpgroup MMA `wgmma.mma_async.sync.aligned.m64nNk16` (A/B from shared memory descriptors, "
              "accumulators in registers; 4 warps issue one instruction).",
    "blackwell": "Also: 5th-gen tensor cores through `tcgen05.mma` (single-thread issue, A/B from shared "
                 "memory descriptors, accumulator in Tensor Memory (TMEM, 256 KB/SM, `tcgen05.alloc` / "
                 "`tcgen05.ld` / `tcgen05.commit`), CTA-pair 2-SM MMA (`cta_group::2`) with cluster launch, "
                 "TMA with multicast, mbarrier-based pipelines. `mma.sync` still works but runs at a small "
                 "fraction of tcgen05 throughput.",
}


def info(arch: str) -> Optional[dict]:
    sm = _sm(arch)
    return HW.get(sm) or (HW[max(k for k in HW if k <= sm)] if sm >= 80 else None)


def arch_opt(arch: str) -> str:
    """The nvcc argument that selects the target (see knowledge.slots._arch_opt): -gencode=arch=compute_100a,code=sm_100a on Blackwell."""
    sm = _sm(arch)
    return "-gencode=arch=compute_100a,code=sm_100a" if sm == 100 else f"-arch={arch_flag(arch)}"


def arch_flag(arch: str) -> str:
    """nvcc -arch value. Hopper/Blackwell tensor-core instructions (wgmma, tcgen05) only assemble for the
    architecture-specific targets sm_90a / sm_100a."""
    sm = _sm(arch)
    return f"sm_{sm}a" if sm in (90, 100) else f"sm_{sm}"


def peaks(arch: str, dtype: str = "fp32", op_class: str = "") -> Optional[tuple]:
    """(GFLOP/s, GB/s) roofline peaks; the fp16 tensor-core peak for tensor-core op classes."""
    h = info(arch)
    if h is None:
        return None
    tc = dtype in ("fp16", "bf16") and op_class in ("gemm", "conv", "attention")
    return (float(h["fp16_tc_gflops"] if tc else h["fp32_gflops"]), float(h["dram_gbs"]))


def hardware_text(arch: str, dtype: str = "fp32") -> str:
    """The "target hardware" block of the system prompt ("" for archs without a table entry)."""
    h, sm = info(arch), _sm(arch)
    if h is None:
        return ""
    lines = [f"Target GPU: {h['name']} ({h['family']}, compute capability {sm // 10}.{sm % 10}, built with "
             f"{arch_opt(arch)}), {h['sms']} SMs, up to {h['smem_kb']} KB shared memory per SM, "
             f"{h['l2_mb']} MB L2, {h['dram_gbs'] / 1000:.2f} TB/s DRAM; fp32 CUDA-core peak "
             f"{h['fp32_gflops'] / 1000:.0f} TFLOP/s" +
             (f", fp16 tensor-core peak {h['fp16_tc_gflops'] / 1000:.0f} TFLOP/s (dense, fp32 accumulate)."
              if dtype in ("fp16", "bf16") else ".")]
    if dtype in ("fp16", "bf16"):
        isa = [_ISA["mma"]]
        if sm >= 90:
            isa.append(_ISA["hopper"])
        if sm >= 100:
            isa.append(_ISA["blackwell"])
        lines.append("Instructions available: " + " ".join(isa))
        if sm < 90:
            lines.append("Not available on this GPU: wgmma, TMA, thread-block clusters, tcgen05. Do not use them.")
        elif sm < 100:
            lines.append("Not available on this GPU: tcgen05 / Tensor Memory. wgmma + TMA are the fast path.")
        else:
            lines.append("On this GPU the fast path is TMA + tcgen05.mma with TMEM accumulators; a kernel built "
                         "from mma.sync alone will stay far below peak.")
        lines.append("Good practice: aim at a high fraction of the tensor-core peak above, check achieved "
                     "TFLOP/s against it, and keep the fallback path for shapes that do not meet the fast path's "
                     "alignment so every instance stays correct.")
    return "\n".join(lines)


# Extra GEMM-class techniques by minimum sm, fp16 only (fp32 stays on CUDA cores: the tolerance rules out TF32).
_TC = [
    (80, "tensor_core_mma", "Use tensor cores: `mma.sync.m16n8k16` (or wmma) with fp16 inputs and fp32 accumulators, "
                            "operands loaded from shared memory with `ldmatrix`."),
    (80, "multi_stage_cp_async_pipeline", "Pipeline global->shared tile loads with `cp.async` over 3-5 stages "
                                          "(commit_group / wait_group) so loads overlap the MMAs."),
    (80, "smem_swizzle_conflict_free", "XOR-swizzle shared-memory tile addresses (instead of padding) so "
                                       "ldmatrix and cp.async are bank-conflict free."),
    (80, "tile_shape_retune", "Retune the CTA tile (e.g. 128x256x32, 256x128x32, BK=64) and the warp tile "
                              "(64x64) for the SM's shared memory and register budget; check occupancy."),
    (80, "l2_block_rasterization", "Remap block ids (grouped / swizzled rasterization) so concurrently running CTAs "
                                   "share A/B tiles in L2."),
    (80, "smem_staged_epilogue", "Write the accumulator tile through shared memory and store full 16-byte rows "
                                 "(coalesced); fuse any bias/activation here."),
    (80, "register_fragment_double_buffering", "Load the next k-slice's ldmatrix fragments while the current slice's "
                                               "MMAs issue (ping-pong register fragments)."),
    (80, "split_k_or_stream_k", "Split the K loop across CTAs (split-K / stream-K with a reduction) when the output "
                                "tile count does not fill all SMs."),
    (90, "tma_bulk_tensor_loads", "Replace cp.async with TMA bulk tensor copies (cp.async.bulk.tensor + mbarrier), "
                                  "one thread issuing whole-tile loads with hardware swizzle."),
    (90, "wgmma_warpgroup_mma", "Use warpgroup MMA (`wgmma.mma_async` m64nNk16, operands from shared-memory "
                                "descriptors, accumulators in registers), overlapped with the next loads."),
    (90, "warp_specialization", "Split warps into a producer (TMA loads) and consumer warpgroups (MMA) "
                                "synchronised with mbarriers; use setmaxnreg to shift registers to the consumers."),
    (90, "cluster_multicast", "Launch thread-block clusters and TMA-multicast A/B tiles to several CTAs, cutting "
                              "L2 traffic."),
    (90, "persistent_tile_scheduler", "Make the kernel persistent (one CTA per SM looping over tiles, "
                                      "overlapping one tile's epilogue with the next tile's loads)."),
    (100, "tcgen05_mma_tmem", "Use 5th-gen tensor cores: `tcgen05.mma` issued by one thread, A/B from shared "
                              "memory descriptors, accumulator in Tensor Memory (tcgen05.alloc/ld/commit); "
                              "epilogue reads TMEM with tcgen05.ld."),
    (100, "cta_pair_2sm_mma", "Use CTA-pair MMA (`cta_group::2`) across a 2-CTA cluster to double the effective "
                              "tile (e.g. 256x256) per MMA and halve B traffic."),
]


# Attention-class (fused attention slots, e.g. ModelBlaster's attention_f16), fp16 tensor cores.
_TC_ATTENTION = [
    (80, "fa2_register_online_softmax", "FlashAttention-2 structure: per CTA 64-128 query rows, Q fragments kept in "
                                        "registers, K/V tiles streamed through cp.async double-buffered smem, "
                                        "S = QK^T and O += PV with mma.sync.m16n8k16 (fp32 accumulate), running "
                                        "row max / sum in fp32 via quad shuffles on the accumulator fragments, P "
                                        "re-packed to fp16 A-fragments in registers; normalise O once at the end."),
    (80, "causal_tile_skipping", "With a causal mask stop each query tile's key loop at the diagonal tile and apply "
                                 "the mask only there: half the FLOPs of the dense loop."),
    (80, "strided_qkv_in_place", "Read q/k/v straight out of the fused projection buffer through their row / head "
                                 "strides and write O in the consumer's layout: no split / transpose copies."),
]


# Recurrent class (gru_seq / lstm_seq with ModelBlaster --recurrent-class on), fp16: the per-step h W_hh^T product.
_TC_RECURRENT = [
    (80, "whh_mma_fragments_in_registers", "Load each block's W_hh slice once as mma.sync.m16n8k16 A-fragments kept "
                                           "in registers for all timesteps; build h_{t-1} B-fragments from the "
                                           "exchange buffer; fp32 accumulators feed the gate math directly."),
]


def recurrent_text(arch: Optional[str]) -> str:
    """Notes for a slot of the recurrent class: what bounds a sequence recurrence on this GPU."""
    h = info(arch) if arch else None
    sms = f"{h['sms']} SMs" if h else "all SMs"
    return ("## Recurrent op notes\n"
            "This op is a sequence recurrence: T timesteps, each depending on the previous hidden state. Its time is "
            "T x (latency of one step), not FLOPs or DRAM bytes; the roofline percentages above will look tiny. "
            "What costs time per step: waiting for other blocks' h_{t-1} through L2 (a few hundred cycles per "
            "round trip), the h W_hh^T product, the gate math, and publishing h_t. Measured on a fast L40S GRU kernel "
            "(1,650 cycles/step): ~640 waiting on L2 for h, ~215 mma + barrier, ~600 gate math + publish.\n"
            "- Keep W_hh and h on chip across steps (one persistent launch); hoist the input projection X W_ih^T for "
            "all steps into one GEMM; exchange h point-to-point (tagged words / flags with release/acquire), not with "
            "grid-wide barriers.\n"
            f"- Spinning blocks must all be co-resident: size the persistent grid to the resident capacity of the SMs "
            f"you may use ({sms} when the op runs alone; call mb_sm_budget() when the harness declares it, since the "
            "op may run beside a concurrent sibling such as the other direction). A non-resident producer = deadlock.\n"
            "- Scratch (gate buffers, flags, step counters) must be slot-owned and safe to reuse across calls "
            "(epoch / step tags instead of resets), because the op is called repeatedly and may be captured in a "
            "CUDA graph.")


def tensor_core_techniques(arch: str, dtype: str, op_class: str) -> list[tuple[str, str]]:
    """GEMM-class techniques valid on ``arch`` (empty unless fp16 gemm/conv/attention on sm_80+; the recurrent
    class gets only its W_hh fragment technique)."""
    sm = _sm(arch)
    if op_class == "recurrent":
        return [(n, d) for lo, n, d in _TC_RECURRENT if sm >= lo] if dtype in ("fp16", "bf16") else []
    if dtype not in ("fp16", "bf16") or op_class not in ("gemm", "conv", "attention") or sm < 80:
        return []
    out = [(n, d) for lo, n, d in _TC if sm >= lo]
    if op_class == "attention":         # flash-style structure first: the GEMM techniques apply inside it
        out = [(n, d) for lo, n, d in _TC_ATTENTION if sm >= lo] + out
    if sm >= 100:   # mma.sync-era techniques that tcgen05 kernels replace
        drop = {"register_fragment_double_buffering"}
        out = [t for t in out if t[0] not in drop]
    return out
