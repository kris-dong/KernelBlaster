## Usage Guidelines for LLM Integration (RISC-V + Zephyr on FPGA / spike)

### For State Analysis
When analyzing modelblaster profile output for a RISC-V target, the LLM should:
1. Extract per-op cycle counts from the `MODELBLASTER_PROFILE_BEGIN/END` CSV block
   (`dispatch_id,name,op,shape,cycles`) — one row per dispatch.
2. Identify the *dominant op* first (>= 50% of total cycles); classify that op's
   bottleneck rather than the whole model. Small elementwise ops (relu, add,
   sigmoid) rarely limit e2e throughput and their bottleneck class is
   uninteresting.
3. When IPC is available (see "Extended Metrics" below), IPC < 0.4 on an
   in-order Rocket / Saturn core strongly indicates memory-bound (load-use
   stalls, cache misses, or bad prefetch). IPC > 1.2 (with `V` extension) or
   > 0.9 (scalar) indicates the compute pipeline is well-fed.
4. Map to the closest seed state (below); when nothing matches, create a
   `discovered_<bottleneck>_<n>` state (the database auto-inherits techniques
   at 0.8× confidence from the best-matching seed).
5. Account for the SoC's memory hierarchy — the target here is a Rocket-family
   in-order core, small L1s, no L2 by default; behavior differs sharply from
   OoO CPUs.

### For Optimization Selection
When selecting optimizations, the LLM should:
1. Prefer techniques whose applicable states include the current state name.
2. If the target has RVV (`rv64gcv*` isa), score V-extension techniques
   higher — vectorization is the largest single lever on this class of core.
3. Score cache-hierarchy techniques (tiling, SoA layout, prefetch) higher when
   the op's working set exceeds the core's L1D capacity (see "Architecture
   Quick Reference").
4. Avoid host-only helpers: no `<math.h>` transcendentals unless linkage against
   picolibc's libm is guaranteed; no `printf`-in-the-hot-loop; no
   `malloc` (Zephyr's minimal libc has no heap by default).
5. Watch precision — fp16 techniques require the Zephyr build to have been
   promoted (`QUANT=fp16` → `_f16` backend variant), which turns on `Zfh`/`Zvfh`.

### For Performance Prediction
The predictions below come from a mix of author-tuned defaults and cross-target
inheritance (from CUDA techniques where the underlying idea transfers). Treat
them as coarse priors, not calibrated values — the RL loop updates
`actual_improvement` as it observes real speedups on this SoC.

---

## RVV Canonical Idioms (single-instruction forms — prefer these)

**RVV 1.0 intrinsic naming** — this toolchain requires ALL vector
intrinsics to be prefixed with `__riscv_`. The pre-1.0 unprefixed names
(`vfmax_vf_f32m8` etc.) fail with `implicit declaration of function` at
compile time; only `__riscv_vfmax_vf_f32m8` and its kin are recognised.

The RVV V extension has direct scalar-argument intrinsics for the common
elementwise patterns. Emitting `__riscv_vmfgt` + `__riscv_vmerge` when a
single-op form exists is a common LLM mistake — it triples the vector-
unit work for no functional gain.

| Op | ONE-OP form | Wrong (3-op) form to avoid |
|---|---|---|
| `y = max(x, 0)` (ReLU) | `__riscv_vfmax_vf_f32m8(x, 0.0f, vl)` | `__riscv_vmfgt_vf_*` + `__riscv_vmerge_vvm_*` |
| `y = min(x, cap)` (clamp-max) | `__riscv_vfmin_vf_f32m8(x, cap, vl)` | `__riscv_vmflt_vf_*` + `__riscv_vmerge_vvm_*` |
| `y = max(x, y)` (elementwise max) | `__riscv_vfmax_vv_f32m8(x, y, vl)` | mask + merge |
| `y = |x|` | `__riscv_vfabs_v_f32m8(x, vl)` | shift + mask hack |
| `y = a * b + c` (FMA) | `__riscv_vfmacc_vv_f32m8(c, a, b, vl)` | `__riscv_vfmul` + `__riscv_vfadd` (2 ops, but breaks fma fusion) |
| `s = sum(x[:])` | `__riscv_vfredosum_vs_f32m8_f32m1(...)` | scalar reduction loop |
| `m = max(x[:])` | `__riscv_vfredmax_vs_f32m8_f32m1(...)` | scalar max loop |
| `y = sign(a) * |b|` | `__riscv_vfsgnj_vv_f32m8(b, a, vl)` | bit manipulation |
| `y = x * scale` (widen memory op) | LMUL=m8 stripmine | LMUL=m2 (misses free bandwidth) |

For any comparison-with-constant that reduces to max/min/clamp/abs/sign,
there is a direct `__riscv_vf*_vf_*` intrinsic. Reach for it FIRST.

Load/store patterns:
- `__riscv_vsetvl_e32m8(n - i)` — request VL for remaining elements
- `__riscv_vle32_v_f32m8(&input[i], vl)` — stride-1 load
- `__riscv_vse32_v_f32m8(&output[i], v, vl)` — stride-1 store
- `__riscv_vle32_v_f32m8_m(mask, &input[i], vl)` — masked load

---

## Memory-vs-Compute Bottleneck Triage (READ BEFORE PICKING A TECHNIQUE)

Before applying any technique, estimate the op's arithmetic intensity:

    intensity = FLOPs / (bytes_read + bytes_written)

For a Rocket + Saturn (DLEN=128, VLEN=256) config with ~1-2 GB/s
effective DDR bandwidth to the target frequency, the compute peak is
~4-8 fp32 GFLOP/s. Break-even intensity is ~2-4 FLOP/byte. Below that
line, the vector unit will sit idle waiting for DDR — compute
optimisations (RVV, unroll, pipeline) CANNOT help; the profile just
shows fewer instructions retired but the same wall time.

Common workload placement:

| Op class | Intensity (FLOP/B) | Regime | Best-shot techniques |
|---|---:|---|---|
| ReLU, clamp, add, mul (pure elementwise fp32) | ~0.25 | **memory-bound** | prefetch, streaming stores, fusion into neighbour |
| Sigmoid, tanh (transcendental elementwise) | ~5 (limb-bound) | compute-bound | polynomial approx, RVV vectorise |
| Reduce sum / max (single-pass) | ~0.5 | memory-bound | tree reduction, `vfredosum_vs` |
| GEMM (large, weight-reused) | 2 * K per output element | compute-bound (K >~ 32) | RVV + register tiling + pipelining |
| Conv2D (weight-reused, small output) | 2 * KH*KW*IC per output | compute-bound | RVV + register tiling + im2col |
| BatchNorm forward | ~2 | borderline | fuse into producer/consumer |
| Attention softmax | mixed (see hybrid_bound_attention) | hybrid | flash-attention style fusion |

**If you're stuck below break-even intensity**, the only levers are
memory-side: prefetch, streaming stores, blocking to fit in L1D, or
fusing to eliminate the intermediate tensor entirely. Adding more
vector lanes just idles them harder.

---

## RISC-V Architecture Quick Reference (Rocket / Saturn defaults)

- **Core**: Single-issue in-order scalar (Rocket) or dual-issue in-order with
  RVV (Saturn). No branch predictor with speculation window (Rocket does
  static predict-not-taken); mispredict cost = 3 cycles.
- **L1I / L1D**: Typically 16 KB / 16 KB, 4-way, 64-byte lines. Line fill
  latency ~15-30 cycles for L2 hit (if L2 present), 60-100 cycles for DRAM.
- **RVV (Saturn/rvv_hetero)**: `V` extension version 1.0 with configurable
  VLEN. On the current Saturn config, VLEN = 128 bits (LMUL up to 8 →
  effective vectors of 4-32 fp32 lanes). Zfh + Zvfh for fp16 vector ops.
- **Load-use latency**: 2-cycle load-to-use bubble on Rocket; use scheduling
  to interleave independent work.
- **FPU**: hardware fp32; hardware fp16 only on Zfh cores; NO hardware
  transcendentals (sin/cos/exp go through libm — SLOW).
- **Instruction pipeline**: 5-stage (Rocket), no dynamic reordering. Compiler
  scheduling matters much more than on OoO cores.
- **No SMT / no hyperthreading** — one hardware thread per core; MP designs
  scale by having multiple cores (typically 1-4 for chipyard configs).
- **Wall-clock timer**: `mtime` CSR at ~10 MHz on default configs (100 ns
  ticks); `mcycle` runs at core frequency (~1 GHz sim clock). The
  `MODELBLASTER_WALL_CYCLES` line reports mtime ticks; per-op CSV cycles are
  mcycle deltas.

---

## Expert Knowledge from RISC-V Kernel Optimization

### Learned Optimization Strategies

The following optimizations are effective for C kernels on RISC-V in-order cores
running under Zephyr on spike / FireSim:

---

### Seed State: memory_bound_stencil
**Primary Bottleneck**: memory_bound
**Secondary Characteristics**: sequential-strided access, large working set
exceeds L1D, low arithmetic intensity per byte loaded.

**Signature**: A dominant op is an elementwise / small-window stencil
(relu, sigmoid, elementwise_add, conv1d, depthwise conv, batchnorm) with input
size >> L1D capacity. Per-op cycles are dominated by load latency; when IPC
data is available, IPC < 0.5. Intensity < 1 FLOP/byte.

**Do-not-bother heuristics for pure-elementwise fp32 stencils (relu, add,
mul, negate, clamp)**: intensity ~0.25 FLOP/byte. Compute optimisations
(unroll, RVV, pipeline) will plateau at 1.1-1.3x because the vector unit
already outruns DDR. Focus on prefetch / streaming stores / fusing into the
neighbour producer or consumer. If those aren't available, expect the RL
loop to top out at ~15% improvement over the reference and move on.

**Recommended Optimizations** (predicted_improvement %):
- `3.3_prefetch_hints` (25%): Software prefetch 64-128 elements ahead
  (`__builtin_prefetch(&a[i + 64], 0, 0)`). Locality hint 0 = no reuse,
  correct for streaming stencils. Rocket's L1 hardware prefetcher is weak.
- `5.1_streaming_stores` (12%): For output-only writes with no reuse, avoid
  the read-for-ownership by writing full cache lines. Modest but stackable.
- `5.2_producer_consumer_fusion` (35%): The largest lever for memory-bound
  elementwise — often not applicable inside a single kernel, but flags the
  isolated-optimisation ceiling to the caller.
- `1.1_loop_tiling_icache` (18%): Tile inner loops so a chunk fits in L1D.
  For a 16 KB L1D holding fp32 data, tile size ≈ 4096 elements — one warm
  cache-line fill lasts many arithmetic ops.
- `1.2_data_layout_soa` (15%): For batch / channel-major inputs, reorder to
  keep stride-1 access on the innermost loop.
- `2.3_rvv_vectorization` (15%): Only widens the memory transaction here —
  won't move the needle if already at DDR peak. Use LMUL=m8 with `vle32_v`
  to amortise the vsetvli overhead over the largest possible batch. USE
  `vfmax_vf` / `vfmin_vf` / `vfabs_v` for elementwise-with-scalar patterns —
  see "RVV Canonical Idioms" above.

---

### Seed State: compute_bound_gemm_small
**Primary Bottleneck**: compute_bound
**Secondary Characteristics**: kernel data fits in L1, arithmetic pipeline is
the limit, FPU throughput dominates.

**Signature**: Dominant op is a matmul / linear / conv2d whose weight tensor
fits in L1D (typically <= 8 KB). Per-op cycles scale ~linearly with FLOP count.
IPC 0.7-1.0 on scalar; > 1.5 with RVV.

**Recommended Optimizations**:
- `2.1_loop_unrolling` (25%): Unroll the innermost loop 4-8× to expose ILP
  and hide the 2-cycle load-use bubble. Rocket has no OoO scheduling — this
  matters a LOT.
- `4.1_register_tiling` (30%): Keep 4×4 accumulators in registers across the
  reduction; write back only at the end. Cuts store traffic 16×.
- `2.3_rvv_vectorization` (45%): `vfmacc.vv` accumulates fp32 dot products;
  4-8 lanes with LMUL=2. Use `vmv1r.v` to seed accumulators.
- `4.2_software_pipelining` (25%): Interleave load(k+1) with fmadd(k) so the
  load latency hides behind arithmetic.

---

### Seed State: compute_bound_transcendental
**Primary Bottleneck**: compute_bound
**Secondary Characteristics**: heavy use of sin/cos/exp/sqrt, softlibm
transcendentals dominate cycles.

**Signature**: Ops like softmax, sigmoid (unfused), gelu, layer_norm, or any
activation using `expf`/`tanhf`/`sqrtf`. Per-op cycles are 10-100× what pure
arithmetic would suggest — because libm is scalar and iterative on RISC-V.

**Recommended Optimizations**:
- `custom_polynomial_approximation` (60%): Replace `expf(x)` with a
  degree-4 minimax polynomial. Accuracy trade-off is workload-dependent;
  KernelBench validates against a tolerance envelope.
- `piecewise_approximation` (40%): Range-reduce (e.g. `sigmoid` → clamp
  to +/-6, otherwise `x/(1+|x|)`).
- `2.3_rvv_vectorization` (30%): Even a scalar polynomial approx becomes
  much faster when the outer loop is vectorized.
- `fuse_activation` (35%): Merge sigmoid/relu/gelu into the preceding
  arithmetic op so the intermediate never round-trips through memory.

---

### Seed State: latency_bound_reduction
**Primary Bottleneck**: latency_bound
**Secondary Characteristics**: reductions with serial dependency chain
(sum, max, argmax), depth ~= working set size.

**Signature**: Ops like reduce_sum, softmax denominator, batchnorm mean/var,
attention_softmax. The dependency chain is O(N) deep; IPC is low regardless
of memory pressure because each iteration waits on the previous.

**Recommended Optimizations**:
- `tree_reduction` (35%): Split the reduction into K independent partial
  sums, combine at the end. Depth drops from N to log(N).
- `2.3_rvv_vectorization` (55%): Use `vfredosum.vs` / `vfredsum.vs` for
  in-vector reduction — one instruction handles LMUL× lanes.
- `2.1_loop_unrolling` (20%): 4× unroll with 4 independent accumulators
  shortens the chain by 4×.
- `3.2_load_use_scheduling` (15%): Prefetch the next chunk while reducing
  the current one.

---

### Seed State: memory_bound_conv2d
**Primary Bottleneck**: memory_bound
**Secondary Characteristics**: 2D stencil with input reuse across a small
kernel window, working set >> L1D.

**Signature**: `conv2d_s8`, `conv2d_f32` with input size > 32 KB. Same load
touched multiple times per output but line evicted before reuse (large stride
along OH between consecutive output rows).

**Recommended Optimizations**:
- `im2col_transform` (40%): Materialize the im2col matrix once, run GEMM on
  it. Turns strided-access into contiguous — much better for L1D + prefetcher.
- `1.1_loop_tiling_icache` (30%): Block along (OH, OW, OC) so a (h, w, oc)
  tile stays resident during its computation. Tile size = (4, 4, 8) is a
  reasonable start for fp32 on 16 KB L1D.
- `direct_conv_row_reuse` (25%): Keep 3 input rows in registers across the
  filter's KH dimension so the OH stride only causes 1 miss per output row
  instead of KH.
- `2.3_rvv_vectorization` (40%): Vectorize the OC dimension; use `vle32.v`
  for input channels and `vfmacc.vv` for the accumulate.

---

### Seed State: hybrid_bound_attention
**Primary Bottleneck**: hybrid_bound
**Secondary Characteristics**: mixed memory + compute + transcendental,
softmax + matmul + reduction chain.

**Signature**: Self-attention block (QK^T → softmax → V matmul). Cycles split
roughly 40% matmul, 40% softmax (exp + reduce), 20% memory shuffle.

**Recommended Optimizations**:
- `fused_attention_block` (50%): Combine QK^T + softmax + V into one pass
  that keeps intermediate tensors in registers (Flash Attention style, but
  scaled down for this SoC).
- `online_softmax_reduction` (30%): Compute softmax in one pass with running
  max + running normaliser; avoids the 2-pass memory traffic.
- `piecewise_approximation` (20%): For softmax's `expf`, range-reduce +
  polynomial approx.
- `2.3_rvv_vectorization` (40%): Vectorize the reduction and the matmul
  independently.

---

### Seed State: latency_bound_indirect_load
**Primary Bottleneck**: latency_bound
**Secondary Characteristics**: pointer-chase / gather / embedding lookup,
each load address depends on a prior load.

**Signature**: `embedding_lookup`, `gather`, `sparse_lookup` — the address
of load N depends on the data returned by load N-1. Rocket has no OoO
speculation; each load stalls the pipeline until it retires. IPC is very low
regardless of anything else.

**Recommended Optimizations**:
- `1.3_reduce_indirect_loads` (35%): If the index sequence is predictable
  (e.g. `idx[i] = i*stride + offset`), inline the address computation and
  eliminate the level-1 load.
- `3.3_prefetch_hints` (30%): Software-prefetch `idx[i+K]` far enough ahead
  that the address is available when needed. K depends on load latency
  (~30 cycles / cycle_per_iteration).
- `batch_gather_transform` (25%): If the workload permits, transform to a
  scatter/broadcast (dense × sparse-mask) that has stride-1 access.

---

### Seed State: latency_bound_branch_heavy
**Primary Bottleneck**: latency_bound
**Secondary Characteristics**: many data-dependent branches per element,
predictor is static (predict-not-taken).

**Signature**: `relu6`, `clamp`, `where`, quantized `saturate`, or any
element-wise op with `if (x > thresh) …`. Each mispredict costs 3 cycles
on Rocket; a heavily-branching hot loop can spend 30-50% of cycles on
mispredicts.

**Recommended Optimizations**:
- `predicated_arithmetic` (30%): Replace `if (x > t) y = a else y = b` with
  `y = a * (x > t) + b * (x <= t)` — branch-free.
- `min_max_via_intrinsics` (20%): Scalar path — use `__builtin_fmax` /
  `__builtin_fmin` (or `fmaxf`/`fminf` when Zfh present); the compiler
  emits a single `fmax.s`/`fmin.s` instead of a branch. Vector path —
  `vfmax_vf_f32m8(v, thresh, vl)` / `vfmin_vf_f32m8(v, thresh, vl)` for
  the direct single-op form; DO NOT expand into vmfgt+vmerge (three ops
  instead of one).
- `sort_or_bucket_data` (25%): If the workload permits reordering, group
  elements by branch outcome so subsequent passes are branch-free.
- `2.3_rvv_vectorization` (40%): RVV masked ops (`vfmax.vv` under mask)
  eliminate scalar branches entirely for elementwise ops.

---

### Seed State: memory_bound_layout_conflict
**Primary Bottleneck**: memory_bound
**Secondary Characteristics**: cache thrashing from bad stride, working set
appears small but effective footprint is huge due to conflict misses.

**Signature**: Op accesses a 2D array with power-of-2 stride equal to a
cache-set stride. E.g. transposed layout with row_stride == 4096 bytes on a
4-way L1D → every 4 rows map to the same set → conflict misses on the 5th
access. Symptom: cycles-per-op far higher than working-set analysis predicts.

**Recommended Optimizations**:
- `1.2_data_layout_soa` (35%): Reorder the offending dimension so stride is
  not a multiple of cache-set stride (add padding, permute axes).
- `padding_between_rows` (30%): Add 1-2 unused columns to break stride
  alignment. Cheap fix if the layout is otherwise correct.
- `blocked_layout_transform` (40%): Materialize a blocked (tile-major) copy
  before the op. Costs one pass; pays back many times if the op is inner-loop.

---

## Extended Metrics (populated when the harness exposes them)

The seed above uses "if IPC is available" and similar conditional language
because the modelblaster harness today emits only per-op mcycle deltas. When
the harness is extended (see `notes/riscv_profile_extension.md` for the
current plan), the additional signals the LLM should key on:

- **IPC** (`cycles / instructions`): computed from `minstret` CSR delta
  alongside `mcycle`. IPC < 0.4 → memory-bound suspected; IPC > 0.9 (scalar)
  or > 1.2 (RVV) → compute-bound suspected.
- **d-cache miss rate**: via HPM counter (`mhpmevent3 = D$_MISS` on Rocket).
  > 0.05 miss/access → confirmed memory-bound.
- **i-cache miss rate**: via HPM counter (`I$_MISS`). Rare in KernelBench
  ops but relevant for large fused kernels; > 0.02 → suggests kernel body
  is larger than L1I capacity.
- **Branch mispredict rate**: via HPM counter. > 0.15 mispredict/branch →
  `latency_bound_branch_heavy` regardless of other signals.
- **Load-use stall rate**: via HPM counter (`LOAD_USE_HAZARD`). > 0.20 →
  `2.1_loop_unrolling` or `4.2_software_pipelining` is high-value.
- **TACIT trace slice depth / duration** (when a TACIT-traced run is
  requested via `--trace=l` on a chipyard-fsim spike): per-function
  timestamps on the reconstructed call stack. Reveals *where inside* an op
  the cycles go — e.g. "conv0's inner loop spends 40% in vsetvl setup" —
  which no counter-based metric captures.

---

### Expert Technique Combinations

- **RVV + register tiling + software pipelining**: The gold-standard for
  compute-bound gemm on Saturn. Vectorize, tile 4×4 accumulators in vector
  registers, pipeline load/fmadd across iterations. Typical 3-4× speedup
  over scalar unrolled baseline.
- **Tiling + prefetch + SoA layout**: The gold-standard for memory-bound
  stencils. Tile so working set fits in L1D, prefetch 2 lines ahead, keep
  stride-1 access. Typical 2× speedup over naive.
- **Predicated arithmetic + RVV mask ops**: Best combo for branch-heavy
  elementwise. Removes both the scalar branch AND vectorizes the whole
  loop.

## Integration with RISC-V + Zephyr Optimization Knowledge

This database is the seed catalog for RISC-V + Zephyr (spike / FireSim) RL
runs. The runtime `optimization_strategies` grow beyond this seed as the LLM
state-summarizer discovers new states and the RL loop measures actual
speedups. Per-technique `actual_improvement` and `confidence_score` are the
authoritative post-run values; the numbers above are priors only.
