## Usage Guidelines for LLM Integration

### For State Analysis
When analyzing NCU reports, the LLM should:
1. Identify primary and secondary bottlenecks
2. Map to the closest state in this database
3. Consider hardware context and kernel characteristics
4. Account for potential state transitions after optimization

### For Optimization Selection
When selecting optimizations, the LLM should:
1. Prioritize high-confidence, high-impact optimizations
2. Consider composite strategies for complex states
3. Account for potential side effects and trade-offs
4. Adapt parameters based on kernel-specific characteristics

### For Performance Prediction
When predicting improvements, the LLM should:
1. Use base predictions as starting points
2. Adjust based on confidence scores and historical accuracy
3. Consider kernel context and similarity to previous cases
4. Provide uncertainty ranges rather than point estimates

## Integration with Comprehensive GPU Optimization Knowledge

This database integrates with the comprehensive GPU optimization decision tree to provide:
- **Hierarchical optimization strategies**: From high-level decisions to specific implementations
- **Context-aware recommendations**: Based on profiling data and performance characteristics
- **Multi-objective optimization**: Balancing performance, accuracy, and maintainability
- **Hardware-specific guidance**: Tailored recommendations for different GPU architectures

The LLM agents use this database as a **living reference** that evolves based on actual optimization results, enabling continuous improvement in optimization strategy selection and performance prediction. 

## Expert Knowledge from KernelBench Leaderboard Analysis

### Learned Optimization Strategies

The following optimizations have been extracted from high-performing solutions in the KernelBench leaderboard:

#### Expert Technique: tensor_core_utilization

**Usage Examples**:
```cuda
#include <mma.h>
using namespace nvcuda;
/*
 wmma_tile_16x16_helper — inner single-tile Tensor Core compute with full tail handling

 IMPORTANT: DO NOT USE ANY WMMA FUNCTIONS OUTSIDE OF THIS HELPER

 Purpose:
 - Computes one 16x16 tile of C (at output block coords (c_row, c_col)).
 - Accumulates across K in 16-wide steps using Tensor Cores.
 - Handles all tail cases: partial M, partial N, and K-tail by packing with zero padding when needed.
 - Stores the 16x16 accumulator tile to accum_out in row-major order (float).

 Inputs (row-major operands):
 - A_tile_base: pointer to A at row c_row and k=0, i.e. &A[c_row * K + 0]
 - lda: leading dimension for A; must be K
 - B_tile_col_ptr: pointer to the start of column c_col in B, i.e. &B[0 * N + c_col]
 - ldb: leading dimension for B; must be N
 - m_eff: number of valid rows in this tile (<= 16)
 - n_eff: number of valid cols in this tile (<= 16)
 - total_k: K (the reduction dimension). Can be any positive integer.

 Output:
 - accum_out: pointer to a 16x16 float tile buffer (row-major). Must be unique per warp
   and 16-byte aligned when placed in shared memory.
 - a_pack, b_pack: per-warp 16x16 half buffers in shared memory used only when K-tail exists;
   must be unique per warp and 16-byte aligned.

 Usage in the outer kernel for tile (c_row, c_col):
   const __half* A_tile_base = A + c_row * K;         // row offset into A
   const __half* B_tile_col_ptr = B + c_col;          // column offset into B (row-major)
   wmma_tile_16x16_helper(A_tile_base, K,
                          B_tile_col_ptr, N,
                          M_eff, N_eff, K,
                          warp_tile,
                          warp_a_pack,
                          warp_b_pack);

 Important notes and pitfalls considered:
 - Operand layouts:
   Both A and B are row_major. The B submatrix used for C(c_row:c_row+16, c_col:c_col+16) at
   K-slice kk is B(kk:kk+16, c_col:c_col+16). With row_major loads, the base pointer and stride
   must be B + c_col + kk * ldb, with ldb = N.

 - Strides and base pointers:
   Passing wrong lda/ldb or mismatched base pointers yields incorrect results.
   A loads use base A + kk with lda = K (because A is row-major MxK).
   B loads use base B + c_col + kk * N with ldb = N (because B is row-major KxN).
   Do NOT use B + b_row + b_col * K unless B is truly column-major.

 - Tile completeness and tails:
   - Fast path: when m_eff==16, n_eff==16, and k_frag==16, load directly from global.
   - Otherwise (edge tiles or K-tail): cooperatively pack A and B blocks into a_pack/b_pack
     with zero padding outside valid extents, then run MMA on the packed tiles.

 - Shared memory staging and alignment:
   Store the accumulator fragment to a per-warp float[16*16] buffer. Declare dynamic
   shared memory as aligned bytes and cast to float* to ensure proper alignment for
   wmma::store_matrix_sync. Use __syncwarp() around producer/consumer steps. a_pack/b_pack
   must be per-warp and non-overlapping.

 - Warp/block tile mapping:
   Each warp must cover a unique set of tiles. Compute base tile indices from blockIdx
   plus per-warp offsets, then iterate local i/j within the warp. Avoid double-adding
   loop indices into the base (which causes overlaps/gaps).

 - Architecture:
   Requires Tensor Cores (sm_70+); this project targets sm_80+.
*/
__device__ __forceinline__ void wmma_tile_16x16_helper(
    const __half* __restrict__ A_tile_base, int lda,
    const __half* __restrict__ B_tile_col_ptr, int ldb,
    int m_eff, int n_eff, int total_k,
    float* __restrict__ accum_out,
    __half* __restrict__ a_pack,   // 16x16 row-major pack buffer in shared memory
    __half* __restrict__ b_pack)   // 16x16 row-major pack buffer in shared memory
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int kk = 0;
    for (; kk < total_k; kk += MMA_K) {
        const int k_frag = min(MMA_K, total_k - kk);

        const bool full_tile = (m_eff == MMA_M) && (n_eff == MMA_N) && (k_frag == MMA_K);
        if (full_tile) {
            // Fast path: direct global loads
            wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, __half, wmma::row_major> b_frag;
            wmma::load_matrix_sync(a_frag, A_tile_base + kk, lda);
            wmma::load_matrix_sync(b_frag, B_tile_col_ptr + kk * ldb, ldb);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        } else {
            // Pack with zero padding for partial M/N or K-tail
            for (int idx = lane_id; idx < MMA_M * MMA_N; idx += WARP_SIZE) {
                a_pack[idx] = __float2half(0.0f);
                b_pack[idx] = __float2half(0.0f);
            }
            __syncwarp();

            // Pack A: rows [0..m_eff-1], cols [0..k_frag-1]
            for (int idx = lane_id; idx < MMA_M * MMA_N; idx += WARP_SIZE) {
                const int ii = idx / MMA_N; // row in 16x16
                const int jj = idx % MMA_N; // col in 16x16
                if (ii < m_eff && jj < k_frag) {
                    a_pack[ii * MMA_N + jj] = A_tile_base[ii * lda + (kk + jj)];
                }
            }
            // Pack B: rows [0..k_frag-1], cols [0..n_eff-1]
            for (int idx = lane_id; idx < MMA_M * MMA_N; idx += WARP_SIZE) {
                const int ii = idx / MMA_N; // row in 16x16 (K-frag)
                const int jj = idx % MMA_N; // col in 16x16 (N-frag)
                if (ii < k_frag && jj < n_eff) {
                    b_pack[ii * MMA_N + jj] = B_tile_col_ptr[(kk + ii) * ldb + jj];
                }
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, __half, wmma::row_major> a_frag_pack;
            wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, __half, wmma::row_major> b_frag_pack;
            wmma::load_matrix_sync(a_frag_pack, a_pack, MMA_N);
            wmma::load_matrix_sync(b_frag_pack, b_pack, MMA_N);
            wmma::mma_sync(c_frag, a_frag_pack, b_frag_pack, c_frag);
        }
    }

    wmma::store_matrix_sync(accum_out, c_frag, MMA_N, wmma::mem_row_major);
}
```

#### Expert Technique: shared_memory_tiling
**Performance Impact**: 0% improvement
**Confidence Score**: 0.98
**Applicable States**: compute_throughput_saturated, memory_compute_balanced, memory_bandwidth_saturated

**Implementation Hints**:
- Optimal tile size: 16
- Highly effective for matrix multiplication workloads
- Consistently achieves high performance gains

**Usage Examples**:
```cuda
// Shared memory from claude-3.5-sonnet
__shared__ float shared_A[TILE_SIZE][TILE_SIZE];
__shared__ float shared_B[TILE_SIZE][TILE_SIZE];
// Kernel launch from claude-3.5-sonnet
matrix_multiply_kernel<<<numBlocks, threadsPerBlock>>>(
// Shared memory from claude-3.5-sonnet
__shared__ float As[16][16];
__shared__ float Bs[16][16];
// Kernel launch from claude-3.5-sonnet
matmul_kernel<<<numBlocks, threadsPerBlock>>>(
```

#### Expert Technique: register_optimization
**Performance Impact**: 0% improvement
**Confidence Score**: 1.00
**Applicable States**: compute_throughput_saturated, memory_compute_balanced, memory_bandwidth_saturated

**Implementation Hints**:
- Highly effective for matrix multiplication workloads
- Consistently achieves high performance gains

**Usage Examples**:
```cuda
// Shared memory from claude-3.5-sonnet
__shared__ float As[16][16];
__shared__ float Bs[16][16];
// Kernel launch from claude-3.5-sonnet
matmul_kernel<<<numBlocks, threadsPerBlock>>>(
// Shared memory from claude-3.5-sonnet
__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
// Kernel launch from claude-3.5-sonnet
tensor_matmul_kernel<<<blocks, threads>>>(
```

#### Expert Technique: memory_coalescing
**Performance Impact**: 0% improvement
**Confidence Score**: 0.83
**Applicable States**: compute_throughput_saturated, memory_compute_balanced, memory_bandwidth_saturated

**Implementation Hints**:
- Optimal tile size: 16
- Highly effective for matrix multiplication workloads
- Consistently achieves high performance gains

**Usage Examples**:
```cuda
// Kernel launch from gpt-o1
batched_matrix_multiply_kernel<<<blocks, threads>>>(
// Kernel launch from gpt-o1
matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
```

#### Expert Technique: occupancy_tuning
**Performance Impact**: 0% improvement
**Confidence Score**: 0.85
**Applicable States**: compute_throughput_saturated, memory_compute_balanced, memory_bandwidth_saturated

**Implementation Hints**:
- Highly effective for matrix multiplication workloads
- Consistently achieves high performance gains

**Usage Examples**:
```cuda
// Kernel launch from deepseek-coder
batched_matmul_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), batch_size, m, k, n);
// Shared memory from claude-3.5-sonnet
extern __shared__ float shared_mem[];
// Kernel launch from claude-3.5-sonnet
square_sum_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
```

#### Expert Technique: dynamic_shared_memory
**Performance Impact**: 0% improvement
**Confidence Score**: 0.44
**Applicable States**: memory_compute_balanced, low_occupancy_register_pressure, memory_latency_bound

**Implementation Hints**:
- Consistently achieves high performance gains

**Usage Examples**:
```cuda
// Shared memory from gpt-o1
extern __shared__ float shared_sum[];  // Shared memory for partial sums
// Kernel launch from gpt-o1
matvec_mul_kernel<<<blocks, threads, shared_mem_size>>>(
// Shared memory from gpt-o1
extern __shared__ float sdata[];
// Kernel launch from gpt-o1
softmax_kernel_batch<<<blocks, threads, shared_mem_size>>>(input_contiguous.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim);
```


### Expert Technique Combinations

**Combination**: register_optimization + shared_memory_tiling
- Average Performance: 0% speedup
- Confidence: 0.60
- Frequency in top solutions: 3 times


## Expert-Learned Optimizations (Auto-Generated)

### Expert Technique: shared_memory_tiling
**Source**: KernelBench Leaderboard Analysis
**Performance Impact**: 0% improvement
**Confidence**: 0.98

**Implementation Strategy**:
- Optimal tile size: 16
- Highly effective for matrix multiplication workloads
- Consistently achieves high performance gains

**Usage Context**:
Best applied to: memory_bandwidth_saturated, memory_compute_balanced, compute_throughput_saturated


### Expert Technique: register_optimization
**Source**: KernelBench Leaderboard Analysis
**Performance Impact**: 0% improvement
**Confidence**: 1.00

**Implementation Strategy**:
- Highly effective for matrix multiplication workloads
- Consistently achieves high performance gains

**Usage Context**:
Best applied to: memory_bandwidth_saturated, memory_compute_balanced, compute_throughput_saturated


### Expert Technique: memory_coalescing
**Source**: KernelBench Leaderboard Analysis
**Performance Impact**: 0% improvement
**Confidence**: 0.83

**Implementation Strategy**:
- Optimal tile size: 16
- Highly effective for matrix multiplication workloads
- Consistently achieves high performance gains

**Usage Context**:
Best applied to: memory_bandwidth_saturated, memory_compute_balanced, compute_throughput_saturated


### Expert Technique: occupancy_tuning
**Source**: KernelBench Leaderboard Analysis
**Performance Impact**: 0% improvement
**Confidence**: 0.85

**Implementation Strategy**:
- Highly effective for matrix multiplication workloads
- Consistently achieves high performance gains

**Usage Context**:
Best applied to: memory_bandwidth_saturated, memory_compute_balanced, compute_throughput_saturated


### Expert Technique: dynamic_shared_memory
**Source**: KernelBench Leaderboard Analysis
**Performance Impact**: 0% improvement
**Confidence**: 0.44

**Implementation Strategy**:
- Consistently achieves high performance gains

**Usage Context**:
Best applied to: memory_latency_bound, memory_compute_balanced, low_occupancy_register_pressure

 