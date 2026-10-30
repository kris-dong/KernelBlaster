## Usage Guidelines for LLM Integration (OpenCL / Adreno)

### For State Analysis
When analyzing OpenCL event profiling output, the LLM should:
1. Identify primary bottlenecks from kernel execution times (memory-bound vs compute-bound)
2. Map to the closest state in this database
3. Consider Adreno hardware context (3 CUs, 32KB local memory, 128-bit memory bus, OpenCL 2.0)
4. Account for potential state transitions after optimization

### For Optimization Selection
When selecting optimizations, the LLM should:
1. Prioritize high-confidence, high-impact optimizations
2. Consider composite strategies for complex states
3. Account for potential side effects and trade-offs
4. Adapt parameters based on kernel-specific characteristics and Adreno constraints

### For Performance Prediction
When predicting improvements, the LLM should:
1. Use base predictions as starting points
2. Adjust based on confidence scores and historical accuracy
3. Consider kernel context and similarity to previous cases
4. Provide uncertainty ranges rather than point estimates

## Integration with OpenCL GPU Optimization Knowledge

This database integrates with Adreno-specific optimization knowledge to provide:
- **Hierarchical optimization strategies**: From high-level decisions to specific implementations
- **Context-aware recommendations**: Based on profiling data and performance characteristics
- **Multi-objective optimization**: Balancing performance, accuracy, and power efficiency
- **Hardware-specific guidance**: Tailored recommendations for Qualcomm Adreno GPU architecture

The LLM agents use this database as a **living reference** that evolves based on actual optimization results, enabling continuous improvement in optimization strategy selection and performance prediction.

## Adreno Architecture Quick Reference

- **Shader Processors (SPs)**: Each CU contains multiple SPs; wavefront size is 32 (similar to NVIDIA warp)
- **Local memory**: 32KB per CU, shared across work-groups scheduled on that CU
- **Global memory bus**: 128-bit LPDDR5; coalesced 128-byte transactions are critical
- **Texture cache**: Read-only path with spatial locality optimisation; use for read-heavy data
- **ALU**: Native float and half (fp16) support; half throughput is 2x float on Adreno 6xx
- **No equivalent to NVIDIA Tensor Cores**: Rely on ALU vectorisation (float4/half4) for throughput

## Expert Knowledge from OpenCL Kernel Optimization

### Learned Optimization Strategies

The following optimizations are effective for OpenCL kernels on Qualcomm Adreno GPUs:

#### Expert Technique: local_memory_tiling
**Performance Impact**: 20-40% improvement (workload-dependent)
**Confidence Score**: 0.95
**Applicable States**: memory_bandwidth_saturated, memory_compute_balanced

**Implementation Hints**:
- Optimal tile size: 16x16 for matmul-like workloads (fits 16×16×4 = 1024 bytes per tile, well within 32KB)
- Use barrier(CLK_LOCAL_MEM_FENCE) between load and compute phases
- Avoid over-allocating local memory — reduces occupancy (fewer work-groups per CU)

**Usage Examples**:
```c
__kernel void matmul_tiled(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M, const int N, const int K)
{
    __local float As[TILE_SIZE][TILE_SIZE];
    __local float Bs[TILE_SIZE][TILE_SIZE];

    int row = get_local_id(1);
    int col = get_local_id(0);
    int globalRow = get_group_id(1) * TILE_SIZE + row;
    int globalCol = get_group_id(0) * TILE_SIZE + col;

    float sum = 0.0f;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int tiledCol = t * TILE_SIZE + col;
        int tiledRow = t * TILE_SIZE + row;
        As[row][col] = (globalRow < M && tiledCol < K) ? A[globalRow * K + tiledCol] : 0.0f;
        Bs[row][col] = (tiledRow < K && globalCol < N) ? B[tiledRow * N + globalCol] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++)
            sum = mad(As[row][k], Bs[k][col], sum);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (globalRow < M && globalCol < N)
        C[globalRow * N + globalCol] = sum;
}
```

#### Expert Technique: vectorized_memory_access
**Performance Impact**: 15-30% improvement
**Confidence Score**: 0.90
**Applicable States**: memory_bandwidth_saturated, memory_compute_balanced

**Implementation Hints**:
- Use float4/int4 to issue 128-bit loads (matches Adreno memory bus width)
- Ensure base addresses are 16-byte aligned for float4
- Each work-item processes 4 elements, reducing total work-items by 4x

**Usage Examples**:
```c
__kernel void vector_add_vec4(
    __global const float4* A,
    __global const float4* B,
    __global float4* C,
    const int n4)  // n4 = N / 4
{
    int gid = get_global_id(0);
    if (gid < n4)
        C[gid] = A[gid] + B[gid];
}
```

#### Expert Technique: work_group_size_tuning
**Performance Impact**: 10-25% improvement
**Confidence Score**: 0.85
**Applicable States**: compute_throughput_saturated, memory_compute_balanced, low_occupancy

**Implementation Hints**:
- Adreno wavefront size is 32; work-group sizes should be multiples of 32
- Common sweet spots: 64, 128, 256 work-items per group
- For 2D kernels: (16,16)=256 or (16,8)=128 are good starting points
- Too-large work-groups reduce occupancy if they exhaust local memory or registers

**Usage Examples**:
```c
// Host-side: choose work-group size
size_t local_work[2] = {16, 16};  // 256 work-items
size_t global_work[2] = {
    ((N + local_work[0] - 1) / local_work[0]) * local_work[0],
    ((M + local_work[1] - 1) / local_work[1]) * local_work[1]
};
clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work, local_work, 0, NULL, &event);
```

#### Expert Technique: register_tiling
**Performance Impact**: 20-35% improvement
**Confidence Score**: 0.85
**Applicable States**: compute_throughput_saturated, memory_compute_balanced

**Implementation Hints**:
- Each work-item computes a small tile (e.g. 4x4 or 2x8) of output using private registers
- Reduces global/local memory traffic per output element
- Combine with local memory tiling for best results

**Usage Examples**:
```c
// Each work-item computes a TILE_R x TILE_C block of C
#define TILE_R 4
#define TILE_C 4

__kernel void matmul_reg_tiled(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M, const int N, const int K)
{
    float acc[TILE_R][TILE_C];
    for (int i = 0; i < TILE_R; i++)
        for (int j = 0; j < TILE_C; j++)
            acc[i][j] = 0.0f;

    int baseRow = get_group_id(1) * (get_local_size(1) * TILE_R) + get_local_id(1) * TILE_R;
    int baseCol = get_group_id(0) * (get_local_size(0) * TILE_C) + get_local_id(0) * TILE_C;

    for (int k = 0; k < K; k++) {
        float a_vals[TILE_R];
        for (int i = 0; i < TILE_R; i++)
            a_vals[i] = (baseRow + i < M) ? A[(baseRow + i) * K + k] : 0.0f;

        float b_vals[TILE_C];
        for (int j = 0; j < TILE_C; j++)
            b_vals[j] = (baseCol + j < N) ? B[k * N + (baseCol + j)] : 0.0f;

        for (int i = 0; i < TILE_R; i++)
            for (int j = 0; j < TILE_C; j++)
                acc[i][j] = mad(a_vals[i], b_vals[j], acc[i][j]);
    }

    for (int i = 0; i < TILE_R; i++)
        for (int j = 0; j < TILE_C; j++)
            if (baseRow + i < M && baseCol + j < N)
                C[(baseRow + i) * N + (baseCol + j)] = acc[i][j];
}
```

#### Expert Technique: half_precision
**Performance Impact**: 30-50% improvement (when precision allows)
**Confidence Score**: 0.75
**Applicable States**: compute_throughput_saturated, memory_bandwidth_saturated

**Implementation Hints**:
- Adreno 6xx has 2x throughput for fp16 vs fp32
- Use half4 for maximum vectorised throughput
- Requires #pragma OPENCL EXTENSION cl_khr_fp16 : enable
- Verify numerical accuracy — not suitable for all workloads

**Usage Examples**:
```c
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void matmul_half(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M, const int N, const int K)
{
    int row = get_global_id(1);
    int col = get_global_id(0);
    if (row < M && col < N) {
        float sum = 0.0f;  // accumulate in fp32 for stability
        for (int k = 0; k < K; k++)
            sum = mad((float)A[row * K + k], (float)B[k * N + col], sum);
        C[row * N + col] = (half)sum;
    }
}
```

#### Expert Technique: mad_fma_usage
**Performance Impact**: 5-15% improvement
**Confidence Score**: 0.90
**Applicable States**: compute_throughput_saturated

**Implementation Hints**:
- Replace `a * b + c` with `mad(a, b, c)` or `fma(a, b, c)`
- Compile with `-cl-mad-enable` to allow the compiler to fuse multiply-adds
- `mad()` may sacrifice precision for speed; `fma()` is IEEE-compliant
- Combine with `-cl-fast-relaxed-math` for maximum throughput (if precision allows)

#### Expert Technique: async_work_group_copy
**Performance Impact**: 10-20% improvement
**Confidence Score**: 0.70
**Applicable States**: memory_bandwidth_saturated, memory_latency_bound

**Implementation Hints**:
- Use async_work_group_copy() for DMA-style global→local transfers
- Enables overlapping compute with memory transfers (double buffering)
- Requires careful event handling with wait_group_events()

**Usage Examples**:
```c
__kernel void compute_with_prefetch(
    __global const float* input,
    __global float* output,
    __local float* buf_a,
    __local float* buf_b,
    const int N)
{
    event_t evt;
    int lid = get_local_id(0);
    int grp = get_group_id(0);
    int lsize = get_local_size(0);

    // Prefetch first tile
    evt = async_work_group_copy(buf_a, input + grp * lsize, lsize, 0);
    wait_group_events(1, &evt);

    // Process buf_a while prefetching next into buf_b ...
    // (double-buffer pattern)
}
```

#### Expert Technique: coalesced_access
**Performance Impact**: 15-30% improvement
**Confidence Score**: 0.90
**Applicable States**: memory_bandwidth_saturated

**Implementation Hints**:
- Consecutive work-items (get_global_id(0), get_global_id(0)+1, ...) should access consecutive memory addresses
- Avoid strided access patterns; transpose data if needed
- Adreno coalesces aligned 128-byte transactions across a wavefront of 32 items

### Expert Technique Combinations

**Combination**: local_memory_tiling + register_tiling
- Expected Performance: 40-60% speedup for GEMM-class workloads
- Confidence: 0.85
- Strategy: Tile into local memory, then each work-item computes a register sub-tile

**Combination**: vectorized_memory_access + work_group_size_tuning
- Expected Performance: 20-40% speedup for bandwidth-limited kernels
- Confidence: 0.80
- Strategy: Use float4 loads with work-group size tuned to match Adreno wavefront

**Combination**: local_memory_tiling + half_precision
- Expected Performance: 50-70% speedup when precision allows
- Confidence: 0.70
- Strategy: Tile in local memory using half type for 2x throughput and 2x data density

## Expert-Learned Optimizations (Auto-Generated)

### Expert Technique: local_memory_tiling
**Source**: OpenCL Kernel Optimization Analysis
**Performance Impact**: 20-40% improvement
**Confidence**: 0.95

**Implementation Strategy**:
- Optimal tile size: 16x16 for matmul-like workloads
- Use barrier(CLK_LOCAL_MEM_FENCE) between phases
- Budget local memory to maintain occupancy

**Usage Context**:
Best applied to: memory_bandwidth_saturated, memory_compute_balanced


### Expert Technique: vectorized_memory_access
**Source**: OpenCL Kernel Optimization Analysis
**Performance Impact**: 15-30% improvement
**Confidence**: 0.90

**Implementation Strategy**:
- Use float4/half4 for 128-bit memory transactions
- Align data to 16 bytes for float4
- Reduce work-item count proportionally

**Usage Context**:
Best applied to: memory_bandwidth_saturated, memory_compute_balanced


### Expert Technique: register_tiling
**Source**: OpenCL Kernel Optimization Analysis
**Performance Impact**: 20-35% improvement
**Confidence**: 0.85

**Implementation Strategy**:
- Each work-item computes a small sub-tile (4x4 or 2x8) in private registers
- Combine with local memory tiling for best results

**Usage Context**:
Best applied to: compute_throughput_saturated, memory_compute_balanced


### Expert Technique: work_group_size_tuning
**Source**: OpenCL Kernel Optimization Analysis
**Performance Impact**: 10-25% improvement
**Confidence**: 0.85

**Implementation Strategy**:
- Keep work-group size a multiple of 32 (Adreno wavefront)
- Common choices: (16,16), (16,8), (8,8) for 2D; 64, 128, 256 for 1D
- Balance occupancy vs register/local memory pressure

**Usage Context**:
Best applied to: compute_throughput_saturated, memory_compute_balanced, low_occupancy


### Expert Technique: half_precision
**Source**: OpenCL Kernel Optimization Analysis
**Performance Impact**: 30-50% improvement
**Confidence**: 0.75

**Implementation Strategy**:
- Adreno 6xx: 2x fp16 throughput vs fp32
- Accumulate in fp32, store in fp16 for numerical stability
- Requires cl_khr_fp16 extension

**Usage Context**:
Best applied to: compute_throughput_saturated, memory_bandwidth_saturated


### Expert Technique: mad_fma_usage
**Source**: OpenCL Kernel Optimization Analysis
**Performance Impact**: 5-15% improvement
**Confidence**: 0.90

**Implementation Strategy**:
- Replace a*b+c with mad(a,b,c) throughout inner loops
- Compile with -cl-mad-enable -cl-fast-relaxed-math

**Usage Context**:
Best applied to: compute_throughput_saturated
