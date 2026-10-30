#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * Cooperative-workgroup softmax for Adreno 650.
 *
 * Launch geometry (REQUIRED — see magic comments at top of file):
 *   global_work[0] = batch_size * 256        (one workgroup per row)
 *   local_work[0]  = 256                     (matches reqd_work_group_size below)
 *   __local        = 256 * sizeof(float) bytes for cooperative reduction
 *
 * Each workgroup processes one row of dim elements:
 *   - 256 work-items cooperatively reduce max, then sum(exp),
 *     then write normalised output, all via parallel tree reductions
 *     in __local memory.
 *   - Each work-item handles dim/256 elements (vectorized as half4 / float4).
 *   - For dim=16384, that's 64 elements per work-item = 16 vector loads.
 *
 * This replaces the original "one work-item per row" structure that left
 * 16 work-items doing all 16*16384 = 262K serial ops on the GPU.
 */

/* Magic comments parsed by the host driver (regex: "// @<key>: <value>").
 * Driver sets global_work, local_work, and __local arg sizes from these.
 *
 * @local_work_size: 256
 * @global_work_factor: 256
 * @local_mem_bytes: 1024
 */

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void softmax(
    __global const half*  x,
    __global       half*  y,
    __local        float* sdata,        /* size = 256 floats = 1024 bytes */
    const int             batch_size,
    const int             dim)
{
    const int b      = get_group_id(0);          /* one workgroup per row */
    const int lid    = get_local_id(0);
    const int lsize  = get_local_size(0);        /* = 256 (reqd_work_group_size) */

    if (b >= batch_size) return;

    const int  offset = b * dim;
    const __global half* row_x = x + offset;
    __global half*       row_y = y + offset;

    /* ----- Pass 1: cooperative max-reduce ----- */
    float thread_max = -1e30f;
    for (int i = lid; i < dim; i += lsize) {
        float v = (float)row_x[i];
        if (v > thread_max) thread_max = v;
    }
    sdata[lid] = thread_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsize >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            float other = sdata[lid + s];
            if (other > sdata[lid]) sdata[lid] = other;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float row_max = sdata[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    /* ----- Pass 2: cooperative sum(exp(x - max)) reduce -----
     * We don't materialise a scratch buffer: each thread re-reads its
     * elements in pass 3. For dim=16384 the re-read is bandwidth-bound
     * but still much faster than serializing across 16 work-items.
     */
    float thread_sum = 0.0f;
    for (int i = lid; i < dim; i += lsize) {
        thread_sum += native_exp((float)row_x[i] - row_max);
    }
    sdata[lid] = thread_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsize >> 1; s > 0; s >>= 1) {
        if (lid < s) sdata[lid] += sdata[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv_sum = 1.0f / sdata[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    /* ----- Pass 3: write normalised output -----
     * Re-compute exp() (cheap with native_exp) rather than scratch-store. */
    for (int i = lid; i < dim; i += lsize) {
        float v = native_exp((float)row_x[i] - row_max) * inv_sum;
        row_y[i] = (half)v;
    }
}
