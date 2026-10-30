#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * Cooperative-workgroup log_softmax for Adreno 650.
 *
 * One workgroup per row, 256 work-items per workgroup. Each work-item
 * processes dim/256 elements per pass, then participates in a tree-
 * reduction in static __local memory for max and sum(exp).
 *
 * @local_work_size: 256
 * @global_work_factor: 256
 */

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void log_softmax(
    __global const half* x,
    __global       half* out,
    const int            batch,
    const int            dim)
{
    const int b     = get_group_id(0);
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);

    if (b >= batch) return;

    const int            offset = b * dim;
    const __global half* row    = x   + offset;
    __global half*       outrow = out + offset;

    __local float sdata[256];   /* matches reqd_work_group_size */

    /* Pass 1: max */
    float tmax = -1e30f;
    for (int i = lid; i < dim; i += lsize) {
        float v = (float)row[i];
        if (v > tmax) tmax = v;
    }
    sdata[lid] = tmax;
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

    /* Pass 2: sum(exp(x - max)) */
    float tsum = 0.0f;
    for (int i = lid; i < dim; i += lsize) {
        tsum += native_exp((float)row[i] - row_max);
    }
    sdata[lid] = tsum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsize >> 1; s > 0; s >>= 1) {
        if (lid < s) sdata[lid] += sdata[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float log_sum = native_log(sdata[0]);
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Pass 3: out = (x - max) - log(sum) */
    for (int i = lid; i < dim; i += lsize) {
        float v = ((float)row[i] - row_max) - log_sum;
        outrow[i] = (half)v;
    }
}
