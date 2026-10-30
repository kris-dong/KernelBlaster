#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * Cooperative-workgroup L1-normalize for Adreno 650.
 *
 * out[i] = x[i] / sum(|x|)  (per row)
 *
 * One workgroup per row; 256 work-items cooperatively reduce sum(|x|).
 *
 * @local_work_size: 256
 * @global_work_factor: 256
 */

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void l1_normalize(
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

    __local float sdata[256];

    /* Pass 1: cooperative sum(|x|) */
    float tsum = 0.0f;
    for (int i = lid; i < dim; i += lsize) {
        tsum += fabs((float)row[i]);
    }
    sdata[lid] = tsum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsize >> 1; s > 0; s >>= 1) {
        if (lid < s) sdata[lid] += sdata[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    /* Avoid div-by-zero by clamping; matches numpy / torch behaviour: when
     * the row is all zeros, the result is also all zeros. */
    const float row_sum = sdata[0] > 0.0f ? sdata[0] : 1.0f;
    const float inv     = 1.0f / row_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Pass 2: write x / sum(|x|) */
    for (int i = lid; i < dim; i += lsize) {
        outrow[i] = (half)((float)row[i] * inv);
    }
}
