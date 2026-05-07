#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void instance_norm(
    __global const half* x,
    __global half* y,
    const int n_slices,
    const int spatial)
{
    __local float l_buf[256];

    int slice_id = get_group_id(0);
    int local_id = get_local_id(0);
    int local_sz = get_local_size(0);

    if (slice_id >= n_slices) return;

    int base = slice_id * spatial;

    float my_sum = 0.0f;
    for (int i = local_id; i < spatial; i += local_sz) {
        my_sum += (float)x[base + i];
    }
    l_buf[local_id] = my_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = local_sz >> 1; stride > 0; stride >>= 1) {
        if (local_id < stride)
            l_buf[local_id] += l_buf[local_id + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = l_buf[0] / (float)spatial;
    barrier(CLK_LOCAL_MEM_FENCE);

    float my_var = 0.0f;
    for (int i = local_id; i < spatial; i += local_sz) {
        float d = (float)x[base + i] - mean;
        my_var += d * d;
    }
    l_buf[local_id] = my_var;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = local_sz >> 1; stride > 0; stride >>= 1) {
        if (local_id < stride)
            l_buf[local_id] += l_buf[local_id + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float var = l_buf[0] / (float)spatial;
    float inv_std = rsqrt(var + 1e-5f);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = local_id; i < spatial; i += local_sz) {
        float val = (float)x[base + i];
        y[base + i] = (half)((val - mean) * inv_std);
    }
}
