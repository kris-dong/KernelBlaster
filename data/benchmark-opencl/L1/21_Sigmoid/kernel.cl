#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void sigmoid_kernel(
    __global const half* x,
    __global half* out,
    const int total)
{
    int idx = get_global_id(0);
    if (idx < total) {
        float val = (float)x[idx];
        float s = 1.0f / (1.0f + exp(-val));
        out[idx] = (half)s;
    }
}
