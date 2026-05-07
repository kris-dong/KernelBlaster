#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void swish(
    __global const half* x,
    __global half* out,
    const int total)
{
    int idx = get_global_id(0);
    if (idx < total) {
        float xf = (float)x[idx];
        float sigmoid_x = 1.0f / (1.0f + exp(-xf));
        out[idx] = (half)(xf * sigmoid_x);
    }
}
