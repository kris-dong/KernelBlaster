#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void relu(
    __global const half* x,
    __global half* out,
    const int n)
{
    int idx = get_global_id(0);
    if (idx < n) {
        float v = (float)x[idx];
        out[idx] = (half)(v > 0.0f ? v : 0.0f);
    }
}
