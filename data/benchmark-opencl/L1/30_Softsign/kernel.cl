#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void softsign(
    __global const half* x,
    __global half* out,
    const int total)
{
    int idx = get_global_id(0);
    if (idx < total) {
        float val = (float)x[idx];
        float result = val / (1.0f + fabs(val));
        out[idx] = (half)result;
    }
}
