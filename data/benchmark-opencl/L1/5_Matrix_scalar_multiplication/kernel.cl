#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void scalar_mul(
    __global const half* A,
    __global half* C,
    const float s,
    const int total)
{
    int idx = get_global_id(0);
    if (idx < total) {
        C[idx] = (half)((float)A[idx] * s);
    }
}
