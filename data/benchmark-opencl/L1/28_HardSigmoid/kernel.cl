#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void hardsigmoid(
    __global const half* x,
    __global half* y,
    const int total)
{
    int idx = get_global_id(0);
    if (idx < total) {
        float val = (float)x[idx];
        /* HardSigmoid: clamp((x + 3) / 6, 0, 1) */
        float result = (val + 3.0f) / 6.0f;
        result = fmax(0.0f, fmin(1.0f, result));
        y[idx] = (half)result;
    }
}
