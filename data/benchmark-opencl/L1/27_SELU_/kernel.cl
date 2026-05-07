#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define SELU_ALPHA  1.6732631921768188f
#define SELU_SCALE  1.0507009873554805f

__kernel void selu_kernel(
    __global const half* x,
    __global half* y,
    const int total)
{
    int idx = get_global_id(0);
    if (idx >= total) return;

    float v = (float)x[idx];
    float result;
    if (v > 0.0f) {
        result = SELU_SCALE * v;
    } else {
        result = SELU_SCALE * SELU_ALPHA * (exp(v) - 1.0f);
    }
    y[idx] = (half)result;
}
