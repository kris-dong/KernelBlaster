#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void elu_kernel(
    __global const half* x,
    __global half* y,
    const float alpha,
    const int total)
{
    int idx = get_global_id(0);
    if (idx < total) {
        float val = (float)x[idx];
        float result;
        if (val >= 0.0f) {
            result = val;
        } else {
            result = alpha * (exp(val) - 1.0f);
        }
        y[idx] = (half)result;
    }
}
