#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void leaky_relu(
    __global const half* x,
    __global half* out,
    const int total,
    const float negative_slope)
{
    int idx = get_global_id(0);
    if (idx < total) {
        float val = (float)x[idx];
        float result = (val >= 0.0f) ? val : val * negative_slope;
        out[idx] = (half)result;
    }
}
