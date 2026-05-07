#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void hardtanh(
    __global const half* x,
    __global half* out,
    const int n,
    const float min_val,
    const float max_val)
{
    int idx = get_global_id(0);
    if (idx < n) {
        float val = (float)x[idx];
        if (val < min_val) val = min_val;
        else if (val > max_val) val = max_val;
        out[idx] = (half)val;
    }
}
