#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void tanh_activation(
    __global const half* x,
    __global half* out,
    const int total)
{
    int idx = get_global_id(0);
    if (idx < total) {
        float val = (float)x[idx];
        /* tanh(x) = (e^x - e^-x) / (e^x + e^-x) */
        float ex  = exp(val);
        float enx = exp(-val);
        float result = (ex - enx) / (ex + enx);
        out[idx] = (half)result;
    }
}
