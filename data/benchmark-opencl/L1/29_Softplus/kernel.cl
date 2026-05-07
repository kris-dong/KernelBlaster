#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void softplus(
    __global const half* x,
    __global half* out,
    const int total)
{
    int idx = get_global_id(0);
    if (idx < total) {
        float val = (float)x[idx];
        /* softplus(x) = log(1 + exp(x))
           Use numerically stable form:
           if x > threshold, softplus(x) ≈ x
           otherwise use log1p(exp(x)) */
        float result;
        if (val > 20.0f) {
            result = val;
        } else {
            result = log1p(exp(val));
        }
        out[idx] = (half)result;
    }
}
