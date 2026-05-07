#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * One work-item per batch element.
 * Normalises over norm_size contiguous elements then applies
 * the per-element affine transform: out = (x - mean) / sqrt(var + eps) * w + b
 */
__kernel void layernorm(
    __global const half* x,
    __global const half* weight,
    __global const half* bias,
    __global       half* out,
    const int            norm_size)
{
    int b = get_global_id(0);

    __global const half* xb = x   + (long)b * norm_size;
    __global       half* ob = out + (long)b * norm_size;

    /* pass 1 – mean */
    float mean = 0.0f;
    int i;
    for (i = 0; i < norm_size; i++)
        mean += (float)xb[i];
    mean /= (float)norm_size;

    /* pass 2 – variance */
    float var = 0.0f;
    for (i = 0; i < norm_size; i++) {
        float d = (float)xb[i] - mean;
        var += d * d;
    }
    var /= (float)norm_size;

    float inv_std = rsqrt(var + 1e-5f);

    /* pass 3 – normalise + affine */
    for (i = 0; i < norm_size; i++) {
        float n = ((float)xb[i] - mean) * inv_std;
        ob[i] = (half)(n * (float)weight[i] + (float)bias[i]);
    }
}
