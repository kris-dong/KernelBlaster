#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/* GELU using tanh approximation:
   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
*/
__kernel void gelu(
    __global const half* x,
    __global half* output,
    const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;

    float val = (float)x[idx];
    float c = 0.7978845608f; /* sqrt(2/pi) */
    float inner = c * (val + 0.044715f * val * val * val);
    float tanh_inner = tanh(inner);
    float result = 0.5f * val * (1.0f + tanh_inner);
    output[idx] = (half)result;
}
