#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define SQRT_2_OVER_PI 0.7978845608028654f
#define COEFF 0.044715f

__kernel void gelu(
    __global const half* input,
    __global half* output,
    const int total)
{
    int idx = get_global_id(0);
    if (idx >= total) return;

    float x = (float)input[idx];
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    float tanh_val = tanh(inner);
    float result = 0.5f * x * (1.0f + tanh_val);
    output[idx] = (half)result;
}
