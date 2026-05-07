__kernel void vector_add(__global const float4 *a,
                         __global const float4 *b,
                         __global float4 *result,
                         const unsigned int n)
{
    unsigned int i = get_global_id(0);
    if (i < n)
    {
        result[i] = a[i] + b[i];
    }
}
