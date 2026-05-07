#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * sum_reduce_dim1
 *
 * Reduces input of shape (batch, dim1, dim2) along dim1, producing (batch, 1, dim2).
 *
 * Global work size: (dim2, batch)
 *   get_global_id(0) -> column index j in [0, dim2)
 *   get_global_id(1) -> batch index b in [0, batch)
 */
__kernel void sum_reduce_dim1(
    __global const half* x,
    __global half*       out,
    const int            batch,
    const int            dim1,
    const int            dim2)
{
    int j = get_global_id(0);
    int b = get_global_id(1);

    if (b >= batch || j >= dim2) return;

    float sum = 0.0f;
    for (int i = 0; i < dim1; i++) {
        sum += (float)x[b * dim1 * dim2 + i * dim2 + j];
    }

    out[b * dim2 + j] = (half)sum;
}
