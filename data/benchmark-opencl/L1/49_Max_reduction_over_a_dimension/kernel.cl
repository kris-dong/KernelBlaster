#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * max_reduce_dim1:
 *   Input:  x[batch, dim1, dim2]  (half)
 *   Output: out[batch, dim2]      (half)
 *   Reduces over dim1 (axis=1), keeping batch and dim2.
 *
 *   Global work: (dim2, batch)
 *   get_global_id(0) -> col  (dim2 index)
 *   get_global_id(1) -> b    (batch index)
 */
__kernel void max_reduce_dim1(
    __global const half* x,
    __global half* out,
    const int batch,
    const int dim1,
    const int dim2)
{
    int col = get_global_id(0);
    int b   = get_global_id(1);

    if (b >= batch || col >= dim2) return;

    float max_val = -INFINITY;
    for (int d1 = 0; d1 < dim1; d1++) {
        float val = (float)x[b * dim1 * dim2 + d1 * dim2 + col];
        if (val > max_val) max_val = val;
    }

    out[b * dim2 + col] = (half)max_val;
}
