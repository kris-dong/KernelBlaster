#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * mean_reduce_dim1:
 *   Input x shape:  (batch, dim1, dim2)
 *   Output shape:   (batch, dim2)
 *   Reduces along dimension 1 (size = dim1).
 *
 *   global_id(0) -> col in [0, dim2)
 *   global_id(1) -> b   in [0, batch)
 */
__kernel void mean_reduce_dim1(
    __global const half* x,
    __global half* out,
    const int batch,
    const int dim1,
    const int dim2)
{
    int col = get_global_id(0);
    int b   = get_global_id(1);

    if (b >= batch || col >= dim2) return;

    float sum = 0.0f;
    int base = b * dim1 * dim2 + col;
    for (int i = 0; i < dim1; i++) {
        sum += (float)x[base + i * dim2];
    }

    out[b * dim2 + col] = (half)(sum / (float)dim1);
}
