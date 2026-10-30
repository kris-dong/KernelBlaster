#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * min_reduce_dim1:
 *   Input x has shape (batch, dim1, dim2) stored row-major.
 *   Output has shape (batch, dim2): for each (b, d2), find min over dim1.
 *
 *   Global work: (dim2, batch)
 *   get_global_id(0) -> d2 index
 *   get_global_id(1) -> batch index
 */
__kernel void min_reduce_dim1(
    __global const half* x,
    __global half* out,
    const int batch,
    const int dim1,
    const int dim2)
{
    int d2 = get_global_id(0);
    int b  = get_global_id(1);

    if (b >= batch || d2 >= dim2) return;

    float min_val = (float)x[b * dim1 * dim2 + 0 * dim2 + d2];

    for (int d1 = 1; d1 < dim1; d1++) {
        float val = (float)x[b * dim1 * dim2 + d1 * dim2 + d2];
        if (val < min_val) min_val = val;
    }

    out[b * dim2 + d2] = (half)min_val;
}
