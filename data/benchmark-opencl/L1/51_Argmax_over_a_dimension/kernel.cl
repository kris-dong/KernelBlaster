#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * argmax over dim=1 for input tensor of shape (batch_size, dim1, dim2)
 * Output shape: (batch_size, dim2)
 *
 * global_id(0) = col index in [0, dim2)
 * global_id(1) = batch index in [0, batch_size)
 */
__kernel void argmax_dim1(
    __global const half* x,
    __global int* out,
    const int batch_size,
    const int dim1,
    const int dim2)
{
    int col = get_global_id(0);
    int b   = get_global_id(1);

    if (b >= batch_size || col >= dim2) return;

    float max_val = (float)x[b * dim1 * dim2 + 0 * dim2 + col];
    int max_idx = 0;

    for (int d = 1; d < dim1; d++) {
        float val = (float)x[b * dim1 * dim2 + d * dim2 + col];
        if (val > max_val) {
            max_val = val;
            max_idx = d;
        }
    }

    out[b * dim2 + col] = max_idx;
}
