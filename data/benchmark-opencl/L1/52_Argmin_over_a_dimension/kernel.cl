#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * argmin along dim=1 for input shape (batch, dim1, dim2)
 * output shape: (batch, dim2)
 * Each work item handles one (batch, dim2) output element.
 * global_id(0) = col index in [0, dim2)
 * global_id(1) = batch index in [0, batch)
 */
__kernel void argmin_dim1(
    __global const half* x,
    __global int* out,
    const int batch,
    const int dim1,
    const int dim2)
{
    int col = get_global_id(0);
    int b   = get_global_id(1);

    if (b >= batch || col >= dim2) return;

    float min_val = (float)x[b * dim1 * dim2 + 0 * dim2 + col];
    int   min_idx = 0;

    for (int d = 1; d < dim1; d++) {
        float val = (float)x[b * dim1 * dim2 + d * dim2 + col];
        if (val < min_val) {
            min_val = val;
            min_idx = d;
        }
    }

    out[b * dim2 + col] = min_idx;
}
