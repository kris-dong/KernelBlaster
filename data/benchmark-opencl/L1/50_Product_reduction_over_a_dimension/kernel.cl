#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * Product reduction over dim1.
 * Input:  x     shape (batch, d1, d2)  -> x[b * d1 * d2 + i * d2 + j]
 * Output: out   shape (batch, d2)      -> out[b * d2 + j]
 *
 * Each work-item handles one (batch, col) pair.
 * global_id(0) = col  in [0, d2)
 * global_id(1) = batch in [0, batch)
 */
__kernel void prod_reduce(
    __global const half* x,
    __global half* out,
    const int batch,
    const int d1,
    const int d2)
{
    int col = get_global_id(0);
    int b   = get_global_id(1);

    if (b >= batch || col >= d2) return;

    float prod = 1.0f;
    int base = b * d1 * d2 + col;
    for (int i = 0; i < d1; i++) {
        prod *= (float)x[base + i * d2];
    }

    out[b * d2 + col] = (half)prod;
}
