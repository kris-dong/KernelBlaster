#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * frob_normalize: 4 args
 *   0: __global const half* x
 *   1: __global half* out
 *   2: float inv_norm
 *   3: int total
 */
__kernel void frob_normalize(
    __global const half* x,
    __global half* out,
    const float inv_norm,
    const int total)
{
    int gid = get_global_id(0);
    if (gid < total) {
        float v = (float)x[gid];
        out[gid] = (half)(v * inv_norm);
    }
}
