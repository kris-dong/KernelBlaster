#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * Group Normalization kernel.
 * Each work-item processes one (batch, group) pair.
 * Input layout: [batch, features, spatial] row-major
 * spatial = dim1 * dim2
 */
__kernel void group_norm(
    __global const half* x,
    __global const half* weight,
    __global const half* bias,
    __global half* out,
    const int batch,
    const int features,
    const int num_groups,
    const int spatial,
    const float eps)
{
    int gid = get_global_id(0);
    int total_groups = batch * num_groups;
    if (gid >= total_groups) return;

    int b = gid / num_groups;
    int g = gid % num_groups;

    int cpg = features / num_groups;   /* channels per group */
    int group_size = cpg * spatial;

    int batch_offset   = b * features * spatial;
    int group_ch_start = g * cpg;

    /* Compute mean */
    float mean = 0.0f;
    for (int c = 0; c < cpg; c++) {
        int ch_offset = batch_offset + (group_ch_start + c) * spatial;
        for (int s = 0; s < spatial; s++) {
            mean += (float)x[ch_offset + s];
        }
    }
    mean /= (float)group_size;

    /* Compute variance */
    float var = 0.0f;
    for (int c = 0; c < cpg; c++) {
        int ch_offset = batch_offset + (group_ch_start + c) * spatial;
        for (int s = 0; s < spatial; s++) {
            float diff = (float)x[ch_offset + s] - mean;
            var += diff * diff;
        }
    }
    var /= (float)group_size;

    float inv_std = rsqrt(var + eps);

    /* Normalize and apply affine transform */
    for (int c = 0; c < cpg; c++) {
        int ch = group_ch_start + c;
        float w  = (float)weight[ch];
        float bi = (float)bias[ch];
        int ch_offset = batch_offset + ch * spatial;
        for (int s = 0; s < spatial; s++) {
            float xn = ((float)x[ch_offset + s] - mean) * inv_std;
            out[ch_offset + s] = (half)(xn * w + bi);
        }
    }
}
