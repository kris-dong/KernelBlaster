#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * Single-pass MSE kernel using a two-level reduction:
 *   1. Each workgroup reduces its chunk into d_partial (float).
 *   2. Work-item 0 of group 0 sums d_partial and writes the mean to d_output.
 *
 * Args (6):
 *   predictions  – input half buffer
 *   targets      – input half buffer
 *   partial      – float scratch (one entry per workgroup)
 *   output       – single half result
 *   total        – total number of elements
 *   scratch      – local float scratch (local_size * sizeof(float))
 */
__kernel void mse(
    __global const half*  predictions,
    __global const half*  targets,
    __global       float* partial,
    __global       half*  output,
    const int             total,
    __local        float* scratch)
{
    int lid        = get_local_id(0);
    int gid        = get_global_id(0);
    int group_id   = get_group_id(0);
    int local_size = get_local_size(0);
    int num_groups = get_num_groups(0);

    /* Each thread computes its squared difference */
    float acc = 0.0f;
    if (gid < total) {
        float p = (float)predictions[gid];
        float t = (float)targets[gid];
        float d = p - t;
        acc = d * d;
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Tree reduction within the workgroup */
    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* First thread of each group writes partial sum */
    if (lid == 0) {
        partial[group_id] = scratch[0];
    }

    /* Synchronise across workgroups using a global barrier:
       only group 0, thread 0 does the final reduction */
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (group_id == 0 && lid == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < num_groups; i++) {
            total_sum += partial[i];
        }
        output[0] = (half)(total_sum / (float)total);
    }
}
