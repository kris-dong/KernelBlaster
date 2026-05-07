#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * smooth_l1_loss: single-pass kernel.
 *
 * Phase 1 (all workgroups): each thread computes element-wise smooth-L1,
 *   then reduces within the workgroup into partial_sums[group_id].
 *
 * Phase 2 (workgroup 0 only): after a global barrier via atomic flag,
 *   workgroup 0 reduces all partial sums and writes the mean as half to output.
 *
 * Parameters (6):
 *   0: predictions  __global const half*
 *   1: targets      __global const half*
 *   2: partial      __global float*        (num_groups floats, scratch)
 *   3: output       __global half*         (1 half scalar result)
 *   4: total        int                    (TOTAL_ELEMENTS)
 *   5: lmem         __local float*         (256 floats, host-allocated local mem)
 */
__kernel void smooth_l1_loss(
    __global const half* predictions,
    __global const half* targets,
    __global volatile float* partial,
    __global half* output,
    const int total,
    __local float* lmem)
{
    int gid      = get_global_id(0);
    int lid      = get_local_id(0);
    int group_id = get_group_id(0);
    int ngroups  = get_num_groups(0);

    /* --- Phase 1: element-wise smooth-L1 + intra-workgroup reduction --- */
    float val = 0.0f;
    if (gid < total) {
        float p        = (float)predictions[gid];
        float t        = (float)targets[gid];
        float d        = p - t;
        float abs_d    = fabs(d);
        val = (abs_d < 1.0f) ? (0.5f * d * d) : (abs_d - 0.5f);
    }

    lmem[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) lmem[lid] += lmem[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partial[group_id] = lmem[0];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    /* --- Phase 2: workgroup 0 reduces partial sums to scalar mean --- */
    if (group_id == 0) {
        float acc = 0.0f;
        for (int i = lid; i < ngroups; i += 256) {
            acc += partial[i];
        }
        lmem[lid] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int stride = 128; stride > 0; stride >>= 1) {
            if (lid < stride) lmem[lid] += lmem[lid + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            output[0] = (half)(lmem[0] / (float)total);
        }
    }
}
