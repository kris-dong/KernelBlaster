#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * cross_entropy_loss
 * Single-workgroup kernel (256 work-items).
 * Each work-item processes a subset of the batch (strided), computes per-sample
 * cross-entropy loss, accumulates a partial sum, then does a parallel reduction
 * to produce the scalar mean loss.
 *
 * Args (6):
 *   0: predictions   __global const half*   [batch_size * num_classes]
 *   1: targets       __global const int*    [batch_size]
 *   2: output        __global half*         [1]  scalar result
 *   3: batch_size    int
 *   4: num_classes   int
 *   5: scratch       __local float*         [local_size]
 */
__kernel void cross_entropy_loss(
    __global const half* predictions,
    __global const int*  targets,
    __global half*       output,
    const int            batch_size,
    const int            num_classes,
    __local float*       scratch)
{
    int lid   = get_local_id(0);
    int lsize = get_local_size(0);

    float partial = 0.0f;

    /* Each work-item handles batch indices: lid, lid+lsize, lid+2*lsize, ... */
    for (int b = lid; b < batch_size; b += lsize) {
        int offset = b * num_classes;

        /* Find max for numerical stability */
        float max_val = (float)predictions[offset];
        for (int c = 1; c < num_classes; c++) {
            float v = (float)predictions[offset + c];
            if (v > max_val) max_val = v;
        }

        /* Sum exp(x - max) */
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            sum_exp += exp((float)predictions[offset + c] - max_val);
        }

        float log_sum_exp = log(sum_exp) + max_val;
        float pred_target = (float)predictions[offset + targets[b]];
        partial += log_sum_exp - pred_target;
    }

    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Parallel tree reduction */
    for (int stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        output[0] = (half)(scratch[0] / (float)batch_size);
    }
}
