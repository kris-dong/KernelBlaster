#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * batchnorm: one work-item per channel.
 * Each work-item computes mean+var over (N,H,W) for its channel,
 * then writes all normalised outputs for that channel.
 *
 * C is fixed at compile time via -DFEATURES=64 baked into the
 * kernel source string.  We keep it simple and pass N,H,W as args.
 *
 * Args (7):
 *   0  x       __global const half*   [N * FEATURES * H * W]
 *   1  weight  __global const half*   [FEATURES]
 *   2  bias    __global const half*   [FEATURES]
 *   3  y       __global       half*   [N * FEATURES * H * W]
 *   4  N       int
 *   5  H       int
 *   6  W       int
 */
__kernel void batchnorm(
    __global const half* x,
    __global const half* weight,
    __global const half* bias,
    __global       half* y,
    const int            N,
    const int            H,
    const int            W)
{
    int c = get_global_id(0);  /* one work-item per channel */

    /* C is the total number of channels = global size = 64 */
    int C   = get_global_size(0);
    int HW  = H * W;
    int NHW = N * HW;

    /* --- pass 1: compute mean --- */
    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        int base = (n * C + c) * HW;
        for (int hw = 0; hw < HW; hw++) {
            sum += (float)x[base + hw];
        }
    }
    float mean = sum / (float)NHW;

    /* --- pass 2: compute variance --- */
    float vsum = 0.0f;
    for (int n = 0; n < N; n++) {
        int base = (n * C + c) * HW;
        for (int hw = 0; hw < HW; hw++) {
            float d = (float)x[base + hw] - mean;
            vsum += d * d;
        }
    }
    float var     = vsum / (float)NHW;
    float inv_std = rsqrt(var + 1e-5f);
    float wc      = (float)weight[c];
    float bc      = (float)bias[c];

    /* --- pass 3: normalise and write output --- */
    for (int n = 0; n < N; n++) {
        int base = (n * C + c) * HW;
        for (int hw = 0; hw < HW; hw++) {
            float xv = (float)x[base + hw];
            float yv = (xv - mean) * inv_std * wc + bc;
            y[base + hw] = (half)yv;
        }
    }
}
