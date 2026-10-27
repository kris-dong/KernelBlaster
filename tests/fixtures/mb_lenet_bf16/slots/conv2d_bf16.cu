// bf16 storage, fp32 math: every element read of a __nv_bfloat16 buffer goes through
// mb_ld; a store of a float/double rounds once (round-to-nearest-even).
__device__ __forceinline__ float mb_ld(__nv_bfloat16 v) { return __bfloat162float(v); }

// one thread per output: blocks of MB_THREADS, halved (down to 32) while that
// leaves an SM without a block -- row-parallel kernels (softmax, layer norm,
// reductions over a dim, small GEMV outputs) have few outputs, and 256-thread
// blocks put them on a handful of SMs. Same per-thread work, so outputs are
// unchanged; large launches keep MB_THREADS.
static inline int mb_sm_count() {
    static int sms = 0;
    if (!sms) {
        int d = 0;
        cudaGetDevice(&d);
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, d) != cudaSuccess || sms <= 0) sms = 1;
    }
    return sms;
}

static inline int mb_threads(long n) {
    int b = MB_THREADS;
    while (b > 32 && (long)b * mb_sm_count() > n) b /= 2;
    return b;
}

// at least one block, as MB_GRID
static inline int mb_blocks(long n) { int b = mb_threads(n); return (int)((n > 0 ? n + b - 1 : b) / b); }

// ---------------------------------------------------------------- conv2d (direct)
// grouped + dilated direct conv; kernel_conv2d is the G=1 case
__global__ void k_convg(const __nv_bfloat16 *x, const __nv_bfloat16 *w, const __nv_bfloat16 *b, __nv_bfloat16 *y,
                        int N, int IC, int IH, int IW, int OC, int OH, int OW, int KH, int KW,
                        int SH, int SW, int PH, int PW, int DH, int DW, int G) {
    long i = MB_TID; long tot = (long)N * OC * OH * OW;
    if (i >= tot) return;
    int ow = i % OW; int oh = (i / OW) % OH; int oc = (i / ((long)OW * OH)) % OC; int n = i / ((long)OW * OH * OC);
    int ICg = IC / G, ic0 = (oc / (OC / G)) * ICg;
    double acc = b ? mb_ld(b[oc]) : 0.0f;
    for (int ic = 0; ic < ICg; ic++)
        for (int kh = 0; kh < KH; kh++) {
            int ih = oh * SH - PH + kh * DH; if (ih < 0 || ih >= IH) continue;
            for (int kw = 0; kw < KW; kw++) {
                int iw = ow * SW - PW + kw * DW; if (iw < 0 || iw >= IW) continue;
                acc += mb_ld(x[(((long)n * IC + ic0 + ic) * IH + ih) * IW + iw]) * mb_ld(w[(((long)oc * ICg + ic) * KH + kh) * KW + kw]);
            }
        }
    y[i] = acc;
}

extern "C" void kernel_conv2d_bf16(const __nv_bfloat16 *input, const __nv_bfloat16 *weight, const __nv_bfloat16 *bias, __nv_bfloat16 *output,
                              int N, int IC, int IH, int IW, int OC, int KH, int KW,
                              int SH, int SW, int PH, int PW, int DH, int DW) {
    int OH = (IH + 2 * PH - DH * (KH - 1) - 1) / SH + 1, OW = (IW + 2 * PW - DW * (KW - 1) - 1) / SW + 1;
    long tot = (long)N * OC * OH * OW;
    k_convg<<<mb_blocks(tot), mb_threads(tot)>>>(input, weight, bias, output, N, IC, IH, IW, OC, OH, OW, KH, KW,
                                          SH, SW, PH, PW, DH, DW, 1);
}
