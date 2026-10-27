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

// ---------------------------------------------------------------- pooling
__global__ void k_maxpool2d(const __nv_bfloat16 *x, __nv_bfloat16 *y, int N, int C, int IH, int IW, int OH, int OW,
                            int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW) {
    long i = MB_TID; long tot = (long)N * C * OH * OW;
    if (i >= tot) return;
    int ow = i % OW; int oh = (i / OW) % OH; long nc = i / ((long)OW * OH);
    float m = -INFINITY;
    for (int kh = 0; kh < KH; kh++) {
        int ih = oh * SH - PH + kh * DH; if (ih < 0 || ih >= IH) continue;
        for (int kw = 0; kw < KW; kw++) {
            int iw = ow * SW - PW + kw * DW; if (iw < 0 || iw >= IW) continue;
            m = fmaxf(m, mb_ld(x[(nc * IH + ih) * IW + iw]));
        }
    }
    y[i] = m;
}

extern "C" void kernel_maxpool2d_bf16(const __nv_bfloat16 *input, __nv_bfloat16 *output, int N, int C, int IH, int IW,
                                 int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW) {
    int OH = (IH + 2 * PH - DH * (KH - 1) - 1) / SH + 1, OW = (IW + 2 * PW - DW * (KW - 1) - 1) / SW + 1;
    long tot = (long)N * C * OH * OW;
    k_maxpool2d<<<mb_blocks(tot), mb_threads(tot)>>>(input, output, N, C, IH, IW, OH, OW, KH, KW, SH, SW, PH, PW, DH, DW);
}
