// fp16 storage, fp32 math: every element read of a __half buffer goes through
// mb_ld; a store of a float/double rounds once (round-to-nearest-even).
__device__ __forceinline__ float mb_ld(__half v) { return __half2float(v); }

// ---------------------------------------------------------------- pooling
__global__ void k_maxpool2d(const __half *x, __half *y, int N, int C, int IH, int IW, int OH, int OW,
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

extern "C" void kernel_maxpool2d_f16(const __half *input, __half *output, int N, int C, int IH, int IW,
                                 int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW) {
    int OH = (IH + 2 * PH - DH * (KH - 1) - 1) / SH + 1, OW = (IW + 2 * PW - DW * (KW - 1) - 1) / SW + 1;
    long tot = (long)N * C * OH * OW;
    k_maxpool2d<<<MB_GRID(tot), MB_THREADS>>>(input, output, N, C, IH, IW, OH, OW, KH, KW, SH, SW, PH, PW, DH, DW);
}
