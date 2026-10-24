// fp16 storage, fp32 math: every element read of a __half buffer goes through
// mb_ld; a store of a float/double rounds once (round-to-nearest-even).
__device__ __forceinline__ float mb_ld(__half v) { return __half2float(v); }

// ---------------------------------------------------------------- conv2d (direct)
// grouped + dilated direct conv; kernel_conv2d is the G=1 case
__global__ void k_convg(const __half *x, const __half *w, const __half *b, __half *y,
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

extern "C" void kernel_conv2d_f16(const __half *input, const __half *weight, const __half *bias, __half *output,
                              int N, int IC, int IH, int IW, int OC, int KH, int KW,
                              int SH, int SW, int PH, int PW, int DH, int DW) {
    int OH = (IH + 2 * PH - DH * (KH - 1) - 1) / SH + 1, OW = (IW + 2 * PW - DW * (KW - 1) - 1) / SW + 1;
    long tot = (long)N * OC * OH * OW;
    k_convg<<<MB_GRID(tot), MB_THREADS>>>(input, weight, bias, output, N, IC, IH, IW, OC, OH, OW, KH, KW,
                                          SH, SW, PH, PW, DH, DW, 1);
}
