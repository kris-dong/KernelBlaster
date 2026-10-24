// ---------------------------------------------------------------- conv2d (direct)
// grouped + dilated direct conv; kernel_conv2d is the G=1 case
__global__ void k_convg(const float *x, const float *w, const float *b, float *y,
                        int N, int IC, int IH, int IW, int OC, int OH, int OW, int KH, int KW,
                        int SH, int SW, int PH, int PW, int DH, int DW, int G) {
    long i = MB_TID; long tot = (long)N * OC * OH * OW;
    if (i >= tot) return;
    int ow = i % OW; int oh = (i / OW) % OH; int oc = (i / ((long)OW * OH)) % OC; int n = i / ((long)OW * OH * OC);
    int ICg = IC / G, ic0 = (oc / (OC / G)) * ICg;
    double acc = b ? b[oc] : 0.0f;
    for (int ic = 0; ic < ICg; ic++)
        for (int kh = 0; kh < KH; kh++) {
            int ih = oh * SH - PH + kh * DH; if (ih < 0 || ih >= IH) continue;
            for (int kw = 0; kw < KW; kw++) {
                int iw = ow * SW - PW + kw * DW; if (iw < 0 || iw >= IW) continue;
                acc += x[(((long)n * IC + ic0 + ic) * IH + ih) * IW + iw] * w[(((long)oc * ICg + ic) * KH + kh) * KW + kw];
            }
        }
    y[i] = acc;
}

extern "C" void kernel_conv2d(const float *input, const float *weight, const float *bias, float *output,
                              int N, int IC, int IH, int IW, int OC, int KH, int KW,
                              int SH, int SW, int PH, int PW, int DH, int DW) {
    int OH = (IH + 2 * PH - DH * (KH - 1) - 1) / SH + 1, OW = (IW + 2 * PW - DW * (KW - 1) - 1) / SW + 1;
    long tot = (long)N * OC * OH * OW;
    k_convg<<<MB_GRID(tot), MB_THREADS>>>(input, weight, bias, output, N, IC, IH, IW, OC, OH, OW, KH, KW,
                                          SH, SW, PH, PW, DH, DW, 1);
}
