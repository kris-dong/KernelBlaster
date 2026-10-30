// ---------------------------------------------------------------- pooling
static mb_program cl_maxpool2d = MB_CL(R"CL(
__kernel void k_maxpool2d(__global const float *x, __global float *y, int N, int C, int IH, int IW, int OH, int OW,
                          int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW) {
    long i = MB_TID; long tot = (long)N * C * OH * OW;
    if (i >= tot) return;
    int ow = i % OW; int oh = (i / OW) % OH; long nc = i / ((long)OW * OH);
    float m = -INFINITY;
    for (int kh = 0; kh < KH; kh++) {
        int ih = oh * SH - PH + kh * DH; if (ih < 0 || ih >= IH) continue;
        for (int kw = 0; kw < KW; kw++) {
            int iw = ow * SW - PW + kw * DW; if (iw < 0 || iw >= IW) continue;
            m = fmax(m, x[(nc * IH + ih) * IW + iw]);
        }
    }
    y[i] = m;
}
)CL");

void kernel_maxpool2d(const float *input, float *output, int N, int C, int IH, int IW,
                      int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW) {
    int OH = (IH + 2 * PH - DH * (KH - 1) - 1) / SH + 1, OW = (IW + 2 * PW - DW * (KW - 1) - 1) / SW + 1;
    long tot = (long)N * C * OH * OW;
    mb_launch(MB_KERNEL(cl_maxpool2d, k_maxpool2d), mb_grid(tot), input, output, N, C, IH, IW, OH, OW,
              KH, KW, SH, SW, PH, PW, DH, DW);
}
