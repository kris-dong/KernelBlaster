#include <float.h>

void kernel_maxpool2d(const float *input, float *output,
                     int N, int C, int IH, int IW,
                     int KH, int KW, int SH, int SW,
                     int PH, int PW, int DH, int DW) {
    int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1;
    int OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    float m = -FLT_MAX;
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh*SH - PH + kh*DH;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow*SW - PW + kw*DW;
                            if (iw < 0 || iw >= IW) continue;
                            float v = input[((n*C + c)*IH + ih)*IW + iw];
                            if (v > m) m = v;
                        }
                    }
                    output[((n*C + c)*OH + oh)*OW + ow] = m;
                }
            }
        }
    }
}
