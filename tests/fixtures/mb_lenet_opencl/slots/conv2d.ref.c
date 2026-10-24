/* conv weights are OIHW unless the backend packed them (generate_skeleton
 * _backend_pack_weight): IHWOC for RVV, HWIO for Gemmini. */
#if defined(MODELBLASTER_GEMMINI_HWIO_WEIGHTS)
#define MB_CONV2D_W(oc, ic, kh, kw) ((((long)(kh)*KW + (kw))*IC + (ic))*OC + (oc))
#elif defined(MODELBLASTER_RVV_IHWOC_WEIGHTS)
#define MB_CONV2D_W(oc, ic, kh, kw) ((((long)(ic)*KH + (kh))*KW + (kw))*OC + (oc))
#else
#define MB_CONV2D_W(oc, ic, kh, kw) ((((long)(oc)*IC + (ic))*KH + (kh))*KW + (kw))
#endif
void kernel_conv2d(const float *input, const float *weight, const float *bias,
                   float *output,
                   int N, int IC, int IH, int IW, int OC,
                   int KH, int KW, int SH, int SW, int PH, int PW,
                   int DH, int DW) {
    int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1;
    int OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    float acc = bias ? bias[oc] : 0.0f;
                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh * DH;
                            if (ih < 0 || ih >= IH) continue;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - PW + kw * DW;
                                if (iw < 0 || iw >= IW) continue;
                                float v = input[((n*IC + ic)*IH + ih)*IW + iw];
                                float w = weight[MB_CONV2D_W(oc, ic, kh, kw)];
                                acc += v * w;
                            }
                        }
                    }
                    output[((n*OC + oc)*OH + oh)*OW + ow] = acc;
                }
            }
        }
    }
}
#undef MB_CONV2D_W
