// ---------------------------------------------------------------- dense
static mb_program cl_linear = MB_CL(R"CL(
__kernel void k_linear(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                       int M, int K, int N) {
    long i = MB_TID; if (i >= (long)M * N) return;
    int m = i / N, n = i % N;
    mb_ksum acc = mb_ksum0();
    for (int k = 0; k < K; k++) mb_kadd(&acc, x[(long)m * K + k] * w[(long)n * K + k]);
    y[i] = acc.s + (b ? b[n] : 0.0f);
}
)CL");

void kernel_linear(const float *input, const float *weight, const float *bias, float *output, int M, int K, int N) {
    mb_launch(MB_KERNEL(cl_linear, k_linear), mb_grid((long)M * N), input, weight, bias, output, M, K, N);
}
