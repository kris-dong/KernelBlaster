// ---------------------------------------------------------------- dense
__global__ void k_linear(const float *x, const float *w, const float *b, float *y, int M, int K, int N) {
    long i = MB_TID; if (i >= (long)M * N) return;
    int m = i / N, n = i % N;
    double acc = b ? b[n] : 0.0f;
    for (int k = 0; k < K; k++) acc += x[(long)m * K + k] * w[(long)n * K + k];
    y[i] = acc;
}

extern "C" void kernel_linear(const float *input, const float *weight, const float *bias, float *output, int M, int K, int N) {
    k_linear<<<MB_GRID((long)M * N), MB_THREADS>>>(input, weight, bias, output, M, K, N);
}
