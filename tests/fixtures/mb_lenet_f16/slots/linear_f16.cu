// fp16 storage, fp32 math: every element read of a __half buffer goes through
// mb_ld; a store of a float/double rounds once (round-to-nearest-even).
__device__ __forceinline__ float mb_ld(__half v) { return __half2float(v); }

// ---------------------------------------------------------------- dense
__global__ void k_linear(const __half *x, const __half *w, const __half *b, __half *y, int M, int K, int N) {
    long i = MB_TID; if (i >= (long)M * N) return;
    int m = i / N, n = i % N;
    double acc = b ? mb_ld(b[n]) : 0.0f;
    for (int k = 0; k < K; k++) acc += mb_ld(x[(long)m * K + k]) * mb_ld(w[(long)n * K + k]);
    y[i] = acc;
}

extern "C" void kernel_linear_f16(const __half *input, const __half *weight, const __half *bias, __half *output, int M, int K, int N) {
    k_linear<<<MB_GRID((long)M * N), MB_THREADS>>>(input, weight, bias, output, M, K, N);
}
