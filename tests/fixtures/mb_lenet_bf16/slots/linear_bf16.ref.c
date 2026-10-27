void kernel_linear_bf16(const __bf16 *input, const __bf16 *weight, const __bf16 *bias,
                   __bf16 *output, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = bias ? bias[n] : 0.0f;
            for (int k = 0; k < K; k++) {
                acc += input[m * K + k] * weight[n * K + k];
            }
            output[m * N + n] = acc;
        }
    }
}
