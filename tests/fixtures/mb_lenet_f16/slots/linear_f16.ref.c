void kernel_linear_f16(const _Float16 *input, const _Float16 *weight,
                       const _Float16 *bias, _Float16 *output,
                       int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = bias ? (float)bias[n] : 0.0f;
            for (int k = 0; k < K; k++) {
                acc += (float)input[m * K + k] * (float)weight[n * K + k];
            }
            output[m * N + n] = (_Float16)acc;
        }
    }
}
