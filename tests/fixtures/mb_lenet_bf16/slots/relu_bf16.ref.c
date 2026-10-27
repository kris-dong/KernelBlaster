void kernel_relu_bf16(const __bf16 *input, __bf16 *output, int n) {
    for (int i = 0; i < n; i++) {
        float v = input[i];
        output[i] = v > 0.0f ? v : 0.0f;
    }
}
