void kernel_relu_f16(const _Float16 *input, _Float16 *output, int n) {
    for (int i = 0; i < n; i++) {
        _Float16 v = input[i];
        output[i] = v > (_Float16)0.0f ? v : (_Float16)0.0f;
    }
}
