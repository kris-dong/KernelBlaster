// ---------------------------------------------------------------- unary/binary
__global__ void k_relu(const float *x, float *y, int n) { long i = MB_TID; if (i < n) y[i] = fmaxf(x[i], 0.0f); }

extern "C" void kernel_relu(const float *input, float *output, int n) { k_relu<<<MB_GRID(n), MB_THREADS>>>(input, output, n); }
