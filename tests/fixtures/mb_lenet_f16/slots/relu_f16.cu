// fp16 storage, fp32 math: every element read of a __half buffer goes through
// mb_ld; a store of a float/double rounds once (round-to-nearest-even).
__device__ __forceinline__ float mb_ld(__half v) { return __half2float(v); }

// ---------------------------------------------------------------- unary/binary
__global__ void k_relu(const __half *x, __half *y, int n) { long i = MB_TID; if (i < n) y[i] = fmaxf(mb_ld(x[i]), 0.0f); }

extern "C" void kernel_relu_f16(const __half *input, __half *output, int n) { k_relu<<<MB_GRID(n), MB_THREADS>>>(input, output, n); }
