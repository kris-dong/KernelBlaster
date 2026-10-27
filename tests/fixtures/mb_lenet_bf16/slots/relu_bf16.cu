// bf16 storage, fp32 math: every element read of a __nv_bfloat16 buffer goes through
// mb_ld; a store of a float/double rounds once (round-to-nearest-even).
__device__ __forceinline__ float mb_ld(__nv_bfloat16 v) { return __bfloat162float(v); }

// one thread per output: blocks of MB_THREADS, halved (down to 32) while that
// leaves an SM without a block -- row-parallel kernels (softmax, layer norm,
// reductions over a dim, small GEMV outputs) have few outputs, and 256-thread
// blocks put them on a handful of SMs. Same per-thread work, so outputs are
// unchanged; large launches keep MB_THREADS.
static inline int mb_sm_count() {
    static int sms = 0;
    if (!sms) {
        int d = 0;
        cudaGetDevice(&d);
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, d) != cudaSuccess || sms <= 0) sms = 1;
    }
    return sms;
}

static inline int mb_threads(long n) {
    int b = MB_THREADS;
    while (b > 32 && (long)b * mb_sm_count() > n) b /= 2;
    return b;
}

// at least one block, as MB_GRID
static inline int mb_blocks(long n) { int b = mb_threads(n); return (int)((n > 0 ? n + b - 1 : b) / b); }

// ---------------------------------------------------------------- unary/binary
__global__ void k_relu(const __nv_bfloat16 *x, __nv_bfloat16 *y, int n) { long i = MB_TID; if (i < n) y[i] = fmaxf(mb_ld(x[i]), 0.0f); }

extern "C" void kernel_relu_bf16(const __nv_bfloat16 *input, __nv_bfloat16 *output, int n) { k_relu<<<mb_blocks(n), mb_threads(n)>>>(input, output, n); }
