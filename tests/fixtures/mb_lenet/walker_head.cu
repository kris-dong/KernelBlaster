// Shared preamble for every slot (from the ModelBlaster hand kernels).
// Hand-written CUDA implementations of ModelBlaster's per-op C ABI.
//
// Every extern "C" launcher below has EXACTLY the signature of the matching
// `KernelSpec.signature` in modelblaster/pipeline/reference_kernels.py, but
// all pointers are device pointers. That is the seam the plan proposes: a
// GPU backend is "the same kernels.h, implemented in CUDA", and a
// KernelBlaster CUDA problem is the same signature renamed to
// launch_gpu_implementation.
//
// These are deliberately naive (one thread per output element, no tiling):
// they are correctness seeds for the KernelBlaster RL loop, not tuned code.
#include <cuda_runtime.h>
#include <math.h>
#define MB_THREADS 256
// at least one block: a zero-size launch is an invalid configuration (every
// kernel bounds-checks its thread id)
#define MB_GRID(n) (int)(((long)(n) > 0 ? (long)(n) + MB_THREADS - 1 : MB_THREADS) / MB_THREADS)
#define MB_TID ((long)blockIdx.x * blockDim.x + threadIdx.x)
// output index i -> (n, c, od, oh, ow) for an [N, C, OD, OH, OW] tensor
#define MB_IDX5(i, C, OD, OH, OW) \
    int ow = (i) % (OW); int oh = ((i) / (OW)) % (OH); int od = ((i) / ((long)(OW) * (OH))) % (OD); \
    int c_ = ((i) / ((long)(OW) * (OH) * (OD))) % (C); int n_ = (i) / ((long)(OW) * (OH) * (OD) * (C));
