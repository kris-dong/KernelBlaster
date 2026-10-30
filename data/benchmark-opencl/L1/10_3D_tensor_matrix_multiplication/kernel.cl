#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * 3D tensor-matrix multiplication:
 *   A: (N, M, K)
 *   B: (K, L)
 *   C: (N, M, L)  where C[n,m,l] = sum_k A[n,m,k] * B[k,l]
 *
 * Global work size: (L, M, N)
 * Each work-item computes one element C[n, m, l].
 */
__kernel void matmul3d(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int N,
    const int M,
    const int K,
    const int L)
{
    int l = get_global_id(0);
    int m = get_global_id(1);
    int n = get_global_id(2);

    if (n < N && m < M && l < L) {
        float sum = 0.0f;
        int a_base = n * M * K + m * K;
        for (int k = 0; k < K; k++) {
            sum += (float)A[a_base + k] * (float)B[k * L + l];
        }
        C[n * M * L + m * L + l] = (half)sum;
    }
}
