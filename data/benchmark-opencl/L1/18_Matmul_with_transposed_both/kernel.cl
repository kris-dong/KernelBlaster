#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/* Compute C = A.T @ B.T
 * A: (K, M) stored row-major -> A[k][m] = A[k * M + m]
 * B: (N, K) stored row-major -> B[n][k] = B[n * K + k]
 * A.T: (M, K) -> A.T[m][k] = A[k * M + m]
 * B.T: (K, N) -> B.T[k][n] = B[n * K + k]
 * C: (M, N) stored row-major -> C[m][n] = C[m * N + n]
 * C[m][n] = sum_k A.T[m][k] * B.T[k][n]
 *          = sum_k A[k * M + m] * B[n * K + k]
 *
 * global_id(0) = n (column index), global_id(1) = m (row index)
 */
__kernel void matmul_transposed(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    int n = get_global_id(0);
    int m = get_global_id(1);

    if (m < M && n < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += (float)A[k * M + m] * (float)B[n * K + k];
        }
        C[m * N + n] = (half)sum;
    }
}
