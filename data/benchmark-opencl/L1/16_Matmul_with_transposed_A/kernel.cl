#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/* C = A.T @ B
   A is (K, M) stored row-major: A[k][m] = A[k*M + m]
   B is (K, N) stored row-major: B[k][n] = B[k*N + n]
   C is (M, N) stored row-major: C[m][n] = C[m*N + n]
   C[m][n] = sum_k A[k*M + m] * B[k*N + n]
*/
__kernel void matmul_transA(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    int col = get_global_id(0);  /* n dimension */
    int row = get_global_id(1);  /* m dimension */

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += (float)A[k * M + row] * (float)B[k * N + col];
        }
        C[row * N + col] = (half)sum;
    }
}
