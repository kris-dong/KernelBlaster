#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/* Matrix multiply with transposed B:
   C = A * B^T
   A: (M, K), B: (N, K), C: (M, N)
   C[row][col] = sum_k A[row*K+k] * B[col*K+k]

   global_work: (N, M)
   get_global_id(0) = col in [0, N)
   get_global_id(1) = row in [0, M)
*/
__kernel void matmul_transB(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    int col = get_global_id(0);  /* output column in [0, N) */
    int row = get_global_id(1);  /* output row in [0, M) */

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += (float)A[row * K + k] * (float)B[col * K + k];
        }
        C[row * N + col] = (half)sum;
    }
}
