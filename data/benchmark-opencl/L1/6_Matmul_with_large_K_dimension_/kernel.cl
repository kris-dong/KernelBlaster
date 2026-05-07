#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int N,
    const int K)
{
    int col = get_global_id(0); /* N dimension (cols) */
    int row = get_global_id(1); /* M dimension (rows) */

    if (row < M && col < N) {
        float sum = 0.0f;
        /* A is M x K, row-major: A[row * K + p]
           B is K x N, row-major: B[p * N + col]
        */
        for (int p = 0; p < K; p++) {
            sum += (float)A[row * K + p] * (float)B[p * N + col];
        }
        C[row * N + col] = (half)sum;
    }
}
