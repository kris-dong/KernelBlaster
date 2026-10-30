#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void matmul(
    __global const half* A, /* M x K */
    __global const half* B, /* K x N */
    __global half* C,       /* M x N */
    const int M,
    const int K,
    const int N)
{
    int col = get_global_id(0); /* N dimension */
    int row = get_global_id(1); /* M dimension */

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int t = 0; t < K; t++) {
            float a = (float)A[row * K + t];
            float b = (float)B[t * N + col];
            sum += a * b;
        }
        C[row * N + col] = (half)sum;
    }
}
