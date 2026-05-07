#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void matmul_mk(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int N,
    const int K)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += (float)A[row * K + k] * (float)B[k * N + col];
        }
        C[row * N + col] = (half)sum;
    }
}
