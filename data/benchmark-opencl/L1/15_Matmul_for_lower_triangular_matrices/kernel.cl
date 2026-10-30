#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void tril_matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int N)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= N || col >= N) return;

    /* Output is lower triangular: upper part is zero */
    if (col > row) {
        C[row * N + col] = (half)0.0f;
        return;
    }

    /* A is lower triangular: A[row][k] != 0 for k <= row
       B is lower triangular: B[k][col] != 0 for col <= k
       Non-zero contribution: col <= k <= row */
    float sum = 0.0f;
    for (int k = col; k <= row; k++) {
        sum += (float)A[row * N + k] * (float)B[k * N + col];
    }
    C[row * N + col] = (half)sum;
}
