#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/* diag_matmul: C = diag(A) * B
   A: shape (N,) - diagonal elements
   B: shape (N, M)
   C: shape (N, M)
   C[row][col] = A[row] * B[row * M + col]
*/
__kernel void diag_matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int N,
    const int M)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row < N && col < M) {
        float a_val = (float)A[row];
        float b_val = (float)B[row * M + col];
        C[row * M + col] = (half)(a_val * b_val);
    }
}
