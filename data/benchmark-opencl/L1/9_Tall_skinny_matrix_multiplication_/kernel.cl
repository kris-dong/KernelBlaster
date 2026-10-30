#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/* Tall-skinny matrix multiplication
 * A: (M, K), B: (K, N), C: (M, N)
 * get_global_id(0) -> col in [0, N)
 * get_global_id(1) -> row in [0, M)
 */
__kernel void tall_skinny_matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
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
