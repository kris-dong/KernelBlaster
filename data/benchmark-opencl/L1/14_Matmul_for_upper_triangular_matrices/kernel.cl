#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * matmul_triu: Computes C = triu(A * B) for upper triangular matrices A, B.
 * Since A and B are upper triangular:
 *   - A[i][k] = 0 for k < i
 *   - B[k][j] = 0 for k > j
 * So C[i][j] = sum_{k=i}^{j} A[i][k] * B[k][j] for i <= j, else 0.
 *
 * global_work: {N, N} with local_work {16, 16}
 * get_global_id(0) = col, get_global_id(1) = row
 */
__kernel void matmul_triu(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int N)
{
    int row = get_global_id(1);
    int col = get_global_id(0);

    if (row >= N || col >= N) return;

    if (col < row) {
        /* Below diagonal: triu forces to zero */
        C[row * N + col] = (half)0.0f;
        return;
    }

    /* Upper triangle: sum from k=row to k=col
     * (A is upper triangular so A[row][k]=0 for k<row,
     *  B is upper triangular so B[k][col]=0 for k>col) */
    float sum = 0.0f;
    for (int k = row; k <= col; k++) {
        sum += (float)A[row * N + k] * (float)B[k * N + col];
    }
    C[row * N + col] = (half)sum;
}
