#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_SIZE 16

__kernel void matmul_symmetric(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int N)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    int local_col = get_local_id(0);
    int local_row = get_local_id(1);

    float sum = 0.0f;

    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int tA_col = t * TILE_SIZE + local_col;
        int tB_row = t * TILE_SIZE + local_row;

        tileA[local_row][local_col] = (row < N && tA_col < N) ? (float)A[row * N + tA_col] : 0.0f;
        tileB[local_row][local_col] = (tB_row < N && col < N) ? (float)B[tB_row * N + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[local_row][k] * tileB[k][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < N && col < N) {
        C[row * N + col] = (half)sum;
    }
}
