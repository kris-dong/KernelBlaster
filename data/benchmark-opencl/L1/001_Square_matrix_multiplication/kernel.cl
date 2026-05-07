#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_SIZE 16

__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int N)
{
    int row = get_global_id(1);
    int col = get_global_id(0);

    int lr = get_local_id(1);
    int lc = get_local_id(0);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int tiledCol = t * TILE_SIZE + lc;
        int tiledRow = t * TILE_SIZE + lr;

        if (row < N && tiledCol < N)
            tileA[lr][lc] = (float)A[row * N + tiledCol];
        else
            tileA[lr][lc] = 0.0f;

        if (tiledRow < N && col < N)
            tileB[lr][lc] = (float)B[tiledRow * N + col];
        else
            tileB[lr][lc] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            sum = mad(tileA[lr][k], tileB[k][lc], sum);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < N && col < N) {
        C[row * N + col] = (half)sum;
    }
}
