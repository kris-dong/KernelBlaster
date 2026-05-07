// Tiled matrix multiplication targeting Adreno
// Uses local memory for tile caching and float4 vectorized loads
#define TILE_SIZE 16

__kernel void matmul_tiled(__global const float *A,
                           __global const float *B,
                           __global float *C,
                           const int M,
                           const int N,
                           const int K)
{
    const int row = get_local_id(1);
    const int col = get_local_id(0);
    const int global_row = get_global_id(1);
    const int global_col = get_global_id(0);

    __local float tile_A[TILE_SIZE][TILE_SIZE];
    __local float tile_B[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;

    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++)
    {
        const int t_offset = t * TILE_SIZE;

        // Load tiles into local memory with bounds checking
        if (global_row < M && (t_offset + col) < K)
            tile_A[row][col] = A[global_row * K + t_offset + col];
        else
            tile_A[row][col] = 0.0f;

        if ((t_offset + row) < K && global_col < N)
            tile_B[row][col] = B[(t_offset + row) * N + global_col];
        else
            tile_B[row][col] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++)
        {
            acc += tile_A[row][k] * tile_B[k][col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_row < M && global_col < N)
    {
        C[global_row * N + global_col] = acc;
    }
}
