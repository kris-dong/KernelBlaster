#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void bmm(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int BATCH,
    const int M,
    const int K,
    const int N)
{
    int col = get_global_id(0);   // N (cols)
    int row = get_global_id(1);   // M (rows)
    int batch = get_global_id(2); // batch

    if (batch < BATCH && row < M && col < N) {
        const int a_batch_offset = batch * M * K;
        const int b_batch_offset = batch * K * N;
        const int c_batch_offset = batch * M * N;

        float sum = 0.0f;
        for (int t = 0; t < K; t++) {
            float va = (float)A[a_batch_offset + row * K + t];
            float vb = (float)B[b_batch_offset + t * N + col];
            sum += va * vb;
        }
        C[c_batch_offset + row * N + col] = (half)sum;
    }
}
