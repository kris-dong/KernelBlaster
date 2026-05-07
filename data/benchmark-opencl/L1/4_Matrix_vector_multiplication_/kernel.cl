#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void matvec(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K)
{
    int col = get_global_id(0); // expected to be 0 (cols=1)
    int row = get_global_id(1);

    if (row < M && col < 1) {
        float sum = 0.0f;
        const int row_off = row * K;
        for (int k = 0; k < K; k++) {
            sum += (float)A[row_off + k] * (float)B[k];
        }
        C[row] = (half)sum;
    }
}
