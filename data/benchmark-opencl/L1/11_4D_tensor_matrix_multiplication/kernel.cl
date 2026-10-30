#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * 4D tensor-matrix multiplication:
 *   C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
 *
 * A layout: [b][i][j][l] -> a[b*(I*J*L) + i*(J*L) + j*L + l]
 * B layout: [l][k]       -> b[l*K + k]
 * C layout: [b][i][j][k] -> c[b*(I*J*K) + i*(J*K) + j*K + k]
 *
 * Work mapping:
 *   get_global_id(0) -> k  [0, K)
 *   get_global_id(1) -> j  [0, J)
 *   get_global_id(2) -> bi [0, B*I) where b = bi/I, i = bi%I
 */
__kernel void tensor_matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int B_DIM,
    const int I_DIM,
    const int J_DIM,
    const int L_DIM,
    const int K_DIM)
{
    int k  = get_global_id(0);
    int j  = get_global_id(1);
    int bi = get_global_id(2);

    if (k >= K_DIM || j >= J_DIM || bi >= B_DIM * I_DIM) return;

    int b = bi / I_DIM;
    int i = bi % I_DIM;

    float sum = 0.0f;
    int a_base = b * (I_DIM * J_DIM * L_DIM) + i * (J_DIM * L_DIM) + j * L_DIM;
    for (int l = 0; l < L_DIM; l++) {
        sum += (float)A[a_base + l] * (float)B[l * K_DIM + k];
    }

    int c_idx = b * (I_DIM * J_DIM * K_DIM) + i * (J_DIM * K_DIM) + j * K_DIM + k;
    C[c_idx] = (half)sum;
}
