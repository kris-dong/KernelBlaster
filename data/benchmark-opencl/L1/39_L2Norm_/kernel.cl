#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void l2norm(
    __global const half* x,
    __global half* out,
    const int batch,
    const int dim)
{
    int b = get_global_id(0);
    if (b >= batch) return;

    int offset = b * dim;

    /* Compute sum of squares in float for accuracy */
    float sum_sq = 0.0f;
    for (int d = 0; d < dim; d++) {
        float v = (float)x[offset + d];
        sum_sq = mad(v, v, sum_sq);
    }

    float norm     = sqrt(sum_sq);
    float inv_norm = (norm > 0.0f) ? (1.0f / norm) : 0.0f;

    /* Write normalised values */
    for (int d = 0; d < dim; d++) {
        float v = (float)x[offset + d];
        out[offset + d] = (half)(v * inv_norm);
    }
}
