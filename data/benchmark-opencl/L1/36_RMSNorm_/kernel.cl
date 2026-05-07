#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void rms_norm(
    __global const half* x,
    __global half* out,
    const int num_features,
    const int dim1,
    const int dim2,
    const float eps)
{
    // Each work item handles one (b, d1_idx, d2_idx) tuple
    int d2_idx = get_global_id(0);
    int d1_idx = get_global_id(1);
    int b      = get_global_id(2);

    if (d2_idx >= dim2 || d1_idx >= dim1 || b >= 16) return;

    // Compute mean of squares across feature dimension
    float mean_sq = 0.0f;
    for (int f = 0; f < num_features; f++) {
        int idx = b * num_features * dim1 * dim2
                + f * dim1 * dim2
                + d1_idx * dim2
                + d2_idx;
        float val = (float)x[idx];
        mean_sq += val * val;
    }
    mean_sq /= (float)num_features;
    float rms = sqrt(mean_sq + eps);
    float inv_rms = 1.0f / rms;

    // Normalize each feature
    for (int f = 0; f < num_features; f++) {
        int idx = b * num_features * dim1 * dim2
                + f * dim1 * dim2
                + d1_idx * dim2
                + d2_idx;
        float val = (float)x[idx];
        out[idx] = (half)(val * inv_rms);
    }
}
