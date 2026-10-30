#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void hinge_loss(
    __global const half* predictions,
    __global const half* targets,
    __global half* output,
    __local  float* local_buf,
    const int n)
{
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    float sum = 0.0f;
    for (int i = lid; i < n; i += local_size) {
        float p = (float)predictions[i];
        float t = (float)targets[i];
        float val = 1.0f - p * t;
        sum += (val > 0.0f) ? val : 0.0f;
    }
    local_buf[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_buf[lid] += local_buf[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        output[0] = (half)(local_buf[0] / (float)n);
    }
}
