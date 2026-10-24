static mb_program cl_relu = MB_CL(R"CL(
__kernel void k_relu(__global const float *input, __global float *output, int n) {
    long i = MB_TID;
    if (i < n) { float v = input[i]; output[i] = fmax(v, 0.0f); }
}
)CL");
void kernel_relu(const float* input, float* output, int n) {
    mb_launch(MB_KERNEL(cl_relu, k_relu), mb_grid(n), input, output, n);
}
