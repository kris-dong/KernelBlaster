// Host-side test program that compiles AND executes OpenCL kernels on the Adreno GPU.
// Supports two modes:
//   Default: execute kernels and verify correctness
//   --profile: also emit per-kernel timing via OpenCL event profiling (for cross-referencing ftrace)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

#define VECTOR_N 1024
#define MATRIX_DIM 64

static int g_profile = 0;

static char *read_file(const char *path, size_t *out_size)
{
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        fprintf(stderr, "ERROR: Cannot open file: %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(size + 1);
    fread(buf, 1, size, f);
    buf[size] = '\0';
    fclose(f);
    *out_size = size;
    return buf;
}

static void print_event_timing(const char *kernel_name, cl_event event)
{
    cl_ulong queued, submit, start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(queued), &queued, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(submit), &submit, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);

    double queue_us = (submit - queued) / 1000.0;
    double submit_us = (start - submit) / 1000.0;
    double exec_us = (end - start) / 1000.0;
    double total_us = (end - queued) / 1000.0;

    printf("  [PROFILE] %s:\n", kernel_name);
    printf("    Queue -> Submit:  %10.2f us\n", queue_us);
    printf("    Submit -> Start:  %10.2f us\n", submit_us);
    printf("    Start -> End:     %10.2f us  (GPU execution time)\n", exec_us);
    printf("    Total:            %10.2f us\n", total_us);
    printf("    Raw timestamps:   start=%llu end=%llu\n",
           (unsigned long long)start, (unsigned long long)end);
}

static cl_program build_program(cl_context ctx, cl_device_id device,
                                const char *kernel_path, const char *build_opts)
{
    size_t src_size;
    char *src = read_file(kernel_path, &src_size);
    if (!src)
        return NULL;

    cl_int err;
    const char *src_ptr = src;
    cl_program program = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_size, &err);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "  ERROR: clCreateProgramWithSource failed (%d)\n", err);
        free(src);
        return NULL;
    }

    err = clBuildProgram(program, 1, &device, build_opts, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "  ERROR: Kernel compilation failed (%d)\n", err);
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        fprintf(stderr, "  Build log:\n%s\n", log);
        free(log);
        clReleaseProgram(program);
        free(src);
        return NULL;
    }

    free(src);
    return program;
}

static int test_vector_add(cl_context ctx, cl_device_id device,
                           cl_command_queue queue, const char *test_dir)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/vector_add.cl", test_dir);
    printf("Testing kernel: vector_add (execute on GPU)\n");

    cl_program program = build_program(ctx, device, path, "-cl-std=CL2.0 -cl-mad-enable");
    if (!program)
        return 1;

    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "  ERROR: clCreateKernel failed (%d)\n", err);
        clReleaseProgram(program);
        return 1;
    }

    // vector_add uses float4, so N elements means N/4 float4 work items
    unsigned int n = VECTOR_N / 4;
    size_t buf_size = VECTOR_N * sizeof(float);

    float *h_a = (float *)malloc(buf_size);
    float *h_b = (float *)malloc(buf_size);
    float *h_result = (float *)malloc(buf_size);

    for (int i = 0; i < VECTOR_N; i++)
    {
        h_a[i] = (float)i;
        h_b[i] = (float)(VECTOR_N - i);
    }

    cl_mem d_a = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size, h_a, &err);
    cl_mem d_b = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size, h_b, &err);
    cl_mem d_result = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, buf_size, NULL, &err);

    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "  ERROR: Buffer creation failed (%d)\n", err);
        free(h_a); free(h_b); free(h_result);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return 1;
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_result);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);

    // Warmup run (JIT compilation happens here)
    size_t global_size = n;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    clFinish(queue);

    // Timed run
    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, &event);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "  ERROR: clEnqueueNDRangeKernel failed (%d)\n", err);
        clReleaseMemObject(d_a); clReleaseMemObject(d_b); clReleaseMemObject(d_result);
        free(h_a); free(h_b); free(h_result);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return 1;
    }

    clWaitForEvents(1, &event);

    if (g_profile)
        print_event_timing("vector_add", event);

    // Read results back
    err = clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0, buf_size, h_result, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "  ERROR: clEnqueueReadBuffer failed (%d)\n", err);
        clReleaseEvent(event);
        clReleaseMemObject(d_a); clReleaseMemObject(d_b); clReleaseMemObject(d_result);
        free(h_a); free(h_b); free(h_result);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return 1;
    }

    // Verify results
    int errors = 0;
    for (int i = 0; i < VECTOR_N; i++)
    {
        float expected = (float)VECTOR_N;
        if (fabsf(h_result[i] - expected) > 1e-5f)
        {
            if (errors < 5)
                fprintf(stderr, "  MISMATCH at [%d]: got %f, expected %f\n", i, h_result[i], expected);
            errors++;
        }
    }

    if (errors == 0)
        printf("  PASS: vector_add executed correctly (%d elements verified)\n", VECTOR_N);
    else
        fprintf(stderr, "  FAIL: %d/%d elements incorrect\n", errors, VECTOR_N);

    clReleaseEvent(event);
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_result);
    free(h_a); free(h_b); free(h_result);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return errors > 0 ? 1 : 0;
}

static int test_matmul(cl_context ctx, cl_device_id device,
                       cl_command_queue queue, const char *test_dir)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/matmul_tiled.cl", test_dir);
    printf("Testing kernel: matmul_tiled (execute on GPU)\n");

    cl_program program = build_program(ctx, device, path,
                                       "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math");
    if (!program)
        return 1;

    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "matmul_tiled", &err);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "  ERROR: clCreateKernel failed (%d)\n", err);
        clReleaseProgram(program);
        return 1;
    }

    int M = MATRIX_DIM, N = MATRIX_DIM, K = MATRIX_DIM;
    size_t mat_size = M * N * sizeof(float);

    float *h_A = (float *)malloc(mat_size);
    float *h_B = (float *)malloc(mat_size);
    float *h_C = (float *)malloc(mat_size);
    float *h_C_ref = (float *)calloc(M * N, sizeof(float));

    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            h_A[i * K + j] = (i == j) ? 1.0f : 0.0f;

    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            h_B[i * N + j] = (float)(i * N + j);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < K; k++)
                h_C_ref[i * N + j] += h_A[i * K + k] * h_B[k * N + j];

    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mat_size, h_A, &err);
    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mat_size, h_B, &err);
    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, mat_size, NULL, &err);

    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "  ERROR: Buffer creation failed (%d)\n", err);
        free(h_A); free(h_B); free(h_C); free(h_C_ref);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return 1;
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &M);
    clSetKernelArg(kernel, 4, sizeof(int), &N);
    clSetKernelArg(kernel, 5, sizeof(int), &K);

    size_t global_work[2] = {N, M};
    size_t local_work[2] = {16, 16};

    // Warmup run (JIT compilation happens here)
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work, local_work, 0, NULL, NULL);
    clFinish(queue);

    // Timed run
    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work, local_work, 0, NULL, &event);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "  ERROR: clEnqueueNDRangeKernel failed (%d)\n", err);
        clReleaseMemObject(d_A); clReleaseMemObject(d_B); clReleaseMemObject(d_C);
        free(h_A); free(h_B); free(h_C); free(h_C_ref);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return 1;
    }

    clWaitForEvents(1, &event);

    if (g_profile)
        print_event_timing("matmul_tiled", event);

    err = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, mat_size, h_C, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "  ERROR: clEnqueueReadBuffer failed (%d)\n", err);
        clReleaseEvent(event);
        clReleaseMemObject(d_A); clReleaseMemObject(d_B); clReleaseMemObject(d_C);
        free(h_A); free(h_B); free(h_C); free(h_C_ref);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return 1;
    }

    int errors = 0;
    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++)
    {
        float diff = fabsf(h_C[i] - h_C_ref[i]);
        if (diff > max_err)
            max_err = diff;
        if (diff > 1e-2f)
        {
            if (errors < 5)
                fprintf(stderr, "  MISMATCH at [%d]: got %f, expected %f (diff=%f)\n",
                        i, h_C[i], h_C_ref[i], diff);
            errors++;
        }
    }

    if (errors == 0)
        printf("  PASS: matmul_tiled executed correctly (%dx%d, max_error=%.6f)\n",
               M, N, max_err);
    else
        fprintf(stderr, "  FAIL: %d/%d elements incorrect (max_error=%.6f)\n",
                errors, M * N, max_err);

    clReleaseEvent(event);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return errors > 0 ? 1 : 0;
}

int main(int argc, char **argv)
{
    const char *test_dir = ".";

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--profile") == 0)
            g_profile = 1;
        else
            test_dir = argv[i];
    }

    printf("=== Adreno OpenCL Kernel Compilation Test ===\n");
    if (g_profile)
        printf("Mode: PROFILING (OpenCL event timing enabled)\n");
    printf("\n");

    cl_int err;

    cl_platform_id platform;
    cl_uint num_platforms;
    err = clGetPlatformIDs(1, &platform, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0)
    {
        printf("No OpenCL platform available (cross-compilation mode)\n");
        printf("Verifying kernel sources can be loaded...\n\n");

        char path[512];
        size_t size;
        int failures = 0;

        snprintf(path, sizeof(path), "%s/vector_add.cl", test_dir);
        char *src = read_file(path, &size);
        if (src) { printf("  vector_add.cl: OK (%zu bytes)\n", size); free(src); }
        else failures++;

        snprintf(path, sizeof(path), "%s/matmul_tiled.cl", test_dir);
        src = read_file(path, &size);
        if (src) { printf("  matmul_tiled.cl: OK (%zu bytes)\n", size); free(src); }
        else failures++;

        printf("\n=== Results: %d/2 passed (source loading only) ===\n", 2 - failures);
        return failures > 0 ? 1 : 0;
    }

    cl_device_id device;
    cl_uint num_devices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0)
    {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0)
        {
            fprintf(stderr, "ERROR: No OpenCL devices found\n");
            return 1;
        }
    }

    char device_name[256];
    char device_version[256];
    cl_uint compute_units;
    size_t max_wg_size;
    cl_ulong local_mem;
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(device_version), device_version, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg_size), &max_wg_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem), &local_mem, NULL);
    printf("Device: %s\n", device_name);
    printf("Version: %s\n", device_version);
    printf("Compute units: %u\n", compute_units);
    printf("Max workgroup size: %zu\n", max_wg_size);
    printf("Local memory: %llu bytes\n", (unsigned long long)local_mem);
    printf("\n");

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "ERROR: clCreateContext failed (%d)\n", err);
        return 1;
    }

    // Create command queue with profiling enabled
    cl_command_queue_properties props = CL_QUEUE_PROFILING_ENABLE;
    cl_command_queue queue = clCreateCommandQueue(ctx, device, props, &err);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "ERROR: clCreateCommandQueue failed (%d)\n", err);
        clReleaseContext(ctx);
        return 1;
    }

    int failures = 0;
    int total = 0;

    total++;
    failures += test_vector_add(ctx, device, queue, test_dir);

    total++;
    failures += test_matmul(ctx, device, queue, test_dir);

    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    printf("\n=== Results: %d/%d passed ===\n", total - failures, total);
    return failures > 0 ? 1 : 0;
}
