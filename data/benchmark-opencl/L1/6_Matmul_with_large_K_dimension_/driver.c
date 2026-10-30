#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

#define M 256
#define N 256
#define K 131072
#define TOLERANCE 1e-1f
#define REFERENCE_FILE "reference_output.bin"

static int g_profile = 0;
static int g_generate_reference = 0;

static char* read_file(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open file: %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(size + 1);
    fread(buf, 1, size, f);
    buf[size] = '\0';
    fclose(f);
    *out_size = size;
    return buf;
}

static float half_to_float(cl_half h) {
    cl_ushort bits = h;
    cl_uint sign = (bits >> 15) & 0x1;
    cl_uint exp  = (bits >> 10) & 0x1f;
    cl_uint mant = bits & 0x3ff;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        float v = (float)mant / 1024.0f;
        v *= (1.0f / 16384.0f);
        return sign ? -v : v;
    }
    if (exp == 31) {
        if (mant == 0) return sign ? -INFINITY : INFINITY;
        return NAN;
    }
    float v = (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)exp - 15.0f);
    return sign ? -v : v;
}

static cl_half float_to_half(float f) {
    union { float f; cl_uint u; } bits;
    bits.f = f;
    cl_uint sign = (bits.u >> 31) & 0x1;
    cl_int  exp  = ((bits.u >> 23) & 0xff) - 127;
    cl_uint mant = bits.u & 0x7fffff;
    if (exp > 15) return (cl_half)((sign << 15) | (31 << 10));
    if (exp < -14) return (cl_half)(sign << 15);
    cl_ushort h_exp  = (cl_ushort)(exp + 15);
    cl_ushort h_mant = (cl_ushort)(mant >> 13);
    return (cl_half)((sign << 15) | (h_exp << 10) | h_mant);
}

static void print_event_timing(const char* kernel_name, cl_event event) {
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    double exec_ms = (end - start) / 1e6;
    printf("[PROFILE] %s: %.3f ms\n", kernel_name, exec_ms);
}

/* Helper: fill buffer with deterministic random floats in [-1,1], converted to half */
static void fill_random_half(cl_half* data, size_t count, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < count; i++) {
        float v = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        data[i] = float_to_half(v);
    }
}

/* CPU reference matmul: C = A * B
   A: M x K, B: K x N, C: M x N
   All stored as cl_half, but accumulation in float.
*/
static void matmul_cpu_half(const cl_half* A, const cl_half* B, cl_half* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                float a = half_to_float(A[(size_t)i * k + p]);
                float b = half_to_float(B[(size_t)p * n + j]);
                sum += a * b;
            }
            C[(size_t)i * n + j] = float_to_half(sum);
        }
    }
}

static int save_reference(const cl_half* data, size_t count, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: Cannot write reference file: %s\n", path); return 0; }
    size_t written = fwrite(data, sizeof(cl_half), count, f);
    fclose(f);
    return (int)written == (int)count;
}

static int load_reference(cl_half* data, size_t count, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    size_t read_count = fread(data, sizeof(cl_half), count, f);
    fclose(f);
    return (int)read_count == (int)count;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--profile") == 0) g_profile = 1;
        else if (strcmp(argv[i], "--generate-reference") == 0) g_generate_reference = 1;
    }

    size_t elems_A = (size_t)M * K;
    size_t elems_B = (size_t)K * N;
    size_t elems_C = (size_t)M * N;
    size_t bytes_A = elems_A * sizeof(cl_half);
    size_t bytes_B = elems_B * sizeof(cl_half);
    size_t bytes_C = elems_C * sizeof(cl_half);

    cl_half* h_A = (cl_half*)malloc(bytes_A);
    cl_half* h_B = (cl_half*)malloc(bytes_B);
    cl_half* h_C = (cl_half*)malloc(bytes_C);
    cl_half* h_ref = (cl_half*)malloc(bytes_C);

    if (!h_A || !h_B || !h_C || !h_ref) {
        fprintf(stderr, "ERROR: Host allocation failed\n");
        return 1;
    }

    /* Deterministic inputs */
    fill_random_half(h_A, elems_A, 42);
    fill_random_half(h_B, elems_B, 123);

    if (g_generate_reference) {
        printf("Computing CPU reference (fp16) for M=%d N=%d K=%d...\n", M, N, K);
        matmul_cpu_half(h_A, h_B, h_ref, M, N, K);
        if (save_reference(h_ref, elems_C, REFERENCE_FILE)) {
            printf("Reference saved to %s (%zu bytes)\n", REFERENCE_FILE, bytes_C);
            printf("passed\n");
        } else {
            printf("failed\n");
        }
        free(h_A); free(h_B); free(h_C); free(h_ref);
        return 0;
    }

    int ref_loaded = load_reference(h_ref, elems_C, REFERENCE_FILE);
    if (!ref_loaded) {
        if (g_profile)
            fprintf(stderr, "WARNING: No cached reference found, computing CPU reference (this may take long)...\n");
        matmul_cpu_half(h_A, h_B, h_ref, M, N, K);
    }

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_uint num_platforms;

    err = clGetPlatformIDs(1, &platform, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "ERROR: No OpenCL platform found\n");
        return 1;
    }

    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) {
            fprintf(stderr, "ERROR: No OpenCL device found\n");
            return 1;
        }
    }

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateContext (%d)\n", err); return 1; }

    cl_command_queue_properties props = g_profile ? CL_QUEUE_PROFILING_ENABLE : 0;
    cl_command_queue queue = clCreateCommandQueue(ctx, device, props, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateCommandQueue (%d)\n", err); clReleaseContext(ctx); return 1; }

    size_t src_size;
    char* src = read_file("kernel.cl", &src_size);
    if (!src) {
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        return 1;
    }

    cl_program program = clCreateProgramWithSource(ctx, 1, (const char**)&src, &src_size, &err);
    free(src);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateProgramWithSource (%d)\n", err); clReleaseCommandQueue(queue); clReleaseContext(ctx); return 1; }

    err = clBuildProgram(program, 1, &device, "-cl-std=CL2.0 -cl-mad-enable", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        fprintf(stderr, "Build error:\n%s\n", log);
        free(log);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "matmul", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel (%d)\n", err); clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(ctx); return 1; }

    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes_A, h_A, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer d_A (%d)\n", err); goto cleanup; }
    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes_B, h_B, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer d_B (%d)\n", err); clReleaseMemObject(d_A); goto cleanup; }
    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes_C, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer d_C (%d)\n", err); clReleaseMemObject(d_A); clReleaseMemObject(d_B); goto cleanup; }

    /* Set kernel args: A, B, C, M, N, K */
    int m_val = M;
    int n_val = N;
    int k_val = K;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &m_val);
    clSetKernelArg(kernel, 4, sizeof(int), &n_val);
    clSetKernelArg(kernel, 5, sizeof(int), &k_val);

    size_t global_work[2] = { (size_t)N, (size_t)M }; /* {cols, rows} */
    size_t local_work[2] = {16, 16};

    /* Warmup run */
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work, local_work, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: warmup enqueue (%d)\n", err); clReleaseMemObject(d_A); clReleaseMemObject(d_B); clReleaseMemObject(d_C); goto cleanup; }
    clFinish(queue);

    /* Timed run */
    cl_event event = NULL;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work, local_work, 0, NULL, g_profile ? &event : NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: enqueue (%d)\n", err); clReleaseMemObject(d_A); clReleaseMemObject(d_B); clReleaseMemObject(d_C); goto cleanup; }
    clFinish(queue);

    if (g_profile && event) {
        print_event_timing("matmul", event);
        clReleaseEvent(event);
    }

    /* Read back result */
    err = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, bytes_C, h_C, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: read buffer (%d)\n", err); clReleaseMemObject(d_A); clReleaseMemObject(d_B); clReleaseMemObject(d_C); goto cleanup; }

    /* Compare to reference */
    int errors = 0;
    float max_rel = 0.0f;
    for (size_t i = 0; i < elems_C; i++) {
        float got = half_to_float(h_C[i]);
        float ref = half_to_float(h_ref[i]);
        float diff = fabsf(got - ref);
        float rel = diff / (fabsf(ref) + 1e-6f);
        if (rel > max_rel) max_rel = rel;
        if (diff > TOLERANCE && rel > TOLERANCE) {
            if (errors < 5) {
                fprintf(stderr, "MISMATCH [%zu]: got %f, expected %f (abs=%f rel=%f)\n",
                        i, got, ref, diff, rel);
            }
            errors++;
        }
    }

    if (errors == 0) printf("passed\n");
    else printf("failed (%d mismatches, max_rel_err=%f)\n", errors, max_rel);

    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);

cleanup:
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    free(h_A); free(h_B); free(h_C); free(h_ref);
    return errors > 0 ? 1 : 0;
}
