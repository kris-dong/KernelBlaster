#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

/* Problem dimensions: A is (N, M, K), B is (K, L), output C is (N, M, L) */
#define TN 16
#define TM 1024
#define TK 2048
#define TL 768

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

static void fill_random_half(cl_half* data, size_t count, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < count; i++) {
        float v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        data[i] = float_to_half(v);
    }
}

/* CPU reference: C[n,m,l] = sum_k A[n,m,k] * B[k,l] */
static void matmul3d_cpu(const cl_half* A, const cl_half* B, cl_half* C,
                          int n, int m, int k, int l) {
    for (int ni = 0; ni < n; ni++) {
        for (int mi = 0; mi < m; mi++) {
            for (int li = 0; li < l; li++) {
                float sum = 0.0f;
                for (int ki = 0; ki < k; ki++) {
                    sum += half_to_float(A[ni * m * k + mi * k + ki]) *
                           half_to_float(B[ki * l + li]);
                }
                C[ni * m * l + mi * l + li] = float_to_half(sum);
            }
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

    size_t A_elems = (size_t)TN * TM * TK;
    size_t B_elems = (size_t)TK * TL;
    size_t C_elems = (size_t)TN * TM * TL;

    size_t A_bytes = A_elems * sizeof(cl_half);
    size_t B_bytes = B_elems * sizeof(cl_half);
    size_t C_bytes = C_elems * sizeof(cl_half);

    cl_half* h_A   = (cl_half*)malloc(A_bytes);
    cl_half* h_B   = (cl_half*)malloc(B_bytes);
    cl_half* h_C   = (cl_half*)malloc(C_bytes);
    cl_half* h_ref = (cl_half*)malloc(C_bytes);

    fill_random_half(h_A, A_elems, 42);
    fill_random_half(h_B, B_elems, 123);

    if (g_generate_reference) {
        printf("Computing CPU reference (fp16) for 3D matmul N=%d M=%d K=%d L=%d...\n", TN, TM, TK, TL);
        matmul3d_cpu(h_A, h_B, h_ref, TN, TM, TK, TL);
        if (save_reference(h_ref, C_elems, REFERENCE_FILE)) {
            printf("Reference saved to %s\n", REFERENCE_FILE);
            printf("passed\n");
        } else {
            printf("failed\n");
        }
        free(h_A); free(h_B); free(h_C); free(h_ref);
        return 0;
    }

    int ref_loaded = load_reference(h_ref, C_elems, REFERENCE_FILE);
    if (!ref_loaded) {
        fprintf(stderr, "No cached reference found, computing CPU reference...\n");
        matmul3d_cpu(h_A, h_B, h_ref, TN, TM, TK, TL);
    }

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_uint num;

    err = clGetPlatformIDs(1, &platform, &num);
    if (err != CL_SUCCESS || num == 0) { fprintf(stderr, "ERROR: No OpenCL platform found\n"); return 1; }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num);
    if (err != CL_SUCCESS || num == 0) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num);
        if (err != CL_SUCCESS || num == 0) { fprintf(stderr, "ERROR: No OpenCL device found\n"); return 1; }
    }

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateContext (%d)\n", err); return 1; }

    cl_command_queue_properties props = g_profile ? CL_QUEUE_PROFILING_ENABLE : 0;
    cl_command_queue queue = clCreateCommandQueue(ctx, device, props, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateCommandQueue (%d)\n", err); return 1; }

    size_t src_size;
    char* src = read_file("kernel.cl", &src_size);
    if (!src) return 1;

    cl_program program = clCreateProgramWithSource(ctx, 1, (const char**)&src, &src_size, &err);
    free(src);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateProgramWithSource (%d)\n", err); return 1; }

    err = clBuildProgram(program, 1, &device, "-cl-std=CL2.0 -cl-mad-enable", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        fprintf(stderr, "Build error:\n%s\n", log);
        free(log);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "matmul3d", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel (%d)\n", err); return 1; }

    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, A_bytes, h_A, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: d_A buffer (%d)\n", err); return 1; }
    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, B_bytes, h_B, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: d_B buffer (%d)\n", err); return 1; }
    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, C_bytes, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: d_C buffer (%d)\n", err); return 1; }

    int n_val = TN;
    int m_val = TM;
    int k_val = TK;
    int l_val = TL;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &n_val);
    clSetKernelArg(kernel, 4, sizeof(int), &m_val);
    clSetKernelArg(kernel, 5, sizeof(int), &k_val);
    clSetKernelArg(kernel, 6, sizeof(int), &l_val);

    /* global: (L, M, N), local: (16, 16, 1) */
    size_t global_work[3] = {(size_t)TL, (size_t)TM, (size_t)TN};
    size_t local_work[3]  = {16, 16, 1};

    /* Warmup */
    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work, local_work, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: warmup enqueue (%d)\n", err); return 1; }
    clFinish(queue);

    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work, local_work, 0, NULL,
                                 g_profile ? &event : NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: enqueue (%d)\n", err); return 1; }
    clFinish(queue);

    if (g_profile) { print_event_timing("matmul3d", event); clReleaseEvent(event); }

    err = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, C_bytes, h_C, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: read buffer (%d)\n", err); return 1; }

    int errors = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < C_elems; i++) {
        float got = half_to_float(h_C[i]);
        float ref = half_to_float(h_ref[i]);
        float diff = fabsf(got - ref);
        float rel = diff / (fabsf(ref) + 1e-6f);
        if (rel > max_err) max_err = rel;
        if (diff > TOLERANCE && rel > TOLERANCE) {
            if (errors < 5)
                fprintf(stderr, "MISMATCH [%zu]: got %f, expected %f (abs=%f rel=%f)\n",
                        i, got, ref, diff, rel);
            errors++;
        }
    }

    if (errors == 0) printf("passed\n");
    else printf("failed (%d mismatches, max_rel_err=%f)\n", errors, max_err);

    clReleaseMemObject(d_A); clReleaseMemObject(d_B); clReleaseMemObject(d_C);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(ctx);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return errors > 0 ? 1 : 0;
}
