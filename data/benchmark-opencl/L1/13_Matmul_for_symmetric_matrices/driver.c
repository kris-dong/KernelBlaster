#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

#define N 4096
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

/* Generate a random float in [-1, 1] using the current srand state */
static float rand_float(void) {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

/* Fill a symmetric matrix: generate NxN random values then symmetrize as (A + A^T) / 2 */
static void fill_symmetric_half(cl_half* data, int n, unsigned int seed) {
    /* Allocate temporary float buffer */
    float* tmp = (float*)malloc((size_t)n * n * sizeof(float));
    srand(seed);
    for (int i = 0; i < n * n; i++) {
        tmp[i] = rand_float();
    }
    /* Symmetrize: data[i][j] = (tmp[i][j] + tmp[j][i]) / 2 */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float v = (tmp[i * n + j] + tmp[j * n + i]) * 0.5f;
            data[i * n + j] = float_to_half(v);
        }
    }
    free(tmp);
}

/* CPU reference: C = A * B (fp16 inputs, float accumulation)
 * Pre-converts to float to avoid repeated half_to_float in inner loop. */
static void matmul_cpu_half(const cl_half* A, const cl_half* B, cl_half* C, int n) {
    size_t elems = (size_t)n * n;
    float* fA = (float*)malloc(elems * sizeof(float));
    float* fB = (float*)malloc(elems * sizeof(float));
    if (!fA || !fB) { fprintf(stderr, "ERROR: malloc failed in matmul_cpu_half\n"); free(fA); free(fB); return; }

    /* Pre-convert to float */
    for (size_t i = 0; i < elems; i++) fA[i] = half_to_float(A[i]);
    for (size_t i = 0; i < elems; i++) fB[i] = half_to_float(B[i]);

    /* Tiled matmul for cache efficiency */
    #define CTILE 64
    float* fC = (float*)calloc(elems, sizeof(float));
    if (!fC) { fprintf(stderr, "ERROR: malloc failed for fC\n"); free(fA); free(fB); return; }
    for (int ii = 0; ii < n; ii += CTILE) {
        int iend = ii + CTILE < n ? ii + CTILE : n;
        for (int kk = 0; kk < n; kk += CTILE) {
            int kend = kk + CTILE < n ? kk + CTILE : n;
            for (int jj = 0; jj < n; jj += CTILE) {
                int jend = jj + CTILE < n ? jj + CTILE : n;
                for (int i = ii; i < iend; i++) {
                    for (int k = kk; k < kend; k++) {
                        float aik = fA[i * n + k];
                        for (int j = jj; j < jend; j++) {
                            fC[i * n + j] += aik * fB[k * n + j];
                        }
                    }
                }
            }
        }
    }
    #undef CTILE

    /* Convert back to half */
    for (size_t i = 0; i < elems; i++) C[i] = float_to_half(fC[i]);
    free(fA); free(fB); free(fC);
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

    size_t mat_elems = (size_t)N * N;
    size_t mat_bytes = mat_elems * sizeof(cl_half);

    cl_half* h_A   = (cl_half*)malloc(mat_bytes);
    cl_half* h_B   = (cl_half*)malloc(mat_bytes);
    cl_half* h_C   = (cl_half*)malloc(mat_bytes);
    cl_half* h_ref = (cl_half*)malloc(mat_bytes);

    if (!h_A || !h_B || !h_C || !h_ref) {
        fprintf(stderr, "ERROR: malloc failed\n");
        return 1;
    }

    /* Generate symmetric matrices: (A + A^T) / 2 */
    fill_symmetric_half(h_A, N, 42);
    fill_symmetric_half(h_B, N, 123);

    if (g_generate_reference) {
        printf("Computing CPU reference (fp16) for N=%d...\n", N);
        matmul_cpu_half(h_A, h_B, h_ref, N);
        if (save_reference(h_ref, mat_elems, REFERENCE_FILE)) {
            printf("Reference saved to %s (%zu bytes)\n", REFERENCE_FILE, mat_bytes);
            printf("passed\n");
        } else { printf("failed\n"); }
        free(h_A); free(h_B); free(h_C); free(h_ref);
        return 0;
    }

    int ref_loaded = load_reference(h_ref, mat_elems, REFERENCE_FILE);
    if (!ref_loaded) {
        if (g_profile)
            fprintf(stderr, "WARNING: No cached reference found, computing CPU reference...\n");
        matmul_cpu_half(h_A, h_B, h_ref, N);
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

    cl_kernel kernel = clCreateKernel(program, "matmul_symmetric", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel (%d)\n", err); return 1; }

    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mat_bytes, h_A, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: d_A buffer creation (%d)\n", err); return 1; }
    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mat_bytes, h_B, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: d_B buffer creation (%d)\n", err); return 1; }
    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, mat_bytes, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: d_C buffer creation (%d)\n", err); return 1; }

    int n_val = N;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &n_val);

    size_t global_work[2] = {N, N};
    size_t local_work[2] = {16, 16};

    /* Warmup */
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work, local_work, 0, NULL, NULL);
    clFinish(queue);

    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work, local_work, 0, NULL,
                                 g_profile ? &event : NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: enqueue (%d)\n", err); return 1; }
    clFinish(queue);

    if (g_profile) { print_event_timing("matmul_symmetric", event); clReleaseEvent(event); }

    err = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, mat_bytes, h_C, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: read buffer (%d)\n", err); return 1; }

    int errors = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < mat_elems; i++) {
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
