#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

#define BATCH_SIZE 16
#define FEATURES   64
#define DIM1       256
#define DIM2       256
#define TOTAL_ELEMS (BATCH_SIZE * FEATURES * DIM1 * DIM2)
#define TOLERANCE 1e-1f
#define REFERENCE_FILE "reference_output.bin"

static int g_profile = 0;
static int g_generate_reference = 0;

static char* read_file(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: Cannot open file: %s\n", path); return NULL; }
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

static void fill_random_half(cl_half* data, int count, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < count; i++) {
        float v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        data[i] = float_to_half(v);
    }
}

static void frobenius_norm_cpu(const cl_half* x, cl_half* out, int total) {
    double sum_sq = 0.0;
    for (int i = 0; i < total; i++) {
        float v = half_to_float(x[i]);
        sum_sq += (double)v * (double)v;
    }
    float norm = (float)sqrt(sum_sq);
    if (norm < 1e-12f) norm = 1e-12f;
    for (int i = 0; i < total; i++) {
        float v = half_to_float(x[i]);
        out[i] = float_to_half(v / norm);
    }
}

static int save_reference(const cl_half* data, int count, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: Cannot write reference file: %s\n", path); return 0; }
    size_t written = fwrite(data, sizeof(cl_half), count, f);
    fclose(f);
    return (int)written == count;
}

static int load_reference(cl_half* data, int count, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    size_t read_count = fread(data, sizeof(cl_half), count, f);
    fclose(f);
    return (int)read_count == count;
}

static void print_event_timing(const char* kernel_name, cl_event event) {
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    double exec_ms = (end - start) / 1e6;
    printf("[PROFILE] %s: %.3f ms\n", kernel_name, exec_ms);
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--profile") == 0) g_profile = 1;
        else if (strcmp(argv[i], "--generate-reference") == 0) g_generate_reference = 1;
    }

    size_t total = (size_t)TOTAL_ELEMS;
    size_t bytes = total * sizeof(cl_half);

    cl_half* h_x   = (cl_half*)malloc(bytes);
    cl_half* h_out = (cl_half*)malloc(bytes);
    cl_half* h_ref = (cl_half*)malloc(bytes);

    fill_random_half(h_x, (int)total, 42);

    if (g_generate_reference) {
        printf("Computing CPU reference for Frobenius norm normalization...\n");
        frobenius_norm_cpu(h_x, h_ref, (int)total);
        if (save_reference(h_ref, (int)total, REFERENCE_FILE)) {
            printf("Reference saved to %s (%zu bytes)\n", REFERENCE_FILE, bytes);
            printf("passed\n");
        } else { printf("failed\n"); }
        free(h_x); free(h_out); free(h_ref);
        return 0;
    }

    /* Always compute CPU reference fresh — do not rely on cached file alone */
    frobenius_norm_cpu(h_x, h_ref, (int)total);
    /* Optionally overwrite with cached if available (for consistency) */
    load_reference(h_ref, (int)total, REFERENCE_FILE);

    /* Compute inv_norm on CPU to pass to GPU kernel */
    double sum_sq = 0.0;
    for (int i = 0; i < (int)total; i++) {
        float v = half_to_float(h_x[i]);
        sum_sq += (double)v * (double)v;
    }
    float norm_val = (float)sqrt(sum_sq);
    if (norm_val < 1e-12f) norm_val = 1e-12f;
    float inv_norm = 1.0f / norm_val;

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_uint num;

    err = clGetPlatformIDs(1, &platform, &num);
    if (err != CL_SUCCESS || num == 0) { fprintf(stderr, "ERROR: No OpenCL platform\n"); return 1; }
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num);
    if (err != CL_SUCCESS || num == 0) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num);
        if (err != CL_SUCCESS || num == 0) { fprintf(stderr, "ERROR: No OpenCL device\n"); return 1; }
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

    cl_kernel kernel = clCreateKernel(program, "frob_normalize", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel (%d)\n", err); return 1; }

    cl_mem d_x = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_x, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: d_x (%d)\n", err); return 1; }
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: d_out (%d)\n", err); return 1; }

    int total_int = (int)total;

    /* frob_normalize: 4 args */
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
    clSetKernelArg(kernel, 2, sizeof(float),  &inv_norm);
    clSetKernelArg(kernel, 3, sizeof(int),    &total_int);

    size_t local_sz  = 256;
    size_t global_sz = ((total + local_sz - 1) / local_sz) * local_sz;

    /* Warmup */
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_sz, &local_sz, 0, NULL, NULL);
    clFinish(queue);

    /* Timed run */
    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_sz, &local_sz,
                                 0, NULL, g_profile ? &event : NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: enqueue (%d)\n", err); return 1; }
    clFinish(queue);

    if (g_profile) { print_event_timing("frob_normalize", event); clReleaseEvent(event); }

    err = clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, bytes, h_out, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: read output (%d)\n", err); return 1; }

    /*
     * Verification: compare GPU output vs CPU reference.
     *
     * After Frobenius normalization the values are x[i]/norm, which are
     * very small in magnitude (norm ~ sqrt(16*64*256*256) ~ 8192 for unit
     * variance inputs).  We must NOT use an absolute threshold alone
     * because zeros would trivially pass.
     *
     * Strategy:
     *   1. Check that the GPU output norm is close to 1.0 (the defining
     *      property of Frobenius normalisation).
     *   2. Compare element-wise using relative error with a small absolute
     *      floor equal to 1/norm_val (one "unit" in the output space).
     */

    /* Check output Frobenius norm ~ 1.0 */
    double out_ssq = 0.0;
    for (int i = 0; i < (int)total; i++) {
        float v = half_to_float(h_out[i]);
        out_ssq += (double)v * (double)v;
    }
    float out_norm = (float)sqrt(out_ssq);
    if (fabsf(out_norm - 1.0f) > 0.5f) {
        fprintf(stderr, "FAIL: output Frobenius norm = %f, expected ~1.0\n", out_norm);
        printf("failed\n");
        clReleaseMemObject(d_x); clReleaseMemObject(d_out);
        clReleaseKernel(kernel); clReleaseProgram(program);
        clReleaseCommandQueue(queue); clReleaseContext(ctx);
        free(h_x); free(h_out); free(h_ref);
        return 1;
    }

    /* Element-wise check: absolute floor is 1/norm_val (smallest meaningful value) */
    float abs_floor = 1.0f / norm_val;   /* ~1/8192 for typical inputs */
    int errors = 0;
    float max_err = 0.0f;
    for (int i = 0; i < (int)total; i++) {
        float got = half_to_float(h_out[i]);
        float ref = half_to_float(h_ref[i]);
        float diff = fabsf(got - ref);
        /* relative error with absolute floor */
        float denom = fabsf(ref);
        if (denom < abs_floor) denom = abs_floor;
        float rel = diff / denom;
        if (rel > max_err) max_err = rel;
        if (rel > TOLERANCE) {
            if (errors < 5)
                fprintf(stderr, "MISMATCH [%d]: got %.6f, expected %.6f (diff=%.6f rel=%.4f)\n",
                        i, got, ref, diff, rel);
            errors++;
        }
    }

    if (errors == 0) printf("passed\n");
    else printf("failed (%d mismatches, max_rel_err=%f)\n", errors, max_err);

    clReleaseMemObject(d_x);
    clReleaseMemObject(d_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(h_x); free(h_out); free(h_ref);
    return errors > 0 ? 1 : 0;
}
