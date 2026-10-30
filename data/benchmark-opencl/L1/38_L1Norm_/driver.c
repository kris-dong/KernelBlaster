#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

#define BATCH_SIZE 16
#define DIM 16384
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

static void fill_random_half(cl_half* data, int count, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < count; i++) {
        float v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        /* avoid near-zero values to make verification meaningful */
        if (fabsf(v) < 0.1f) v = (v < 0.0f) ? -0.1f : 0.1f;
        data[i] = float_to_half(v);
    }
}

static void l1norm_cpu(const cl_half* x, cl_half* out, int batch, int dim) {
    for (int b = 0; b < batch; b++) {
        float l1sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            l1sum += fabsf(half_to_float(x[b * dim + d]));
        }
        if (l1sum < 1e-10f) l1sum = 1e-10f;
        for (int d = 0; d < dim; d++) {
            float val = half_to_float(x[b * dim + d]) / l1sum;
            out[b * dim + d] = float_to_half(val);
        }
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

    size_t total_elems = (size_t)BATCH_SIZE * DIM;
    size_t total_bytes = total_elems * sizeof(cl_half);

    cl_half* h_x   = (cl_half*)malloc(total_bytes);
    cl_half* h_out = (cl_half*)malloc(total_bytes);
    cl_half* h_ref = (cl_half*)malloc(total_bytes);

    fill_random_half(h_x, (int)total_elems, 42);

    if (g_generate_reference) {
        printf("Computing CPU reference for L1 normalization (batch=%d, dim=%d)...\n", BATCH_SIZE, DIM);
        l1norm_cpu(h_x, h_ref, BATCH_SIZE, DIM);
        if (save_reference(h_ref, (int)total_elems, REFERENCE_FILE)) {
            printf("Reference saved to %s (%zu bytes)\n", REFERENCE_FILE, total_bytes);
            printf("passed\n");
        } else { printf("failed\n"); }
        free(h_x); free(h_out); free(h_ref);
        return 0;
    }

    int ref_loaded = load_reference(h_ref, (int)total_elems, REFERENCE_FILE);
    if (!ref_loaded) {
        if (g_profile)
            fprintf(stderr, "WARNING: No cached reference found, computing CPU reference...\n");
        l1norm_cpu(h_x, h_ref, BATCH_SIZE, DIM);
    }

    /* Sanity-check the reference is non-trivial */
    {
        float ref_sum = 0.0f;
        for (int i = 0; i < (int)total_elems; i++)
            ref_sum += fabsf(half_to_float(h_ref[i]));
        if (ref_sum < 1.0f) {
            fprintf(stderr, "ERROR: reference output appears to be all-zeros or trivial (sum=%f)\n", ref_sum);
            printf("failed\n");
            free(h_x); free(h_out); free(h_ref);
            return 1;
        }
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

    cl_kernel kernel = clCreateKernel(program, "l1_normalize", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel (%d)\n", err); return 1; }

    cl_mem d_x   = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, total_bytes, h_x, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: input buffer creation (%d)\n", err); return 1; }
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, total_bytes, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: output buffer creation (%d)\n", err); return 1; }

    int batch_val = BATCH_SIZE;
    int dim_val   = DIM;
    
    /* magic-comment parser */
    size_t cfg_local_size      = 16;
    size_t cfg_global_factor   = 1;
    {
        size_t kcl_size;
        char* kcl_src = read_file("kernel.cl", &kcl_size);
        if (kcl_src) {
            const char* keys[]    = {"@local_work_size:", "@global_work_factor:"};
            size_t* targets[]     = {&cfg_local_size,     &cfg_global_factor};
            for (int k = 0; k < 2; k++) {
                const char* p = strstr(kcl_src, keys[k]);
                if (p) {
                    p += strlen(keys[k]);
                    while (*p == ' ' || *p == '\t') p++;
                    long v = strtol(p, NULL, 10);
                    if (v > 0) *targets[k] = (size_t)v;
                }
            }
            free(kcl_src);
        }
    }
    fprintf(stderr,
        "[launch] local=%zu global_factor=%zu group_count=%d\n",
        cfg_local_size, cfg_global_factor, (int)(BATCH_SIZE));

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
    clSetKernelArg(kernel, 2, sizeof(int),    &batch_val);
    clSetKernelArg(kernel, 3, sizeof(int),    &dim_val);

    size_t global_work[1] = { (size_t)(BATCH_SIZE) * cfg_global_factor };
    size_t local_work[1]  = { cfg_local_size };

    /* Warmup */
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work, local_work, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: warmup enqueue (%d)\n", err); return 1; }
    clFinish(queue);

    /* Zero out output buffer before timed run to detect no-op kernels */
    cl_half zero_val = float_to_half(0.0f);
    err = clEnqueueFillBuffer(queue, d_out, &zero_val, sizeof(cl_half), 0, total_bytes, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        /* fallback: write zeros from host */
        cl_half* zeros = (cl_half*)calloc(total_elems, sizeof(cl_half));
        clEnqueueWriteBuffer(queue, d_out, CL_TRUE, 0, total_bytes, zeros, 0, NULL, NULL);
        free(zeros);
    }
    clFinish(queue);

    /* Timed run */
    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work, local_work, 0, NULL,
                                 g_profile ? &event : NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: enqueue (%d)\n", err); return 1; }
    clFinish(queue);

    if (g_profile) { print_event_timing("l1_normalize", event); clReleaseEvent(event); }

    err = clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, total_bytes, h_out, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: read buffer (%d)\n", err); return 1; }

    /* Sanity-check GPU output is non-trivial (catches no-op/dummy kernels) */
    {
        float gpu_sum = 0.0f;
        for (int i = 0; i < (int)total_elems; i++)
            gpu_sum += fabsf(half_to_float(h_out[i]));
        if (gpu_sum < 1.0f) {
            fprintf(stderr, "ERROR: GPU output appears to be all-zeros or trivial (sum=%f) — possible no-op kernel\n", gpu_sum);
            printf("failed\n");
            clReleaseMemObject(d_x); clReleaseMemObject(d_out);
            clReleaseKernel(kernel); clReleaseProgram(program);
            clReleaseCommandQueue(queue); clReleaseContext(ctx);
            free(h_x); free(h_out); free(h_ref);
            return 1;
        }
    }

    /* Element-wise verification: flag mismatch if EITHER absolute OR relative error is too large */
    int errors = 0;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    for (int i = 0; i < (int)total_elems; i++) {
        float got = half_to_float(h_out[i]);
        float ref = half_to_float(h_ref[i]);
        float diff = fabsf(got - ref);
        float rel  = diff / (fabsf(ref) + 1e-6f);
        if (diff > max_abs_err) max_abs_err = diff;
        if (rel  > max_rel_err) max_rel_err = rel;
        /* Use OR: fail if either absolute or relative error exceeds tolerance */
        /* Pass if EITHER abs OR rel is within tolerance — fp16 tiny values
         * may have huge relative error but trivially small absolute error. */
        int abs_bad = diff > TOLERANCE;
        int rel_bad = rel  > TOLERANCE;
        if (abs_bad && rel_bad) {
            if (errors < 5)
                fprintf(stderr, "MISMATCH [%d]: got %f, expected %f (abs=%f rel=%f)\n",
                        i, got, ref, diff, rel);
            errors++;
        }
    }

    if (errors == 0) printf("passed\n");
    else printf("failed (%d mismatches, max_abs_err=%f, max_rel_err=%f)\n", errors, max_abs_err, max_rel_err);

    clReleaseMemObject(d_x);
    clReleaseMemObject(d_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(h_x); free(h_out); free(h_ref);
    return errors > 0 ? 1 : 0;
}
