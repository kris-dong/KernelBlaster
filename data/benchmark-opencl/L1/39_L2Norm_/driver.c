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

static void print_event_timing(const char* kernel_name, cl_event event) {
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    double exec_ms = (end - start) / 1e6;
    printf("[PROFILE] %s: %.3f ms\n", kernel_name, exec_ms);
}

static void fill_random_half(cl_half* data, int count, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < count; i++) {
        float v = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        data[i] = float_to_half(v);
    }
}

static void l2norm_cpu(const cl_half* x, cl_half* out, int batch, int dim) {
    for (int b = 0; b < batch; b++) {
        float sum_sq = 0.0f;
        for (int d = 0; d < dim; d++) {
            float v = half_to_float(x[b * dim + d]);
            sum_sq += v * v;
        }
        float norm = sqrtf(sum_sq);
        float inv_norm = (norm > 0.0f) ? (1.0f / norm) : 0.0f;
        for (int d = 0; d < dim; d++) {
            float v = half_to_float(x[b * dim + d]);
            out[b * dim + d] = float_to_half(v * inv_norm);
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

    /* Initialize output buffer to a known non-zero sentinel so we can
       detect a no-op kernel that leaves the buffer untouched. */
    for (size_t i = 0; i < total_elems; i++)
        h_out[i] = float_to_half(999.0f);

    fill_random_half(h_x, (int)total_elems, 42);

    if (g_generate_reference) {
        printf("Computing CPU reference for batch_size=%d, dim=%d...\n", BATCH_SIZE, DIM);
        l2norm_cpu(h_x, h_ref, BATCH_SIZE, DIM);
        if (save_reference(h_ref, (int)total_elems, REFERENCE_FILE)) {
            printf("Reference saved to %s\n", REFERENCE_FILE);
            printf("passed\n");
        } else {
            printf("failed\n");
        }
        free(h_x); free(h_out); free(h_ref);
        return 0;
    }

    int ref_loaded = load_reference(h_ref, (int)total_elems, REFERENCE_FILE);
    if (!ref_loaded) {
        fprintf(stderr, "No cached reference found, computing CPU reference...\n");
        l2norm_cpu(h_x, h_ref, BATCH_SIZE, DIM);
    }

    /* Quick sanity-check the reference: L2-normalised rows should have
       unit norm (within fp16 precision). */
    {
        int ref_ok = 1;
        for (int b = 0; b < BATCH_SIZE && ref_ok; b++) {
            float ss = 0.0f;
            for (int d = 0; d < DIM; d++) {
                float v = half_to_float(h_ref[b * DIM + d]);
                ss += v * v;
            }
            float n = sqrtf(ss);
            if (fabsf(n - 1.0f) > 0.2f) {
                fprintf(stderr,
                    "ERROR: CPU reference row %d has norm %f (expected ~1.0). "
                    "Reference is invalid!\n", b, n);
                ref_ok = 0;
            }
        }
        if (!ref_ok) {
            /* Recompute fresh, ignoring the cached file. */
            l2norm_cpu(h_x, h_ref, BATCH_SIZE, DIM);
        }
    }

    cl_int err;
    cl_platform_id platform;
    cl_uint num_platforms;
    err = clGetPlatformIDs(1, &platform, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "ERROR: No OpenCL platform found\n"); return 1;
    }

    cl_device_id device;
    cl_uint num_devices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) {
            fprintf(stderr, "ERROR: No OpenCL device found\n"); return 1;
        }
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

    cl_kernel kernel = clCreateKernel(program, "l2norm", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel (%d)\n", err); return 1; }

    cl_mem d_x = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 total_bytes, h_x, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: d_x buffer (%d)\n", err); return 1; }

    /* Initialise output buffer on device with sentinel value too. */
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  total_bytes, h_out, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: d_out buffer (%d)\n", err); return 1; }

    int batch_val = BATCH_SIZE;
    int dim_val   = DIM;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
    clSetKernelArg(kernel, 2, sizeof(int), &batch_val);
    clSetKernelArg(kernel, 3, sizeof(int), &dim_val);

    size_t global_work[1] = { (size_t)BATCH_SIZE };
    size_t local_work[1]  = { 16 };

    /* Warmup */
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work, local_work, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: warmup enqueue (%d)\n", err); return 1; }
    clFinish(queue);

    /* Timed run */
    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work, local_work, 0, NULL,
                                 g_profile ? &event : NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: enqueue (%d)\n", err); return 1; }
    clFinish(queue);

    if (g_profile) { print_event_timing("l2norm", event); clReleaseEvent(event); }

    err = clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, total_bytes, h_out, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: read buffer (%d)\n", err); return 1; }

    /* ------------------------------------------------------------------
     * Verification
     * For L2-normalised output every element should be in [-1, 1].
     * We flag a mismatch when EITHER:
     *   absolute error  > TOLERANCE, OR
     *   relative error  > TOLERANCE
     * (using OR, not AND, so a no-op / garbage kernel cannot sneak past).
     * ------------------------------------------------------------------ */
    int errors = 0;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;

    for (size_t i = 0; i < total_elems; i++) {
        float got = half_to_float(h_out[i]);
        float ref = half_to_float(h_ref[i]);
        float diff = fabsf(got - ref);
        float rel  = diff / (fabsf(ref) + 1e-3f);  /* 1e-3 avoids /0 for near-zero ref */

        if (diff > max_abs_err) max_abs_err = diff;
        if (rel  > max_rel_err) max_rel_err = rel;

        /* Fail if EITHER absolute OR relative error is too large */
        if (diff > TOLERANCE || rel > TOLERANCE) {
            if (errors < 10)
                fprintf(stderr,
                    "MISMATCH [%zu] (batch=%zu, dim=%zu): got=%.5f ref=%.5f "
                    "abs=%.5f rel=%.5f\n",
                    i, i / DIM, i % DIM, got, ref, diff, rel);
            errors++;
        }
    }

    /* Extra sanity: verify GPU output rows have unit norm */
    for (int b = 0; b < BATCH_SIZE && errors == 0; b++) {
        float ss = 0.0f;
        for (int d = 0; d < DIM; d++) {
            float v = half_to_float(h_out[b * DIM + d]);
            ss += v * v;
        }
        float n = sqrtf(ss);
        if (fabsf(n - 1.0f) > 0.2f) {
            fprintf(stderr,
                "NORM CHECK FAIL: GPU output row %d has norm %f (expected ~1.0)\n", b, n);
            errors++;
        }
    }

    if (errors == 0)
        printf("passed\n");
    else
        printf("failed (%d mismatches, max_abs_err=%f, max_rel_err=%f)\n",
               errors, max_abs_err, max_rel_err);

    clReleaseMemObject(d_x);
    clReleaseMemObject(d_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(h_x); free(h_out); free(h_ref);
    return errors > 0 ? 1 : 0;
}
