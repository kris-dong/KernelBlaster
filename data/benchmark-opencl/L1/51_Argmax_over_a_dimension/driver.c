#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

#define BATCH_SIZE 16
#define DIM1 256
#define DIM2 256
#define ARGMAX_DIM 1
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

/* CPU reference: argmax over dim=1 for shape (BATCH_SIZE, DIM1, DIM2) */
static void argmax_cpu(const cl_half* x, int* out) {
    /* output shape: (BATCH_SIZE, DIM2) */
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < DIM2; c++) {
            float max_val = half_to_float(x[b * DIM1 * DIM2 + 0 * DIM2 + c]);
            int max_idx = 0;
            for (int d = 1; d < DIM1; d++) {
                float val = half_to_float(x[b * DIM1 * DIM2 + d * DIM2 + c]);
                if (val > max_val) {
                    max_val = val;
                    max_idx = d;
                }
            }
            out[b * DIM2 + c] = max_idx;
        }
    }
}

static int save_reference(const int* data, int count, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: Cannot write reference file: %s\n", path); return 0; }
    size_t written = fwrite(data, sizeof(int), count, f);
    fclose(f);
    return (int)written == count;
}

static int load_reference(int* data, int count, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    size_t read_count = fread(data, sizeof(int), count, f);
    fclose(f);
    return (int)read_count == count;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--profile") == 0) g_profile = 1;
        else if (strcmp(argv[i], "--generate-reference") == 0) g_generate_reference = 1;
    }

    size_t input_elems = (size_t)BATCH_SIZE * DIM1 * DIM2;
    size_t input_bytes = input_elems * sizeof(cl_half);
    size_t output_elems = (size_t)BATCH_SIZE * DIM2;
    size_t output_bytes = output_elems * sizeof(cl_int);

    cl_half* h_x   = (cl_half*)malloc(input_bytes);
    cl_int*  h_out = (cl_int*)malloc(output_bytes);
    int*     h_ref = (int*)malloc(output_elems * sizeof(int));

    fill_random_half(h_x, (int)input_elems, 42);

    if (g_generate_reference) {
        printf("Computing CPU reference argmax for shape (%d, %d, %d), dim=1...\n",
               BATCH_SIZE, DIM1, DIM2);
        argmax_cpu(h_x, h_ref);
        if (save_reference(h_ref, (int)output_elems, REFERENCE_FILE)) {
            printf("Reference saved to %s\n", REFERENCE_FILE);
            printf("passed\n");
        } else {
            printf("failed\n");
        }
        free(h_x); free(h_out); free(h_ref);
        return 0;
    }

    int ref_loaded = load_reference(h_ref, (int)output_elems, REFERENCE_FILE);
    if (!ref_loaded) {
        fprintf(stderr, "No cached reference found, computing CPU reference...\n");
        argmax_cpu(h_x, h_ref);
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

    cl_kernel kernel = clCreateKernel(program, "argmax_dim1", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel (%d)\n", err); return 1; }

    cl_mem d_x   = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_bytes, h_x, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: d_x buffer creation (%d)\n", err); return 1; }
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, output_bytes, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: d_out buffer creation (%d)\n", err); return 1; }

    int batch_size_val = BATCH_SIZE;
    int dim1_val = DIM1;
    int dim2_val = DIM2;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
    clSetKernelArg(kernel, 2, sizeof(int), &batch_size_val);
    clSetKernelArg(kernel, 3, sizeof(int), &dim1_val);
    clSetKernelArg(kernel, 4, sizeof(int), &dim2_val);

    /* global: (DIM2, BATCH_SIZE), local: (16, 16) */
    size_t global_work[2] = {DIM2, BATCH_SIZE};
    size_t local_work[2]  = {16, 16};

    /* Warmup */
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work, local_work, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: warmup enqueue (%d)\n", err); return 1; }
    clFinish(queue);

    /* Timed run */
    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work, local_work, 0, NULL,
                                 g_profile ? &event : NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: enqueue (%d)\n", err); return 1; }
    clFinish(queue);

    if (g_profile) { print_event_timing("argmax_dim1", event); clReleaseEvent(event); }

    err = clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, output_bytes, h_out, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: read buffer (%d)\n", err); return 1; }

    int errors = 0;
    for (int i = 0; i < (int)output_elems; i++) {
        int got = (int)h_out[i];
        int ref = h_ref[i];
        if (got != ref) {
            if (errors < 5) {
                fprintf(stderr, "MISMATCH [%d]: got %d, expected %d\n", i, got, ref);
            }
            errors++;
        }
    }

    if (errors == 0) printf("passed\n");
    else printf("failed (%d mismatches)\n", errors);

    clReleaseMemObject(d_x);
    clReleaseMemObject(d_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(h_x); free(h_out); free(h_ref);
    return errors > 0 ? 1 : 0;
}
