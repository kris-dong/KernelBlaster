#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

#define BATCH_SIZE 16
#define FEATURES   64
#define DIM1       256
#define DIM2       256
#define EPS_VAL    1e-5f
#define TOLERANCE  1e-1f
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

static void print_event_timing(const char* kernel_name, cl_event event) {
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    double exec_ms = (end - start) / 1e6;
    printf("[PROFILE] %s: %.3f ms\n", kernel_name, exec_ms);
}

static void instance_norm_cpu(const cl_half* x, cl_half* y,
                               int batch, int channels, int h, int w) {
    int spatial = h * w;
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            int offset = (b * channels + c) * spatial;
            float mean = 0.0f;
            for (int i = 0; i < spatial; i++)
                mean += half_to_float(x[offset + i]);
            mean /= (float)spatial;
            float var = 0.0f;
            for (int i = 0; i < spatial; i++) {
                float d = half_to_float(x[offset + i]) - mean;
                var += d * d;
            }
            var /= (float)spatial;
            float inv_std = 1.0f / sqrtf(var + EPS_VAL);
            for (int i = 0; i < spatial; i++) {
                float val = (half_to_float(x[offset + i]) - mean) * inv_std;
                y[offset + i] = float_to_half(val);
            }
        }
    }
}

static int save_reference(const cl_half* data, int count, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: Cannot write reference: %s\n", path); return 0; }
    size_t written = fwrite(data, sizeof(cl_half), count, f);
    fclose(f);
    return (int)written == count;
}

static int load_reference(cl_half* data, int count, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    size_t n = fread(data, sizeof(cl_half), count, f);
    fclose(f);
    return (int)n == count;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--profile") == 0) g_profile = 1;
        else if (strcmp(argv[i], "--generate-reference") == 0) g_generate_reference = 1;
    }

    int total_elems = BATCH_SIZE * FEATURES * DIM1 * DIM2;
    size_t data_bytes = (size_t)total_elems * sizeof(cl_half);
    int n_slices = BATCH_SIZE * FEATURES;
    int spatial  = DIM1 * DIM2;

    cl_half* h_x   = (cl_half*)malloc(data_bytes);
    cl_half* h_y   = (cl_half*)malloc(data_bytes);
    cl_half* h_ref = (cl_half*)malloc(data_bytes);

    fill_random_half(h_x, total_elems, 42);

    if (g_generate_reference) {
        printf("Computing CPU reference for InstanceNorm...\n");
        instance_norm_cpu(h_x, h_ref, BATCH_SIZE, FEATURES, DIM1, DIM2);
        if (save_reference(h_ref, total_elems, REFERENCE_FILE)) {
            printf("Reference saved to %s\n", REFERENCE_FILE);
            printf("passed\n");
        } else {
            printf("failed\n");
        }
        free(h_x); free(h_y); free(h_ref);
        return 0;
    }

    if (!load_reference(h_ref, total_elems, REFERENCE_FILE)) {
        fprintf(stderr, "No cached reference, computing CPU reference...\n");
        instance_norm_cpu(h_x, h_ref, BATCH_SIZE, FEATURES, DIM1, DIM2);
    }

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_uint num_found;

    err = clGetPlatformIDs(1, &platform, &num_found);
    if (err != CL_SUCCESS || num_found == 0) {
        fprintf(stderr, "ERROR: No OpenCL platform\n"); return 1;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_found);
    if (err != CL_SUCCESS || num_found == 0) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_found);
        if (err != CL_SUCCESS || num_found == 0) {
            fprintf(stderr, "ERROR: No OpenCL device\n"); return 1;
        }
    }

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateContext (%d)\n", err); return 1; }

    cl_command_queue_properties qprops = g_profile ? CL_QUEUE_PROFILING_ENABLE : 0;
    cl_command_queue queue = clCreateCommandQueue(ctx, device, qprops, &err);
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
        free(log); return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "instance_norm", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel instance_norm (%d)\n", err); return 1; }

    cl_mem d_x = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data_bytes, h_x, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: d_x alloc (%d)\n", err); return 1; }
    cl_mem d_y = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, data_bytes, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: d_y alloc (%d)\n", err); return 1; }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_y);
    clSetKernelArg(kernel, 2, sizeof(cl_int), &n_slices);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &spatial);

    size_t local_size  = 256;
    size_t global_size = (size_t)n_slices * local_size;

    /* Warmup */
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: warmup enqueue (%d)\n", err); return 1; }
    clFinish(queue);

    /* Timed run */
    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL,
                                  g_profile ? &event : NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: enqueue (%d)\n", err); return 1; }
    clFinish(queue);

    if (g_profile) {
        print_event_timing("instance_norm", event);
        clReleaseEvent(event);
    }

    err = clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0, data_bytes, h_y, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: read buffer (%d)\n", err); return 1; }

    int errors = 0;
    float max_err = 0.0f;
    for (int i = 0; i < total_elems; i++) {
        float got = half_to_float(h_y[i]);
        float ref = half_to_float(h_ref[i]);
        float diff = fabsf(got - ref);
        float rel  = diff / (fabsf(ref) + 1e-6f);
        if (rel > max_err) max_err = rel;
        if (diff > TOLERANCE && rel > TOLERANCE) {
            if (errors < 5)
                fprintf(stderr, "MISMATCH [%d]: got %f, expected %f (abs=%f rel=%f)\n",
                        i, got, ref, diff, rel);
            errors++;
        }
    }

    if (errors == 0) printf("passed\n");
    else printf("failed (%d mismatches, max_rel_err=%f)\n", errors, max_err);

    clReleaseMemObject(d_x);
    clReleaseMemObject(d_y);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(h_x); free(h_y); free(h_ref);
    return errors > 0 ? 1 : 0;
}
