#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

#define BATCH_SIZE 4096
#define NUM_CLASSES 10
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
        float v = ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
        data[i] = float_to_half(v);
    }
}

static void fill_random_int(int* data, int count, int max_val, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < count; i++) {
        data[i] = rand() % max_val;
    }
}

static float cross_entropy_cpu(const cl_half* predictions, const int* targets, int batch, int num_cls) {
    float total_loss = 0.0f;
    for (int b = 0; b < batch; b++) {
        const cl_half* row = predictions + b * num_cls;
        float max_val = half_to_float(row[0]);
        for (int c = 1; c < num_cls; c++) {
            float v = half_to_float(row[c]);
            if (v > max_val) max_val = v;
        }
        float sum_exp = 0.0f;
        for (int c = 0; c < num_cls; c++) {
            sum_exp += expf(half_to_float(row[c]) - max_val);
        }
        float log_sum_exp = logf(sum_exp) + max_val;
        float pred_target = half_to_float(row[targets[b]]);
        total_loss += log_sum_exp - pred_target;
    }
    return total_loss / (float)batch;
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

    int pred_count = BATCH_SIZE * NUM_CLASSES;

    cl_half* h_predictions = (cl_half*)malloc(pred_count * sizeof(cl_half));
    int*     h_targets     = (int*)malloc(BATCH_SIZE * sizeof(int));
    cl_half* h_output      = (cl_half*)malloc(sizeof(cl_half));
    cl_half* h_ref         = (cl_half*)malloc(sizeof(cl_half));

    fill_random_half(h_predictions, pred_count, 42);
    fill_random_int(h_targets, BATCH_SIZE, NUM_CLASSES, 123);

    if (g_generate_reference) {
        printf("Computing CPU reference cross entropy...\n");
        float loss = cross_entropy_cpu(h_predictions, h_targets, BATCH_SIZE, NUM_CLASSES);
        h_ref[0] = float_to_half(loss);
        printf("CPU reference loss = %f\n", loss);
        if (save_reference(h_ref, 1, REFERENCE_FILE)) {
            printf("Reference saved to %s\n", REFERENCE_FILE);
            printf("passed\n");
        } else { printf("failed\n"); }
        free(h_predictions); free(h_targets); free(h_output); free(h_ref);
        return 0;
    }

    int ref_loaded = load_reference(h_ref, 1, REFERENCE_FILE);
    if (!ref_loaded) {
        fprintf(stderr, "INFO: No cached reference, computing CPU reference...\n");
        float loss = cross_entropy_cpu(h_predictions, h_targets, BATCH_SIZE, NUM_CLASSES);
        h_ref[0] = float_to_half(loss);
    }

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

    /*
     * Single kernel: cross_entropy_loss
     * Args (6): predictions, targets, output, batch_size, num_classes, scratch(local)
     */
    cl_kernel kernel = clCreateKernel(program, "cross_entropy_loss", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel (%d)\n", err); return 1; }

    size_t pred_bytes = pred_count * sizeof(cl_half);
    size_t tgt_bytes  = BATCH_SIZE * sizeof(int);
    size_t out_bytes  = sizeof(cl_half);

    cl_mem d_predictions = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, pred_bytes, h_predictions, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: d_predictions (%d)\n", err); return 1; }
    cl_mem d_targets     = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, tgt_bytes,  h_targets,     &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: d_targets (%d)\n", err); return 1; }
    cl_mem d_output      = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,                        out_bytes,  NULL,          &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: d_output (%d)\n", err); return 1; }

    int batch_val   = BATCH_SIZE;
    int num_cls_val = NUM_CLASSES;
    int local_size  = 256;

    /* 6 args total — must match kernel signature exactly */
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_predictions);          /* arg 0 */
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_targets);              /* arg 1 */
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_output);               /* arg 2 */
    clSetKernelArg(kernel, 3, sizeof(int),    &batch_val);              /* arg 3 */
    clSetKernelArg(kernel, 4, sizeof(int),    &num_cls_val);            /* arg 4 */
    clSetKernelArg(kernel, 5, local_size * sizeof(float), NULL);        /* arg 5: local scratch */

    size_t global_work = (size_t)local_size;
    size_t local_work  = (size_t)local_size;

    /* Warmup */
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work, &local_work, 0, NULL, NULL);
    clFinish(queue);

    /* Timed run */
    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work, &local_work, 0, NULL,
                                 g_profile ? &event : NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: enqueue (%d)\n", err); return 1; }
    clFinish(queue);

    if (g_profile) { print_event_timing("cross_entropy_loss", event); clReleaseEvent(event); }

    err = clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, out_bytes, h_output, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: read buffer (%d)\n", err); return 1; }

    float got  = half_to_float(h_output[0]);
    float ref  = half_to_float(h_ref[0]);
    float diff = fabsf(got - ref);
    float rel  = diff / (fabsf(ref) + 1e-6f);

    fprintf(stderr, "GPU loss=%.6f  CPU loss=%.6f  abs=%.6f  rel=%.6f\n", got, ref, diff, rel);

    int passed = (diff <= TOLERANCE || rel <= TOLERANCE);
    if (passed) printf("passed\n");
    else        printf("failed (got=%f, expected=%f, abs=%f, rel=%f)\n", got, ref, diff, rel);

    clReleaseMemObject(d_predictions);
    clReleaseMemObject(d_targets);
    clReleaseMemObject(d_output);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(h_predictions); free(h_targets); free(h_output); free(h_ref);
    return passed ? 0 : 1;
}
