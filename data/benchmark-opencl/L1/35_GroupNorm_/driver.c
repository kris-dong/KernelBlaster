#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

#define BATCH_SIZE 16
#define FEATURES   64
#define NUM_GROUPS 8
#define DIM1       256
#define DIM2       256
#define CHANNELS_PER_GROUP (FEATURES / NUM_GROUPS)
#define SPATIAL    (DIM1 * DIM2)
#define TOTAL_ELEMENTS ((size_t)BATCH_SIZE * FEATURES * DIM1 * DIM2)
#define TOLERANCE  1e-1f
#define REFERENCE_FILE "reference_output.bin"
#define EPS 1e-5f

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

static void fill_ones_half(cl_half* data, int count) {
    for (int i = 0; i < count; i++) {
        data[i] = float_to_half(1.0f);
    }
}

static void fill_zeros_half(cl_half* data, int count) {
    for (int i = 0; i < count; i++) {
        data[i] = float_to_half(0.0f);
    }
}

static void group_norm_cpu(
    const cl_half* x,
    const cl_half* weight,
    const cl_half* bias,
    cl_half* out,
    int batch, int features, int num_groups, int spatial, float eps)
{
    int cpg = features / num_groups;
    int group_size = cpg * spatial;

    for (int b = 0; b < batch; b++) {
        for (int g = 0; g < num_groups; g++) {
            int group_ch_start = g * cpg;
            int batch_offset = b * features * spatial;

            /* Compute mean */
            float mean = 0.0f;
            for (int c = 0; c < cpg; c++) {
                int ch = group_ch_start + c;
                int ch_offset = batch_offset + ch * spatial;
                for (int s = 0; s < spatial; s++) {
                    mean += half_to_float(x[ch_offset + s]);
                }
            }
            mean /= (float)group_size;

            /* Compute variance */
            float var = 0.0f;
            for (int c = 0; c < cpg; c++) {
                int ch = group_ch_start + c;
                int ch_offset = batch_offset + ch * spatial;
                for (int s = 0; s < spatial; s++) {
                    float diff = half_to_float(x[ch_offset + s]) - mean;
                    var += diff * diff;
                }
            }
            var /= (float)group_size;

            float inv_std = 1.0f / sqrtf(var + eps);

            /* Normalize and apply affine */
            for (int c = 0; c < cpg; c++) {
                int ch = group_ch_start + c;
                float w  = half_to_float(weight[ch]);
                float bi = half_to_float(bias[ch]);
                int ch_offset = batch_offset + ch * spatial;
                for (int s = 0; s < spatial; s++) {
                    float xn = (half_to_float(x[ch_offset + s]) - mean) * inv_std;
                    out[ch_offset + s] = float_to_half(xn * w + bi);
                }
            }
        }
    }
}

static int save_reference(const cl_half* data, size_t count, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: Cannot write reference file: %s\n", path); return 0; }
    size_t written = fwrite(data, sizeof(cl_half), count, f);
    fclose(f);
    return written == count;
}

static int load_reference(cl_half* data, size_t count, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    size_t read_count = fread(data, sizeof(cl_half), count, f);
    fclose(f);
    return read_count == count;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--profile") == 0) g_profile = 1;
        else if (strcmp(argv[i], "--generate-reference") == 0) g_generate_reference = 1;
    }

    size_t input_elems  = TOTAL_ELEMENTS;
    size_t input_bytes  = input_elems * sizeof(cl_half);
    size_t weight_elems = (size_t)FEATURES;
    size_t weight_bytes = weight_elems * sizeof(cl_half);
    size_t output_elems = TOTAL_ELEMENTS;
    size_t output_bytes = output_elems * sizeof(cl_half);

    cl_half* h_x      = (cl_half*)malloc(input_bytes);
    cl_half* h_weight = (cl_half*)malloc(weight_bytes);
    cl_half* h_bias   = (cl_half*)malloc(weight_bytes);
    cl_half* h_out    = (cl_half*)malloc(output_bytes);
    cl_half* h_ref    = (cl_half*)malloc(output_bytes);

    if (!h_x || !h_weight || !h_bias || !h_out || !h_ref) {
        fprintf(stderr, "ERROR: malloc failed\n");
        return 1;
    }

    /* Generate input data deterministically */
    fill_random_half(h_x, (int)input_elems, 42);

    /* GroupNorm default: weight=1, bias=0 */
    fill_ones_half(h_weight, (int)weight_elems);
    fill_zeros_half(h_bias,  (int)weight_elems);

    if (g_generate_reference) {
        printf("Computing CPU reference for Group Normalization...\n");
        group_norm_cpu(h_x, h_weight, h_bias, h_ref,
                       BATCH_SIZE, FEATURES, NUM_GROUPS, SPATIAL, EPS);
        if (save_reference(h_ref, output_elems, REFERENCE_FILE)) {
            printf("Reference saved to %s\n", REFERENCE_FILE);
            printf("passed\n");
        } else {
            printf("failed\n");
        }
        free(h_x); free(h_weight); free(h_bias); free(h_out); free(h_ref);
        return 0;
    }

    int ref_loaded = load_reference(h_ref, output_elems, REFERENCE_FILE);
    if (!ref_loaded) {
        fprintf(stderr, "No cached reference found, computing CPU reference...\n");
        group_norm_cpu(h_x, h_weight, h_bias, h_ref,
                       BATCH_SIZE, FEATURES, NUM_GROUPS, SPATIAL, EPS);
    }

    /* OpenCL setup */
    cl_int err;
    cl_platform_id platform;
    cl_uint num;

    err = clGetPlatformIDs(1, &platform, &num);
    if (err != CL_SUCCESS || num == 0) {
        fprintf(stderr, "ERROR: No OpenCL platform found\n");
        return 1;
    }

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num);
    if (err != CL_SUCCESS || num == 0) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num);
        if (err != CL_SUCCESS || num == 0) {
            fprintf(stderr, "ERROR: No OpenCL device found\n");
            return 1;
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

    cl_kernel kernel = clCreateKernel(program, "group_norm", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel (%d)\n", err); return 1; }

    cl_mem d_x      = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, input_bytes,  h_x,      &err);
    cl_mem d_weight = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, weight_bytes, h_weight, &err);
    cl_mem d_bias   = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, weight_bytes, h_bias,   &err);
    cl_mem d_out    = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,                         output_bytes, NULL,     &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: buffer creation (%d)\n", err); return 1; }

    int arg_batch      = BATCH_SIZE;
    int arg_features   = FEATURES;
    int arg_num_groups = NUM_GROUPS;
    int arg_spatial    = SPATIAL;
    float arg_eps      = EPS;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_weight);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_bias);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_out);
    clSetKernelArg(kernel, 4, sizeof(int),    &arg_batch);
    clSetKernelArg(kernel, 5, sizeof(int),    &arg_features);
    clSetKernelArg(kernel, 6, sizeof(int),    &arg_num_groups);
    clSetKernelArg(kernel, 7, sizeof(int),    &arg_spatial);
    clSetKernelArg(kernel, 8, sizeof(float),  &arg_eps);

    /* Each work-item handles one (batch, group) pair */
    size_t total_work = (size_t)BATCH_SIZE * NUM_GROUPS;
    size_t local_work = (size_t)NUM_GROUPS;
    size_t global_work = total_work;

    /* Warmup */
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work, &local_work, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: warmup enqueue (%d)\n", err); return 1; }
    clFinish(queue);

    /* Timed run */
    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work, &local_work, 0, NULL,
                                 g_profile ? &event : NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: enqueue (%d)\n", err); return 1; }
    clFinish(queue);

    if (g_profile) { print_event_timing("group_norm", event); clReleaseEvent(event); }

    err = clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, output_bytes, h_out, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: read buffer (%d)\n", err); return 1; }

    /* Verify */
    int errors = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < output_elems; i++) {
        float got  = half_to_float(h_out[i]);
        float ref  = half_to_float(h_ref[i]);
        float diff = fabsf(got - ref);
        float rel  = diff / (fabsf(ref) + 1e-6f);
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

    clReleaseMemObject(d_x);
    clReleaseMemObject(d_weight);
    clReleaseMemObject(d_bias);
    clReleaseMemObject(d_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(h_x); free(h_weight); free(h_bias); free(h_out); free(h_ref);
    return errors > 0 ? 1 : 0;
}
