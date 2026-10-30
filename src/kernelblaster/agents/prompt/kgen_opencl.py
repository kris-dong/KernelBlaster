"""System prompt and helpers for the OpenCL kgen translation agent.

The prompt is parameterized by precision ("fp16" or "fp32").  bf16 is not
supported on Adreno GPUs.
"""

# ── Precision-dependent fragments ──────────────────────────────────────────

_FP16_BUFFER_GUIDANCE = """\
   - Uses fp16 (cl_half) for all data buffers, with float for CPU-side accumulation
   - Verifies GPU output vs CPU reference with `TOLERANCE 1e-1f` (both absolute and relative)"""

_FP32_BUFFER_GUIDANCE = """\
   - Uses fp32 (float) for all data buffers
   - Verifies GPU output vs CPU reference with `TOLERANCE 1e-3f` (both absolute and relative)"""

_FP16_KERNEL_GUIDANCE = """\
   - Starts with `#pragma OPENCL EXTENSION cl_khr_fp16 : enable`
   - Uses `half` type for input/output buffers, `float` for internal accumulation"""

_FP32_KERNEL_GUIDANCE = """\
   - Uses `float` type for all input/output buffers and computation"""

_FP16_HELPERS = r"""
You MUST include these exact helper functions (copy them verbatim — they handle IEEE 754 half-precision correctly):

```c
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
```"""

_FP32_HELPERS = r"""
You MUST include these exact helper functions (copy them verbatim):

```c
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

static void print_event_timing(const char* kernel_name, cl_event event) {
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    double exec_ms = (end - start) / 1e6;
    printf("[PROFILE] %s: %.3f ms\n", kernel_name, exec_ms);
}
```"""

_FP16_MATMUL_OP = """`torch.matmul(A, B)` → nested loops with float accumulation: `sum += half_to_float(A[...]) * half_to_float(B[...])`"""
_FP32_MATMUL_OP = """`torch.matmul(A, B)` → nested loops: `sum += A[...] * B[...]`"""

_FP16_PARAM_NOTE = "For any `nn.Module` with learnable parameters: generate them deterministically using `srand(seed)` with a unique seed per parameter tensor. The kernel receives these as additional `__global const half*` buffers."
_FP32_PARAM_NOTE = "For any `nn.Module` with learnable parameters: generate them deterministically using `srand(seed)` with a unique seed per parameter tensor. The kernel receives these as additional `__global const float*` buffers."

# ── Precision-dependent examples ───────────────────────────────────────────

_FP16_EXAMPLE_REF = """\
```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)

N = 2048

def get_inputs():
    A = torch.randn(N, N, dtype=torch.float16)
    B = torch.randn(N, N, dtype=torch.float16)
    return [A, B]

def get_init_inputs():
    return []
```"""

_FP16_EXAMPLE_DRIVER = r"""```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

#define N 2048
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
        data[i] = float_to_half(v);
    }
}

static void matmul_cpu_half(const cl_half* A, const cl_half* B, cl_half* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += half_to_float(A[i * n + k]) * half_to_float(B[k * n + j]);
            }
            C[i * n + j] = float_to_half(sum);
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

    size_t mat_elems = (size_t)N * N;
    size_t mat_bytes = mat_elems * sizeof(cl_half);

    cl_half* h_A   = (cl_half*)malloc(mat_bytes);
    cl_half* h_B   = (cl_half*)malloc(mat_bytes);
    cl_half* h_C   = (cl_half*)malloc(mat_bytes);
    cl_half* h_ref = (cl_half*)malloc(mat_bytes);

    fill_random_half(h_A, mat_elems, 42);
    fill_random_half(h_B, mat_elems, 123);

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

    cl_kernel kernel = clCreateKernel(program, "matmul", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel (%d)\n", err); return 1; }

    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mat_bytes, h_A, &err);
    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mat_bytes, h_B, &err);
    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, mat_bytes, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: buffer creation (%d)\n", err); return 1; }

    int n_val = N;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &n_val);

    size_t global_work[2] = {N, N};
    size_t local_work[2] = {16, 16};

    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work, local_work, 0, NULL, NULL);
    clFinish(queue);

    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work, local_work, 0, NULL,
                                 g_profile ? &event : NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: enqueue (%d)\n", err); return 1; }
    clFinish(queue);

    if (g_profile) { print_event_timing("matmul", event); clReleaseEvent(event); }

    err = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, mat_bytes, h_C, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: read buffer (%d)\n", err); return 1; }

    int errors = 0;
    float max_err = 0.0f;
    for (int i = 0; i < (int)mat_elems; i++) {
        float got = half_to_float(h_C[i]);
        float ref = half_to_float(h_ref[i]);
        float diff = fabsf(got - ref);
        float rel = diff / (fabsf(ref) + 1e-6f);
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

    clReleaseMemObject(d_A); clReleaseMemObject(d_B); clReleaseMemObject(d_C);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(ctx);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return errors > 0 ? 1 : 0;
}
```"""

_FP16_EXAMPLE_KERNEL = r"""```opencl
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int N)
{
    int row = get_global_id(1);
    int col = get_global_id(0);

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += (float)A[row * N + k] * (float)B[k * N + col];
        }
        C[row * N + col] = (half)sum;
    }
}
```"""

_FP32_EXAMPLE_REF = """\
```python
import torch
import torch.nn as nn

# Use fp32 datatype for all tensors
torch.set_default_dtype(torch.float32)

class Model(nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)

N = 2048

def get_inputs():
    A = torch.randn(N, N)
    B = torch.randn(N, N)
    return [A, B]

def get_init_inputs():
    return []
```"""

_FP32_EXAMPLE_DRIVER = r"""```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

#define N 2048
#define TOLERANCE 1e-3f
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

static void fill_random_float(float* data, int count, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < count; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

static void matmul_cpu(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

static int save_reference(const float* data, int count, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: Cannot write reference file: %s\n", path); return 0; }
    size_t written = fwrite(data, sizeof(float), count, f);
    fclose(f);
    return (int)written == count;
}

static int load_reference(float* data, int count, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    size_t read_count = fread(data, sizeof(float), count, f);
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

    size_t mat_elems = (size_t)N * N;
    size_t mat_bytes = mat_elems * sizeof(float);

    float* h_A   = (float*)malloc(mat_bytes);
    float* h_B   = (float*)malloc(mat_bytes);
    float* h_C   = (float*)malloc(mat_bytes);
    float* h_ref = (float*)malloc(mat_bytes);

    fill_random_float(h_A, mat_elems, 42);
    fill_random_float(h_B, mat_elems, 123);

    if (g_generate_reference) {
        printf("Computing CPU reference (fp32) for N=%d...\n", N);
        matmul_cpu(h_A, h_B, h_ref, N);
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
        matmul_cpu(h_A, h_B, h_ref, N);
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

    cl_kernel kernel = clCreateKernel(program, "matmul", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel (%d)\n", err); return 1; }

    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mat_bytes, h_A, &err);
    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mat_bytes, h_B, &err);
    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, mat_bytes, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: buffer creation (%d)\n", err); return 1; }

    int n_val = N;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &n_val);

    size_t global_work[2] = {N, N};
    size_t local_work[2] = {16, 16};

    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work, local_work, 0, NULL, NULL);
    clFinish(queue);

    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work, local_work, 0, NULL,
                                 g_profile ? &event : NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: enqueue (%d)\n", err); return 1; }
    clFinish(queue);

    if (g_profile) { print_event_timing("matmul", event); clReleaseEvent(event); }

    err = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, mat_bytes, h_C, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: read buffer (%d)\n", err); return 1; }

    int errors = 0;
    float max_err = 0.0f;
    for (int i = 0; i < (int)mat_elems; i++) {
        float got = h_C[i];
        float ref = h_ref[i];
        float diff = fabsf(got - ref);
        float rel = diff / (fabsf(ref) + 1e-6f);
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

    clReleaseMemObject(d_A); clReleaseMemObject(d_B); clReleaseMemObject(d_C);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(ctx);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return errors > 0 ? 1 : 0;
}
```"""

_FP32_EXAMPLE_KERNEL = r"""```opencl
__kernel void matmul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int N)
{
    int row = get_global_id(1);
    int col = get_global_id(0);

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```"""

# ── Required headers (same for both) ──────────────────────────────────────

_REQUIRED_HEADERS = """\
## Required headers in driver.c

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>
```"""

# ── Output format (same for both) ─────────────────────────────────────────

_OUTPUT_FORMAT = """\
## Output format

Output driver.c in a ```c code block.
Output kernel.cl in a ```opencl code block.
Do not include any other code blocks."""

# ── Shared guidance fragments ─────────────────────────────────────────────

_SHAPE_GUIDANCE = """\
## Translating tensor shapes to C arrays

- PyTorch tensors are row-major. A tensor of shape (M, N) maps to `{ctype} arr[M * N]` with `arr[i * N + j]`.
- For 3D+ tensors (B, M, N), flatten as `arr[b * M * N + i * N + j]`.
- `get_inputs()` defines the input shapes and count. Each input becomes a separate host buffer and device buffer.
- `get_init_inputs()` defines constructor parameters (kernel_size, etc.) — pass these as scalar kernel args."""

_WORKSIZE_GUIDANCE = """\
## Global/local work sizes

- For 1D elementwise operations: `global_work = {total_elements}`, `local_work = {256}`
- For 2D operations (matmul, conv2d output): `global_work = {cols, rows}`, `local_work = {16, 16}`
- Always check `if (idx < total)` or `if (row < M && col < N)` in the kernel"""


# ── Build the system prompt ───────────────────────────────────────────────

def build_system_prompt(precision: str = "fp16") -> str:
    if precision == "fp16":
        buffer_guidance = _FP16_BUFFER_GUIDANCE
        kernel_guidance = _FP16_KERNEL_GUIDANCE
        helpers = _FP16_HELPERS
        matmul_op = _FP16_MATMUL_OP
        param_note = _FP16_PARAM_NOTE
        example_ref = _FP16_EXAMPLE_REF
        example_driver = _FP16_EXAMPLE_DRIVER
        example_kernel = _FP16_EXAMPLE_KERNEL
        ctype = "cl_half"
        tolerance_note = "1e-1 for fp16"
    elif precision == "fp32":
        buffer_guidance = _FP32_BUFFER_GUIDANCE
        kernel_guidance = _FP32_KERNEL_GUIDANCE
        helpers = _FP32_HELPERS
        matmul_op = _FP32_MATMUL_OP
        param_note = _FP32_PARAM_NOTE
        example_ref = _FP32_EXAMPLE_REF
        example_driver = _FP32_EXAMPLE_DRIVER
        example_kernel = _FP32_EXAMPLE_KERNEL
        ctype = "float"
        tolerance_note = "1e-3 for fp32"
    else:
        raise ValueError(f"Unsupported precision for OpenCL kgen: {precision!r} (use 'fp16' or 'fp32')")

    shape_guidance = _SHAPE_GUIDANCE.replace("{ctype}", ctype)

    return f"""\
You are an expert C and OpenCL programmer targeting Qualcomm Adreno mobile GPUs. Given a PyTorch reference model, generate two files:

1. **driver.c** — A plain C host program (no C++, no LibTorch) that:
   - Compiles with: `gcc -o main driver.c -lOpenCL -lm -DCL_TARGET_OPENCL_VERSION=200`
   - Loads `kernel.cl` at runtime via `read_file("kernel.cl", &size)`
   - Implements the CPU reference computation in plain C for verification
{buffer_guidance}
   - Generates deterministic inputs using `srand(seed)` with fixed seeds (42, 123, etc.)
   - Prints `"passed"` or `"failed"` to stdout
   - Supports `--profile` flag (enables CL_QUEUE_PROFILING_ENABLE, prints `[PROFILE] kernel_name: X.XXX ms`)
   - Supports `--generate-reference` flag (computes CPU reference, saves to `reference_output.bin`, exits)
   - Loads cached `reference_output.bin` if available (avoids recomputing slow CPU reference)
   - Includes proper OpenCL boilerplate: platform/device discovery, context, queue, program build, kernel creation
   - Calls `clBuildProgram` with flags `"-cl-std=CL2.0 -cl-mad-enable"`
   - Does a warmup kernel dispatch before the timed run
   - Cleans up all OpenCL resources before exit

2. **kernel.cl** — An OpenCL C kernel that:
{kernel_guidance}
   - Defines `__kernel void <name>(...)` matching the driver's `clCreateKernel` call
   - Has correct boundary checks using `get_global_id()`
   - Argument order and types must exactly match the driver's `clSetKernelArg` sequence

## Required helper functions in driver.c
{helpers}

{_REQUIRED_HEADERS}

{_OUTPUT_FORMAT}

## Translating PyTorch operations to C

- {matmul_op}
- `torch.relu(x)` → `max(0, x)` per element
- `torch.softmax(x, dim)` → compute max, subtract, exp, sum, divide per row/slice
- `torch.sum(x, dim)` → accumulate along the specified dimension
- `torch.nn.LayerNorm` → mean, variance, normalize, scale+shift (generate weight/bias with srand)
- `torch.nn.Conv2d` → sliding window with weight kernel (generate weights with srand)
- {param_note}

{shape_guidance}

{_WORKSIZE_GUIDANCE}

## Complete example

Given this PyTorch reference:
{example_ref}

The driver.c should be:
{example_driver}

And the kernel.cl should be:
{example_kernel}
"""


# Keep SYSTEM_PROMPT as the fp16 default for backward compatibility
SYSTEM_PROMPT = build_system_prompt("fp16")


def build_user_prompt(reference_code: str, precision: str = "fp16") -> str:
    """Build the user prompt from the PyTorch reference code."""
    prec_label = "fp16 (half-precision)" if precision == "fp16" else "fp32 (single-precision)"
    prompt = (
        f"Translate the following PyTorch model to OpenCL for a Qualcomm Adreno GPU using {prec_label}.\n"
        "Output driver.c in a ```c code block and kernel.cl in a ```opencl code block.\n\n"
        "PyTorch reference:\n"
        f"```python\n{reference_code}\n```"
    )

    if "get_init_inputs" in reference_code:
        init_match = reference_code.split("get_init_inputs")[1] if "get_init_inputs" in reference_code else ""
        if "return []" not in init_match[:100]:
            prompt += (
                "\n\nNote: This model has constructor parameters (see get_init_inputs). "
                "Generate these parameters deterministically in the driver using "
                "srand() with fixed seeds, matching the parameter shapes from the PyTorch model."
            )

    if "nn.BatchNorm" in reference_code:
        prompt += (
            "\n\nBatchNorm hint: Generate weight, bias, running_mean, running_var "
            "arrays deterministically. The CPU reference must implement: "
            "y = (x - mean) / sqrt(var + eps) * weight + bias"
        )

    if "nn.Conv2d" in reference_code:
        prompt += (
            "\n\nConv2d hint: Generate weight (and bias if present) deterministically. "
            "The CPU reference must implement the sliding window convolution with "
            "proper padding/stride handling."
        )

    return prompt
