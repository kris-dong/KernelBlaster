"""System + user prompt for the single-stage CUDA kgen translation agent.

Mirrors ``kgen_opencl`` but targets NVIDIA GPUs and the existing
``compile_and_run_cu_file`` transport: the driver is a LibTorch C++ program
(``driver.cpp``) that computes the reference via ``torch`` ops and compares
GPU output with ``torch::allclose``; the kernel lives in ``final_cuda.cu`` and
must export a ``launch_gpu_implementation`` host launcher whose signature
matches the forward declaration in the driver.

Supported precisions: fp16 (default), fp32, bf16. bf16 is real here because
CUDA supports it natively — unlike Adreno/OpenCL.
"""

# ── Precision-dependent fragments ──────────────────────────────────────────

_PREC_TABLE = {
    "fp16": {
        "torch_dtype": "torch::kFloat16",
        "ctype": "half",
        "cuda_header": "#include <cuda_fp16.h>",
        "buffer_guidance": (
            "   - Uses fp16 (`torch::kFloat16` / `half`) for all tensor buffers, "
            "with float32 accumulation inside the kernel"
        ),
        "dtype_snippet": "torch::Dtype dtype = torch::kFloat16;",
    },
    "fp32": {
        "torch_dtype": "torch::kFloat32",
        "ctype": "float",
        "cuda_header": "#include <cuda_runtime.h>",
        "buffer_guidance": (
            "   - Uses fp32 (`torch::kFloat32` / `float`) for all tensor buffers"
        ),
        "dtype_snippet": "torch::Dtype dtype = torch::kFloat32;",
    },
    "bf16": {
        "torch_dtype": "torch::kBFloat16",
        "ctype": "__nv_bfloat16",
        "cuda_header": "#include <cuda_bf16.h>",
        "buffer_guidance": (
            "   - Uses bf16 (`torch::kBFloat16` / `__nv_bfloat16`) for tensor buffers, "
            "with float32 accumulation inside the kernel"
        ),
        "dtype_snippet": "torch::Dtype dtype = torch::kBFloat16;",
    },
}


# ── Tolerance table — keyed by (precision, problem_class) ─────────────────
#
# Floors derived from K·ε analysis with fp32 accumulation:
#   fp32 ε ≈ 1.2e-7;  fp16 ε ≈ 5e-4;  bf16 ε ≈ 4e-3
# `l1`   : single op, K up to ~4096
# `l2`   : composite forward, ~5 chained matmuls / reductions
# `deep` : backward / sol-level2 / >5 chained matmuls; cuBLAS-vs-naive
#          reduction-tree drift dominates fp32 numerics

_TOLERANCE_TABLE = {
    "l1":   {"fp32": "1e-3", "fp16": "1e-1", "bf16": "1e-1"},
    "l2":   {"fp32": "5e-3", "fp16": "1e-1", "bf16": "1e-1"},
    "deep": {"fp32": "1e-2", "fp16": "1e-1", "bf16": "1e-1"},
}


def resolve_tolerance(precision: str, problem_class: str = "l1") -> str:
    """Return the tolerance string (e.g. '1e-3') for a given precision + class."""
    if problem_class not in _TOLERANCE_TABLE:
        raise ValueError(
            f"Unknown problem_class {problem_class!r}; "
            f"expected one of {sorted(_TOLERANCE_TABLE)}"
        )
    if precision not in _PREC_TABLE:
        raise ValueError(
            f"Unsupported precision {precision!r}; "
            f"expected one of {sorted(_PREC_TABLE)}"
        )
    return _TOLERANCE_TABLE[problem_class][precision]


# ── Worked example: single-square-matmul at N=2048 ─────────────────────────

_EXAMPLE_REF_FP16 = """\
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

_EXAMPLE_DRIVER_FP16 = r"""```cpp
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <iostream>
#include "cuda_model.cuh"

// Forward declaration — must match final_cuda.cu's definition exactly.
void launch_gpu_implementation(
    void* output,
    void* A,
    void* B,
    int64_t N
);

int main() {
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);
    int64_t N = 2048;

    // Deterministic inputs: seed torch's CUDA RNG so driver + rerun match.
    torch::manual_seed(42);

    auto A = torch::randn({N, N}, torch::TensorOptions().dtype(dtype).device(device));
    auto B = torch::randn({N, N}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference via LibTorch. Matmul already respects the input dtype.
    auto ref_output = torch::matmul(A, B);

    // Pre-init to zeros so a no-op kernel is detectable (output != ref unless ref is 0).
    auto gpu_output = torch::zeros_like(ref_output);

    launch_gpu_implementation(
        gpu_output.data_ptr(),
        A.data_ptr(),
        B.data_ptr(),
        N
    );
    cudaDeviceSynchronize();

    bool passed = torch::allclose(gpu_output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;
    return passed ? 0 : 1;
}
```"""

_EXAMPLE_KERNEL_FP16 = r"""```cuda
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void matmul_fp16_kernel(
    half* __restrict__ C,
    const half* __restrict__ A,
    const half* __restrict__ B,
    int64_t N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += __half2float(A[row * N + k]) * __half2float(B[k * N + col]);
    }
    C[row * N + col] = __float2half_rn(sum);
}

// Host launcher — name + signature must exactly match driver.cpp's declaration.
void launch_gpu_implementation(
    void* output,
    void* A,
    void* B,
    int64_t N)
{
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    matmul_fp16_kernel<<<grid, block>>>(
        reinterpret_cast<half*>(output),
        reinterpret_cast<const half*>(A),
        reinterpret_cast<const half*>(B),
        N
    );
}
```"""


# ── Shared guidance ───────────────────────────────────────────────────────

_OUTPUT_FORMAT = """\
## Output format

Output driver.cpp in a ```cpp code block.
Output final_cuda.cu in a ```cuda code block (```cu is also accepted).
Do not include any other code blocks."""

_SHAPE_GUIDANCE = """\
## Translating tensor shapes

- PyTorch tensors are row-major. A 2D tensor (M, N) maps to a flat `T*` buffer
  where `T arr[i * N + j] == tensor[i, j]`.
- For 3D+ tensors (B, M, N), flatten as `arr[b * M * N + i * N + j]`.
- `get_inputs()` defines the input shapes and count. Each input becomes a
  separate `torch::Tensor` on CUDA. Pass each one's `data_ptr()` to the launcher.
- `get_init_inputs()` defines constructor params (kernel_size, etc.) — pass
  them as additional scalar/int64 kernel args.
- For any `nn.Module` with learnable parameters (weights, biases, running
  stats): materialize them with `torch::manual_seed(seed); torch::randn(...)`
  deterministically in the driver and pass their `data_ptr()` to the launcher."""

_OP_GUIDANCE = """\
## Translating PyTorch operations

- `torch.matmul(A, B)` → reference: `torch::matmul(A, B)`; kernel: plain
  nested-loop matmul. Use the SAME accumulation precision as the buffers —
  do not introduce a higher-precision accumulator unless the buffers are
  fp16/bf16 and accumulating in their native type would overflow.
- `torch.relu(x)` → reference: `torch::relu(x)`; kernel: elementwise `max(0, x)`.
- `torch.softmax(x, dim)` → reference: `torch::softmax(x, dim)`; kernel:
  per-slice max/subtract/exp/sum/divide (numerically stable).
- `torch.triu` / `torch.tril` → apply in the CPU reference inside the driver
  AND enforce the masking in the kernel so both zero out the same half.
- `torch.einsum("bijl,lk->bijk", A, B)` → reference: `torch::einsum(...)`;
  kernel: flatten leading dims into a single M=b*i*j, treat as (M, L) @ (L, K).
- `torch.nn.Conv2d` etc. — generate weights with `torch::manual_seed + torch::randn`,
  implement the sliding-window convolution in the kernel, use the same
  weight tensor in the driver's reference (`torch::conv2d`)."""

_QUALITY_GUIDANCE = """\
## Quality guidance — baseline reference, NOT performant

These kernels are baseline reference implementations. They must be correct
and as straightforward as possible. They are NOT meant to be fast — a later
optimization pass is responsible for performance.

Hard rules:
- Do NOT use any external library. No cuBLAS, no cuBLASLt, no cuDNN, no
  Thrust, no CUB. Use only `<cuda_runtime.h>` and the precision-specific
  header (`<cuda_fp16.h>` or `<cuda_bf16.h>` when needed). Do NOT link
  `-lcublas` / `-lcudnn`.
- Do NOT use Tensor Cores / WMMA / mma instructions. Plain `__global__`
  kernels with arithmetic per thread.
- Do NOT introduce shared-memory tiling, warp-level intrinsics
  (`__shfl_*`), atomicAdd-based reductions, or async copies for performance.
  One thread per output element is the default pattern.
- Do NOT change the storage precision. If the buffers are fp32, accumulate
  in fp32. If the buffers are fp16/bf16, accumulate in fp32 inside the kernel
  but never store fp32 intermediate tensors — read fp16/bf16 from global
  memory, accumulate in a local fp32 register, write fp16/bf16 back.
- ABSOLUTELY NO fp64 (`double`, `kFloat64`). The tolerance for this problem
  class is set deliberately wide enough to admit fp32 reduction-order drift.
  Do NOT introduce f32↔f64 conversion kernels or fp64 buffers to "fix"
  accuracy — if you can't pass at the configured tolerance with same-precision
  arithmetic, the answer is to match the driver's reference algorithm to
  the kernel's algorithm (write the reference as a naive loop instead of
  `torch::matmul`), NOT to upgrade precision.
- Do NOT fuse ops for speed. One torch op per kernel is fine. But do NOT
  proliferate unnecessary helpers either: do not write multiple variants of
  the same op (e.g. four transposes for slightly different shapes — use one
  parameterized transpose, or inline the indexing into the consumer).
- Prefer the minimal number of kernels that each do one clear thing. A small
  device-side helper (`__device__ inline`) is fine for index math; a
  separate `__global__` launch for a 5-line operation is usually wasteful.

When in doubt, write the simplest correct kernel."""


_LAUNCHER_GUIDANCE = """\
## Launcher contract

- The host function MUST be named `launch_gpu_implementation`.
- Its signature MUST match exactly between driver.cpp (forward declaration)
  and final_cuda.cu (definition): same return type (`void`), same parameter
  order, same parameter types.
- Parameter convention: tensor data pointers as `void*` (cast inside the .cu),
  scalars as their native type (`float`, `int64_t`, etc.).
- The driver passes `tensor.data_ptr()` for tensor arguments. Inside the
  launcher, `reinterpret_cast` to the appropriate element type (e.g.
  `half*`, `float*`, `__nv_bfloat16*`).
- Do NOT use `extern "C"` — both files are C++ and use the same mangling.
- Always `cudaDeviceSynchronize()` in the driver after the launcher call,
  before `torch::allclose`."""

_DRIVER_REQUIREMENTS = """\
## driver.cpp requirements

- `#include <torch/torch.h>` and `#include <cuda_runtime.h>` (and
  `"cuda_model.cuh"` — the compile server renames final_cuda.cu into that
  include target at build time).
- Has `int main(...)` that:
  1. Sets the tensor dtype/device.
  2. Uses `torch::manual_seed(<fixed_seed>)` before any `torch::randn(...)`.
  3. Allocates inputs on CUDA with the configured dtype.
  4. Computes the reference via LibTorch ops (exactly matching the PyTorch
     reference's forward).
  5. Allocates `gpu_output` with `torch::zeros_like(ref_output)` (so a no-op
     kernel produces a visibly different result).
  6. Calls `launch_gpu_implementation(...)`, then `cudaDeviceSynchronize()`.
  7. `torch::allclose(gpu_output, ref_output, rtol=TOL, atol=TOL)`.
  8. Prints `passed` or `failed` to stdout.
  9. Returns 0 on pass, non-zero on fail."""

_KERNEL_REQUIREMENTS = """\
## final_cuda.cu requirements

- Includes `<cuda_runtime.h>` plus the precision-specific header
  (`<cuda_fp16.h>` for fp16, `<cuda_bf16.h>` for bf16).
- Defines one or more `__global__` kernels as needed.
- Defines `launch_gpu_implementation(...)` as the single host entry point,
  with signature matching driver.cpp's forward declaration exactly.
- Never calls any LibTorch API from inside the .cu (LibTorch headers are not
  available in this translation unit)."""


# ── Builders ──────────────────────────────────────────────────────────────


_PROBLEM_CLASS_DESCRIPTION = {
    "l1": (
        "L1 problem class: a single PyTorch op (matmul, softmax, conv, etc.). "
        "Tolerance is tight; expect bit-near agreement when algorithms match."
    ),
    "l2": (
        "L2 problem class: composite forward (≤5 chained matmuls / reductions). "
        "Tolerance is moderate to admit reduction-order drift."
    ),
    "deep": (
        "Deep problem class: backward passes, sol-level2, or any computation "
        "with >5 chained matmuls. Tolerance is wide because cuBLAS's parallel "
        "reduction tree (used by torch::matmul) drifts from a naive sequential "
        "kernel by O(K·ε) per matmul, which compounds. The widened tolerance "
        "is the budget — do not exceed it via fp64 intermediates."
    ),
}


def build_system_prompt(precision: str = "fp16", problem_class: str = "l1") -> str:
    if precision not in _PREC_TABLE:
        raise ValueError(
            f"Unsupported precision for CUDA kgen: {precision!r} "
            f"(use 'fp16', 'fp32', or 'bf16')"
        )
    if problem_class not in _PROBLEM_CLASS_DESCRIPTION:
        raise ValueError(
            f"Unknown problem_class: {problem_class!r} "
            f"(use 'l1', 'l2', or 'deep')"
        )
    p = _PREC_TABLE[precision]
    tol = resolve_tolerance(precision, problem_class)
    pc_blurb = _PROBLEM_CLASS_DESCRIPTION[problem_class]

    return f"""\
You are an expert C++ and CUDA programmer targeting NVIDIA GPUs (Ada / Hopper / Ampere).
Given a PyTorch reference model, produce two files in a single response.

Problem class: **{problem_class}** — {pc_blurb}
Tolerance: rtol = atol = {tol} for `torch::allclose`.

1. **driver.cpp** — A LibTorch C++ host program that:
   - Compiles with nvcc + LibTorch
{p["buffer_guidance"]}
   - Computes the reference using `torch::` ops (matmul, softmax, etc.)
   - Allocates `gpu_output` as `torch::zeros_like(ref_output)` before calling the kernel
   - Calls `launch_gpu_implementation(...)` — forward-declared at the top of driver.cpp
   - Synchronizes with `cudaDeviceSynchronize()` before comparison
   - Verifies GPU output with `torch::allclose(gpu_output, ref_output, /*rtol=*/{tol}, /*atol=*/{tol})`
   - Prints `"passed"` or `"failed"` to stdout

2. **final_cuda.cu** — A CUDA translation unit that:
   {p["cuda_header"]}
   - Defines one or more `__global__` kernels
   - Exports `launch_gpu_implementation(...)` as the single host entry point

{_QUALITY_GUIDANCE}

{_LAUNCHER_GUIDANCE}

{_DRIVER_REQUIREMENTS}

{_KERNEL_REQUIREMENTS}

{_SHAPE_GUIDANCE}

{_OP_GUIDANCE}

{_OUTPUT_FORMAT}

## Complete example

Given this PyTorch reference:
{_EXAMPLE_REF_FP16}

The driver.cpp should be:
{_EXAMPLE_DRIVER_FP16}

And the final_cuda.cu should be:
{_EXAMPLE_KERNEL_FP16}
"""


# Back-compat default
SYSTEM_PROMPT = build_system_prompt("fp16", "l1")


def build_user_prompt(
    reference_code: str, precision: str = "fp16", problem_class: str = "l1"
) -> str:
    """Build the user prompt from the PyTorch reference code."""
    prec_label = {
        "fp16": "fp16 (half-precision)",
        "fp32": "fp32 (single-precision)",
        "bf16": "bf16 (bfloat16)",
    }.get(precision, precision)
    tol = resolve_tolerance(precision, problem_class)

    prompt = (
        f"Translate the following PyTorch model to CUDA for an NVIDIA GPU using {prec_label}.\n"
        f"Problem class: {problem_class}. Tolerance: rtol = atol = {tol}.\n"
        "Output driver.cpp in a ```cpp code block and final_cuda.cu in a ```cuda code block.\n\n"
        "PyTorch reference:\n"
        f"```python\n{reference_code}\n```"
    )

    if "get_init_inputs" in reference_code:
        init_match = reference_code.split("get_init_inputs", 1)[1]
        if "return []" not in init_match[:120]:
            prompt += (
                "\n\nNote: This model has constructor parameters (see `get_init_inputs`). "
                "In driver.cpp, materialize them as `torch::Tensor` using "
                "`torch::manual_seed(<unique>); torch::randn(...)` with deterministic seeds, "
                "then pass their `data_ptr()` to the launcher alongside the inputs."
            )

    if "nn.BatchNorm" in reference_code:
        prompt += (
            "\n\nBatchNorm hint: generate `weight`, `bias`, `running_mean`, `running_var` "
            "tensors deterministically with `torch::manual_seed`. The reference side can "
            "use `torch::batch_norm(..., training=false)`; the kernel must implement "
            "`y = (x - running_mean) / sqrt(running_var + eps) * weight + bias`."
        )

    if "nn.Conv2d" in reference_code:
        prompt += (
            "\n\nConv2d hint: generate `weight` (and `bias` if present) deterministically. "
            "The reference side can use `torch::conv2d(...)`; the kernel implements the "
            "sliding window with proper padding/stride handling."
        )

    return prompt
