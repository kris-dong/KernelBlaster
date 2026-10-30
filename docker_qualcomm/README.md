# Qualcomm Adreno OpenCL Docker Environment

Docker build files for compiling and running OpenCL kernels on the Qualcomm RB5 dev board (QRB5165, Adreno 650).

Targets **Ubuntu 20.04 arm64** natively — no cross-compilation needed.

## What's Included

- **OpenCL ICD loader + Adreno SDK headers** (including `cl_ext_qcom.h`)
- **Adreno libOpenCL.so** from the SDK for linking
- **pyopencl** for testing kernels from Python
- **Same Python/LLM stack** as the CUDA image (LangChain, Anthropic, OpenAI, etc.)

## Prerequisites

1. Download the Adreno OpenCL SDK from [Qualcomm Developer](https://developer.qualcomm.com/software/adreno-gpu-sdk)
2. Place the zip in `docker_qualcomm/sdk/` (e.g., `Adreno_OpenCL_SDK.Core.2.4.1.All-AnyCPU-opencl-sdk-2.4.1.zip`)

## Build

```bash
./docker_qualcomm/build.sh
```

## Run

```bash
# Development mode (interactive shell)
./docker_qualcomm/run.sh dev

# Compilation server
./docker_qualcomm/run.sh compile
```

The run script auto-detects `/dev/kgsl-3d0` and passes it into the container.

## Test

Run the test suite inside the container to verify OpenCL compilation works:

```bash
./docker_qualcomm/run.sh dev
# then inside the container:
bash /kernelblaster/docker_qualcomm/test.sh
```

The test builds a C host program that loads two `.cl` kernels (vector_add, tiled matmul) and compiles them via the OpenCL runtime against the Adreno GPU.

## Key Differences from CUDA Docker

| Aspect | CUDA (`docker/`) | Qualcomm (`docker_qualcomm/`) |
|--------|------------------|-------------------------------|
| Base image | `nvcr.io/nvidia/pytorch:25.01-py3` (x86) | `arm64v8/ubuntu:20.04` |
| Target board | Any NVIDIA GPU server | Qualcomm RB5 (QRB5165 / Adreno 650) |
| Compiler | `nvcc` | OpenCL C (compiled JIT by Adreno driver) |
| Runtime | CUDA driver + toolkit | OpenCL ICD + Adreno driver (`libOpenCL.so`) |
| Device access | `--gpus all` (NVIDIA) | `--device /dev/kgsl-3d0` |
| Kernel format | `.cu` (CUDA C++) | `.cl` (OpenCL C) |
| Architecture | x86_64 | aarch64 (native on board) |
