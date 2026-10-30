/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "cuda_model.cuh"
#include <torch/torch.h>
#include <iostream>

// Declaration for the GPU implementation.
// The input and output pointers are expected to point to GPU memory (torch::kCUDA).
void launch_gpu_implementation(
    void* output,            // Pointer to output tensor memory (GPU)
    void* input,             // Pointer to input tensor memory (GPU)
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation
);

int main() {
    // Set the default dtype to float16 (half)
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Model and input parameters
    int batch_size = 16;
    int channels = 32;
    int height = 128;
    int width = 128;
    int kernel_size = 2;
    int stride = 2;
    int padding = 1;
    int dilation = 3;

    // Create input tensor with random values on CUDA, dtype float16
    torch::Tensor input = torch::randn({batch_size, channels, height, width}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference implementation using libtorch
    torch::nn::MaxPool2d maxpool(torch::nn::MaxPool2dOptions(kernel_size)
                                 .stride(stride)
                                 .padding(padding)
                                 .dilation(dilation));
    maxpool->to(device, dtype);

    torch::Tensor ref_output = maxpool->forward(input);

    // Allocate output tensor for CUDA kernel
    torch::Tensor gpu_output = torch::empty_like(ref_output);

    // Call the CUDA implementation (to be implemented elsewhere)
    launch_gpu_implementation(
        gpu_output.data_ptr(),          // void* output
        input.data_ptr(),               // void* input
        batch_size,
        channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        dilation
    );

    // Compare outputs using torch::allclose with rtol=1e-1, atol=1e-1 for fp16
    bool passed = torch::allclose(gpu_output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
