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
#include <torch/torch.h>
#include <iostream>
#include "cuda_model.cuh"

// Declaration for the GPU implementation of 2D Average Pooling
void launch_gpu_implementation(
    void* output,                // pointer to output tensor memory (GPU)
    void* input,                 // pointer to input tensor memory (GPU)
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding
);

int main() {
    // Set default dtype to float16
    torch::Dtype dtype = torch::kFloat16;

    // Device is CUDA
    torch::Device device(torch::kCUDA);

    // Model parameters
    int batch_size = 16;
    int channels = 64;
    int height = 256;
    int width = 256;
    int kernel_size = 3;
    int stride = kernel_size; // As in PyTorch default if stride is None
    int padding = 0;

    // Create random input tensor on GPU with float16 dtype
    torch::Tensor input = torch::randn({batch_size, channels, height, width}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference model (libtorch)
    torch::nn::AvgPool2d avg_pool_layer(torch::nn::AvgPool2dOptions(kernel_size).stride(stride).padding(padding));
    avg_pool_layer->to(device, dtype);

    // Compute reference output
    torch::Tensor ref_output = avg_pool_layer->forward(input);

    // Allocate output tensor for the CUDA implementation (same shape and dtype as reference output)
    torch::Tensor output = torch::empty_like(ref_output, torch::TensorOptions().device(device));

    // Call the CUDA implementation (output and input pointers, and all relevant parameters)
    launch_gpu_implementation(
        output.data_ptr(),                     // void* output
        input.data_ptr(),                      // void* input
        batch_size,
        channels,
        height,
        width,
        kernel_size,
        stride,
        padding
    );

    // Compare outputs using torch::allclose with rtol=1e-1 and atol=1e-1 for fp16
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
