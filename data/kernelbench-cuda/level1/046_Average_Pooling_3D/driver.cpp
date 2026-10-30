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

// Declaration of the GPU implementation
void launch_gpu_implementation(
    void* output,           // Output tensor pointer (GPU memory)
    void* input,            // Input tensor pointer (GPU memory)
    int batch_size,
    int channels,
    int depth,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding
);

int main() {
    // Set device and dtype
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model/input parameters
    int batch_size = 16;
    int channels = 32;
    int depth = 64;
    int height = 64;
    int width = 64;
    int kernel_size = 3;
    int stride = 2;
    int padding = 1;

    // Create input tensor on GPU
    torch::Tensor input = torch::randn({batch_size, channels, depth, height, width}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference implementation using libtorch
    torch::nn::AvgPool3d avg_pool(torch::nn::AvgPool3dOptions(kernel_size).stride(stride).padding(padding));
    avg_pool->to(device, dtype);

    torch::Tensor ref_output = avg_pool->forward(input);

    // Allocate output tensor for CUDA implementation
    torch::Tensor output = torch::empty_like(ref_output);

    // Call the CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(), 
        input.data_ptr(), 
        batch_size, channels, depth, height, width,
        kernel_size, stride, padding
    );

    // Compare results using torch::allclose with rtol=1e-1, atol=1e-1 for fp16
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
