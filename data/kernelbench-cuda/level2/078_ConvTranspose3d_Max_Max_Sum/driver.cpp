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
#include "cuda_model.cuh"

void launch_gpu_implementation(void* output, void* input, 
                              int in_channels, int out_channels,
                              int kernel_size, int stride, int padding,
                              void* weight, void* bias);

int main() {
    // Setup model parameters
    const int batch_size = 16;
    const int in_channels = 8;
    const int out_channels = 16;
    const int depth = 16, height = 32, width = 32;
    const int kernel_size = 3;
    const int stride = 2;
    const int padding = 1;

    // Create model and move to CUDA
    torch::nn::ConvTranspose3d conv_transpose(
        torch::nn::ConvTranspose3dOptions(in_channels, out_channels, kernel_size)
            .stride(stride).padding(padding));
    torch::nn::MaxPool3d max_pool1(2);
    torch::nn::MaxPool3d max_pool2(3);
    
    // Move all modules to CUDA and set float16 dtype
    conv_transpose->to(torch::kCUDA, torch::kFloat16);
    max_pool1->to(torch::kCUDA);
    max_pool2->to(torch::kCUDA);

    // Generate input tensor on GPU
    auto input = torch::randn({batch_size, in_channels, depth, height, width}, 
                             torch::dtype(torch::kFloat16).device(torch::kCUDA));

    // Run reference implementation
    auto ref_output = conv_transpose(input);
    ref_output = max_pool1(ref_output);
    ref_output = max_pool2(ref_output);
    ref_output = torch::sum(ref_output, {1}, /*keepdim=*/true);

    // Get model parameters
    auto weight = conv_transpose->weight;
    auto bias = conv_transpose->bias;

    // Allocate output tensor for GPU implementation
    auto output = torch::empty_like(ref_output, torch::dtype(torch::kFloat16).device(torch::kCUDA));

    // Call CUDA implementation
    launch_gpu_implementation(output.data_ptr(), input.data_ptr(),
                             in_channels, out_channels,
                             kernel_size, stride, padding,
                             weight.data_ptr(), bias.defined() ? bias.data_ptr() : nullptr);

    // Verify results with relaxed tolerance for fp16
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
