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

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias,
                               int batch_size, int in_channels, int out_channels,
                               int height, int width, int kernel_size);

int main() {
    // Initialize parameters
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int height = 32;
    const int width = 32;
    const int kernel_size = 3;

    // Create input tensor on CUDA with FP16
    auto input = torch::randn({batch_size, in_channels, height, width}, 
                             torch::device(torch::kCUDA).dtype(torch::kHalf));

    // Create and configure Conv2d module
    auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size));
    conv->to(torch::kCUDA, torch::kHalf);  // Move to CUDA and convert weights to FP16

    // Run reference implementation
    auto x = conv->forward(input);
    x = torch::relu(x);
    x = x * torch::clamp((x + 3) / 6, 0, 1);
    auto reference_output = x;

    // Get parameter pointers
    auto weight = conv->weight;
    auto bias = conv->bias;

    // Allocate output tensor for CUDA kernel
    auto output = torch::empty_like(reference_output);

    // Launch custom GPU implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size
    );

    // Verify results with relaxed tolerances for FP16
    bool passed = torch::allclose(reference_output, output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
