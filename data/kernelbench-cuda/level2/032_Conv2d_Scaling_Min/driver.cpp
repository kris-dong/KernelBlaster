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

void launch_gpu_implementation(void* output, void* input, void* conv_weight, void* conv_bias, float scale_factor,
                               int in_channels, int out_channels, int kernel_size,
                               int batch_size, int height, int width);

int main() {
    // Initialize parameters from Python spec
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int height = 32, width = 32;
    const int kernel_size = 3;
    const float scale_factor = 2.0f;

    // Create CUDA tensor with float16 dtype
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);
    auto input = torch::randn({batch_size, in_channels, height, width}, options);

    // Create reference model
    torch::nn::Conv2d conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).stride(1).padding(kernel_size/2));
    conv->to(torch::kCUDA, torch::kHalf);
    conv->eval();

    // Run reference implementation
    auto x = conv->forward(input);
    x = x * scale_factor;
    auto min_result = torch::min(x, 1, /*keepdim=*/true);  // Get tuple of (values, indices)
    auto ref_output = std::get<0>(min_result);  // Extract values tensor

    // Get model parameters for kernel launch
    void* conv_weight = conv->weight.data_ptr();
    void* conv_bias = conv->bias.data_ptr();

    // Allocate output tensor for CUDA kernel
    auto output = torch::empty_like(ref_output);

    // Launch custom implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        conv_weight,
        conv_bias,
        scale_factor,
        in_channels,
        out_channels,
        kernel_size,
        batch_size,
        height,
        width
    );

    // Verify results with fp16 tolerances
    bool passed = torch::allclose(ref_output, output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
