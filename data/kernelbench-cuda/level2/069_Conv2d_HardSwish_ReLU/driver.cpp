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

void launch_gpu_implementation(void* output, void* input, void* conv_weight, void* conv_bias, 
                              int64_t in_channels, int64_t out_channels, int64_t kernel_size,
                              int64_t batch_size, int64_t height, int64_t width);

int main() {
    // Model parameters
    const int64_t in_channels = 3;
    const int64_t out_channels = 16;
    const int64_t kernel_size = 3;
    const int64_t batch_size = 128;
    const int64_t height = 32;
    const int64_t width = 32;

    // Create model and move to CUDA with FP16
    auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size));
    conv->to(torch::kCUDA, torch::kFloat16);

    // Create input tensor on CUDA with FP16
    auto input_tensor = torch::randn({batch_size, in_channels, height, width}, 
                                    torch::dtype(torch::kFloat16).device(torch::kCUDA));

    // Reference implementation
    auto conv_out = conv->forward(input_tensor);
    auto hardswish_out = torch::hardswish(conv_out);  // Fixed namespace
    auto reference_output = torch::relu(hardswish_out);

    // Allocate output tensor for GPU implementation
    auto output_tensor = torch::empty_like(reference_output);

    // Get parameter pointers
    auto conv_weight = conv->weight;
    auto conv_bias = conv->bias;

    // Call GPU implementation
    launch_gpu_implementation(
        output_tensor.data_ptr(),
        input_tensor.data_ptr(),
        conv_weight.data_ptr(),
        conv_bias.data_ptr(),
        in_channels,
        out_channels,
        kernel_size,
        batch_size,
        height,
        width
    );

    // Verify results
    bool passed = torch::allclose(reference_output, output_tensor, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
