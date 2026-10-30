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

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, int in_channels, int out_channels, int kernel_size, int batch_size, int height, int width);

int main() {
    // Configuration parameters
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int height = 32;
    const int width = 32;
    const int kernel_size = 3;

    // Create input tensor on GPU with FP16
    auto input = torch::randn({batch_size, in_channels, height, width}, 
                             torch::TensorOptions().device(torch::kCUDA).dtype(torch::kHalf));

    // Create reference model
    auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size));
    conv->to(torch::kCUDA, torch::kHalf);  // Move to CUDA and convert weights to FP16

    // Run reference implementation
    auto ref_output = conv->forward(input);
    ref_output = torch::mish(ref_output);
    ref_output = torch::mish(ref_output);

    // Allocate output tensor for CUDA implementation
    auto output = torch::empty_like(ref_output);

    // Get raw pointers for CUDA implementation
    void* input_ptr = input.data_ptr();
    void* output_ptr = output.data_ptr();
    void* weight_ptr = conv->weight.data_ptr();
    void* bias_ptr = conv->bias.defined() ? conv->bias.data_ptr() : nullptr;

    // Execute CUDA kernel
    launch_gpu_implementation(output_ptr, input_ptr, weight_ptr, bias_ptr,
                             in_channels, out_channels, kernel_size,
                             batch_size, height, width);

    // Verify results with relaxed tolerances for FP16
    bool passed = torch::allclose(ref_output, output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
