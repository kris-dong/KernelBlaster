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
    int in_channels, int out_channels, int kernel_size,
    int stride, int padding,
    const void* conv_weight, const void* conv_bias);

int main() {
    // Configuration parameters
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int depth = 16, height = 32, width = 32;
    const int kernel_size = 3;
    const int stride = 1;
    const int padding = 1;

    // Create input tensor on GPU with FP16
    torch::Tensor input = torch::randn({batch_size, in_channels, depth, height, width},
        torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Create and configure Conv3D module
    auto conv = torch::nn::Conv3d(torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
        .stride(stride).padding(padding));
    conv->to(torch::kCUDA, torch::kHalf);  // Move to CUDA and cast to FP16

    // Create reference output
    torch::Tensor x = conv->forward(input);
    x = torch::nn::functional::max_pool3d(x, torch::nn::functional::MaxPool3dFuncOptions(2).stride(2));
    x = torch::logsumexp(x, 1, true);
    x = torch::relu(x);
    torch::Tensor ref_output = x;

    // Allocate CUDA output tensor
    torch::Tensor cuda_output = torch::empty_like(ref_output);

    // Get raw pointers for kernel parameters
    const void* conv_weight_ptr = conv->weight.data_ptr();
    const void* conv_bias_ptr = conv->bias.defined() ? conv->bias.data_ptr() : nullptr;

    // Execute CUDA implementation
    launch_gpu_implementation(
        cuda_output.data_ptr(),
        input.data_ptr(),
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        conv_weight_ptr,
        conv_bias_ptr
    );

    // Verify results with relaxed tolerances for FP16
    bool passed = torch::allclose(ref_output, cuda_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
