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

void launch_gpu_implementation(
    void* output, void* input,
    void* conv_weight, void* conv_bias,
    void* bn_weight, void* bn_bias,
    void* bn_running_mean, void* bn_running_var,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size,
    float eps, float momentum);

int main() {
    // Configuration parameters
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int height = 32, width = 32;
    const int kernel_size = 3;
    const float eps = 1e-5f;
    const float momentum = 0.1f;
    const at::ScalarType dtype = torch::kHalf;

    // Create input tensor on GPU
    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
    auto input = torch::randn({batch_size, in_channels, height, width}, options);

    // Create reference model components
    auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).stride(1).padding(kernel_size/2));
    auto bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels).eps(eps).momentum(momentum));
    
    // Move modules to GPU
    conv->to(torch::kCUDA, dtype);
    bn->to(torch::kCUDA, dtype);

    // Run reference implementation
    auto x = conv->forward(input);
    x = torch::tanh(torch::softplus(x)) * x;
    auto ref_output = bn->forward(x);

    // Get all parameter pointers
    auto conv_weight = conv->weight.data_ptr();
    auto conv_bias = conv->bias.data_ptr();
    auto bn_weight = bn->weight.data_ptr();
    auto bn_bias = bn->bias.data_ptr();
    auto bn_running_mean = bn->running_mean.data_ptr();
    auto bn_running_var = bn->running_var.data_ptr();

    // Prepare output tensor for CUDA implementation
    auto cuda_output = torch::empty_like(ref_output, options);

    // Launch CUDA implementation
    launch_gpu_implementation(
        cuda_output.data_ptr(), input.data_ptr(),
        conv_weight, conv_bias,
        bn_weight, bn_bias,
        bn_running_mean, bn_running_var,
        batch_size, in_channels, out_channels,
        height, width, kernel_size,
        eps, momentum);

    // Verify results
    bool passed = torch::allclose(cuda_output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
