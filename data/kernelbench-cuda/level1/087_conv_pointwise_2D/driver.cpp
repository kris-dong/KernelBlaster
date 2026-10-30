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

void launch_gpu_implementation(
    void* output, 
    void* input, 
    void* weight, 
    void* bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    bool has_bias
);

int main() {
    // Set default dtype to float16
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Model parameters
    int batch_size = 16;
    int in_channels = 3;
    int out_channels = 64;
    int height = 256;
    int width = 256;
    bool has_bias = false;

    // Create input
    auto input = torch::randn({batch_size, in_channels, height, width}, torch::TensorOptions().dtype(dtype).device(device));

    // Instantiate model
    torch::nn::Conv2dOptions conv_options(in_channels, out_channels, /*kernel_size=*/1);
    conv_options.stride(1).padding(0).bias(has_bias);
    auto conv = torch::nn::Conv2d(conv_options);
    conv->to(dtype);
    conv->to(device);

    // Reference output
    auto ref_output = conv->forward(input);

    // Extract weights and bias
    auto weight = conv->weight.detach().clone().to(device);
    torch::Tensor bias;
    if (has_bias) {
        bias = conv->bias.detach().clone().to(device);
    } else {
        // Allocate a dummy bias tensor (will not be used)
        bias = torch::empty({0}, torch::TensorOptions().dtype(dtype).device(device));
    }

    // Allocate output tensor
    auto output = torch::empty_like(ref_output);

    // Call CUDA implementation
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
        has_bias
    );

    // Compare outputs
    double atol = 1e-1;
    double rtol = 1e-1;

    bool passed = torch::allclose(output, ref_output, rtol, atol);
    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }
    return 0;
}
