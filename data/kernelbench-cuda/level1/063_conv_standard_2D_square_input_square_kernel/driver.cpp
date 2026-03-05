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
#include <cmath>
#include "cuda_model.cuh"

// Declaration for CUDA implementation
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
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    bool has_bias
);

int main() {
    // Set the default dtype to float16
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Model parameters
    int batch_size = 16;
    int in_channels = 3;
    int out_channels = 64;
    int kernel_size = 3;
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    int groups = 1;
    bool has_bias = false;

    int height = 256;
    int width = 256;

    // Instantiate the model and move to CUDA
    torch::nn::Conv2dOptions conv_options(in_channels, out_channels, kernel_size);
    conv_options.stride(stride).padding(padding).dilation(dilation).groups(groups).bias(has_bias);

    torch::nn::Conv2d conv2d(conv_options);
    conv2d->to(device, dtype);

    // Generate input
    auto input = torch::randn({batch_size, in_channels, height, width}, torch::TensorOptions().dtype(dtype).device(device));

    // Extract weights and bias pointers
    auto weight = conv2d->weight.detach().clone().to(device, dtype);
    void* weight_ptr = weight.data_ptr();
    void* bias_ptr = nullptr;
    if (has_bias) {
        auto bias = conv2d->bias.detach().clone().to(device, dtype);
        bias_ptr = bias.data_ptr();
    }

    // Run reference implementation
    auto ref_output = conv2d->forward(input);

    // Prepare output tensor for CUDA kernel
    auto output = torch::empty_like(ref_output);

    // Call CUDA kernel implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        weight_ptr,
        bias_ptr,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        has_bias
    );

    // Compare outputs
    double rtol = 1e-1;
    double atol = 1e-1;

    bool passed = torch::allclose(output, ref_output, rtol, atol);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }
    return 0;
}
