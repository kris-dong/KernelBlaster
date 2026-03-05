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

// Declaration for the GPU implementation
void launch_gpu_implementation(
    void* output,                         // Output tensor pointer
    void* input,                          // Input tensor pointer
    void* weight,                         // Weight tensor pointer
    void* bias,                           // Bias tensor pointer (can be nullptr if no bias)
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_size_h,
    int kernel_size_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
);

int main() {
    // Set device to CUDA and dtype to float16
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model parameters
    int batch_size = 16;
    int in_channels = 3;
    int out_channels = in_channels;
    int kernel_size_h = 3;
    int kernel_size_w = 5;
    int input_width = 256;
    int input_height = 128;
    int stride_h = 1;
    int stride_w = 1;
    int padding_h = 0;
    int padding_w = 0;
    int dilation_h = 1;
    int dilation_w = 1;
    int groups = in_channels;
    bool bias_flag = false;

    // Create input tensor
    auto input = torch::randn(
        {batch_size, in_channels, input_height, input_width},
        torch::TensorOptions().dtype(dtype).device(device)
    );

    // Create depthwise Conv2d weight and (optional) bias
    // Weight shape: (in_channels, 1, kernel_size_h, kernel_size_w)
    auto weight = torch::randn(
        {in_channels, 1, kernel_size_h, kernel_size_w},
        torch::TensorOptions().dtype(dtype).device(device)
    );
    torch::Tensor bias;
    if (bias_flag) {
        bias = torch::randn({in_channels}, torch::TensorOptions().dtype(dtype).device(device));
    } else {
        bias = torch::Tensor(); // undefined tensor
    }

    // Reference output using libtorch
    auto ref_output = torch::conv2d(
        input,
        weight,
        bias_flag ? bias : torch::Tensor(),
        {stride_h, stride_w},
        {padding_h, padding_w},
        {dilation_h, dilation_w},
        groups
    );

    // Allocate output tensor for GPU implementation
    auto output = torch::empty_like(ref_output);

    // Launch custom CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        weight.data_ptr(),
        bias_flag ? bias.data_ptr() : nullptr,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size_h,
        kernel_size_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups
    );

    // Compare results
    bool passed = torch::allclose(
        output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1
    );

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
