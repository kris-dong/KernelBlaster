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
#include <vector>
#include "cuda_model.cuh"

// Declaration for your CUDA kernel launcher
void launch_gpu_implementation(
    void* output, 
    void* input, 
    void* weight, 
    void* bias, 
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int height_in, 
    int width_in, 
    int kernel_h, 
    int kernel_w, 
    int stride_h, 
    int stride_w, 
    int pad_h, 
    int pad_w, 
    int output_pad_h, 
    int output_pad_w, 
    int dilation_h, 
    int dilation_w, 
    int groups, 
    bool bias_enabled
);

int main() {
    // Set device and dtype
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model parameters
    const int batch_size = 16;
    const int in_channels = 32;
    const int out_channels = 64;
    const int kernel_h = 3;
    const int kernel_w = 5;
    const int height_in = 16;
    const int width_in = 32;
    const int stride_h = 1, stride_w = 1;
    const int pad_h = 0, pad_w = 0;
    const int output_pad_h = 0, output_pad_w = 0;
    const int dilation_h = 1, dilation_w = 1;
    const int groups = 1;
    const bool bias_enabled = false;

    // Instantiate model and set to CUDA
    torch::nn::ConvTranspose2dOptions conv_options(
        in_channels, out_channels, {kernel_h, kernel_w});
    conv_options.stride({stride_h, stride_w});
    conv_options.padding({pad_h, pad_w});
    conv_options.output_padding({output_pad_h, output_pad_w});
    conv_options.dilation({dilation_h, dilation_w});
    conv_options.groups(groups);
    conv_options.bias(bias_enabled);

    torch::nn::ConvTranspose2d conv_transpose(conv_options);
    conv_transpose->to(device, dtype);

    // Prepare input
    auto input = torch::randn({batch_size, in_channels, height_in, width_in}, torch::TensorOptions().dtype(dtype).device(device));

    // Get weight and bias pointers
    auto weight = conv_transpose->weight.detach().clone().to(device, dtype);
    void* weight_ptr = weight.data_ptr();
    void* bias_ptr = nullptr;
    if (bias_enabled) {
        auto bias = conv_transpose->bias.detach().clone().to(device, dtype);
        bias_ptr = bias.data_ptr();
    }

    // Reference output using libtorch
    auto ref_output = conv_transpose->forward(input);

    // Prepare output tensor for CUDA kernel
    // Output shape calculation
    // output = (input - 1) * stride - 2*padding + dilation*(kernel-1) + output_padding + 1
    int height_out = (height_in - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + output_pad_h + 1;
    int width_out = (width_in - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + output_pad_w + 1;

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, torch::TensorOptions().dtype(dtype).device(device));

    // Call CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        weight_ptr,
        bias_ptr,
        batch_size,
        in_channels,
        out_channels,
        height_in,
        width_in,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        output_pad_h,
        output_pad_w,
        dilation_h,
        dilation_w,
        groups,
        bias_enabled
    );

    // Compare outputs
    double rtol = 1e-1, atol = 1e-1; // For fp16
    bool passed = torch::allclose(output, ref_output, rtol, atol);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
