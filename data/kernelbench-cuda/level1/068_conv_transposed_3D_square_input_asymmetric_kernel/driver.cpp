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
#include <cuda_model.cuh>

void launch_gpu_implementation(
    void* output,                   // Output tensor (GPU memory)
    void* input,                    // Input tensor (GPU memory)
    void* weight,                   // Weight tensor (GPU memory)
    void* bias,                     // Bias tensor (GPU memory, nullptr if bias is not used)
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t depth,
    int64_t width,
    int64_t height,
    int64_t kernel_depth,
    int64_t kernel_width,
    int64_t kernel_height,
    int64_t stride_d,
    int64_t stride_w,
    int64_t stride_h,
    int64_t padding_d,
    int64_t padding_w,
    int64_t padding_h,
    int64_t output_padding_d,
    int64_t output_padding_w,
    int64_t output_padding_h,
    int64_t groups
); // declaration only

int main() {
    // Set device
    torch::Device device(torch::kCUDA);

    // Parameters
    using scalar_t = at::Half; // fp16
    constexpr int64_t batch_size = 16;
    constexpr int64_t in_channels = 32;
    constexpr int64_t out_channels = 64;
    constexpr int64_t kernel_depth = 3;
    constexpr int64_t kernel_width = 5;
    constexpr int64_t kernel_height = 5;
    constexpr int64_t depth = 64;
    constexpr int64_t width = 64;
    constexpr int64_t height = 64;
    constexpr int64_t stride_d = 1, stride_w = 1, stride_h = 1;
    constexpr int64_t padding_d = 0, padding_w = 0, padding_h = 0;
    constexpr int64_t output_padding_d = 0, output_padding_w = 0, output_padding_h = 0;
    constexpr int64_t groups = 1;
    constexpr bool use_bias = false;

    // Create input
    auto input = torch::randn(
        {batch_size, in_channels, depth, width, height},
        torch::TensorOptions().dtype(torch::kFloat16).device(device)
    );

    // Create ConvTranspose3d module
    torch::nn::ConvTranspose3dOptions conv_options(
        in_channels, out_channels, {kernel_depth, kernel_width, kernel_height}
    );
    conv_options.stride({stride_d, stride_w, stride_h});
    conv_options.padding({padding_d, padding_w, padding_h});
    conv_options.output_padding({output_padding_d, output_padding_w, output_padding_h});
    conv_options.groups(groups);
    conv_options.bias(use_bias);

    auto conv = torch::nn::ConvTranspose3d(conv_options);
    conv->to(device, torch::kFloat16);

    // Copy weights and bias to ensure we can pass pointers
    auto weight = conv->weight.detach().clone().to(torch::kFloat16).to(device);
    torch::Tensor bias;
    if (use_bias) {
        bias = conv->bias.detach().clone().to(torch::kFloat16).to(device);
    } else {
        bias = torch::Tensor();
    }

    // Reference output using libtorch
    torch::NoGradGuard no_grad;
    auto ref_output = conv->forward(input);

    // Prepare output tensor for CUDA implementation
    auto output = torch::empty_like(ref_output, torch::TensorOptions().dtype(torch::kFloat16).device(device));

    // Launch custom CUDA implementation
    launch_gpu_implementation(
        output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        use_bias ? bias.data_ptr<scalar_t>() : nullptr,
        batch_size,
        in_channels,
        out_channels,
        depth,
        width,
        height,
        kernel_depth,
        kernel_width,
        kernel_height,
        stride_d,
        stride_w,
        stride_h,
        padding_d,
        padding_w,
        padding_h,
        output_padding_d,
        output_padding_w,
        output_padding_h,
        groups
    );

    // Compare results using torch::allclose
    double rtol = 1e-1, atol = 1e-1; // fp16 tolerance
    bool passed = torch::allclose(output, ref_output, rtol, atol);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
