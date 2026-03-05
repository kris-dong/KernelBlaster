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
#include <ATen/ATen.h>
#include <iostream>
#include "cuda_model.cuh"

// Declaration for the CUDA implementation
void launch_gpu_implementation(
    void* output,
    void* input,
    void* weight,
    void* bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth,
    int height,
    int width,
    int kernel_size_d,
    int kernel_size_h,
    int kernel_size_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int output_padding_d,
    int output_padding_h,
    int output_padding_w,
    int groups,
    bool has_bias
);

int main() {
    // Set device and dtype
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model and input parameters
    const int batch_size = 16;
    const int in_channels = 32;
    const int out_channels = 64;
    const int kernel_size_d = 3, kernel_size_h = 5, kernel_size_w = 7;
    const int depth = 16, height = 32, width = 64;
    const int stride_d = 2, stride_h = 2, stride_w = 2;
    const int padding_d = 1, padding_h = 2, padding_w = 3;
    const int output_padding_d = 1, output_padding_h = 1, output_padding_w = 1;
    const int groups = 4;
    const bool has_bias = false;

    // Instantiate inputs
    auto input = torch::randn({batch_size, in_channels, depth, height, width},
        torch::TensorOptions().dtype(dtype).device(device));

    // Instantiate model
    torch::nn::ConvTranspose3dOptions options(in_channels, out_channels, {kernel_size_d, kernel_size_h, kernel_size_w});
    options.stride({stride_d, stride_h, stride_w})
           .padding({padding_d, padding_h, padding_w})
           .output_padding({output_padding_d, output_padding_h, output_padding_w})
           .groups(groups)
           .bias(has_bias);

    torch::nn::ConvTranspose3d conv_transpose3d(options);
    conv_transpose3d->to(device, dtype);

    // Reference output
    auto ref_output = conv_transpose3d->forward(input);

    // Extract pointers to weight and bias (use at::Half for fp16)
    auto weight = conv_transpose3d->weight.detach();
    void* weight_ptr = weight.data_ptr<at::Half>();
    void* bias_ptr = nullptr;
    if (has_bias) {
        bias_ptr = conv_transpose3d->bias.detach().data_ptr<at::Half>();
    }

    // Allocate output tensor for CUDA kernel
    auto output = torch::empty_like(ref_output);

    // Launch CUDA implementation
    launch_gpu_implementation(
        output.data_ptr<at::Half>(),
        input.data_ptr<at::Half>(),
        weight_ptr,
        bias_ptr,
        batch_size,
        in_channels,
        out_channels,
        depth,
        height,
        width,
        kernel_size_d,
        kernel_size_h,
        kernel_size_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        output_padding_d,
        output_padding_h,
        output_padding_w,
        groups,
        has_bias
    );

    // Compare outputs using torch::allclose (rtol/atol = 1e-1 for fp16)
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
