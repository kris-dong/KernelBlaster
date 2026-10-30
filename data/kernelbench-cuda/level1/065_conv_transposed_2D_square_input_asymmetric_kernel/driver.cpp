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

// Declaration only -- do not implement here
void launch_gpu_implementation(
    void* output, void* input,
    void* weight, void* bias,
    int batch_size, int in_channels, int out_channels,
    int height, int width,
    int kernel_h, int kernel_w,
    int stride, int padding, int output_padding, int groups, bool has_bias);

int main() {
    using namespace torch;

    // Set device and dtype
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model parameters (keep in sync with Python version)
    int batch_size = 16;
    int in_channels = 32;
    int out_channels = 64;
    int kernel_h = 3;
    int kernel_w = 5;
    int height = 128;
    int width = 128;
    int stride = 1;
    int padding = 0;
    int output_padding = 0;
    int groups = 1;
    bool has_bias = false;

    // Random input tensor
    auto input = torch::randn({batch_size, in_channels, height, width}, torch::TensorOptions().dtype(dtype).device(device));

    // Define the ConvTranspose2d layer and move to CUDA
    torch::nn::ConvTranspose2d conv_transpose2d(
        torch::nn::ConvTranspose2dOptions(in_channels, out_channels, {kernel_h, kernel_w})
            .stride(stride)
            .padding(padding)
            .output_padding(output_padding)
            .groups(groups)
            .bias(has_bias)
    );
    conv_transpose2d->to(device, dtype);

    // Get weight and bias pointers
    auto weight = conv_transpose2d->weight;
    void* weight_ptr = weight.data_ptr();
    void* bias_ptr = nullptr;
    if (has_bias) {
        bias_ptr = conv_transpose2d->bias.data_ptr();
    }

    // Reference output using libtorch
    auto ref_output = conv_transpose2d->forward(input);

    // Allocate output tensor for CUDA kernel
    auto output = torch::empty_like(ref_output);

    // Launch CUDA kernel (declaration only)
    launch_gpu_implementation(
        output.data_ptr(), input.data_ptr(),
        weight_ptr, bias_ptr,
        batch_size, in_channels, out_channels,
        height, width,
        kernel_h, kernel_w,
        stride, padding, output_padding, groups, has_bias
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
