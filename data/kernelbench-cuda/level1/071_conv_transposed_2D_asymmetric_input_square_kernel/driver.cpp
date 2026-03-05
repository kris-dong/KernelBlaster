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
    void* output,                 // Output tensor (fp16, CUDA)
    void* input,                  // Input tensor (fp16, CUDA)
    void* weight,                 // Weight tensor (fp16, CUDA)
    void* bias,                   // Bias tensor (nullptr if bias is not used, fp16, CUDA)
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int height_in,
    int width_in,
    int stride,
    int padding,
    int output_padding,
    int groups
);

int main() {
    // Model configuration
    constexpr int batch_size = 16;
    constexpr int in_channels = 32;
    constexpr int out_channels = 64;
    constexpr int kernel_size = 3;
    constexpr int height_in = 128;
    constexpr int width_in = 256;
    constexpr int stride = 1;
    constexpr int padding = 0;
    constexpr int output_padding = 0;
    constexpr int groups = 1;
    constexpr bool bias = false;

    // Set device and dtype
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Create input tensor on GPU
    auto input = torch::randn({batch_size, in_channels, height_in, width_in}, torch::TensorOptions().dtype(dtype).device(device));

    // Instantiate the ConvTranspose2d layer (square kernel, no bias)
    auto conv_transpose2d = torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .output_padding(output_padding)
            .groups(groups)
            .bias(bias)
    );
    // Move parameters to CUDA and fp16
    conv_transpose2d->to(device, dtype);

    // Get weight and bias pointers (bias is nullptr if not used)
    auto weight = conv_transpose2d->weight.detach().clone().to(device, dtype).contiguous();
    void* weight_ptr = weight.data_ptr();
    void* bias_ptr = nullptr;
    if (bias) {
        auto bias_tensor = conv_transpose2d->bias.detach().clone().to(device, dtype).contiguous();
        bias_ptr = bias_tensor.data_ptr();
    }

    // Run reference output using libtorch
    auto ref_output = conv_transpose2d->forward(input);

    // Allocate output tensor for CUDA kernel
    auto output = torch::empty_like(ref_output, torch::TensorOptions().dtype(dtype).device(device)).contiguous();

    // Launch GPU implementation
    launch_gpu_implementation(
        output.data_ptr(),           // output
        input.data_ptr(),            // input
        weight_ptr,                  // weight
        bias_ptr,                    // bias (nullptr if not used)
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        height_in,
        width_in,
        stride,
        padding,
        output_padding,
        groups
    );

    // Validate result using torch::allclose with rtol/atol=1e-1 for fp16
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
