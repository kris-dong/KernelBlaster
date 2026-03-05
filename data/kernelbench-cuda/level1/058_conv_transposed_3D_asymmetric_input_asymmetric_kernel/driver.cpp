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
#include <cstdint>
#include "cuda_model.cuh"

// Declaration for the GPU implementation.
// All parameters are passed as pointers to GPU memory.
void launch_gpu_implementation(
    void* output,            // Output tensor, shape: (batch_size, out_channels, depth_out, height_out, width_out), fp16
    void* input,             // Input tensor, shape: (batch_size, in_channels, depth_in, height_in, width_in), fp16
    void* weight,            // Weight tensor, shape: (in_channels, out_channels/groups, kD, kH, kW), fp16
    void* bias,              // Bias tensor, shape: (out_channels), fp16 or nullptr if bias is not used
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t depth_in,
    int64_t height_in,
    int64_t width_in,
    int64_t groups,
    int64_t kD,
    int64_t kH,
    int64_t kW,
    int64_t strideD,
    int64_t strideH,
    int64_t strideW,
    int64_t padD,
    int64_t padH,
    int64_t padW,
    int64_t outpadD,
    int64_t outpadH,
    int64_t outpadW
);

int main() {
    // Set device to CUDA and dtype to Half (fp16)
    torch::Device device(torch::kCUDA);

    // Model and input parameters
    int64_t batch_size = 16;
    int64_t in_channels = 32;
    int64_t out_channels = 16;
    int64_t kD = 3, kH = 5, kW = 7; // Kernel size (asymmetric)
    int64_t depth_in = 16, height_in = 32, width_in = 64;
    int64_t strideD = 1, strideH = 1, strideW = 1;
    int64_t padD = 0, padH = 0, padW = 0;
    int64_t outpadD = 0, outpadH = 0, outpadW = 0;
    int64_t groups = 1;
    bool bias_flag = false;

    // Create input tensor on GPU, fp16
    auto input = torch::randn(
        {batch_size, in_channels, depth_in, height_in, width_in},
        torch::TensorOptions().dtype(torch::kFloat16).device(device)
    );

    // Instantiate ConvTranspose3d module
    torch::nn::ConvTranspose3d conv_transpose3d(
        torch::nn::ConvTranspose3dOptions(in_channels, out_channels, {kD, kH, kW})
            .stride({strideD, strideH, strideW})
            .padding({padD, padH, padW})
            .output_padding({outpadD, outpadH, outpadW})
            .groups(groups)
            .bias(bias_flag)
    );
    conv_transpose3d->to(device, torch::kFloat16);

    // Get weight and bias pointers (always on GPU)
    auto weight = conv_transpose3d->weight.detach().clone().to(device, torch::kFloat16);
    torch::Tensor bias;
    if (bias_flag) {
        bias = conv_transpose3d->bias.detach().clone().to(device, torch::kFloat16);
    }

    // Forward pass (reference output)
    auto ref_output = conv_transpose3d->forward(input);

    // Allocate output tensor for CUDA kernel
    auto output = torch::empty_like(ref_output, torch::TensorOptions().device(device).dtype(torch::kFloat16));

    // Launch custom CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),                      // void* output
        input.data_ptr(),                       // void* input
        weight.data_ptr(),                      // void* weight
        bias_flag ? bias.data_ptr() : nullptr,  // void* bias (nullptr if not used)
        batch_size,
        in_channels,
        out_channels,
        depth_in,
        height_in,
        width_in,
        groups,
        kD, kH, kW,
        strideD, strideH, strideW,
        padD, padH, padW,
        outpadD, outpadH, outpadW
    );

    // Compare results: use rtol/atol=1e-1 for fp16
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
