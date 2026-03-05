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

// Declaration of the GPU implementation
void launch_gpu_implementation(
    void* output,
    void* input,
    void* weight,
    void* bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int width,
    int height,
    int depth,
    int kernel_width,
    int kernel_height,
    int kernel_depth,
    int stride,
    int padding,
    int dilation,
    int groups,
    bool has_bias
);

int main() {
    // Check CUDA availability
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available!" << std::endl;
        return 1;
    }

    // Parameters
    const int batch_size = 16;
    const int in_channels = 3;
    const int out_channels = 64;
    const int kernel_width = 3;
    const int kernel_height = 5;
    const int kernel_depth = 7;
    const int width = 64;
    const int height = 64;
    const int depth = 64;
    const int stride = 1;
    const int padding = 0;
    const int dilation = 1;
    const int groups = 1;
    const bool has_bias = false;

    // Set dtype and device
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Input tensor
    auto input = torch::randn(
        {batch_size, in_channels, width, height, depth},
        torch::TensorOptions().dtype(dtype).device(device)
    );

    // Weight tensor
    auto weight = torch::randn(
        {out_channels, in_channels / groups, kernel_width, kernel_height, kernel_depth},
        torch::TensorOptions().dtype(dtype).device(device)
    );
    torch::Tensor bias;
    if (has_bias) {
        bias = torch::randn(
            {out_channels},
            torch::TensorOptions().dtype(dtype).device(device)
        );
    }

    // Reference Conv3d implementation
    torch::nn::Conv3dOptions conv3d_options(
        in_channels, out_channels, {kernel_width, kernel_height, kernel_depth}
    );
    conv3d_options.stride(stride).padding(padding).dilation(dilation).groups(groups).bias(has_bias);

    auto conv3d = std::make_shared<torch::nn::Conv3dImpl>(conv3d_options);
    conv3d->to(device, dtype);

    // Copy weights/bias
    {
        torch::NoGradGuard no_grad;
        conv3d->weight.copy_(weight);
        if (has_bias) {
            conv3d->bias.copy_(bias);
        }
    }

    auto ref_output = conv3d->forward(input);

    // Allocate output tensor for CUDA kernel
    auto gpu_output = torch::empty_like(ref_output);

    // Launch custom GPU implementation
    launch_gpu_implementation(
        gpu_output.data_ptr(),
        input.data_ptr(),
        weight.data_ptr(),
        has_bias ? bias.data_ptr() : nullptr,
        batch_size,
        in_channels,
        out_channels,
        width,
        height,
        depth,
        kernel_width,
        kernel_height,
        kernel_depth,
        stride,
        padding,
        dilation,
        groups,
        has_bias
    );

    // Check correctness (rtol/atol = 1e-1 for fp16)
    bool passed = torch::allclose(gpu_output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
