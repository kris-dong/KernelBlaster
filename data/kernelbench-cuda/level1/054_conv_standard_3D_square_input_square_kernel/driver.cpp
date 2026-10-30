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

// Declaration for the CUDA kernel launcher
void launch_gpu_implementation(
    void* output,                      // Output tensor (GPU memory)
    void* input,                       // Input tensor (GPU memory)
    void* weight,                      // Weight tensor (GPU memory)
    void* bias,                        // Bias tensor (GPU memory, nullptr if not used)
    int batch_size,
    int in_channels,
    int out_channels,
    int depth,
    int width,
    int height,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups
);

int main() {
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model parameters
    int batch_size = 16;
    int in_channels = 3;
    int out_channels = 64;
    int kernel_size = 3;
    int depth = 64;
    int width = 64;
    int height = 64;
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    int groups = 1;
    bool bias_flag = false;

    // Input tensor
    auto input = torch::randn({batch_size, in_channels, depth, width, height},
                              torch::TensorOptions().dtype(dtype).device(device));

    // Weight and bias tensors
    auto weight = torch::randn({out_channels, in_channels / groups, kernel_size, kernel_size, kernel_size},
                               torch::TensorOptions().dtype(dtype).device(device));
    torch::Tensor bias;
    if (bias_flag) {
        bias = torch::randn({out_channels}, torch::TensorOptions().dtype(dtype).device(device));
    }

    // Setup Conv3d and copy weights/bias
    torch::nn::Conv3dOptions conv_options(in_channels, out_channels, kernel_size);
    conv_options.stride(stride).padding(padding).dilation(dilation).groups(groups).bias(bias_flag);
    auto conv3d = torch::nn::Conv3d(conv_options);
    conv3d->to(device, dtype);

    // Copy weights and (optionally) bias
    {
        torch::NoGradGuard no_grad;
        conv3d->weight.copy_(weight);
        if (bias_flag) {
            conv3d->bias.copy_(bias);
        }
    }

    // Reference output
    auto ref_output = conv3d->forward(input);

    // Output tensor for CUDA kernel
    auto output = torch::empty_like(ref_output);

    // Launch CUDA kernel
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        weight.data_ptr(),
        bias_flag ? bias.data_ptr() : nullptr,
        batch_size,
        in_channels,
        out_channels,
        depth,
        width,
        height,
        kernel_size,
        stride,
        padding,
        dilation,
        groups
    );

    // Compare results (use relaxed tolerances for fp16)
    double rtol = 1e-1, atol = 1e-1;
    bool passed = torch::allclose(output, ref_output, rtol, atol);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
