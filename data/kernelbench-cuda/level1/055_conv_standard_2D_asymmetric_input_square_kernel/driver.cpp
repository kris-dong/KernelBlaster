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

// Declaration for the CUDA implementation.
// All pointers are expected to point to GPU memory.
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
    // Check CUDA availability
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available." << std::endl;
        return 1;
    }

    // Set default dtype to float16
    auto dtype = torch::kFloat16;
    auto device = torch::Device(torch::kCUDA);

    // Model parameters
    int batch_size = 16;
    int in_channels = 3;
    int out_channels = 64;
    int kernel_size = 3;
    int width = 256;
    int height = 128;
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    int groups = 1;
    bool has_bias = false;

    // Input tensor (on GPU, float16)
    auto input = torch::randn({batch_size, in_channels, height, width},
                              torch::TensorOptions().dtype(dtype).device(device));

    // Create Conv2d layer and move to CUDA/fp16
    torch::nn::Conv2dOptions conv_options(in_channels, out_channels, kernel_size);
    conv_options.stride(stride).padding(padding).dilation(dilation).groups(groups).bias(has_bias);
    torch::nn::Conv2d conv(conv_options);
    conv->to(dtype);
    conv->to(device);

    // Extract weights and bias (if used)
    auto weight = conv->weight.detach().clone().to(device, dtype);
    torch::Tensor bias_tensor;
    if (has_bias) {
        bias_tensor = conv->bias.detach().clone().to(device, dtype);
    }

    // Reference output using libtorch
    auto ref_output = conv->forward(input);

    // Allocate output tensor for CUDA kernel
    auto output = torch::empty_like(ref_output);

    // Launch custom CUDA implementation
    launch_gpu_implementation(
        output.data_ptr<at::Half>(), 
        input.data_ptr<at::Half>(), 
        weight.data_ptr<at::Half>(), 
        has_bias ? bias_tensor.data_ptr<at::Half>() : nullptr,
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

    // Validate output using torch::allclose
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
