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
    void* output,                  // output tensor (float16, GPU)
    void* input,                   // input tensor (float16, GPU)
    void* weight,                  // conv2d weight (float16, GPU)
    void* bias,                    // conv2d bias (nullptr, since bias=False)
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t height_in,
    int64_t width_in,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t height_out,
    int64_t width_out
);

int main() {
    // Parameters
    const int64_t batch_size = 16;
    const int64_t in_channels = 3;
    const int64_t out_channels = 3;
    const int64_t kernel_size = 3;
    const int64_t width_in = 256;
    const int64_t height_in = 128;
    const int64_t stride = 1;
    const int64_t padding = 0;
    const bool use_bias = false;

    // Set default dtype to float16, but explicitly set dtype for tensors
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Input tensor
    auto input = torch::randn({batch_size, in_channels, height_in, width_in}, torch::TensorOptions().dtype(dtype).device(device));

    // Conv2d parameters (depthwise)
    // Weight shape: (out_channels, 1, kernel_size, kernel_size)
    auto weight = torch::randn({out_channels, 1, kernel_size, kernel_size}, torch::TensorOptions().dtype(dtype).device(device));
    torch::Tensor bias;
    if (use_bias) {
        bias = torch::randn({out_channels}, torch::TensorOptions().dtype(dtype).device(device));
    } else {
        bias = torch::Tensor(); // undefined tensor
    }

    // Libtorch reference: depthwise Conv2d
    // groups = in_channels, out_channels = in_channels for depthwise
    auto conv = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .groups(in_channels)
            .bias(use_bias)
    );
    // Copy weights and bias to conv module
    conv->to(device, dtype);
    conv->weight.set_data(weight);
    if (use_bias) {
        conv->bias.set_data(bias);
    }

    // Reference output
    auto ref_output = conv->forward(input);

    // Compute output shape
    int64_t height_out = (height_in + 2 * padding - kernel_size) / stride + 1;
    int64_t width_out = (width_in + 2 * padding - kernel_size) / stride + 1;

    // Allocate output tensor for cuda kernel
    auto output = torch::empty_like(ref_output);

    // Call custom CUDA kernel (pass raw pointers to data)
    launch_gpu_implementation(
        output.data_ptr(),                          // output
        input.data_ptr(),                           // input
        weight.data_ptr(),                          // weight
        use_bias ? bias.data_ptr() : nullptr,       // bias
        batch_size,
        in_channels,
        out_channels,
        height_in,
        width_in,
        kernel_size,
        stride,
        padding,
        height_out,
        width_out
    );

    // Compare outputs (torch::allclose), use rtol=1e-1, atol=1e-1 for fp16
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
