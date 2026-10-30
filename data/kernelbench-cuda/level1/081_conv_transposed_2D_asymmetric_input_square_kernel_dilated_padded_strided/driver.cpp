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

// Declaration only, do not implement here
void launch_gpu_implementation(
    void* output,                     // output tensor (float16, GPU)
    void* input,                      // input tensor (float16, GPU)
    void* weight,                     // weight tensor (float16, GPU)
    void* bias,                       // bias tensor (nullptr if bias==false, float16, GPU)
    int batch_size,                   // e.g., 16
    int in_channels,                  // e.g., 32
    int out_channels,                 // e.g., 64
    int height_in,                    // e.g., 64
    int width_in,                     // e.g., 128
    int kernel_size,                  // e.g., 3
    int stride,                       // e.g., 5
    int padding,                      // e.g., 1
    int dilation,                     // e.g., 2
    bool has_bias                     // whether bias is present
);

int main() {
    // Parameters
    int batch_size = 16;
    int in_channels = 32;
    int out_channels = 64;
    int kernel_size = 3;
    int height_in = 64;
    int width_in = 128;
    int stride = 5;
    int padding = 1;
    int dilation = 2;
    bool has_bias = false;

    // Set default dtype to float16 (fp16)
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device = torch::Device(torch::kCUDA);

    // Input tensor
    auto input = torch::randn({batch_size, in_channels, height_in, width_in}, torch::TensorOptions().dtype(dtype).device(device));

    // ConvTranspose2d weight and bias
    // Weight shape: (in_channels, out_channels/groups, kernel_size[0], kernel_size[1]) for torch, but for ConvTranspose2d: (in_channels, out_channels, kH, kW)
    auto weight = torch::randn({in_channels, out_channels, kernel_size, kernel_size}, torch::TensorOptions().dtype(dtype).device(device));
    torch::Tensor bias;
    if (has_bias) {
        bias = torch::randn({out_channels}, torch::TensorOptions().dtype(dtype).device(device));
    } else {
        bias = torch::Tensor(); // undefined tensor
    }

    // Reference output using libtorch (ConvTranspose2d)
    torch::nn::ConvTranspose2dOptions conv_opts(in_channels, out_channels, kernel_size);
    conv_opts.stride(stride).padding(padding).dilation(dilation).bias(has_bias);
    auto conv = torch::nn::ConvTranspose2d(conv_opts);

    // Assign the weights and bias from the above tensors
    conv->to(device, dtype);
    conv->weight.set_data(weight);
    if (has_bias) {
        conv->bias.set_data(bias);
    }

    auto ref_output = conv->forward(input);

    // Prepare output tensor for CUDA kernel
    auto output = torch::empty_like(ref_output);

    // Call the CUDA kernel launcher
    launch_gpu_implementation(
        output.data_ptr(),                      // output (float16, GPU)
        input.data_ptr(),                       // input (float16, GPU)
        weight.data_ptr(),                      // weight (float16, GPU)
        has_bias ? bias.data_ptr() : nullptr,   // bias (float16, GPU) or nullptr
        batch_size,
        in_channels,
        out_channels,
        height_in,
        width_in,
        kernel_size,
        stride,
        padding,
        dilation,
        has_bias
    );

    // Compare the outputs
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }
    return 0;
}
