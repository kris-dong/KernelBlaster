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

// Declaration for the CUDA implementation
void launch_gpu_implementation(
    void* output, // output tensor (GPU memory, fp16)
    void* input,  // input tensor (GPU memory, fp16)
    void* weight, // conv_transpose2d weight (GPU memory, fp16)
    void* bias,   // conv_transpose2d bias (GPU memory, fp16 or nullptr)
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    int dilation_height,
    int dilation_width,
    int groups
);

int main() {
    // Set device and dtype
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model and input parameters (matching Python code)
    int batch_size = 16;
    int in_channels = 32;
    int out_channels = 64;
    int kernel_height = 3;
    int kernel_width = 5;
    int input_height = 128;
    int input_width = 256;
    int stride_height = 2;
    int stride_width = 3;
    int padding_height = 1;
    int padding_width = 2;
    int dilation_height = 2;
    int dilation_width = 1;
    int groups = 4;
    bool has_bias = false;

    // Create input tensor (GPU, fp16)
    auto input = torch::randn({batch_size, in_channels, input_height, input_width},
        torch::TensorOptions().dtype(dtype).device(device));

    // Create ConvTranspose2d layer with specified parameters
    torch::nn::ConvTranspose2dOptions conv_options(in_channels, out_channels, {kernel_height, kernel_width});
    conv_options.stride({stride_height, stride_width});
    conv_options.padding({padding_height, padding_width});
    conv_options.dilation({dilation_height, dilation_width});
    conv_options.groups(groups);
    conv_options.bias(has_bias);

    torch::nn::ConvTranspose2d conv(conv_options);
    conv->to(device, dtype);

    // Reference weights and bias (copy to GPU, fp16)
    auto weight = conv->weight.detach().clone().to(device, dtype);
    void* bias_ptr = nullptr;
    if (has_bias) {
        auto bias_tensor = conv->bias.detach().clone().to(device, dtype);
        bias_ptr = bias_tensor.data_ptr();
    }

    // Reference output using libtorch
    auto ref_output = conv->forward(input);

    // Allocate output tensor for CUDA kernel (same shape as ref_output)
    auto out_shape = ref_output.sizes();
    auto gpu_output = torch::empty(out_shape, torch::TensorOptions().dtype(dtype).device(device));

    // Call CUDA implementation
    launch_gpu_implementation(
        gpu_output.data_ptr(), // output
        input.data_ptr(),      // input
        weight.data_ptr(),     // weight
        bias_ptr,              // bias or nullptr
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        dilation_height,
        dilation_width,
        groups
    );

    // Compare outputs using torch::allclose
    // Use atol/rtol = 1e-1 for fp16
    bool passed = torch::allclose(
        gpu_output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1
    );

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }
    return 0;
}
