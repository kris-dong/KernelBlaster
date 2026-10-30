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

// Declaration for the CUDA kernel interface
void launch_gpu_implementation(
    void* output,
    void* input,
    void* weight,
    void* bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool has_bias
);

int main() {
    // Set default dtype to Half (float16)
    torch::Dtype dtype = torch::kFloat16;

    // Device setup
    torch::Device device(torch::kCUDA);

    // Model parameters
    const int batch_size = 16;
    const int in_channels = 3;
    const int out_channels = 64;
    const int kernel_size = 5;
    const int input_length = 256;
    const int stride = 1;
    const int padding = 0;
    const int dilation = 3;
    const bool has_bias = false;

    // Instantiate the model (no bias)
    torch::nn::ConvTranspose1d conv1d_transpose(
        torch::nn::ConvTranspose1dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .bias(has_bias)
    );
    conv1d_transpose->to(device, dtype);

    // Random input tensor (on GPU, fp16)
    auto input = torch::randn({batch_size, in_channels, input_length}, torch::TensorOptions().dtype(dtype).device(device));

    // Get reference output using libtorch
    auto ref_output = conv1d_transpose->forward(input);

    // Prepare parameter pointers for CUDA kernel
    auto weight = conv1d_transpose->weight.detach().clone().to(device, dtype).contiguous();
    void* weight_ptr = weight.data_ptr();
    void* bias_ptr = nullptr;
    if (has_bias) {
        auto bias = conv1d_transpose->bias.detach().clone().to(device, dtype).contiguous();
        bias_ptr = bias.data_ptr();
    }

    // Prepare output tensor for CUDA kernel
    auto output_shape = ref_output.sizes().vec();
    auto output = torch::empty(output_shape, torch::TensorOptions().dtype(dtype).device(device));
    void* output_ptr = output.data_ptr();
    void* input_ptr = input.data_ptr();

    // Call the CUDA kernel interface
    launch_gpu_implementation(
        output_ptr,
        input_ptr,
        weight_ptr,
        bias_ptr,
        batch_size,
        in_channels,
        out_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        dilation,
        has_bias
    );

    // Compare with torch reference implementation using torch::allclose
    double rtol = 1e-1, atol = 1e-1; // For float16
    bool passed = torch::allclose(output, ref_output, rtol, atol);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
