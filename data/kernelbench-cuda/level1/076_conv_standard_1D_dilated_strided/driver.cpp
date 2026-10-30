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
    void* output,                   // Output tensor pointer (float16)
    void* input,                    // Input tensor pointer (float16)
    void* weight,                   // Weight tensor pointer (float16)
    void* bias,                     // Bias tensor pointer (float16 or nullptr)
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t input_length,
    int64_t stride,
    int64_t dilation,
    bool has_bias                  // Indicates if bias is present
);

// Helper function to compare tensors using torch::allclose with proper tolerances
bool allclose_fp16(const at::Tensor& a, const at::Tensor& b) {
    // For fp16, use rtol=1e-1, atol=1e-1 as per guideline
    return torch::allclose(a, b, /*rtol=*/1e-1, /*atol=*/1e-1);
}

int main() {
    // Set device and dtype
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model and input parameters (from get_init_inputs)
    int64_t in_channels = 3;
    int64_t out_channels = 64;
    int64_t kernel_size = 3;
    int64_t stride = 3;
    int64_t dilation = 4;
    int64_t batch_size = 16;
    int64_t input_length = 256;
    bool has_bias = false;

    // Random input tensor (from get_inputs)
    at::Tensor input = torch::randn({batch_size, in_channels, input_length}, device).to(dtype);

    // Construct the Conv1d layer manually (to extract weight/bias)
    // Use torch::nn::Conv1dOptions for clarity
    torch::nn::Conv1dOptions options(in_channels, out_channels, kernel_size);
    options.stride(stride).dilation(dilation).bias(has_bias);
    torch::nn::Conv1d conv(options);
    conv->to(device, dtype);

    // Reference output (libtorch implementation)
    at::Tensor ref_output = conv->forward(input);

    // Prepare pointers for GPU implementation
    at::Tensor weight = conv->weight.detach().clone().to(device, dtype);
    at::Tensor bias = has_bias ? conv->bias.detach().clone().to(device, dtype) : at::Tensor();

    // Prepare output tensor for the CUDA kernel
    int64_t output_length = (input_length + 2*0 - dilation*(kernel_size-1) - 1) / stride + 1;
    at::Tensor output = torch::empty({batch_size, out_channels, output_length}, device).to(dtype);

    // Call GPU implementation
    launch_gpu_implementation(
        output.data_ptr(),             // Output pointer
        input.data_ptr(),              // Input pointer
        weight.data_ptr(),             // Weight pointer
        has_bias ? bias.data_ptr() : nullptr, // Bias pointer (or nullptr)
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        input_length,
        stride,
        dilation,
        has_bias
    );

    // Compare outputs
    if (allclose_fp16(output, ref_output)) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
