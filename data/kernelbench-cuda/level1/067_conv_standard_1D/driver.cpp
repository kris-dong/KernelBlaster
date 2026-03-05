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

// Declaration for GPU implementation
void launch_gpu_implementation(
    void* output,                 // Output tensor (fp16, CUDA)
    void* input,                  // Input tensor (fp16, CUDA)
    void* weight,                 // Conv1d weight (fp16, CUDA)
    void* bias,                   // Conv1d bias (nullptr since bias=False)
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t length,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
);

int main() {
    // Set up parameters
    const int64_t batch_size = 16;
    const int64_t in_channels = 3;
    const int64_t out_channels = 64;
    const int64_t kernel_size = 3;
    const int64_t length = 512;
    const int64_t stride = 1;
    const int64_t padding = 0;
    const int64_t dilation = 1;
    const int64_t groups = 1;
    const bool bias = false;

    // Set default dtype to float16
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Create input tensor
    auto input = torch::randn({batch_size, in_channels, length}, torch::TensorOptions().dtype(dtype).device(device));

    // Create Conv1d module and extract weights, bias
    auto conv = torch::nn::Conv1d(
        torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
        .stride(stride).padding(padding).dilation(dilation).groups(groups).bias(bias)
    );
    conv->to(device, dtype);

    // Reference output (PyTorch)
    auto ref_output = conv->forward(input);

    // Get raw pointers to weights and bias
    auto weight = conv->weight.detach().clone().contiguous();
    void* weight_ptr = weight.data_ptr();
    void* bias_ptr = nullptr; // Since bias=False

    // Prepare output tensor for GPU implementation
    auto output = torch::empty_like(ref_output);

    // Call the GPU implementation
    launch_gpu_implementation(
        output.data_ptr(),                   // output
        input.data_ptr(),                    // input
        weight_ptr,                          // weight
        bias_ptr,                            // bias (nullptr)
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        length,
        stride,
        padding,
        dilation,
        groups
    );

    // Compare outputs
    double rtol = 1e-1, atol = 1e-1; // fp16
    bool passed = torch::allclose(output, ref_output, rtol, atol);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
