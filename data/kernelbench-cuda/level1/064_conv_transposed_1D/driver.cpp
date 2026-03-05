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

// Declaration of the GPU implementation.
// Pass all weights and bias pointers (if present), and all relevant sizes and hyperparameters.
void launch_gpu_implementation(
    void* output,                  // Output tensor pointer (fp16, GPU)
    void* input,                   // Input tensor pointer (fp16, GPU)
    void* weight,                  // Weight tensor pointer (fp16, GPU)
    void* bias,                    // Bias tensor pointer (fp16, GPU) (nullptr if no bias)
    int batch_size,                // Batch size
    int in_channels,               // Number of input channels
    int out_channels,              // Number of output channels
    int input_length,              // Input sequence length
    int kernel_size,               // Convolution kernel size
    int stride,                    // Stride
    int padding,                   // Padding
    int output_padding,            // Output padding
    int groups,                    // Number of groups
    bool has_bias                  // Bias flag
);

int main() {
    // Set default dtype to float16 and device to CUDA
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Model hyperparameters
    int batch_size = 16;
    int in_channels = 64;
    int out_channels = 3;
    int kernel_size = 3;
    int stride = 1;
    int padding = 0;
    int output_padding = 0;
    int groups = 1;
    bool has_bias = false;

    // Input tensor
    int input_length = 128;
    auto input = torch::randn({batch_size, in_channels, input_length}, torch::TensorOptions().dtype(dtype).device(device));

    // Define ConvTranspose1d layer (weights and bias randomly initialized)
    torch::nn::ConvTranspose1d conv(
        torch::nn::ConvTranspose1dOptions(in_channels, out_channels, kernel_size)
        .stride(stride)
        .padding(padding)
        .output_padding(output_padding)
        .groups(groups)
        .bias(has_bias)
    );
    conv->to(device, dtype);

    // Get weight and bias pointers
    auto weight = conv->weight.detach().clone().to(device, dtype).contiguous();
    void* weight_ptr = weight.data_ptr();

    torch::Tensor bias;
    void* bias_ptr = nullptr;
    if (has_bias) {
        bias = conv->bias.detach().clone().to(device, dtype).contiguous();
        bias_ptr = bias.data_ptr();
    }

    // Reference output using libtorch
    auto ref_output = conv->forward(input);

    // Prepare output tensor for GPU implementation
    auto gpu_output = torch::empty_like(ref_output);

    // Launch GPU implementation
    launch_gpu_implementation(
        gpu_output.data_ptr(),           // output
        input.data_ptr(),                // input
        weight_ptr,                      // weight
        bias_ptr,                        // bias (nullptr if no bias)
        batch_size,
        in_channels,
        out_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        has_bias
    );

    // Check output (use rtol=1e-1, atol=1e-1 for fp16)
    bool passed = torch::allclose(ref_output, gpu_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
