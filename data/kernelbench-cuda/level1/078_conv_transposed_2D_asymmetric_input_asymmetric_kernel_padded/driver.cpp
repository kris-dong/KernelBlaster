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

// Declaration for the CUDA GPU implementation
void launch_gpu_implementation(
    void* output, // pointer to output tensor (GPU memory)
    void* input,  // pointer to input tensor (GPU memory)
    void* weight, // pointer to conv_transpose2d weight (GPU memory)
    void* bias,   // pointer to conv_transpose2d bias (GPU memory, can be nullptr if no bias)
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
    int padding_width
);

int main() {
    // Set CUDA device and default dtype
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model hyperparameters
    int batch_size = 16;
    int in_channels = 32;
    int out_channels = 64;
    int kernel_height = 3, kernel_width = 5;
    int height = 128, width = 256;
    int stride_height = 1, stride_width = 1;
    int padding_height = 1, padding_width = 2;
    bool use_bias = false;

    // Input tensor
    torch::Tensor x = torch::randn({batch_size, in_channels, height, width}, torch::TensorOptions().dtype(dtype).device(device));

    // Instantiate the ConvTranspose2d layer
    torch::nn::ConvTranspose2dOptions options(in_channels, out_channels, {kernel_height, kernel_width});
    options.stride({stride_height, stride_width});
    options.padding({padding_height, padding_width});
    options.bias(use_bias);
    torch::nn::ConvTranspose2d conv_transpose2d(options);
    conv_transpose2d->to(device, dtype);

    // Reference output using libtorch
    torch::Tensor ref_output = conv_transpose2d->forward(x);

    // Get weights and bias pointers (ensure contiguous memory for passing to CUDA)
    torch::Tensor weight = conv_transpose2d->weight.detach().contiguous().to(device, dtype);
    torch::Tensor bias;
    void* bias_ptr = nullptr;
    if (use_bias) {
        bias = conv_transpose2d->bias.detach().contiguous().to(device, dtype);
        bias_ptr = bias.data_ptr();
    }

    // Allocate output tensor for CUDA kernel
    torch::Tensor output = torch::empty_like(ref_output, torch::TensorOptions().device(device).dtype(dtype));

    // Launch CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),         // output
        x.data_ptr(),              // input
        weight.data_ptr(),         // weight
        bias_ptr,                  // bias or nullptr
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width
    );

    // Compare outputs: use rtol=1e-1, atol=1e-1 for fp16
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
