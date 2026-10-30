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
    void* output,                        // Output tensor (GPU memory)
    void* input,                         // Input tensor (GPU memory)
    void* weight,                        // Conv2d weight (GPU memory)
    void* bias,                          // Conv2d bias (GPU memory), nullptr if bias is not used
    int batch_size,
    int in_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding
);

int main() {
    // Set the default dtype to float16 for all tensors
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device = torch::kCUDA;

    // Model parameters
    int batch_size = 16;
    int in_channels = 3;
    int kernel_size = 3;
    int width = 256;
    int height = 256;
    int stride = 1;
    int padding = 0;
    bool bias = false; // as per the Python code

    // Instantiate model and move to GPU with float16
    torch::nn::Conv2dOptions conv_options(in_channels, in_channels, kernel_size);
    conv_options.stride(stride).padding(padding).groups(in_channels).bias(bias);
    torch::nn::Conv2d conv2d(conv_options);
    conv2d->to(device, dtype);

    // Prepare input tensor on GPU
    torch::Tensor input = torch::randn({batch_size, in_channels, height, width}, torch::TensorOptions().dtype(dtype).device(device));

    // Forward pass with PyTorch (reference)
    torch::NoGradGuard no_grad;
    torch::Tensor ref_output = conv2d->forward(input);

    // Prepare output tensor for GPU implementation
    torch::Tensor gpu_output = torch::empty_like(ref_output, torch::TensorOptions().device(device));

    // Extract weight and bias pointers
    torch::Tensor weight = conv2d->weight;
    void* weight_ptr = weight.data_ptr();
    void* bias_ptr = nullptr;
    if (bias) {
        bias_ptr = conv2d->bias.data_ptr();
    }

    // Launch the GPU implementation
    launch_gpu_implementation(
        gpu_output.data_ptr(),
        input.data_ptr(),
        weight_ptr,
        bias_ptr,
        batch_size,
        in_channels,
        height,
        width,
        kernel_size,
        stride,
        padding
    );

    // Compare outputs using torch::allclose
    // For fp16 use rtol=1e-1, atol=1e-1
    bool passed = torch::allclose(gpu_output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
