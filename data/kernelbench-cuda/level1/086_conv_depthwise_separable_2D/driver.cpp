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
#include "cuda_model.cuh"
#include <torch/torch.h>
#include <iostream>

// Declaration of the GPU implementation
void launch_gpu_implementation(
    void* output,                        // Output tensor (float16, GPU memory)
    void* input,                         // Input tensor (float16, GPU memory)
    void* depthwise_weight,              // Depthwise Conv2d weight (float16, GPU memory)
    void* depthwise_bias,                // Depthwise Conv2d bias (float16, GPU memory) or nullptr if no bias
    void* pointwise_weight,              // Pointwise Conv2d weight (float16, GPU memory)
    void* pointwise_bias,                // Pointwise Conv2d bias (float16, GPU memory) or nullptr if no bias
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation
);

int main() {
    // Set device and dtype
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model parameters
    int batch_size = 16;
    int in_channels = 3;
    int out_channels = 64;
    int kernel_size = 3;
    int width = 256;
    int height = 256;
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    bool bias = false; // As per the Python code

    // Create input tensor on GPU
    torch::Tensor input = torch::randn({batch_size, in_channels, height, width}, torch::TensorOptions().dtype(dtype).device(device));

    // Create depthwise Conv2d
    torch::nn::Conv2dOptions depthwise_options(in_channels, in_channels, kernel_size);
    depthwise_options.stride(stride).padding(padding).dilation(dilation).groups(in_channels).bias(bias);
    torch::nn::Conv2d depthwise(depthwise_options);
    depthwise->to(device, dtype);

    // Create pointwise Conv2d
    torch::nn::Conv2dOptions pointwise_options(in_channels, out_channels, 1);
    pointwise_options.bias(bias);
    torch::nn::Conv2d pointwise(pointwise_options);
    pointwise->to(device, dtype);

    // Copy weights to GPU and set dtype to float16
    depthwise->weight = torch::randn_like(depthwise->weight, torch::TensorOptions().dtype(dtype).device(device));
    if (bias) {
        depthwise->bias = torch::randn_like(depthwise->bias, torch::TensorOptions().dtype(dtype).device(device));
    }
    pointwise->weight = torch::randn_like(pointwise->weight, torch::TensorOptions().dtype(dtype).device(device));
    if (bias) {
        pointwise->bias = torch::randn_like(pointwise->bias, torch::TensorOptions().dtype(dtype).device(device));
    }

    // Save pointers to weights/biases
    void* depthwise_weight_ptr = depthwise->weight.data_ptr();
    void* depthwise_bias_ptr = bias ? depthwise->bias.data_ptr() : nullptr;
    void* pointwise_weight_ptr = pointwise->weight.data_ptr();
    void* pointwise_bias_ptr = bias ? pointwise->bias.data_ptr() : nullptr;

    // Libtorch reference output
    torch::NoGradGuard no_grad;
    torch::Tensor ref = depthwise->forward(input);
    ref = pointwise->forward(ref);

    // Allocate output tensor for the CUDA implementation
    torch::Tensor output = torch::empty_like(ref);

    // Call the CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        depthwise_weight_ptr,
        depthwise_bias_ptr,
        pointwise_weight_ptr,
        pointwise_bias_ptr,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        dilation
    );

    // Compare outputs using torch::allclose (rtol/atol = 1e-1 for fp16)
    bool passed = torch::allclose(output, ref, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
