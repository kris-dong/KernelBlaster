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
#include "cuda_model.cuh"

// Declaration for the CUDA implementation.
// All pointers are to GPU memory. Weights and biases are passed as void* for flexibility.
void launch_gpu_implementation(
    void* output,            // Output tensor, shape: (batch_size, out_channels, out_h, out_w)
    void* input,             // Input tensor, shape: (batch_size, in_channels, height, width)
    void* weight,            // Conv2d weights, shape: (out_channels, in_channels, kernel_h, kernel_w)
    void* bias,              // Conv2d bias, shape: (out_channels,) or nullptr if bias == false
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_h,
    int kernel_w,
    int stride,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    bool has_bias
);

int main() {
    // Set default dtype to float16 for all tensors
    torch::Dtype dtype = torch::kFloat16;

    // Parameters from the Python code
    int batch_size = 16;
    int in_channels = 3;
    int out_channels = 64;
    int kernel_h = 3, kernel_w = 5; // Asymmetric kernel
    int input_height = 256, input_width = 256;
    int stride = 1;
    int pad_h = 1, pad_w = 2; // Asymmetric padding
    int dilation_h = 2, dilation_w = 1; // Asymmetric dilation
    bool has_bias = false;

    // Set device to CUDA
    torch::Device device(torch::kCUDA);

    // Create input tensor on GPU
    auto input = torch::randn({batch_size, in_channels, input_height, input_width}, torch::TensorOptions().dtype(dtype).device(device));

    // Create Conv2d weights and bias on GPU
    auto weight = torch::randn({out_channels, in_channels, kernel_h, kernel_w}, torch::TensorOptions().dtype(dtype).device(device));
    torch::Tensor bias;
    if (has_bias) {
        bias = torch::randn({out_channels}, torch::TensorOptions().dtype(dtype).device(device));
    }

    // Reference implementation using libtorch Conv2d
    torch::nn::Conv2dOptions conv_options(in_channels, out_channels, {kernel_h, kernel_w});
    conv_options.stride(stride).padding({pad_h, pad_w}).dilation({dilation_h, dilation_w}).bias(has_bias);

    torch::nn::Conv2d conv(conv_options);
    // Copy weights and bias to Conv2d module (on GPU)
    conv->to(device, dtype);
    conv->weight.set_data(weight);
    if (has_bias) {
        conv->bias.set_data(bias);
    }

    // Forward pass (reference)
    auto ref_output = conv->forward(input);

    // Prepare output tensor for CUDA implementation (same shape as ref_output)
    auto output = torch::empty_like(ref_output);

    // Call the CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),                // output
        input.data_ptr(),                 // input
        weight.data_ptr(),                // weight
        has_bias ? bias.data_ptr() : nullptr, // bias (nullptr if not used)
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_h,
        kernel_w,
        stride,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        has_bias
    );

    // Compare outputs using torch::allclose
    double rtol = 1e-1, atol = 1e-1; // For fp16
    bool passed = torch::allclose(output, ref_output, rtol, atol);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
