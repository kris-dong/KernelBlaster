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
#include <cuda_model.cuh>

// Declaration for launch_gpu_implementation
void launch_gpu_implementation(
    void* output,           // Output tensor (GPU memory)
    void* input,            // Input tensor (GPU memory)
    void* weight,           // conv2d weight tensor (GPU memory)
    void* bias,             // conv2d bias tensor (GPU memory or nullptr)
    int64_t batch_size,
    int64_t in_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation
);

int main() {
    // Set device to CUDA and dtype to float16
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model and input parameters
    int64_t batch_size = 16;
    int64_t in_channels = 3;
    int64_t kernel_size = 3;
    int64_t width = 256;
    int64_t height = 256;
    int64_t stride = 1;
    int64_t padding = 0;
    int64_t dilation = 1;
    bool bias = false;

    // Create input tensor on GPU
    torch::Tensor input = torch::randn({batch_size, in_channels, height, width}, torch::TensorOptions().dtype(dtype).device(device));

    // Instantiate the model and move to CUDA
    torch::nn::Conv2dOptions conv_options(in_channels, in_channels, {kernel_size, 1});
    conv_options.stride(stride).padding(padding).dilation(dilation).groups(in_channels).bias(bias);
    torch::nn::Conv2d conv2d(conv_options);
    conv2d->to(device, dtype);

    // Forward pass (reference)
    torch::Tensor ref_output = conv2d->forward(input);

    // Extract parameters (weight and bias)
    torch::Tensor weight = conv2d->weight.clone().detach().to(device, dtype);
    torch::Tensor bias_tensor;
    if (bias) {
        bias_tensor = conv2d->bias.clone().detach().to(device, dtype);
    } else {
        bias_tensor = torch::Tensor();
    }

    // Prepare output tensor for kernel
    torch::Tensor output = torch::empty_like(ref_output, torch::TensorOptions().device(device).dtype(dtype));

    // Call the kernel implementation
    launch_gpu_implementation(
        output.data_ptr(),                                   // output
        input.data_ptr(),                                    // input
        weight.data_ptr(),                                   // weight
        bias ? bias_tensor.data_ptr() : nullptr,             // bias (or nullptr)
        batch_size,
        in_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        dilation
    );

    // Compare output with reference using torch::allclose (rtol/atol = 1e-1 for fp16)
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
