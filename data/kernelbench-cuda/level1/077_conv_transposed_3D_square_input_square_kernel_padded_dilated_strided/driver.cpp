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

// Declaration for launch_gpu_implementation.
// All pointers passed are assumed to be device pointers (CUDA memory).
void launch_gpu_implementation(
    void* output,                     // Output tensor (float16, GPU)
    const void* input,                // Input tensor (float16, GPU)
    const void* weight,               // ConvTranspose3d weight tensor (float16, GPU)
    const void* bias,                 // ConvTranspose3d bias tensor (float16, GPU, can be nullptr)
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool has_bias
);

int main() {
    // Set dtype and device
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Model parameters
    int batch_size = 16;
    int in_channels = 32;
    int out_channels = 64;
    int kernel_size = 3;
    int in_depth = 16;
    int in_height = 32;
    int in_width = 32;
    int stride = 2;
    int padding = 1;
    int dilation = 2;
    bool has_bias = false;

    // Create input tensor on GPU with float16
    torch::Tensor input = torch::randn({batch_size, in_channels, in_depth, in_height, in_width}, torch::TensorOptions().dtype(dtype).device(device));

    // Create ConvTranspose3d layer and get parameters
    torch::nn::ConvTranspose3d conv_transpose3d(
        torch::nn::ConvTranspose3dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .bias(has_bias)
    );
    conv_transpose3d->to(device, dtype);

    // Get weights and bias pointers (weights and bias are on GPU)
    torch::Tensor weight = conv_transpose3d->weight.detach().clone().to(device, dtype);
    torch::Tensor bias = has_bias ? conv_transpose3d->bias.detach().clone().to(device, dtype) : torch::Tensor();

    // Compute the reference output using libtorch (on GPU)
    torch::Tensor ref_output = conv_transpose3d->forward(input);

    // Allocate output tensor for custom CUDA kernel
    torch::Tensor output = torch::empty_like(ref_output, torch::TensorOptions().device(device).dtype(dtype));

    // Call the CUDA kernel implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        weight.data_ptr(),
        has_bias ? bias.data_ptr() : nullptr,
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        kernel_size,
        stride,
        padding,
        dilation,
        has_bias
    );

    // Compare outputs using torch::allclose with appropriate tolerances for fp16
    bool is_close = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (is_close) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
