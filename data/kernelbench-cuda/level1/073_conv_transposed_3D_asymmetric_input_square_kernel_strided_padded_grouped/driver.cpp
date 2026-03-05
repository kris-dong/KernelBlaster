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

// Declaration for the CUDA implementation
void launch_gpu_implementation(
    void* output,                  // Output tensor (float16 on CUDA)
    void* input,                   // Input tensor (float16 on CUDA)
    void* weight,                  // Weight tensor (float16 on CUDA)
    void* bias,                    // Bias tensor (nullptr since bias=False)
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t depth,
    int64_t height,
    int64_t width,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups
); // declaration only

int main() {
    // Parameters
    const int64_t batch_size = 16;
    const int64_t in_channels = 32;
    const int64_t out_channels = 64;
    const int64_t kernel_size = 3;
    const int64_t depth = 16;
    const int64_t height = 32;
    const int64_t width = 32;
    const int64_t stride = 2;
    const int64_t padding = 3;
    const int64_t output_padding = 0;
    const int64_t groups = 4;
    const bool bias = false;

    // Set dtype to float16
    auto dtype = torch::kFloat16;
    auto device = torch::kCUDA;

    // Random input tensor (on CUDA, float16)
    torch::Tensor input = torch::randn({batch_size, in_channels, depth, height, width}, torch::TensorOptions().dtype(dtype).device(device));

    // ConvTranspose3d weights and bias (on CUDA, float16)
    torch::Tensor weight = torch::randn({in_channels, out_channels / groups, kernel_size, kernel_size, kernel_size}, torch::TensorOptions().dtype(dtype).device(device));
    torch::Tensor bias_tensor;
    if (bias) {
        bias_tensor = torch::randn({out_channels}, torch::TensorOptions().dtype(dtype).device(device));
    } else {
        bias_tensor = torch::Tensor(); // Empty tensor, will pass nullptr
    }

    // Libtorch reference: ConvTranspose3d
    torch::nn::ConvTranspose3dOptions options(in_channels, out_channels, {kernel_size, kernel_size, kernel_size});
    options.stride(stride).padding(padding).groups(groups).bias(bias);
    torch::nn::ConvTranspose3d conv_transpose3d(options);
    // Assign weights and bias
    conv_transpose3d->to(device, dtype);
    conv_transpose3d->weight.set_data(weight);
    if (bias) {
        conv_transpose3d->bias.set_data(bias_tensor);
    }

    torch::Tensor ref_output = conv_transpose3d->forward(input);

    // Prepare output tensor for CUDA implementation
    torch::Tensor output = torch::empty_like(ref_output);

    // Call CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),                // Output pointer
        input.data_ptr(),                 // Input pointer
        weight.data_ptr(),                // Weight pointer
        bias ? bias_tensor.data_ptr() : nullptr, // Bias pointer or nullptr
        batch_size,
        in_channels,
        out_channels,
        depth,
        height,
        width,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups
    );

    // Compare outputs using torch::allclose with rtol and atol of 1e-1 for fp16
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }
    return 0;
}
