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
    void* output,
    void* input,
    void* weight,
    void* bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    bool has_bias
);

int main() {
    // Set default dtype to float16 for all tensors
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Model hyperparameters
    int batch_size = 16;
    int in_channels = 32;
    int out_channels = 64;
    int kernel_size = 3;
    int width = 128;
    int height = 128;
    int stride = 1;
    int padding = 0;
    int output_padding = 0;
    int groups = 1;
    bool bias = false;

    // Input tensor (on GPU, float16)
    auto input = torch::randn({batch_size, in_channels, height, width}, torch::TensorOptions().dtype(dtype).device(device));

    // Instantiate the module (on GPU)
    torch::nn::ConvTranspose2dOptions conv_options(in_channels, out_channels, kernel_size);
    conv_options.stride(stride).padding(padding).output_padding(output_padding).groups(groups).bias(bias);
    torch::nn::ConvTranspose2d conv_transpose2d(conv_options);
    conv_transpose2d->to(device, dtype);

    // Reference output using libtorch (on GPU)
    auto ref_output = conv_transpose2d->forward(input);

    // Get weights and bias pointers
    auto weight = conv_transpose2d->weight.detach();
    void* weight_ptr = weight.data_ptr();
    void* bias_ptr = nullptr;
    if (bias) {
        bias_ptr = conv_transpose2d->bias.detach().data_ptr();
    }

    // Allocate output tensor for GPU implementation (on GPU, float16)
    auto output = torch::empty_like(ref_output);

    // Call the CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        weight_ptr,
        bias_ptr,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        bias
    );

    // Compare outputs (use rtol and atol of 1e-1 for fp16)
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }
    return 0;
}
