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

// Declaration for the CUDA implementation
void launch_gpu_implementation(
    void* output,                     // output tensor (fp16, GPU)
    void* input,                      // input tensor (fp16, GPU)
    void* weight,                     // conv2d weight (fp16, GPU)
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t padding_height,
    int64_t padding_width,
    int64_t dilation_height,
    int64_t dilation_width,
    int64_t groups
    // no bias parameter, as bias=False
);

int main() {
    // Set device and dtype
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model hyperparameters
    int64_t batch_size = 16;
    int64_t in_channels = 3;
    int64_t out_channels = 64;
    int64_t kernel_height = 3;
    int64_t kernel_width = 5;
    int64_t input_height = 256;
    int64_t input_width = 128;
    int64_t stride_height = 1;
    int64_t stride_width = 1;
    int64_t padding_height = 0;
    int64_t padding_width = 0;
    int64_t dilation_height = 1;
    int64_t dilation_width = 1;
    int64_t groups = 1;

    // Create input tensor
    auto input = torch::randn({batch_size, in_channels, input_height, input_width}, torch::TensorOptions().dtype(dtype).device(device));

    // Create weight tensor (no bias)
    auto weight = torch::randn({out_channels, in_channels, kernel_height, kernel_width}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference output using libtorch
    torch::nn::Conv2dOptions conv2d_options(in_channels, out_channels, {kernel_height, kernel_width});
    conv2d_options.stride({stride_height, stride_width});
    conv2d_options.padding({padding_height, padding_width});
    conv2d_options.dilation({dilation_height, dilation_width});
    conv2d_options.groups(groups);
    conv2d_options.bias(false);

    torch::nn::Conv2d conv2d(conv2d_options);
    // Copy initialized weights to the conv2d module
    conv2d->weight = torch::nn::Parameter(weight);

    conv2d->to(device, dtype);

    auto ref_output = conv2d->forward(input);

    // Allocate output tensor for CUDA kernel
    auto output = torch::empty_like(ref_output, torch::TensorOptions().dtype(dtype).device(device));

    // Call CUDA kernel implementation
    launch_gpu_implementation(
        output.data_ptr(),                    // output tensor (float16*)
        input.data_ptr(),                     // input tensor (float16*)
        weight.data_ptr(),                    // weight tensor (float16*)
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        dilation_height,
        dilation_width,
        groups
        // no bias pointer, as bias=False
    );

    // Compare outputs using torch::allclose with rtol and atol for fp16
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
