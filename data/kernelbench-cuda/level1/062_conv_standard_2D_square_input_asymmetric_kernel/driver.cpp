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
    void* output, 
    void* input, 
    void* weight, 
    void* bias,
    int64_t batch_size, 
    int64_t in_channels, 
    int64_t out_channels, 
    int64_t input_height, 
    int64_t input_width,
    int64_t kernel_height, 
    int64_t kernel_width,
    int64_t stride_h, 
    int64_t stride_w, 
    int64_t padding_h, 
    int64_t padding_w,
    int64_t dilation_h, 
    int64_t dilation_w,
    int64_t groups,
    bool has_bias
);

int main() {
    // Set device
    torch::Device device(torch::kCUDA);

    // Model and input parameters
    const int64_t batch_size = 16;
    const int64_t in_channels = 3;
    const int64_t out_channels = 64;
    const int64_t kernel_height = 3;
    const int64_t kernel_width = 5;
    const int64_t input_height = 256;
    const int64_t input_width = 256;
    const int64_t stride_h = 1, stride_w = 1;
    const int64_t padding_h = 0, padding_w = 0;
    const int64_t dilation_h = 1, dilation_w = 1;
    const int64_t groups = 1;
    const bool has_bias = false;

    // Set default dtype to float16
    torch::Dtype dtype = torch::kFloat16;

    // Create input tensor on GPU with fp16
    auto input = torch::randn({batch_size, in_channels, input_height, input_width}, torch::TensorOptions().dtype(dtype).device(device));

    // Create weight tensor as in nn.Conv2d (out_channels, in_channels, kernel_h, kernel_w)
    auto weight = torch::randn({out_channels, in_channels, kernel_height, kernel_width}, torch::TensorOptions().dtype(dtype).device(device));

    // Optionally create bias
    torch::Tensor bias;
    if (has_bias) {
        bias = torch::randn({out_channels}, torch::TensorOptions().dtype(dtype).device(device));
    } else {
        bias = torch::Tensor(); // undefined
    }

    // Reference output using libtorch Conv2d
    torch::nn::Conv2dOptions conv_opts(in_channels, out_channels, {kernel_height, kernel_width});
    conv_opts.stride({stride_h, stride_w});
    conv_opts.padding({padding_h, padding_w});
    conv_opts.dilation({dilation_h, dilation_w});
    conv_opts.groups(groups);
    conv_opts.bias(has_bias);

    torch::nn::Conv2d conv(conv_opts);
    // Move weights and (optionally) bias to GPU and set values
    conv->to(device, dtype);
    conv->weight.set_data(weight.clone());
    if (has_bias) {
        conv->bias.set_data(bias.clone());
    }

    // Compute reference output
    auto ref_output = conv->forward(input);

    // Prepare output tensor for the GPU implementation
    auto output = torch::empty_like(ref_output);

    // Launch CUDA implementation (output, input, weight, bias, ... kernel params)
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        weight.data_ptr(),
        has_bias ? bias.data_ptr() : nullptr,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        has_bias
    );

    // Compare outputs (use relaxed tolerance for fp16)
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
