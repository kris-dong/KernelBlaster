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
    int input_height,
    int input_width,
    int input_depth,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    bool use_bias
);

int main() {
    // Set device and dtype
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model and input params
    int batch_size = 16;
    int in_channels = 3;
    int out_channels = 64;
    int kernel_size = 3;
    int width = 256;
    int height = 256;
    int depth = 10;
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    int groups = 1;
    bool use_bias = false;

    // Create input tensor (on GPU, float16)
    auto input = torch::randn({batch_size, in_channels, height, width, depth}, torch::TensorOptions().dtype(dtype).device(device));

    // Create Conv3d module and move to GPU
    auto options = torch::nn::Conv3dOptions(in_channels, out_channels, {kernel_size, kernel_size, 1})
        .stride(stride)
        .padding(padding)
        .dilation(dilation)
        .groups(groups)
        .bias(use_bias);

    torch::nn::Conv3d conv3d(options);
    conv3d->to(device, dtype);

    // Extract weights and bias (bias may be undefined if not used)
    auto weight = conv3d->weight.detach().clone().to(device, dtype);
    void* weight_ptr = weight.data_ptr();
    void* bias_ptr = nullptr;
    torch::Tensor bias;
    if (use_bias) {
        bias = conv3d->bias.detach().clone().to(device, dtype);
        bias_ptr = bias.data_ptr();
    }

    // Reference output
    auto ref_output = conv3d->forward(input);

    // Prepare output tensor for GPU implementation
    auto output_shape = ref_output.sizes();
    auto output = torch::empty(output_shape, torch::TensorOptions().dtype(dtype).device(device));
    void* output_ptr = output.data_ptr();
    void* input_ptr = input.data_ptr();

    // Call launch_gpu_implementation
    launch_gpu_implementation(
        output_ptr,
        input_ptr,
        weight_ptr,
        bias_ptr,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        depth,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        use_bias
    );

    // Compare outputs
    double rtol = 1e-1, atol = 1e-1; // fp16
    bool result = torch::allclose(output, ref_output, rtol, atol);

    if (result) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
