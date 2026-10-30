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

void launch_gpu_implementation(
    void* output, void* input,
    void* conv_weight, void* conv_bias,
    void* bn_weight, void* bn_bias, void* bn_running_mean, void* bn_running_var,
    int64_t batch_size, int64_t in_channels, int64_t out_channels,
    int64_t kernel_size, int64_t stride, int64_t padding,
    int64_t input_depth, int64_t input_height, int64_t input_width);

int main() {
    // Setup parameters
    const int64_t batch_size = 128;
    const int64_t in_channels = 3;
    const int64_t out_channels = 16;
    const int64_t kernel_size = 3;
    const int64_t stride = 2;
    const int64_t padding = 1;
    const int64_t depth = 32, height = 32, width = 32;
    const torch::ScalarType dtype = torch::kFloat16;
    const torch::Device device(torch::kCUDA);

    // Create model components with proper configuration
    auto conv_options = torch::nn::ConvTranspose3dOptions(in_channels, out_channels, kernel_size)
                            .stride(stride)
                            .padding(padding);
    auto conv = torch::nn::ConvTranspose3d(conv_options);
    auto bn = torch::nn::BatchNorm3d(out_channels);
    auto pool1 = torch::nn::AvgPool3d(2);
    auto pool2 = torch::nn::AvgPool3d(2);

    // Move to CUDA and set data type
    conv->to(device, dtype);
    bn->to(device, dtype);
    bn->eval();

    // Create input tensor
    auto input = torch::randn({batch_size, in_channels, depth, height, width}, 
                            torch::dtype(dtype).device(device));

    // Run reference implementation
    auto x = conv->forward(input);
    x = bn->forward(x);
    x = pool1->forward(x);
    auto reference_output = pool2->forward(x);

    // Get parameter pointers
    auto conv_weight = conv->weight.data_ptr();
    auto conv_bias = conv->bias.data_ptr();
    auto bn_weight = bn->weight.data_ptr();
    auto bn_bias = bn->bias.data_ptr();
    auto bn_mean = bn->running_mean.data_ptr();
    auto bn_var = bn->running_var.data_ptr();

    // Prepare output tensor
    auto output = torch::empty_like(reference_output);

    // Call GPU implementation
    launch_gpu_implementation(
        output.data_ptr(), input.data_ptr(),
        conv_weight, conv_bias,
        bn_weight, bn_bias, bn_mean, bn_var,
        batch_size, in_channels, out_channels,
        kernel_size, stride, padding,
        depth, height, width
    );

    // Verify results
    bool passed = torch::allclose(reference_output, output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
