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
    void* group_norm_weight, void* group_norm_bias,
    int batch_size, int in_channels, int height, int width,
    int out_channels, int kernel_size,
    int groups, float eps);

int main() {
    // Initialize parameters
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int height = 32, width = 32;
    const int kernel_size = 3;
    const int groups = 8;
    const float eps = 1e-5f;

    // Create model and move to CUDA
    auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size));
    auto group_norm = torch::nn::GroupNorm(torch::nn::GroupNormOptions(groups, out_channels).eps(eps));
    conv->to(torch::kCUDA, torch::kFloat16);
    group_norm->to(torch::kCUDA, torch::kFloat16);

    // Create input tensor on GPU
    auto input = torch::randn({batch_size, in_channels, height, width}, 
                            torch::dtype(torch::kFloat16).device(torch::kCUDA));

    // Reference implementation
    auto x_conv = conv->forward(input);
    auto x_norm = group_norm->forward(x_conv);
    auto x_tanh = torch::tanh(x_norm);
    auto x_hard_swish = torch::hardshrink(x_tanh, 1.0/6.0);  // Equivalent to HardSwish
    auto x_res = x_conv + x_hard_swish;
    auto reference_output = torch::logsumexp(x_res, 1, true);

    // Allocate GPU memory for CUDA implementation output
    auto output_cuda = torch::empty_like(reference_output);

    // Get data pointers
    auto* conv_weight_ptr = conv->weight.data_ptr();
    auto* conv_bias_ptr = conv->bias.data_ptr();
    auto* group_norm_weight_ptr = group_norm->weight.data_ptr();
    auto* group_norm_bias_ptr = group_norm->bias.data_ptr();

    // Launch CUDA implementation
    launch_gpu_implementation(
        output_cuda.data_ptr(),
        input.data_ptr(),
        conv_weight_ptr,
        conv_bias_ptr,
        group_norm_weight_ptr,
        group_norm_bias_ptr,
        batch_size,
        in_channels,
        height,
        width,
        out_channels,
        kernel_size,
        groups,
        eps
    );

    // Verify results
    bool is_close = torch::allclose(reference_output, output_cuda, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (is_close ? "passed" : "failed") << std::endl;

    return 0;
}
