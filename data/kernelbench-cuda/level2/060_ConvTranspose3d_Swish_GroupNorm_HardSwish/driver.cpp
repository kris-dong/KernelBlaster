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
    void* output,
    void* input,
    const void* conv_weight,
    const void* conv_bias,
    const void* gn_weight,
    const void* gn_bias,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t groups,
    double eps,
    int64_t batch_size,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width
);

int main() {
    // Initialize parameters
    const int64_t batch_size = 128;
    const int64_t in_channels = 3;
    const int64_t out_channels = 16;
    const int64_t depth = 16, height = 32, width = 32;
    const int64_t kernel_size = 3;
    const int64_t stride = 2;
    const int64_t padding = 1;
    const int64_t groups = 4;
    const double eps = 1e-5;
    const auto input = torch::randn({batch_size, in_channels, depth, height, width},
                                  torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Create reference model
    auto conv_transpose = torch::nn::ConvTranspose3d(
        torch::nn::ConvTranspose3dOptions(in_channels, out_channels, {kernel_size, kernel_size, kernel_size})
            .stride({stride, stride, stride})
            .padding({padding, padding, padding})
            .bias(true)
    );
    auto group_norm = torch::nn::GroupNorm(
        torch::nn::GroupNormOptions(groups, out_channels).eps(eps)
    );

    // Move to CUDA and convert to half precision
    conv_transpose->to(torch::kCUDA, torch::kHalf);
    group_norm->to(torch::kCUDA, torch::kHalf);
    conv_transpose->eval();
    group_norm->eval();

    // Run reference forward pass
    auto x = input.clone();
    x = conv_transpose->forward(x);
    x = torch::sigmoid(x) * x;  // Swish
    x = group_norm->forward(x);
    x = torch::hardswish(x);  // Corrected HardSwish activation
    const auto reference_output = x;

    // Prepare custom implementation
    auto custom_output = torch::empty_like(reference_output);
    launch_gpu_implementation(
        custom_output.data_ptr(),
        input.data_ptr(),
        conv_transpose->weight.data_ptr(),
        conv_transpose->bias.data_ptr(),
        group_norm->weight.data_ptr(),
        group_norm->bias.data_ptr(),
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups,
        eps,
        batch_size,
        depth,
        height,
        width
    );

    // Verify results
    const bool all_close = torch::allclose(reference_output, custom_output, 1e-1, 1e-1);
    std::cout << (all_close ? "passed" : "failed") << std::endl;

    return 0;
}
