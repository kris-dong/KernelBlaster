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

void launch_gpu_implementation(void* output, void* input, const void* conv_weight, const void* gn_weight, const void* gn_bias, 
                               int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t groups, bool bias);

int main() {
    // Initialize parameters
    const int64_t batch_size = 16;
    const int64_t in_channels = 64;
    const int64_t out_channels = 128;
    const int64_t D = 8, H = 16, W = 16;
    const int64_t kernel_size = 3;
    const int64_t groups = 8;
    const bool bias = false;

    // Create input tensor on GPU with fp16
    auto input = torch::randn({batch_size, in_channels, D, H, W}, 
                             torch::device(torch::kCUDA).dtype(torch::kHalf));

    // Create reference model components
    auto conv = torch::nn::ConvTranspose3d(
        torch::nn::ConvTranspose3dOptions(in_channels, out_channels, {kernel_size, kernel_size, kernel_size})
        .bias(bias));
    auto group_norm = torch::nn::GroupNorm(
        torch::nn::GroupNormOptions(groups, out_channels));

    // Move parameters to GPU and convert to fp16
    conv->to(torch::kCUDA, torch::kHalf);
    group_norm->to(torch::kCUDA, torch::kHalf);

    // Run reference implementation
    auto x = conv->forward(input);
    x = torch::relu(x);
    auto ref_output = group_norm->forward(x);

    // Prepare CUDA output tensor
    auto output_cuda = torch::empty_like(ref_output, torch::device(torch::kCUDA).dtype(torch::kHalf));

    // Get raw pointers to parameters
    auto conv_weight_ptr = conv->weight.data_ptr();
    auto gn_weight_ptr = group_norm->weight.data_ptr();
    auto gn_bias_ptr = group_norm->bias.data_ptr();

    // Call CUDA implementation
    launch_gpu_implementation(
        output_cuda.data_ptr(),
        input.data_ptr(),
        conv_weight_ptr,
        gn_weight_ptr,
        gn_bias_ptr,
        in_channels,
        out_channels,
        kernel_size,
        groups,
        bias
    );

    // Verify results with fp16 tolerance
    bool is_close = torch::allclose(ref_output, output_cuda, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (is_close ? "passed" : "failed") << std::endl;

    return 0;
}
