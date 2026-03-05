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
    void* gn_weight, void* gn_bias,
    int batch_size, int in_channels, int out_channels,
    int kernel_size, int stride, int padding, int num_groups,
    int D, int H, int W);

int main() {
    // Initialize parameters from Python
    const int in_channels = 3;
    const int out_channels = 16;
    const int kernel_size = 3;
    const int num_groups = 8;
    const int batch_size = 128;
    const int D = 16, H = 32, W = 32;
    const int stride = 1;
    const int padding = 0;

    // Create input tensor on GPU with fp16
    auto input = torch::randn({batch_size, in_channels, D, H, W},
        torch::dtype(torch::kFloat16).device(torch::kCUDA));

    // Create reference model components
    auto conv = torch::nn::Conv3d(torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
        .stride(stride).padding(padding));
    conv->to(torch::kCUDA, torch::kFloat16);
    
    auto gn = torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, out_channels));
    gn->to(torch::kCUDA, torch::kFloat16);

    // Run reference implementation
    auto ref_output = conv->forward(input);
    ref_output = gn->forward(ref_output);
    ref_output = ref_output.mean({1, 2, 3, 4});  // Reduce to [batch_size]

    // Prepare parameters for GPU kernel
    void* conv_weight_ptr = conv->weight.data_ptr();
    void* conv_bias_ptr = conv->bias.data_ptr();
    void* gn_weight_ptr = gn->weight.data_ptr();
    void* gn_bias_ptr = gn->bias.data_ptr();

    // Allocate GPU output tensor
    auto output = torch::empty({batch_size}, 
        torch::dtype(torch::kFloat16).device(torch::kCUDA));
    void* output_ptr = output.data_ptr();

    // Launch custom GPU implementation
    launch_gpu_implementation(
        output_ptr, input.data_ptr(),
        conv_weight_ptr, conv_bias_ptr,
        gn_weight_ptr, gn_bias_ptr,
        batch_size, in_channels, out_channels,
        kernel_size, stride, padding, num_groups,
        D, H, W);

    // Verify results with fp16 tolerance
    bool is_close = torch::allclose(ref_output, output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (is_close ? "passed" : "failed") << std::endl;

    return 0;
}
