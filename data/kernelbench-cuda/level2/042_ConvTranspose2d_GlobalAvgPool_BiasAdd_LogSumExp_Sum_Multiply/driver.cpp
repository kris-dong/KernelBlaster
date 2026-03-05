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

void launch_gpu_implementation(void* output, void* input,
                               void* conv_weight, void* conv_bias, void* model_bias,
                               int in_channels, int out_channels, int kernel_size,
                               int64_t bias_c, int64_t bias_h, int64_t bias_w);

int main() {
    // Initialize parameters from Python equivalent
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int kernel_size = 3;
    const std::vector<int64_t> bias_shape{out_channels, 1, 1};
    const int height = 32, width = 32;

    // Create GPU tensors with float16 dtype
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    
    // Create input tensor
    auto input = torch::randn({batch_size, in_channels, height, width}, options);
    
    // Initialize model parameters
    auto conv_weight = torch::randn({in_channels, out_channels, kernel_size, kernel_size}, options);
    auto conv_bias = torch::randn({out_channels}, options);
    auto model_bias = torch::randn(bias_shape, options);

    // Reference implementation using libtorch
    auto x = torch::conv_transpose2d(input, conv_weight, conv_bias, 1, 0);
    x = torch::mean(x, {2, 3}, /*keepdim=*/true);
    x = x + model_bias;
    x = torch::logsumexp(x, 1, /*keepdim=*/true);
    x = torch::sum(x, {2, 3});
    auto reference_output = x * 10.0;

    // Create output tensor for GPU implementation
    auto output = torch::empty_like(reference_output);

    // Call GPU implementation with all parameters
    launch_gpu_implementation(output.data_ptr(), input.data_ptr(),
                             conv_weight.data_ptr(), conv_bias.data_ptr(), model_bias.data_ptr(),
                             in_channels, out_channels, kernel_size,
                             bias_shape[0], bias_shape[1], bias_shape[2]);

    // Verify results with fp16 tolerance
    bool passed = torch::allclose(reference_output, output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
