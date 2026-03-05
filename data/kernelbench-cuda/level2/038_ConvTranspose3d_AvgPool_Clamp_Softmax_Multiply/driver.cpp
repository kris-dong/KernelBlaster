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
                              void* conv_weight, void* conv_bias,
                              int in_channels, int out_channels,
                              int kernel_size, int stride, 
                              int padding, int output_padding,
                              int pool_kernel_size,
                              float clamp_min, float clamp_max);

int main() {
    // Setup parameters
    const int batch_size = 16;
    const int in_channels = 8;
    const int out_channels = 16;
    const int depth = 16, height = 32, width = 32;
    const int kernel_size = 3;
    const int stride = 2;
    const int padding = 1;
    const int output_padding = 1;
    const int pool_kernel_size = 2;
    const float clamp_min = 0.0f;
    const float clamp_max = 1.0f;

    // Create input tensor on GPU
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);
    torch::Tensor input = torch::randn({batch_size, in_channels, depth, height, width}, options);

    // Create reference model with corrected ConvTranspose3d initialization
    auto conv_transpose = torch::nn::ConvTranspose3d(
        torch::nn::ConvTranspose3dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .output_padding(output_padding));
    
    auto avg_pool = torch::nn::AvgPool3d(
        torch::nn::AvgPool3dOptions(pool_kernel_size)
    );

    // Move modules to GPU
    conv_transpose->to(torch::kCUDA, torch::kHalf);
    avg_pool->to(torch::kCUDA);

    // Run reference implementation
    torch::Tensor x = conv_transpose->forward(input);
    x = avg_pool(x);
    x = torch::clamp(x, clamp_min, clamp_max);
    x = torch::softmax(x, 1);
    torch::Tensor reference_output = x * 2;

    // Prepare GPU implementation
    torch::Tensor gpu_output = torch::empty_like(reference_output);
    
    // Get weight and bias pointers from conv_transpose
    auto conv_weight = conv_transpose->weight.data_ptr();
    auto conv_bias = conv_transpose->bias.data_ptr();

    // Call GPU implementation
    launch_gpu_implementation(
        gpu_output.data_ptr(),
        input.data_ptr(),
        conv_weight,
        conv_bias,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        pool_kernel_size,
        clamp_min,
        clamp_max
    );

    // Verify results
    bool passed = torch::allclose(gpu_output, reference_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
