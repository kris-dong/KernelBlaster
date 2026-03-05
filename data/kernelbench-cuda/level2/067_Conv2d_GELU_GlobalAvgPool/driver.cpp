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

void launch_gpu_implementation(void* output, void* input, const void* weight, const void* bias,
                               int in_channels, int out_channels, int kernel_size,
                               int batch_size, int height, int width);

int main() {
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int height = 32, width = 32;
    const int kernel_size = 3;

    // Create FP16 tensors on CUDA
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);
    torch::Tensor input = torch::randn({batch_size, in_channels, height, width}, options);

    // Create reference model
    torch::nn::Conv2d conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size));
    conv->to(torch::kCUDA);
    conv->to(torch::kHalf);

    // Run forward pass
    torch::Tensor x = conv->forward(input);
    x = torch::gelu(x);
    x = torch::adaptive_avg_pool2d(x, {1, 1});
    x = x.squeeze(-1).squeeze(-1);
    torch::Tensor reference_output = x;

    // Get parameter pointers
    torch::Tensor weight = conv->weight;
    torch::Tensor bias = conv->bias;

    // Allocate output tensor for CUDA implementation
    torch::Tensor output = torch::empty_like(reference_output);

    // Call CUDA implementation
    launch_gpu_implementation(output.data_ptr(),
                             input.data_ptr(),
                             weight.data_ptr(),
                             bias.data_ptr(),
                             in_channels,
                             out_channels,
                             kernel_size,
                             batch_size,
                             height,
                             width);

    // Verify results with FP16-appropriate tolerances
    bool passed = torch::allclose(reference_output, output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
