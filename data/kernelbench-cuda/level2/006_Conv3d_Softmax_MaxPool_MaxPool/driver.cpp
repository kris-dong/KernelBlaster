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

void launch_gpu_implementation(void* output, void* input, int in_channels, int out_channels, 
                              int kernel_size, int pool_kernel_size, void* conv_weight, void* conv_bias);

int main() {
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int depth = 16, height = 32, width = 32;
    const int kernel_size = 3;
    const int pool_kernel_size = 2;

    // Create input tensor on GPU with FP16
    auto input = torch::randn({batch_size, in_channels, depth, height, width}, 
                            torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Create reference model components
    auto conv = torch::nn::Conv3d(torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size));
    auto pool1 = torch::nn::MaxPool3d(pool_kernel_size);
    auto pool2 = torch::nn::MaxPool3d(pool_kernel_size);

    // Move modules to CUDA and convert parameters to FP16
    conv->to(torch::kCUDA, torch::kHalf);
    pool1->to(torch::kCUDA);
    pool2->to(torch::kCUDA);

    // Run reference implementation
    auto x = conv->forward(input);
    x = torch::softmax(x, 1);
    x = pool1->forward(x);
    x = pool2->forward(x);
    auto ref_output = x.clone();

    // Create output tensor for CUDA kernel
    auto output = torch::empty_like(ref_output);

    // Get parameter pointers
    auto conv_weight = conv->weight;
    auto conv_bias = conv->bias;

    // Call CUDA implementation
    launch_gpu_implementation(output.data_ptr(), input.data_ptr(),
                             in_channels, out_channels, kernel_size, pool_kernel_size,
                             conv_weight.data_ptr(), conv_bias.data_ptr());

    // Verify results with FP16 tolerance
    if (torch::allclose(ref_output, output, /*rtol=*/1e-1, /*atol=*/1e-1)) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
