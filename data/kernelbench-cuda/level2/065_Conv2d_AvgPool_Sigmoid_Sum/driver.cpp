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

void launch_gpu_implementation(void* output, void* input, void* conv_weight, void* conv_bias, 
                              int batch_size, int in_channels, int out_channels, 
                              int input_height, int input_width, int conv_kernel_size, 
                              int pool_kernel_size);

int main() {
    // Initialize parameters from Python
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int height = 32, width = 32;
    const int kernel_size = 3;
    const int pool_kernel_size = 2;

    // Create input tensor with FP16 dtype on CUDA
    torch::Tensor input = torch::randn({batch_size, in_channels, height, width}, 
                                      torch::device(torch::kCUDA).dtype(torch::kHalf));

    // Create reference model components
    auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                      .stride(1).padding(0));
    conv->to(torch::kCUDA, torch::kHalf);  // Move to CUDA and cast to half precision
    
    auto avg_pool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(pool_kernel_size)
                                        .stride(pool_kernel_size));

    // Run reference implementation
    auto ref_output = conv->forward(input);
    ref_output = avg_pool(ref_output);
    ref_output = torch::sigmoid(ref_output);
    ref_output = torch::sum(ref_output, {1, 2, 3});  // Sum over CHW dimensions

    // Prepare CUDA implementation output buffer
    torch::Tensor cuda_output = torch::empty({batch_size}, 
                                            torch::device(torch::kCUDA).dtype(torch::kHalf));

    // Get raw data pointers for GPU memory
    void* input_ptr = input.data_ptr();
    void* output_ptr = cuda_output.data_ptr();
    void* weight_ptr = conv->weight.data_ptr();
    void* bias_ptr = conv->bias.data_ptr();

    // Launch custom CUDA implementation
    launch_gpu_implementation(output_ptr, input_ptr, weight_ptr, bias_ptr,
                             batch_size, in_channels, out_channels,
                             height, width, kernel_size, pool_kernel_size);

    // Verify results with FP16-appropriate tolerances
    bool passed = torch::allclose(cuda_output, ref_output, 
                                 /*rtol=*/1e-1, /*atol=*/1e-1);

    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
