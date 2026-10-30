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

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, 
                              int in_channels, int out_channels, int kernel_size, int dim);

int main() {
    // Initialize parameters
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int D = 16, H = 32, W = 32;
    const int kernel_size = 3;
    const int dim = 2;

    // Create input tensor on GPU with fp16
    auto input = torch::randn({batch_size, in_channels, D, H, W}, 
                            torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Create reference model
    auto conv = torch::nn::Conv3d(torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size));
    conv->to(torch::kCUDA, torch::kHalf);

    // Run reference implementation
    auto conv_out = conv->forward(input);
    auto min_out = torch::min(conv_out, dim, /*keepdim=*/false);
    auto min_values = std::get<0>(min_out);  // Get values from tuple
    auto reference_output = torch::softmax(min_values, 1);

    // Create output tensor for CUDA implementation
    auto cuda_output = torch::empty_like(reference_output);

    // Get raw data pointers
    void* input_ptr = input.data_ptr();
    void* output_ptr = cuda_output.data_ptr();
    void* weight_ptr = conv->weight.data_ptr();
    void* bias_ptr = conv->bias.data_ptr();

    // Launch CUDA kernel
    launch_gpu_implementation(output_ptr, input_ptr, weight_ptr, bias_ptr,
                            in_channels, out_channels, kernel_size, dim);

    // Compare results with fp16 tolerance
    bool passed = torch::allclose(reference_output, cuda_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
