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
#include <torch/script.h>
#include <torch/torch.h>
#include "cuda_model.cuh"

void launch_gpu_implementation(void* output, void* input, void* conv_weight, void* conv_bias, 
                              float subtract1_value, float subtract2_value, int kernel_size_pool);

int main() {
    // Configuration parameters
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int height = 32, width = 32;
    const int kernel_size = 3;
    const float subtract1_value = 0.5f;
    const float subtract2_value = 0.2f;
    const int kernel_size_pool = 2;

    // Set up device and data type
    torch::ScalarType dtype = torch::kHalf;
    torch::Device device(torch::kCUDA);
    torch::TensorOptions options = torch::TensorOptions().dtype(dtype).device(device);

    // Create input tensor
    auto input = torch::randn({batch_size, in_channels, height, width}, options);

    // Initialize model components
    auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size));
    conv->to(device, dtype);  // Move to CUDA and set dtype
    auto avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(kernel_size_pool));

    // LibTorch reference implementation
    torch::Tensor x = input.clone();
    x = conv->forward(x);
    x = x - subtract1_value;
    x = torch::tanh(x);
    x = x - subtract2_value;
    x = avgpool->forward(x);
    torch::Tensor ref_output = x;

    // Get parameter pointers
    torch::Tensor conv_weight = conv->weight;
    torch::Tensor conv_bias = conv->bias;

    // Allocate output tensor for CUDA implementation
    torch::Tensor output = torch::empty_like(ref_output);

    // Get raw pointers for GPU memory
    void* input_ptr = input.data_ptr();
    void* output_ptr = output.data_ptr();
    void* conv_weight_ptr = conv_weight.data_ptr();
    void* conv_bias_ptr = conv_bias.data_ptr();

    // Execute CUDA implementation
    launch_gpu_implementation(output_ptr, input_ptr, conv_weight_ptr, conv_bias_ptr,
                             subtract1_value, subtract2_value, kernel_size_pool);

    // Verify results
    bool is_allclose = torch::allclose(ref_output, output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (is_allclose ? "passed" : "failed") << std::endl;

    return 0;
}
