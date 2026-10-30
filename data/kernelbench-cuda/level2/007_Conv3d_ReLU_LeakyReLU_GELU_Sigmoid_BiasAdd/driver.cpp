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
    void* conv_weight,
    void* conv_bias,
    void* model_bias,
    int in_channels,
    int out_channels,
    int kernel_size,
    const std::vector<int64_t>& bias_shape,
    const std::vector<int64_t>& input_shape
);

int main() {
    // Configuration parameters
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int depth = 16, height = 32, width = 32;
    const int kernel_size = 3;
    const std::vector<int64_t> bias_shape{out_channels, 1, 1, 1};
    const std::vector<int64_t> input_shape{batch_size, in_channels, depth, height, width};
    
    // Create CUDA tensors with float16 dtype
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    torch::Tensor input = torch::randn(input_shape, options);
    
    // Initialize reference model components
    auto conv = torch::nn::Conv3d(torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size).bias(true));
    conv->to(torch::kCUDA, torch::kFloat16);
    torch::Tensor model_bias = torch::randn(bias_shape, options);
    
    // Run reference implementation
    torch::Tensor ref_output = conv->forward(input);
    ref_output = torch::relu(ref_output);
    ref_output = torch::leaky_relu(ref_output, 0.01);
    ref_output = torch::gelu(ref_output);
    ref_output = torch::sigmoid(ref_output);
    ref_output += model_bias;
    
    // Prepare output tensor for CUDA kernel
    torch::Tensor output = torch::empty_like(ref_output, options);
    
    // Get raw pointers for kernel launch
    void* conv_bias_ptr = conv->bias.defined() ? conv->bias.data_ptr() : nullptr;
    
    // Launch CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        conv->weight.data_ptr(),
        conv_bias_ptr,
        model_bias.data_ptr(),
        in_channels,
        out_channels,
        kernel_size,
        bias_shape,
        input_shape
    );
    
    // Verify results with fp16 thresholds
    bool passed = torch::allclose(ref_output, output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;
    
    return 0;
}
