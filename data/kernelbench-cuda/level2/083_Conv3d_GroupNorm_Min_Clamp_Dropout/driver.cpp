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
    int kernel_size, int groups,
    float min_value, float max_value, float dropout_p
);

int main() {
    // Configuration parameters with adjusted min_value to prevent all-zero output
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int depth = 16, height = 32, width = 32;
    const int kernel_size = 3;
    const int groups = 8;
    const float min_value = -0.5f;  // Changed from 0.0 to create non-zero output
    const float max_value = 1.0f;
    const float dropout_p = 0.2f;

    // Create input tensor on GPU with fp16
    auto input = torch::randn({batch_size, in_channels, depth, height, width}, 
                            torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));

    // Create and configure model components
    auto conv = torch::nn::Conv3d(torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size));
    auto norm = torch::nn::GroupNorm(torch::nn::GroupNormOptions(groups, out_channels));
    auto dropout = torch::nn::Dropout(dropout_p);

    // Move parameters to CUDA and convert to fp16
    conv->to(torch::kCUDA, torch::kFloat16);
    norm->to(torch::kCUDA, torch::kFloat16);
    dropout->to(torch::kCUDA);

    // Set model to evaluation mode
    conv->eval();
    norm->eval();
    dropout->eval();

    // Run reference implementation
    auto x = input.clone();
    x = conv->forward(x);
    x = norm->forward(x);
    x = torch::min(x, torch::tensor(min_value, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA)));
    x = torch::clamp(x, min_value, max_value);
    x = dropout->forward(x);
    auto reference_output = x;

    // Verify reference contains non-zero values
    auto zero_count = torch::sum(reference_output == 0.0).item<int>();
    if (zero_count == reference_output.numel()) {
        std::cerr << "Test invalid: Reference output is all zeros\n";
        return 1;
    }

    // Get parameter pointers
    void* conv_weight_ptr = conv->weight.data_ptr();
    void* conv_bias_ptr = conv->bias.data_ptr();
    void* gn_weight_ptr = norm->weight.data_ptr();
    void* gn_bias_ptr = norm->bias.data_ptr();

    // Prepare output tensor for CUDA implementation
    auto cuda_output = torch::empty_like(reference_output);
    void* cuda_output_ptr = cuda_output.data_ptr();

    // Launch GPU implementation
    launch_gpu_implementation(
        cuda_output_ptr,
        input.data_ptr(),
        conv_weight_ptr,
        conv_bias_ptr,
        gn_weight_ptr,
        gn_bias_ptr,
        kernel_size,
        groups,
        min_value,
        max_value,
        dropout_p
    );

    // Verify results with explicit check
    bool passed = torch::allclose(reference_output, cuda_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    if (!passed) {
        std::cout << "Validation failed - differences detected:\n";
        auto diff = (reference_output - cuda_output).abs();
        std::cout << "Max difference: " << diff.max().item<float>() << "\n";
        std::cout << "Mean difference: " << diff.mean().item<float>() << "\n";
    }
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
