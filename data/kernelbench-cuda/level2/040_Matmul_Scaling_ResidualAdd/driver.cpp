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

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, float scaling_factor, int batch_size, int in_features, int out_features);

int main() {
    // Configuration parameters
    const int batch_size = 128;
    const int in_features = 64;
    const int out_features = 128;
    const float scaling_factor = 0.5f;

    // Create input tensor on GPU with fp16
    auto input = torch::randn({batch_size, in_features}, torch::dtype(torch::kFloat16).device(torch::kCUDA));

    // Create and initialize reference model
    auto linear = torch::nn::Linear(in_features, out_features);
    linear->to(torch::kCUDA, torch::kFloat16);
    
    // Manually initialize with same parameters
    torch::NoGradGuard no_grad;
    linear->weight.normal_();
    linear->bias.normal_();

    // Reference forward pass
    auto x = linear->forward(input);
    auto original_x = x.clone();
    x = x * scaling_factor + original_x;
    auto reference_output = x;

    // Allocate GPU memory for CUDA implementation output
    auto cuda_output = torch::empty_like(reference_output);

    // Get raw pointers for GPU memory
    auto input_ptr = input.data_ptr();
    auto weight_ptr = linear->weight.data_ptr();
    auto bias_ptr = linear->bias.data_ptr();
    auto output_ptr = cuda_output.data_ptr();

    // Launch CUDA implementation
    launch_gpu_implementation(
        output_ptr,
        input_ptr,
        weight_ptr,
        bias_ptr,
        scaling_factor,
        batch_size,
        in_features,
        out_features
    );

    // Verify results
    bool passed = torch::allclose(reference_output, cuda_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
