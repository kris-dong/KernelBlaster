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

void launch_gpu_implementation(void* output, void* input, void* weights, void* bias, float multiplier, float negative_slope);

int main() {
    const int64_t in_features = 1024;
    const int64_t out_features = 512;
    const float multiplier = 2.0f;
    const float negative_slope = 0.1f;
    const int batch_size = 128;

    // Create model and convert to FP16
    auto linear = torch::nn::Linear(in_features, out_features);
    linear->to(torch::kCUDA, torch::kFloat16);  // Fixed device/dtype specification
    auto weights = linear->weight;
    auto bias = linear->bias;

    // Create input tensor on GPU
    auto input = torch::randn({batch_size, in_features}, 
                            torch::dtype(torch::kFloat16).device(torch::kCUDA));

    // Reference implementation
    auto x = linear->forward(input);
    x = x * multiplier;
    x = torch::leaky_relu(x, negative_slope);
    auto ref_output = x;

    // Allocate output tensor for GPU implementation
    auto test_output = torch::empty_like(ref_output);

    // Get raw pointers for GPU implementation
    auto input_ptr = input.data_ptr();
    auto output_ptr = test_output.data_ptr();
    auto weights_ptr = linear->weight.data_ptr();
    auto bias_ptr = linear->bias.defined() ? linear->bias.data_ptr() : nullptr;

    // Launch custom GPU kernel
    launch_gpu_implementation(output_ptr, input_ptr, weights_ptr, bias_ptr, 
                            multiplier, negative_slope);

    // Verify results with FP16 tolerance
    bool passed = torch::allclose(ref_output, test_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
