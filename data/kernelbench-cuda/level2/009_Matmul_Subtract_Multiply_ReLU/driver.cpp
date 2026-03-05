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

void launch_gpu_implementation(void* output, void* input, void* linear_weight, void* linear_bias, 
                              void* subtract_value, void* multiply_value, int batch_size, 
                              int in_features, int out_features);

int main() {
    // Configuration parameters
    const int batch_size = 128;
    const int in_features = 10;
    const int out_features = 5;
    const float subtract_value = 2.0f;
    const float multiply_value = 1.5f;

    // Create model components on GPU with float16
    torch::nn::Linear linear(in_features, out_features);
    linear->to(torch::kCUDA, torch::kFloat16);
    
    // Create scalar tensors for arithmetic operations
    auto subtract_val = torch::full({}, subtract_value, torch::dtype(torch::kFloat16).device(torch::kCUDA));
    auto multiply_val = torch::full({}, multiply_value, torch::dtype(torch::kFloat16).device(torch::kCUDA));

    // Generate random input tensor on GPU
    auto input = torch::randn({batch_size, in_features}, 
                             torch::dtype(torch::kFloat16).device(torch::kCUDA)).contiguous();

    // LibTorch reference implementation
    auto x = torch::linear(input, linear->weight, linear->bias);
    x = x.sub(subtract_val);
    x = x.mul(multiply_val);
    auto reference_output = torch::relu(x).contiguous();

    // Prepare output tensor
    auto output = torch::empty_like(reference_output).contiguous();

    // Call CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        linear->weight.contiguous().data_ptr(),
        linear->bias.contiguous().data_ptr(),
        subtract_val.data_ptr(),
        multiply_val.data_ptr(),
        batch_size,
        in_features,
        out_features
    );

    // Verify results
    bool passed = torch::allclose(reference_output, output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
