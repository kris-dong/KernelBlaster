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

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, void* scaling_factor, int in_features, int out_features, int batch_size);

int main() {
    const int batch_size = 128;
    const int in_features = 1024;
    const int out_features = 512;
    const float scaling_factor_val = 2.0f;

    // Create CUDA tensors with FP16 dtype
    auto input_tensor = torch::randn({batch_size, in_features}, 
                                   torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Create and configure model components
    auto linear = torch::nn::Linear(in_features, out_features);
    linear->to(torch::kCUDA, torch::kHalf);  // Move to CUDA and convert weights to FP16

    // Create scaling factor tensor in FP16
    auto scaling_factor = torch::full({}, scaling_factor_val, 
                                    torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Reference implementation
    auto x = linear->forward(input_tensor);
    x = x * torch::sigmoid(x);
    x = x * scaling_factor;

    // Allocate output tensor for CUDA kernel
    auto output_tensor = torch::empty_like(x);

    // Get raw pointers for kernel launch
    launch_gpu_implementation(
        output_tensor.data_ptr(),
        input_tensor.data_ptr(),
        linear->weight.data_ptr(),
        linear->bias.data_ptr(),
        scaling_factor.data_ptr(),
        in_features,
        out_features,
        batch_size
    );

    // Verify results with relaxed tolerances for FP16
    bool passed = torch::allclose(x, output_tensor, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
