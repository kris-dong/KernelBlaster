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
#include "cuda_model.cuh"
#include <torch/torch.h>
#include <cmath>
#include <iostream>

// Declaration for the CUDA implementation.
// All pointers are assumed to point to GPU memory (CUDA tensors).
void launch_gpu_implementation(
    void* output,           // Output tensor, shape: [batch_size, dim], dtype: torch::kFloat16
    void* input,            // Input tensor,  shape: [batch_size, dim], dtype: torch::kFloat16
    int64_t batch_size,
    int64_t dim
);

int main() {
    // Set default dtype to float16 for all tensors (as in Python code)
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Model hyperparameters
    const int64_t batch_size = 2000;
    const int64_t dim = 2000;

    // Generate random input on GPU with fp16
    torch::Tensor input = torch::randn({batch_size, dim}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference implementation using PyTorch's formula (GELU, OpenAI GPT flavor)
    // y = 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    torch::Tensor x_cube = torch::pow(input, 3.0);
    torch::Tensor inner = input + 0.044715f * x_cube;
    torch::Tensor tanh_out = torch::tanh(sqrt_2_over_pi * inner);
    torch::Tensor ref_output = 0.5f * input * (1.0f + tanh_out);

    // Allocate output tensor for CUDA kernel
    torch::Tensor output = torch::empty({batch_size, dim}, torch::TensorOptions().dtype(dtype).device(device));

    // Call CUDA implementation (to be implemented elsewhere)
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        batch_size,
        dim
    );

    // Compare outputs: use rtol=1e-1, atol=1e-1 for fp16
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }
    return 0;
}
