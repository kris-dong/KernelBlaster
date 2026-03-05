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
#include <iostream>
#include "cuda_model.cuh"

// Declaration for the GPU implementation
// All tensors are fp16 (torch::kHalf), device is CUDA
void launch_gpu_implementation(
    void* output,                 // (M, N) fp16 CUDA tensor
    void* input_A,                // (M, K) fp16 CUDA tensor
    void* input_B,                // (K, N) fp16 CUDA tensor
    int64_t M, int64_t N, int64_t K // matrix dimensions
);

int main() {
    // Matrix dimensions
    const int64_t M = 16384;
    const int64_t N = 16384;
    const int64_t K = 32;

    // Use fp16 for all tensors
    torch::Dtype dtype = torch::kHalf;
    torch::Device device(torch::kCUDA);

    // Generate random inputs on GPU
    torch::Tensor A = torch::randn({M, K}, torch::TensorOptions().dtype(dtype).device(device));
    torch::Tensor B = torch::randn({K, N}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference output using libtorch
    torch::Tensor ref_output = torch::matmul(A, B);

    // Allocate output tensor for kernel
    torch::Tensor output = torch::empty({M, N}, torch::TensorOptions().dtype(dtype).device(device));

    // Run GPU implementation
    launch_gpu_implementation(
        output.data_ptr(),
        A.data_ptr(),
        B.data_ptr(),
        M, N, K
    );

    // Compare output
    // Use rtol and atol of 1e-1 for fp16
    bool is_close = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (is_close) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
