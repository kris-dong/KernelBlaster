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

// Declaration for the CUDA kernel launcher (to be implemented separately)
void launch_gpu_implementation(
    void* output,         // Output: (M, N), float16
    void* input_A,        // Input: (K, M), float16
    void* input_B,        // Input: (K, N), float16
    int64_t M, 
    int64_t K, 
    int64_t N
);

int main() {
    // Model dimensions
    const int64_t M = 1024;
    const int64_t K = 4096;
    const int64_t N = 2048;

    // Use float16 (Half) for all tensors and CUDA device
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device = torch::kCUDA;

    // Create inputs on GPU
    torch::Tensor A = torch::randn({K, M}, torch::TensorOptions().dtype(dtype).device(device));
    torch::Tensor B = torch::randn({K, N}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference implementation using libtorch (C = A^T * B)
    // A: (K, M) --> A.T: (M, K)
    // B: (K, N)
    // Result: (M, N)
    torch::Tensor ref_output = torch::matmul(A.transpose(0, 1), B);

    // Allocate output tensor for CUDA kernel
    torch::Tensor output = torch::empty({M, N}, torch::TensorOptions().dtype(dtype).device(device));

    // Launch the CUDA kernel (assume column-major/row-major as in PyTorch)
    launch_gpu_implementation(
        output.data_ptr(),       // Output: (M, N)
        A.data_ptr(),            // Input A: (K, M)
        B.data_ptr(),            // Input B: (K, N)
        M, K, N
    );

    // Compare outputs
    // Use rtol=1e-1, atol=1e-1 for fp16
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }
    return 0;
}
