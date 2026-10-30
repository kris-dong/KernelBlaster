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
#include <iomanip>
#include "cuda_model.cuh"

// Declaration for the GPU implementation
void launch_gpu_implementation(
    void* output,            // Output: float16 tensor (M, N)
    void* input_A,           // Input: float16 tensor (K, M)
    void* input_B,           // Input: float16 tensor (N, K)
    int64_t M,
    int64_t N,
    int64_t K
);

int main() {
    // Set device to CUDA
    torch::Device device(torch::kCUDA);

    // Set dtype to float16
    torch::Dtype dtype = torch::kFloat16;

    // Problem sizes
    const int64_t M = 1024;
    const int64_t K = 4096;
    const int64_t N = 2048;

    // Generate input tensors (K, M) and (N, K) on CUDA, float16
    torch::Tensor A = torch::randn({K, M}, torch::TensorOptions().dtype(dtype).device(device));
    torch::Tensor B = torch::randn({N, K}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference output using the PyTorch logic: C = matmul(A.T, B.T)
    // A.T: (M, K), B.T: (K, N), result: (M, N)
    torch::Tensor ref_output = torch::matmul(A.transpose(0, 1), B.transpose(0, 1));

    // Allocate output for CUDA kernel
    torch::Tensor output = torch::empty({M, N}, torch::TensorOptions().dtype(dtype).device(device));

    // Call the CUDA kernel (pass pointers to the raw data)
    launch_gpu_implementation(
        output.data_ptr(),    // void* output
        A.data_ptr(),         // void* input_A
        B.data_ptr(),         // void* input_B
        M,
        N,
        K
    );

    // Compare outputs using torch::allclose with rtol=1e-1, atol=1e-1 for fp16
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
