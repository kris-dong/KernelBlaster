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

// Declaration for GPU implementation
// All pointers point to GPU memory (CUDA tensors)
void launch_gpu_implementation(
    void* output,        // (N, M) float16, output tensor
    void* input_A,       // (N) float16, diagonal vector A
    void* input_B,       // (N, M) float16, matrix B
    int64_t N,           // size of A, first dim of B
    int64_t M            // second dim of B
);

int main() {
    // Set device and dtype
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Problem size
    const int64_t N = 4096;
    const int64_t M = 4096;

    // Create random inputs on GPU
    torch::Tensor A = torch::randn({N}, torch::TensorOptions().dtype(dtype).device(device));
    torch::Tensor B = torch::randn({N, M}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference implementation using libtorch
    // C_ref = diag(A) @ B
    torch::Tensor diagA = torch::diag(A);  // (N, N)
    torch::Tensor C_ref = torch::matmul(diagA, B); // (N, M)

    // Allocate output tensor for the CUDA kernel
    torch::Tensor C_out = torch::empty({N, M}, torch::TensorOptions().dtype(dtype).device(device));

    // Call the CUDA implementation
    launch_gpu_implementation(
        C_out.data_ptr(),   // output
        A.data_ptr(),       // input_A
        B.data_ptr(),       // input_B
        N,
        M
    );

    // Compare results: use rtol/atol=1e-1 for fp16
    bool passed = torch::allclose(C_ref, C_out, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
