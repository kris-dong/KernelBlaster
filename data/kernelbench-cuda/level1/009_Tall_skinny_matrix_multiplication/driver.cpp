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

// Declaration for the GPU implementation.
// Since we are using fp16, we use at::Half for torch::kFloat16 tensors.
void launch_gpu_implementation(
    void* output,           // Output: (M, M), at::Half*
    void* input_A,          // Input: (M, N), at::Half*
    void* input_B,          // Input: (N, M), at::Half*
    int64_t M,              // Rows of A, cols of B, output rows/cols
    int64_t N               // Cols of A, rows of B
);

int main() {
    // Set device to CUDA and dtype to Half (float16)
    torch::Device device(torch::kCUDA);
    using half_t = at::Half;

    // Matrix sizes
    const int64_t M = 16384;
    const int64_t N = 16;

    // Generate test inputs in fp16 on CUDA
    torch::Tensor A = torch::randn({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(device));
    torch::Tensor B = torch::randn({N, M}, torch::TensorOptions().dtype(torch::kFloat16).device(device));

    // Reference output using libtorch
    torch::Tensor ref_output = torch::matmul(A, B);

    // Allocate output tensor for the CUDA kernel
    torch::Tensor output = torch::empty({M, M}, torch::TensorOptions().dtype(torch::kFloat16).device(device));

    // Call the GPU implementation
    launch_gpu_implementation(
        output.data_ptr<half_t>(),
        A.data_ptr<half_t>(),
        B.data_ptr<half_t>(),
        M,
        N
    );

    // Use torch::allclose to compare outputs with relaxed tolerances for fp16
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
