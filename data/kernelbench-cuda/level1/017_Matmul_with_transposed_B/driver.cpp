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

void launch_gpu_implementation(
    void* output,            // output: (M, N), fp16
    void* input_A,           // input A: (M, K), fp16
    void* input_B,           // input B: (N, K), fp16
    int64_t M, 
    int64_t K, 
    int64_t N
);

int main() {
    // Use CUDA and set default dtype to Half (fp16)
    torch::Device device(torch::kCUDA);
    using torch_dtype = at::Half;

    // Problem dimensions
    const int64_t M = 1024;
    const int64_t K = 4096;
    const int64_t N = 2048;

    // Create random inputs on GPU
    auto A = torch::randn({M, K}, torch::TensorOptions().dtype(torch::kFloat16).device(device));
    auto B = torch::randn({N, K}, torch::TensorOptions().dtype(torch::kFloat16).device(device));

    // Reference output using libtorch (C = A * B^T)
    auto B_T = B.transpose(0, 1); // (K, N)
    auto ref_output = torch::matmul(A, B_T); // (M, N)

    // Allocate output tensor for GPU kernel
    auto output = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(device));

    // Call the CUDA kernel launcher (user implementation)
    launch_gpu_implementation(
        output.data_ptr(),   // output: (M, N)
        A.data_ptr(),        // input: (M, K)
        B.data_ptr(),        // input: (N, K)
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
