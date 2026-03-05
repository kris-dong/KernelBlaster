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
void launch_gpu_implementation(
    void* output,             // (N, M, L) - fp16, GPU
    void* input_A,            // (N, M, K) - fp16, GPU
    void* input_B,            // (K, L)    - fp16, GPU
    int64_t N,
    int64_t M,
    int64_t K,
    int64_t L
);

int main() {
    // Set device to CUDA and dtype to float16
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model parameters (dimensions)
    const int64_t N = 16;
    const int64_t M = 1024;
    const int64_t K = 2048;
    const int64_t L = 768;

    // Generate random input tensors (on GPU, fp16)
    torch::Tensor A = torch::randn({N, M, K}, torch::TensorOptions().dtype(dtype).device(device));
    torch::Tensor B = torch::randn({K, L}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference output using libtorch
    torch::Tensor ref_output = torch::matmul(A, B);

    // Allocate output tensor for CUDA kernel
    torch::Tensor output = torch::empty({N, M, L}, torch::TensorOptions().dtype(dtype).device(device));

    // Call GPU implementation
    launch_gpu_implementation(
        output.data_ptr(),    // output (N, M, L), fp16
        A.data_ptr(),         // input A (N, M, K), fp16
        B.data_ptr(),         // input B (K, L), fp16
        N, M, K, L
    );

    // Compare outputs using torch::allclose (rtol=1e-1, atol=1e-1 for fp16)
    bool is_close = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (is_close) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
