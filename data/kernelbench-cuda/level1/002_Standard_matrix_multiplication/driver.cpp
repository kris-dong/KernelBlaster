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
#include <cuda_runtime.h>
#include "cuda_model.cuh"

void launch_gpu_implementation(
    void* output,
    void* input_A,
    void* input_B,
    int64_t M,
    int64_t K,
    int64_t N
);

int main() {
    // Set default dtype to float16
    torch::Dtype default_dtype = torch::kFloat16;

    // Set device to CUDA
    torch::Device device(torch::kCUDA);

    // Model dimensions
    constexpr int64_t M = 1024;
    constexpr int64_t K = 4096;
    constexpr int64_t N = 2048;

    // Create input tensors on GPU with fp16
    auto A = torch::randn({M, K}, torch::TensorOptions().dtype(default_dtype).device(device));
    auto B = torch::randn({K, N}, torch::TensorOptions().dtype(default_dtype).device(device));

    // Reference libtorch implementation (on GPU)
    auto ref_output = torch::matmul(A, B);

    // Prepare output tensor for GPU implementation
    auto gpu_output = torch::zeros_like(ref_output);

    // Launch GPU implementation
    launch_gpu_implementation(
        gpu_output.data_ptr(),
        A.data_ptr(),
        B.data_ptr(),
        M,
        K,
        N
    );

    // Compare outputs using torch::allclose with rtol=1e-1, atol=1e-1 for fp16
    bool passed = torch::allclose(gpu_output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
