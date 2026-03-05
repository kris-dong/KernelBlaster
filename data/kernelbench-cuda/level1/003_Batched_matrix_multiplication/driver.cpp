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
// All tensors are fp16 (at::Half) and on CUDA
void launch_gpu_implementation(void* output, void* A, void* B, int batch_size, int m, int k, int n);

// Main test code
int main() {
    // Set device to CUDA
    torch::Device device(torch::kCUDA);

    // Use fp16 for all tensors
    using dtype = at::Half;

    // Problem sizes
    const int batch_size = 128;
    const int m = 128;
    const int k = 256;
    const int n = 512;

    // Generate random fp16 inputs on CUDA
    torch::Tensor A = torch::randn({batch_size, m, k}, torch::TensorOptions().dtype(torch::kFloat16).device(device));
    torch::Tensor B = torch::randn({batch_size, k, n}, torch::TensorOptions().dtype(torch::kFloat16).device(device));

    // Reference implementation using libtorch (torch::bmm)
    torch::Tensor ref_output = torch::bmm(A, B);

    // Allocate output tensor for GPU implementation
    torch::Tensor gpu_output = torch::empty({batch_size, m, n}, torch::TensorOptions().dtype(torch::kFloat16).device(device));

    // Call the launch_gpu_implementation; pass raw pointers to data
    launch_gpu_implementation(
        gpu_output.data_ptr<dtype>(),
        A.data_ptr<dtype>(),
        B.data_ptr<dtype>(),
        batch_size, m, k, n
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
