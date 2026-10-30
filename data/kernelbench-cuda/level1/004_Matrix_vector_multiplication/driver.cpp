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
// All pointers are to GPU memory (fp16)
void launch_gpu_implementation(void* output, void* input_A, void* input_B, int M, int K);

// Main test code
int main() {
    // Set default dtype to float16 and device to CUDA
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Matrix dimensions
    const int M = 256;
    const int K = 131072;

    // Create random inputs on GPU
    torch::Tensor A = torch::randn({M, K}, torch::TensorOptions().dtype(dtype).device(device));
    torch::Tensor B = torch::randn({K, 1}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference output using libtorch
    torch::Tensor ref_output = torch::matmul(A, B);

    // Allocate output tensor for GPU kernel
    torch::Tensor output = torch::empty({M, 1}, torch::TensorOptions().dtype(dtype).device(device));

    // Launch custom CUDA implementation (inputs and output as raw pointers)
    launch_gpu_implementation(
        output.data_ptr(),               // void* output
        A.data_ptr(),                    // void* input_A
        B.data_ptr(),                    // void* input_B
        M, K
    );

    // Compare outputs using torch::allclose with rtol=1e-1, atol=1e-1 (fp16)
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
