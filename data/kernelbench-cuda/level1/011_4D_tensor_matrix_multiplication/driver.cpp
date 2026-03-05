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
    void* output,         // Output tensor (b, i, j, k), type: at::Half*
    void* input_A,        // Input tensor A (b, i, j, l), type: at::Half*
    void* input_B,        // Input matrix B (l, k), type: at::Half*
    uint64_t b, uint64_t i, uint64_t j, uint64_t l, uint64_t k // Tensor shapes
);

int main() {
    // Set device and dtype
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Problem sizes
    int b = 16;
    int i = 256;
    int j = 512;
    int l = 256;
    int k = 768;

    // Create inputs on GPU
    torch::Tensor A = torch::randn({b, i, j, l}, torch::TensorOptions().dtype(dtype).device(device));
    torch::Tensor B = torch::randn({l, k}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference output (libtorch)
    torch::Tensor ref_output = torch::einsum("bijl,lk->bijk", {A, B});

    // Allocate output tensor for GPU kernel
    torch::Tensor output = torch::empty({b, i, j, k}, torch::TensorOptions().dtype(dtype).device(device));

    // Call the GPU implementation
    launch_gpu_implementation(
        output.data_ptr(),    // output: (b, i, j, k)
        A.data_ptr(),         // input_A: (b, i, j, l)
        B.data_ptr(),         // input_B: (l, k)
        b, i, j, l, k
    );

    // Compare outputs (using torch::allclose)
    // For fp16, use rtol and atol of 1e-1
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
