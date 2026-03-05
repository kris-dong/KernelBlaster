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
    void* output, // pointer to output tensor (C)
    void* input_A, // pointer to input tensor A
    void* input_B, // pointer to input tensor B
    int64_t N // matrix size
);

int main() {
    // Set device to CUDA
    torch::Device device(torch::kCUDA);

    // Use fp16 (Half) for all tensors
    using dtype = at::Half;

    // Matrix size
    const int64_t N = 4096;

    // Generate two random symmetric matrices on GPU
    torch::Tensor A = torch::randn({N, N}, torch::TensorOptions().dtype(torch::kFloat16).device(device));
    A = (A + A.transpose(0, 1)) / 2;
    torch::Tensor B = torch::randn({N, N}, torch::TensorOptions().dtype(torch::kFloat16).device(device));
    B = (B + B.transpose(0, 1)) / 2;

    // Libtorch reference output
    torch::Tensor ref_output = torch::matmul(A, B);

    // Prepare output tensor for CUDA kernel
    torch::Tensor cuda_output = torch::empty({N, N}, torch::TensorOptions().dtype(torch::kFloat16).device(device));

    // Launch the custom CUDA implementation
    launch_gpu_implementation(
        cuda_output.data_ptr(), // output
        A.data_ptr(),           // input A
        B.data_ptr(),           // input B
        N                       // matrix size
    );

    // Compare outputs using torch::allclose
    // For fp16, use rtol=1e-1, atol=1e-1
    bool passed = torch::allclose(ref_output, cuda_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }
    return 0;
}
