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

// Declaration of the GPU implementation
void launch_gpu_implementation(
    void* output,
    void* input,
    float s,
    int64_t M,
    int64_t N
);

int main() {
    // Set the default dtype to float16 for all tensors
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Model parameters
    const int64_t M = 16384;
    const int64_t N = 4096;
    const float s = 3.14f;

    // Generate random input on GPU with fp16
    torch::Tensor A = torch::randn({M, N}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference output using libtorch (on GPU)
    torch::Tensor ref_output = A * s; // (M, N) * scalar

    // Output tensor for GPU kernel
    torch::Tensor gpu_output = torch::empty_like(ref_output);

    // Call the GPU implementation
    launch_gpu_implementation(
        gpu_output.data_ptr(),    // output
        A.data_ptr(),             // input
        s,                        // scalar
        M,                        // rows
        N                         // cols
    );

    // Compare results
    double rtol = 1e-1, atol = 1e-1; // fp16 tolerance
    bool passed = torch::allclose(gpu_output, ref_output, rtol, atol);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
