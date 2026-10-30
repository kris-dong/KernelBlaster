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
#include "cuda_model.cuh"
#include <torch/torch.h>
#include <iostream>

void launch_gpu_implementation(void* output, void* input_A, void* input_B, int64_t N);

int main() {
    // Set device and dtype
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;
    const int64_t M = 4096;

    // Generate random lower triangular matrices on GPU, fp16
    auto A_full = torch::randn({M, M}, torch::TensorOptions().dtype(dtype).device(device));
    auto B_full = torch::randn({M, M}, torch::TensorOptions().dtype(dtype).device(device));
    auto A = torch::tril(A_full);
    auto B = torch::tril(B_full);

    // Reference implementation (libtorch)
    auto ref = torch::tril(torch::matmul(A, B));

    // Allocate output tensor for CUDA implementation, zero initialized for test safety
    auto out = torch::zeros({M, M}, torch::TensorOptions().dtype(dtype).device(device));

    // Call CUDA kernel launcher (not implemented yet, so output should be all zeros)
    launch_gpu_implementation(
        out.data_ptr(),                   // output pointer
        A.data_ptr(),                     // input A pointer
        B.data_ptr(),                     // input B pointer
        M                                 // matrix size
    );

    // Compare results with tolerance for fp16 (rtol=1e-1, atol=1e-1)
    bool passed = torch::allclose(out, ref, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
