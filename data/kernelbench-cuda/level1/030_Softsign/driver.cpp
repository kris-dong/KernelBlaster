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

void launch_gpu_implementation(void* output, void* input, int64_t batch_size, int64_t dim);

int main() {
    // Set default dtype to float16
    torch::Dtype dtype = torch::kFloat16;

    // Set device to CUDA
    torch::Device device(torch::kCUDA);

    // Model parameters
    const int64_t batch_size = 16;
    const int64_t dim = 16384;

    // Generate input tensor on GPU
    torch::Tensor x = torch::randn({batch_size, dim}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference output using PyTorch (Softsign: x / (1 + |x|))
    torch::Tensor ref_output = x / (1 + torch::abs(x));

    // Allocate output tensor for CUDA kernel
    torch::Tensor output = torch::empty_like(ref_output);

    // Launch CUDA implementation
    // Pass raw pointers for input and output
    launch_gpu_implementation(
        output.data_ptr(),         // void* output
        x.data_ptr(),              // void* input
        batch_size,                // int64_t batch_size
        dim                        // int64_t dim
    );

    // Compare outputs using torch::allclose
    bool is_close = torch::allclose(
        output,
        ref_output,
        /*rtol=*/1e-1,
        /*atol=*/1e-1
    );

    if (is_close) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
