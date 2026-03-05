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
    // Set device to CUDA and use float16
    torch::Device device(torch::kCUDA);
    using scalar_t = at::Half; // fp16
    
    int64_t batch_size = 16;
    int64_t dim = 16384;

    // Create input tensor on GPU, fp16
    torch::Tensor x = torch::randn({batch_size, dim}, torch::TensorOptions().dtype(torch::kFloat16).device(device));

    // Reference implementation using libtorch (Swish activation)
    torch::Tensor ref_output = x * torch::sigmoid(x);

    // Allocate output tensor for CUDA kernel
    torch::Tensor y = torch::empty_like(ref_output);

    // Call the CUDA implementation (passing raw pointers)
    launch_gpu_implementation(
        y.data_ptr<scalar_t>(),
        x.data_ptr<scalar_t>(),
        batch_size,
        dim
    );

    // Compare outputs using torch::allclose. Use rtol and atol of 1e-1 for fp16.
    bool passed = torch::allclose(y, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
