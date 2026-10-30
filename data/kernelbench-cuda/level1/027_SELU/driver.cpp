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
#include <ATen/ATen.h>
#include <iostream>
#include "cuda_model.cuh"

// Declaration for the launch_gpu_implementation function
void launch_gpu_implementation(void* output, void* input, int64_t batch_size, int64_t dim);

int main() {
    // Set device to CUDA
    torch::Device device(torch::kCUDA);

    // Input dimensions
    const int64_t batch_size = 16;
    const int64_t dim = 16384;

    // Create input tensor on GPU with fp16
    torch::Tensor x = torch::randn({batch_size, dim}, torch::dtype(torch::kHalf).device(device));

    // Reference implementation using libtorch (SELU activation)
    torch::Tensor ref_output = torch::selu(x);

    // Allocate output tensor for CUDA implementation
    torch::Tensor gpu_output = torch::empty_like(ref_output);

    // Launch CUDA implementation
    launch_gpu_implementation(
        gpu_output.data_ptr<at::Half>(),
        x.data_ptr<at::Half>(),
        batch_size,
        dim
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
