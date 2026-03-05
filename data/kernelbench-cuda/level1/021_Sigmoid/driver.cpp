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

// Declaration for the CUDA kernel launch function
void launch_gpu_implementation(
    void* output,           // Output tensor (float16, GPU memory)
    void* input,            // Input tensor (float16, GPU memory)
    int64_t batch_size,     // Batch size
    int64_t dim             // Dimension
);

int main() {
    // Set the default dtype to float16 for all tensors
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Model parameters
    const int64_t batch_size = 16;
    const int64_t dim = 16384;

    // Generate random input tensor on GPU with float16 dtype
    torch::Tensor input = torch::randn({batch_size, dim}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference implementation using libtorch
    torch::Tensor ref_output = torch::sigmoid(input);

    // Allocate output tensor for CUDA kernel
    torch::Tensor output = torch::empty({batch_size, dim}, torch::TensorOptions().dtype(dtype).device(device));

    // Launch the CUDA kernel implementation
    launch_gpu_implementation(
        output.data_ptr(),    // void* output
        input.data_ptr(),     // void* input
        batch_size,
        dim
    );

    // Compare outputs using torch::allclose with fp16 tolerances
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
