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
#include <cuda_runtime.h>

// Declaration for the GPU implementation
// All tensors are half precision (torch::kHalf) and reside on CUDA device.
void launch_gpu_implementation(
    void* output,        // output: pointer to GPU memory, shape [batch_size, dim], type half
    void* input,         // input: pointer to GPU memory, shape [batch_size, dim], type half
    int64_t batch_size,  // batch size (16)
    int64_t dim          // feature dimension (16384)
);

int main() {
    torch::Dtype dtype = torch::kHalf;
    torch::Device device(torch::kCUDA);

    int64_t batch_size = 16;
    int64_t dim = 16384;

    // Create input tensor on GPU with half precision
    torch::Tensor input = torch::randn({batch_size, dim}, torch::TensorOptions().dtype(dtype).device(device));
    torch::Tensor ref_input = input.clone();

    // Reference output using PyTorch (L2 normalization along dim=1)
    torch::Tensor norm = torch::norm(ref_input, 2, /*dim=*/1, /*keepdim=*/true);
    torch::Tensor ref_output = ref_input / norm;

    // Allocate output tensor for kernel
    torch::Tensor output = torch::empty_like(input);

    // Fill output with random data to avoid accidental allclose passing from uninitialized memory
    output.uniform_(-10.0, 10.0);

    // Call GPU implementation (not implemented yet)
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        batch_size,
        dim
    );

    // Synchronize to ensure kernel (even if not implemented) has finished
    cudaDeviceSynchronize();

    // Compare (should fail, as kernel is not implemented and output is random)
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
