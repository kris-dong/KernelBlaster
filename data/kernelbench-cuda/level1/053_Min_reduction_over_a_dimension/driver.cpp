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
#include <vector>
#include "cuda_model.cuh"

// Declaration for the GPU implementation
void launch_gpu_implementation(
    void* output,
    void* input,
    int64_t dim,
    int64_t batch_size,
    int64_t dim1,
    int64_t dim2
);

int main() {
    // Set the default dtype to float16 (Half)
    torch::Dtype dtype = torch::kFloat16;

    // Model parameters
    int64_t batch_size = 16;
    int64_t dim1 = 256;
    int64_t dim2 = 256;
    int64_t reduce_dim = 1; // Example, change as needed

    // Create input on GPU
    torch::Tensor input = torch::randn({batch_size, dim1, dim2}, torch::TensorOptions().dtype(dtype).device(torch::kCUDA));

    // Reference output using libtorch (min reduction over the specified dimension)
    torch::Tensor ref_output = std::get<0>(torch::min(input, /*dim=*/reduce_dim));

    // Prepare output tensor for GPU kernel (same shape as ref_output)
    torch::Tensor gpu_output = torch::empty_like(ref_output, torch::TensorOptions().device(torch::kCUDA));

    // Call GPU implementation
    launch_gpu_implementation(
        gpu_output.data_ptr(),
        input.data_ptr(),
        reduce_dim,
        batch_size,
        dim1,
        dim2
    );

    // Compare outputs using torch::allclose with rtol/atol for fp16
    bool passed = torch::allclose(
        gpu_output, ref_output,
        /*rtol=*/1e-1, /*atol=*/1e-1
    );

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
