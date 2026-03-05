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

// Declaration for GPU implementation
void launch_gpu_implementation(
    void* output,  // Output tensor pointer (fp16)
    void* input,   // Input tensor pointer (fp16)
    int64_t batch_size,
    int64_t dim
);

int main() {
    // Set default dtype to float16
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Model parameters
    const int64_t batch_size = 16;
    const int64_t dim = 16384;

    // Generate input tensor on GPU, dtype float16
    torch::Tensor input = torch::randn({batch_size, dim}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference output: HardSigmoid activation (libtorch)
    torch::Tensor ref_output = at::hardsigmoid(input);

    // Allocate output tensor for CUDA kernel
    torch::Tensor output = torch::empty_like(ref_output);

    // Launch CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),    // output pointer (fp16)
        input.data_ptr(),     // input pointer (fp16)
        batch_size,
        dim
    );

    // Compare the outputs using torch::allclose
    bool is_close = torch::allclose(
        output,
        ref_output,
        /*rtol=*/1e-1,  // Use 1e-1 for fp16
        /*atol=*/1e-1
    );

    if (is_close) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
