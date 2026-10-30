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

void launch_gpu_implementation(
    void* output, 
    void* input, 
    int dim, 
    int64_t batch_size, 
    int64_t input_size
); // declaration only

int main() {
    // Set the default dtype to float16
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Model parameters
    int64_t batch_size = 128;
    int64_t input_size = 4000;
    int dim = 1;

    // Create random input tensor on GPU
    torch::Tensor input = torch::randn({batch_size, input_size}, torch::TensorOptions().dtype(dtype).device(device));
    torch::Tensor reference_output = torch::cumprod(input, dim);

    // Allocate output tensor for CUDA kernel
    torch::Tensor gpu_output = torch::empty_like(reference_output);

    // Call the CUDA kernel launcher
    launch_gpu_implementation(
        gpu_output.data_ptr(), 
        input.data_ptr(), 
        dim, 
        batch_size, 
        input_size
    );

    // Compare outputs using torch::allclose
    double rtol = 1e-1;
    double atol = 1e-1;
    bool is_close = torch::allclose(gpu_output, reference_output, rtol, atol);

    if (is_close) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
