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

// Declaration for the CUDA kernel launcher
void launch_gpu_implementation(void* output, void* input, int64_t batch_size, int64_t dim);

int main() {
    // Set the device to CUDA
    torch::Device device(torch::kCUDA);

    // Set default dtype to float16
    torch::Dtype dtype = torch::kFloat16;
    torch::TensorOptions options = torch::TensorOptions().dtype(dtype).device(device);

    // Model parameters
    const int64_t batch_size = 16;
    const int64_t dim = 16384;

    // Generate input tensor on GPU
    torch::Tensor input = torch::randn({batch_size, dim}, options);

    // Reference output using libtorch (Tanh activation)
    torch::Tensor ref_output = torch::tanh(input);

    // Allocate output tensor for CUDA kernel result
    torch::Tensor gpu_output = torch::empty({batch_size, dim}, options);

    // Call the CUDA kernel launcher
    launch_gpu_implementation(
        gpu_output.data_ptr(),   // void* output
        input.data_ptr(),        // void* input
        batch_size,
        dim
    );

    // Compare outputs with appropriate tolerance for fp16
    bool is_close = torch::allclose(
        gpu_output, ref_output,
        /*rtol=*/1e-1,   // relaxed tolerance for fp16
        /*atol=*/1e-1
    );

    if (is_close) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
