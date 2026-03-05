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

void launch_gpu_implementation(void* output, void* input, int64_t batch_size, int64_t dim, int64_t logsoftmax_dim);

int main() {
    // Set default dtype to float16
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Model configuration
    int64_t batch_size = 16;
    int64_t dim = 16384;
    int64_t logsoftmax_dim = 1;

    // Generate input tensor on GPU, dtype float16
    torch::Tensor x = torch::randn({batch_size, dim}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference output using libtorch implementation
    torch::Tensor ref_output = torch::log_softmax(x, logsoftmax_dim);

    // Allocate output tensor for CUDA implementation
    torch::Tensor output = torch::empty_like(ref_output);

    // Call the CUDA implementation (user will implement this)
    launch_gpu_implementation(
        output.data_ptr(),        // void* output
        x.data_ptr(),             // void* input
        batch_size,
        dim,
        logsoftmax_dim
    );

    // Compare the CUDA implementation output to the reference
    // Use rtol and atol = 1e-1 for fp16
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
