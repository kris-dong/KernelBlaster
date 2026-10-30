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
    // Set default dtype to float16 for all tensors
    torch::Dtype dtype = torch::kFloat16;

    // Model parameters
    const int64_t batch_size = 16;
    const int64_t dim = 16384;

    // Generate input on CUDA
    torch::Tensor x = torch::randn({batch_size, dim}, torch::TensorOptions().dtype(dtype).device(torch::kCUDA));

    // Reference output using libtorch's Softplus
    torch::Tensor ref_output = torch::nn::functional::softplus(x);

    // Allocate output tensor for CUDA kernel
    torch::Tensor output = torch::empty_like(ref_output);

    // Launch custom CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),                   // output pointer
        x.data_ptr(),                        // input pointer
        batch_size,
        dim
    );

    // Compare outputs using torch::allclose with appropriate tolerances for fp16
    const double rtol = 1e-1;
    const double atol = 1e-1;
    bool passed = torch::allclose(output, ref_output, rtol, atol);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
