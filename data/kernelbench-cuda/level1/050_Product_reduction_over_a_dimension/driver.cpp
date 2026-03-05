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

// Declaration: output and input are GPU pointers, reduction_dim is reduction axis.
void launch_gpu_implementation(
    void* output,           // output tensor (fp16)
    void* input,            // input tensor (fp16)
    int64_t batch_size,
    int64_t dim1,
    int64_t dim2,
    int64_t reduction_dim   // dimension to reduce over
);

int main() {
    // Use fp16 for all tensors
    using scalar_t = at::Half;

    torch::manual_seed(0);

    // Model and input parameters
    const int64_t batch_size = 16;
    const int64_t dim1 = 256;
    const int64_t dim2 = 256;
    const int64_t reduction_dim = 1;

    // Create input tensor on GPU, fp16
    auto x = torch::randn({batch_size, dim1, dim2}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    
    // Reference using libtorch
    auto ref_output = torch::prod(x, reduction_dim);

    // Prepare output tensor for CUDA kernel, initialize with a known value
    auto output_sizes = x.sizes().vec();
    output_sizes.erase(output_sizes.begin() + reduction_dim);
    auto y = torch::full(output_sizes, -42.0, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));

    // Call CUDA kernel implementation
    launch_gpu_implementation(
        y.data_ptr<scalar_t>(),
        x.data_ptr<scalar_t>(),
        batch_size,
        dim1,
        dim2,
        reduction_dim
    );

    // Compare results
    // For fp16, use rtol=1e-1, atol=1e-1
    bool passed = torch::allclose(y, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }
    return 0;
}
