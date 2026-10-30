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

// Declaration for the CUDA kernel launcher
void launch_gpu_implementation(
    void* output,
    void* input,
    int64_t dim,
    int64_t batch_size,
    int64_t dim1,
    int64_t dim2
);

int main() {
    // Use fp16 for all tensors
    using scalar_t = at::Half;

    // Model parameters
    constexpr int64_t batch_size = 16;
    constexpr int64_t dim1 = 256;
    constexpr int64_t dim2 = 256;
    constexpr int64_t reduce_dim = 1; // As per get_init_inputs()

    // Set device to CUDA
    torch::Device device(torch::kCUDA);

    // Generate input tensor (fp16)
    torch::Tensor input = torch::randn({batch_size, dim1, dim2}, torch::dtype(torch::kFloat16).device(device));
    torch::Tensor ref_input = input.clone(); // for reference

    // Reference output using PyTorch
    torch::Tensor ref_output = torch::mean(ref_input, reduce_dim);

    // Allocate output tensor for custom CUDA kernel
    std::vector<int64_t> output_shape = input.sizes().vec();
    output_shape.erase(output_shape.begin() + reduce_dim);
    torch::Tensor output = torch::empty(output_shape, torch::dtype(torch::kFloat16).device(device));

    // Launch CUDA kernel implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        reduce_dim,
        batch_size,
        dim1,
        dim2
    );

    // Compare outputs using torch::allclose (rtol/atol = 1e-1 for fp16)
    bool is_close = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (is_close) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
