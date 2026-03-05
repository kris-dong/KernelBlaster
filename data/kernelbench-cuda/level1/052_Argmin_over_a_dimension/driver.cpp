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

// Declaration for the GPU implementation
// x: input tensor (fp16), output: indices (int64), dim: dimension along which to perform argmin
void launch_gpu_implementation(
    void* output,          // int64_t*
    void* input,           // at::Half*
    int64_t batch_size,
    int64_t dim1,
    int64_t dim2,
    int64_t dim
);

int main() {
    // Use fp16 for all tensors
    using torch_dtype = at::Half;

    const int64_t batch_size = 16;
    const int64_t dim1 = 256;
    const int64_t dim2 = 256;
    const int64_t dim = 1;

    // Create input tensor on CUDA, fp16
    torch::Tensor x = torch::randn({batch_size, dim1, dim2}, torch::dtype(torch::kFloat16).device(torch::kCUDA));

    // Reference output using libtorch (argmin along dim)
    torch::Tensor ref_output = torch::argmin(x, dim, /*keepdim=*/false);

    // Prepare output tensor for CUDA kernel (int64)
    torch::Tensor gpu_output = torch::empty(ref_output.sizes(), torch::dtype(torch::kInt64).device(torch::kCUDA));

    // Launch custom GPU implementation
    launch_gpu_implementation(
        gpu_output.data_ptr<int64_t>(),
        x.data_ptr<at::Half>(),
        batch_size,
        dim1,
        dim2,
        dim
    );

    // Convert outputs to float32 for allclose comparison (since argmin returns indices)
    torch::Tensor ref_output_f = ref_output.to(torch::kFloat32);
    torch::Tensor gpu_output_f = gpu_output.to(torch::kFloat32);

    // Use torch::allclose for comparison (rtol, atol = 1e-1 for fp16)
    bool passed = torch::allclose(gpu_output_f, ref_output_f, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
