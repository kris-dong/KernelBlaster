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

// Declaration for the GPU kernel launcher.
// Since the operation is a sum reduction, and there are no learned parameters, only input/output and reduction dimension are needed.
// We use void* for input/output for flexibility, but in practice these will be torch::Half* pointers.
void launch_gpu_implementation(
    void* output,        // Output tensor data pointer (fp16)
    const void* input,   // Input tensor data pointer (fp16)
    int64_t batch_size,
    int64_t dim1,
    int64_t dim2,
    int64_t reduce_dim   // reduction dimension (in this case, 1)
);

int main() {
    // Set device and default dtype for fp16
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model and dimensions
    int64_t batch_size = 16;
    int64_t dim1 = 256;
    int64_t dim2 = 256;
    int64_t reduce_dim = 1;

    // Create input tensor on CUDA with fp16
    torch::Tensor x = torch::randn({batch_size, dim1, dim2}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference output using PyTorch (libtorch)
    torch::Tensor ref_output = torch::sum(x, reduce_dim, /*keepdim=*/true);

    // Prepare output tensor for CUDA kernel
    torch::Tensor gpu_output = torch::empty_like(ref_output);

    // Launch the custom CUDA implementation
    launch_gpu_implementation(
        gpu_output.data_ptr(),
        x.data_ptr(),
        batch_size,
        dim1,
        dim2,
        reduce_dim
    );

    // Compare outputs using torch::allclose with rtol=1e-1, atol=1e-1 for fp16
    bool passed = torch::allclose(gpu_output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
