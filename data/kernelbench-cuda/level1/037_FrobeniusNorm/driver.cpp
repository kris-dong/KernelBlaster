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
#include <cuda_runtime.h>
#include "cuda_model.cuh"

// Declaration for the CUDA implementation
void launch_gpu_implementation(
    void* output,                // Output tensor (GPU memory)
    void* input,                 // Input tensor (GPU memory)
    int64_t batch_size,
    int64_t features,
    int64_t dim1,
    int64_t dim2
);

int main() {
    // Model and input configuration
    using scalar_t = at::Half; // fp16
    constexpr int64_t batch_size = 16;
    constexpr int64_t features = 64;
    constexpr int64_t dim1 = 256;
    constexpr int64_t dim2 = 256;

    // Generate random input tensor on GPU (fp16)
    torch::Tensor input = torch::randn({batch_size, features, dim1, dim2}, torch::dtype(torch::kFloat16).device(torch::kCUDA));
    torch::Tensor ref_output;

    // Reference output using libtorch
    {
        // Compute Frobenius norm: equivalent to L2 norm over all elements
        torch::Tensor norm = torch::norm(input, 2); // 2-norm over all elements (default dims and keepdim=false)
        ref_output = input / norm;
    }

    // Allocate output tensor on GPU and fill with a known value
    torch::Tensor output = torch::full_like(input, -10000.0);

    // Call the CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        batch_size,
        features,
        dim1,
        dim2
    );

    // Make sure all CUDA ops are done before comparison
    cudaDeviceSynchronize();

    // Compare the CUDA output to the reference using torch::allclose
    double rtol = 1e-1, atol = 1e-1; // fp16
    bool passed = torch::allclose(output, ref_output, rtol, atol);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
        // Print a small part of the tensors for debugging
        std::cout << "Reference output (flattened, first 5): " << ref_output.flatten().slice(0, 0, 5) << std::endl;
        std::cout << "CUDA output (flattened, first 5): " << output.flatten().slice(0, 0, 5) << std::endl;
    }

    return 0;
}
