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
void launch_gpu_implementation(
    void* output,                   // Output tensor pointer (float16 on CUDA)
    void* input,                    // Input tensor pointer (float16 on CUDA)
    int64_t batch_size,             // Number of batches
    int64_t num_features,           // Number of features (channels)
    int64_t dim1,                   // First spatial dimension
    int64_t dim2,                   // Second spatial dimension
    float eps                       // Epsilon for numerical stability
);

int main() {
    // Set device to CUDA and use float16
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model and input sizes
    const int64_t batch_size = 16;
    const int64_t num_features = 64;
    const int64_t dim1 = 256;
    const int64_t dim2 = 256;
    const float eps = 1e-5f;

    // Generate input tensor on CUDA, float16
    torch::Tensor input = torch::randn(
        {batch_size, num_features, dim1, dim2},
        torch::TensorOptions().dtype(dtype).device(device)
    );

    // Reference output using PyTorch RMSNorm logic
    // rms = sqrt(mean(x ** 2, dim=1, keepdim=True) + eps)
    torch::Tensor squared = input.pow(2);
    torch::Tensor mean_squared = squared.mean(1, /*keepdim=*/true);
    torch::Tensor rms = (mean_squared + eps).sqrt();
    torch::Tensor ref_output = input / rms;

    // Allocate output tensor for custom CUDA kernel
    torch::Tensor output = torch::empty_like(input);

    // Call the CUDA kernel (pass raw pointers)
    launch_gpu_implementation(
        output.data_ptr(),                   // Output pointer
        input.data_ptr(),                    // Input pointer
        batch_size,
        num_features,
        dim1,
        dim2,
        eps
    );

    // Compare outputs using torch::allclose (rtol/atol = 1e-1 for fp16)
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
