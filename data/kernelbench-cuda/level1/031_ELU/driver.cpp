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

// Declaration of the CUDA implementation
void launch_gpu_implementation(
    void* output,           // Output tensor (fp16, GPU memory)
    void* input,            // Input tensor (fp16, GPU memory)
    float alpha,            // ELU alpha parameter
    int64_t batch_size,     // Batch size
    int64_t dim             // Feature dimension
);

int main() {
    // Parameters
    const int64_t batch_size = 16;
    const int64_t dim = 16384;
    const float alpha = 1.0f;

    // Set default dtype to float16
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device = torch::kCUDA;

    // Create random input tensor on GPU, fp16
    torch::Tensor input = torch::randn({batch_size, dim}, torch::TensorOptions().dtype(dtype).device(device));

    // Run reference ELU using libtorch (on GPU, fp16)
    torch::Tensor ref_output = torch::nn::functional::elu(input, torch::nn::functional::ELUFuncOptions().alpha(alpha));

    // Allocate output tensor for CUDA kernel (same shape, dtype, device)
    torch::Tensor cuda_output = torch::empty({batch_size, dim}, torch::TensorOptions().dtype(dtype).device(device));

    // Call CUDA kernel (implementation to be provided elsewhere)
    launch_gpu_implementation(
        cuda_output.data_ptr(),   // Output
        input.data_ptr(),         // Input
        alpha,                    // ELU alpha
        batch_size,               // Batch size
        dim                       // Feature dimension
    );

    // Compare outputs
    bool passed = torch::allclose(
        cuda_output, ref_output,
        /*rtol=*/1e-1,  // fp16: higher tolerance
        /*atol=*/1e-1
    );

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
