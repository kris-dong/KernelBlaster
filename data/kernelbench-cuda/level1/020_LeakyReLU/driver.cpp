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

void launch_gpu_implementation(
    void* output, 
    void* input, 
    float negative_slope, 
    int64_t batch_size, 
    int64_t dim
); // declaration only

int main() {
    // Set the device to CUDA and dtype to float16
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model parameters
    const float negative_slope = 0.01f;
    const int64_t batch_size = 16;
    const int64_t dim = 16384;

    // Input tensor
    torch::Tensor x = torch::randn({batch_size, dim}, torch::TensorOptions().device(device).dtype(dtype));

    // Reference output using libtorch
    torch::Tensor ref_output = torch::nn::functional::leaky_relu(x, torch::nn::functional::LeakyReLUFuncOptions().negative_slope(negative_slope));

    // Allocate output tensor for CUDA kernel
    torch::Tensor output = torch::empty_like(ref_output);

    // Launch custom CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),      // void* output
        x.data_ptr(),           // void* input
        negative_slope,         // float negative_slope
        batch_size,             // int64_t batch_size
        dim                     // int64_t dim
    );

    // Compare results. Use rtol/atol=1e-1 for fp16
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
