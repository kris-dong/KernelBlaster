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
    void* output,                    // Output tensor (GPU memory, fp16)
    void* input,                     // Input tensor (GPU memory, fp16)
    int64_t batch_size,
    int64_t in_channels,
    int64_t input_length,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding
); // declaration only

int main() {
    // Set default dtype to Half (fp16)
    torch::Dtype dtype = torch::kFloat16;

    // Device
    torch::Device device(torch::kCUDA);

    // Model parameters
    int64_t batch_size = 16;
    int64_t in_channels = 32;
    int64_t input_length = 128;
    int64_t kernel_size = 4;
    int64_t stride = 2;
    int64_t padding = 1;

    // Create random input on GPU
    torch::Tensor input = torch::randn({batch_size, in_channels, input_length}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference implementation using libtorch
    torch::nn::AvgPool1d avg_pool(torch::nn::AvgPool1dOptions(kernel_size).stride(stride).padding(padding));
    avg_pool->to(device, dtype);

    torch::Tensor ref_output = avg_pool->forward(input);

    // Allocate output tensor for CUDA kernel
    torch::Tensor output = torch::empty_like(ref_output, torch::TensorOptions().device(device).dtype(dtype));

    // Launch CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),      // output
        input.data_ptr(),       // input
        batch_size,
        in_channels,
        input_length,
        kernel_size,
        stride,
        padding
    );

    // Compare outputs
    double rtol = 1e-1;
    double atol = 1e-1;
    bool passed = torch::allclose(output, ref_output, rtol, atol);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
