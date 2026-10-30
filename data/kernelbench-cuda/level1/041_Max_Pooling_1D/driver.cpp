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

// Declaration of the GPU implementation.
void launch_gpu_implementation(
    void* output, 
    void* input, 
    int64_t batch_size, 
    int64_t features, 
    int64_t sequence_length, 
    int64_t kernel_size, 
    int64_t stride, 
    int64_t padding, 
    int64_t dilation, 
    bool return_indices
);

int main() {
    // Set device and dtype
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model and pooling parameters
    const int64_t batch_size = 16;
    const int64_t features = 64;
    const int64_t sequence_length = 128;
    const int64_t kernel_size = 4;
    const int64_t stride = 2;
    const int64_t padding = 2;
    const int64_t dilation = 3;
    const bool return_indices = false;

    // Generate input tensor on GPU
    torch::Tensor input = torch::randn({batch_size, features, sequence_length}, torch::TensorOptions().dtype(dtype).device(device));
    
    // Reference implementation using libtorch
    torch::nn::MaxPool1d maxpool(torch::nn::MaxPool1dOptions(kernel_size)
                                     .stride(stride)
                                     .padding(padding)
                                     .dilation(dilation));
    maxpool->to(device, dtype);

    torch::Tensor ref_output = maxpool->forward(input);

    // Prepare output tensor for CUDA kernel (same shape and dtype as reference)
    torch::Tensor output = torch::empty_like(ref_output, torch::TensorOptions().dtype(dtype).device(device));

    // Launch the CUDA kernel (user will implement this in a separate file)
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        batch_size,
        features,
        sequence_length,
        kernel_size,
        stride,
        padding,
        dilation,
        return_indices
    );

    // Compare outputs using allclose with rtol=1e-1, atol=1e-1 for fp16
    bool close = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (close) {
        std::cout << "passed\n";
    } else {
        std::cout << "failed\n";
    }

    return 0;
}
