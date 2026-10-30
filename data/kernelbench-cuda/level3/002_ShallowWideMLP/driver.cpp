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
#include <cuda_runtime.h>
#include "cuda_model.cuh"

// Declaration only. Do not implement here.
void launch_gpu_implementation(
    void* output,
    void* input,
    void* w1,
    void* b1,
    void* w2,
    void* b2,
    void* w3,
    void* b3,
    int64_t batch_size,
    int64_t input_size,
    int64_t hidden_size1,
    int64_t hidden_size2,
    int64_t output_size
);

int main() {
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available. Test cannot run on GPU. Exiting (return code 0 as requested)." << std::endl;
        return 0;
    }

    torch::Device device(torch::kCUDA);
    auto dtype = torch::kFloat16; // Use fp16 as per torch.set_default_dtype(torch.float16)
    auto options = torch::TensorOptions().dtype(dtype).device(device);

    // Problem parameters from the provided Python code
    int64_t batch_size = 1;
    int64_t input_size = 1000;
    std::vector<int64_t> hidden_layer_sizes = {2000, 2000};
    int64_t output_size = 10;

    int64_t hidden_size1 = hidden_layer_sizes[0];
    int64_t hidden_size2 = hidden_layer_sizes[1];

    // Inputs on GPU
    auto input = torch::randn({batch_size, input_size}, options).contiguous();

    // Randomly initialized weights and biases on GPU (to be passed to the CUDA kernel)
    auto W1 = torch::randn({hidden_size1, input_size}, options).contiguous();
    auto b1 = torch::randn({hidden_size1}, options).contiguous();

    auto W2 = torch::randn({hidden_size2, hidden_size1}, options).contiguous();
    auto b2 = torch::randn({hidden_size2}, options).contiguous();

    auto W3 = torch::randn({output_size, hidden_size2}, options).contiguous();
    auto b3 = torch::randn({output_size}, options).contiguous();

    // Reference implementation using libtorch (GPU)
    auto y1 = torch::matmul(input, W1.transpose(0, 1)) + b1;
    y1 = torch::relu(y1);

    auto y2 = torch::matmul(y1, W2.transpose(0, 1)) + b2;
    y2 = torch::relu(y2);

    auto ref_output = torch::matmul(y2, W3.transpose(0, 1)) + b3;

    // Output tensor for the GPU implementation (initialized to zeros to avoid uninitialized memory)
    auto gpu_output = torch::zeros_like(ref_output);

    // Launch the user-provided CUDA implementation
    launch_gpu_implementation(
        static_cast<void*>(gpu_output.data_ptr()),
        static_cast<void*>(input.data_ptr()),
        static_cast<void*>(W1.data_ptr()),
        static_cast<void*>(b1.data_ptr()),
        static_cast<void*>(W2.data_ptr()),
        static_cast<void*>(b2.data_ptr()),
        static_cast<void*>(W3.data_ptr()),
        static_cast<void*>(b3.data_ptr()),
        batch_size,
        input_size,
        hidden_size1,
        hidden_size2,
        output_size
    );

    // Ensure all CUDA work is complete before comparison
    cudaDeviceSynchronize();

    // Since dtype is fp16, use rtol=1e-1 and atol=1e-1
    bool ok = torch::allclose(gpu_output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (ok) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    // Always return 0 as requested
    return 0;
}
