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
#include <iostream>
#include <vector>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include "cuda_model.cuh"

// Declaration only. Do not implement here.
void launch_gpu_implementation(
    void* output,
    const void* input,
    const void* w1,
    const void* b1,
    const void* w2,
    const void* b2,
    const void* w3,
    const void* b3,
    int64_t batch_size,
    int64_t input_size,
    int64_t hidden1_size,
    int64_t hidden2_size,
    int64_t output_size
);

int main() {
    // Configuration matching the provided Python code
    const int64_t batch_size = 1;
    const int64_t input_size = 1000;
    const int64_t hidden1_size = 400;
    const int64_t hidden2_size = 800;
    const int64_t output_size = 500;

    // Use CUDA device and fp16 dtype as default
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kHalf;

    // Create layers equivalent to the Python nn.Sequential:
    // Linear(1000->400) + ReLU + Linear(400->800) + ReLU + Linear(800->500)
    auto fc1 = torch::nn::Linear(torch::nn::LinearOptions(input_size, hidden1_size));
    auto fc2 = torch::nn::Linear(torch::nn::LinearOptions(hidden1_size, hidden2_size));
    auto fc3 = torch::nn::Linear(torch::nn::LinearOptions(hidden2_size, output_size));

    // Move layers to CUDA and set dtype to fp16
    fc1->to(device, dtype);
    fc2->to(device, dtype);
    fc3->to(device, dtype);

    // Create input on GPU with fp16 dtype
    auto x = torch::randn({batch_size, input_size}, torch::TensorOptions().device(device).dtype(dtype));

    // Reference forward pass using libtorch (matches the Python model structure)
    auto y1 = torch::relu(fc1->forward(x));
    auto y2 = torch::relu(fc2->forward(y1));
    auto y_ref = fc3->forward(y2);

    // Allocate output tensor for GPU implementation (initialized to zeros)
    auto y_gpu = torch::zeros_like(y_ref);

    // Extract parameter tensors (ensure on GPU and fp16)
    auto w1 = fc1->weight; // [hidden1_size, input_size]
    auto b1 = fc1->bias;   // [hidden1_size]
    auto w2 = fc2->weight; // [hidden2_size, hidden1_size]
    auto b2 = fc2->bias;   // [hidden2_size]
    auto w3 = fc3->weight; // [output_size, hidden2_size]
    auto b3 = fc3->bias;   // [output_size]

    // Launch user's GPU implementation
    launch_gpu_implementation(
        static_cast<void*>(y_gpu.data_ptr()),
        static_cast<const void*>(x.data_ptr()),
        static_cast<const void*>(w1.data_ptr()),
        static_cast<const void*>(b1.data_ptr()),
        static_cast<const void*>(w2.data_ptr()),
        static_cast<const void*>(b2.data_ptr()),
        static_cast<const void*>(w3.data_ptr()),
        static_cast<const void*>(b3.data_ptr()),
        batch_size,
        input_size,
        hidden1_size,
        hidden2_size,
        output_size
    );

    // Ensure kernel completion before checking results
    cudaDeviceSynchronize();

    // Validate results with appropriate tolerances for fp16
    const double rtol = 1e-1;
    const double atol = 1e-1;
    bool ok = torch::allclose(y_ref, y_gpu, rtol, atol);
    if (ok) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    // Always exit with return code 0
    return 0;
}
