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

// Declaration for the CUDA implementation.
// The output and input pointers are assumed to be device pointers (GPU memory).
// All tensors are fp16 (at::Half).
void launch_gpu_implementation(
    void* output,          // [batch_size, input_shape] (or scalar) fp16, device pointer
    void* predictions,     // [batch_size, input_shape] fp16, device pointer
    void* targets,         // [batch_size, input_shape] fp16, device pointer
    int64_t batch_size,
    int64_t input0,
    int64_t input1
);

int main() {
    // Set up parameters
    constexpr int64_t batch_size = 128;
    constexpr int64_t input0 = 4096;
    constexpr int64_t input1 = 1; // input_shape is (4096,), so single dimension

    // Use fp16 (Half) for all tensors
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Generate random test inputs on GPU
    auto predictions = torch::randn({batch_size, input0}, torch::TensorOptions().dtype(dtype).device(device));
    auto targets = torch::randn({batch_size, input0}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference output using PyTorch's Smooth L1 Loss (Huber Loss)
    // Default reduction is 'mean' in torch.nn.functional.smooth_l1_loss
    auto reference_output = torch::nn::functional::smooth_l1_loss(
        predictions, targets, torch::nn::functional::SmoothL1LossFuncOptions());

    // Allocate output tensor for CUDA implementation (result is scalar, [1])
    auto cuda_output = torch::empty({}, torch::TensorOptions().dtype(dtype).device(device));

    // Call CUDA implementation (passing raw device pointers)
    launch_gpu_implementation(
        cuda_output.data_ptr(),
        predictions.data_ptr(),
        targets.data_ptr(),
        batch_size,
        input0,
        input1
    );

    // Compare outputs using torch::allclose.
    // Use rtol=1e-1, atol=1e-1 for fp16.
    bool is_close = torch::allclose(cuda_output, reference_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    if (is_close) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
        std::cout << "Reference output: " << reference_output.item<float>()
                  << ", CUDA output: " << cuda_output.item<float>() << std::endl;
    }
    return 0;
}
