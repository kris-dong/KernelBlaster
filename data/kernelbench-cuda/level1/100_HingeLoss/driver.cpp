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

// Declaration of GPU kernel launcher for Hinge Loss
void launch_gpu_implementation(
    void* output, 
    void* predictions, 
    void* targets, 
    int batch_size, 
    int input_dim
);

int main() {
    // Set device to CUDA and dtype to float16
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model input parameters
    constexpr int batch_size = 128;
    constexpr int input_dim = 1;

    // Generate inputs
    auto predictions = torch::randn({batch_size, input_dim}, torch::dtype(dtype).device(device));
    auto targets = (torch::randint(0, 2, {batch_size, 1}, torch::dtype(torch::kFloat16).device(device)) * 2 - 1).to(dtype);

    // Reference implementation using libtorch
    // Hinge Loss: mean(clamp(1 - predictions * targets, min=0))
    auto hinge_term = 1.0f - predictions * targets;
    auto hinge_clamped = torch::clamp(hinge_term, 0);
    auto ref_output = torch::mean(hinge_clamped);

    // Storage for GPU implementation output
    auto gpu_output = torch::empty({}, torch::dtype(dtype).device(device)); // scalar

    // Call GPU implementation (output is a scalar)
    launch_gpu_implementation(
        gpu_output.data_ptr(), 
        predictions.data_ptr(), 
        targets.data_ptr(), 
        batch_size, 
        input_dim
    );

    // Compare outputs
    bool passed = torch::allclose(
        gpu_output, 
        ref_output, 
        /*rtol=*/1e-1, 
        /*atol=*/1e-1
    );

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
        std::cout << "Reference output: " << ref_output.item<float>() << std::endl;
        std::cout << "GPU output: " << gpu_output.item<float>() << std::endl;
    }

    return 0;
}
