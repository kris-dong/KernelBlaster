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

// Declaration for the GPU implementation of the KL divergence model
void launch_gpu_implementation(
    void* output,                // Output tensor (float16, CUDA)
    void* predictions,           // Input predictions tensor (float16, CUDA)
    void* targets,               // Input targets tensor (float16, CUDA)
    int64_t batch_size,          // Batch size
    int64_t feature_size         // Feature dimension (4096)
);

int main() {
    // Set default dtype to float16
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Model parameters
    constexpr int64_t batch_size = 128;
    constexpr int64_t feature_size = 4096;

    // Generate input tensors using softmax for valid distributions
    auto predictions = torch::randn({batch_size, feature_size}, torch::TensorOptions().dtype(dtype).device(device));
    predictions = torch::softmax(predictions, -1);
    auto targets = torch::randn({batch_size, feature_size}, torch::TensorOptions().dtype(dtype).device(device));
    targets = torch::softmax(targets, -1);

    // Reference implementation using libtorch
    auto predictions_log = torch::log(predictions);
    auto ref_output = torch::nn::functional::kl_div(
        predictions_log, targets,
        torch::nn::functional::KLDivFuncOptions().reduction(torch::kBatchMean)
    );

    // Allocate output tensor for GPU implementation
    auto output = torch::empty_like(ref_output);

    // Call GPU implementation
    launch_gpu_implementation(
        output.data_ptr(), 
        predictions.data_ptr(), 
        targets.data_ptr(),
        batch_size,
        feature_size
    );

    // Compare outputs (use rtol=1e-1, atol=1e-1 for fp16)
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
