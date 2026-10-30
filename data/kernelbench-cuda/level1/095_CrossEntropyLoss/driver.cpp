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

// Declaration for the GPU implementation.
// - output: pointer to float16 (half) scalar loss
// - predictions: pointer to batch_size x num_classes float16 (half) tensor
// - targets: pointer to batch_size int64 tensor (class indices)
// - batch_size, num_classes: dimensions
void launch_gpu_implementation(
    void* output,                // Output: scalar loss, float16
    void* predictions,           // Input: [batch_size, num_classes], float16
    void* targets,               // Input: [batch_size], int64
    int64_t batch_size,
    int64_t num_classes
);

int main() {
    using torch::indexing::Slice;
    torch::manual_seed(42);

    // Set up CUDA device and dtype
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model and input sizes
    int64_t batch_size = 4096;
    int64_t num_classes = 10;

    // Generate random predictions and targets on GPU
    torch::Tensor predictions = torch::randn({batch_size, num_classes}, torch::TensorOptions().dtype(dtype).device(device));
    torch::Tensor targets = torch::randint(0, num_classes, {batch_size}, torch::TensorOptions().dtype(torch::kLong).device(device));

    // Reference: PyTorch cross_entropy (reduction='mean')
    torch::Tensor ref_loss = torch::nn::functional::cross_entropy(predictions, targets);

    // Allocate output for GPU kernel (scalar float16)
    torch::Tensor output = torch::empty({1}, torch::TensorOptions().dtype(dtype).device(device));

    // Launch GPU implementation
    launch_gpu_implementation(
        output.data_ptr(),
        predictions.data_ptr(),
        targets.data_ptr(),
        batch_size,
        num_classes
    );

    // Move both outputs to float32 for comparison (torch::allclose does not support float16 directly)
    torch::Tensor output_fp32 = output.to(torch::kFloat32);
    torch::Tensor ref_loss_fp32 = ref_loss.to(torch::kFloat32);

    // Use rtol and atol suitable for fp16
    double rtol = 1e-1, atol = 1e-1;
    bool passed = torch::allclose(output_fp32, ref_loss_fp32, rtol, atol);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
        std::cout << "Reference: " << ref_loss_fp32.item<float>() << std::endl;
        std::cout << "GPU Output: " << output_fp32.item<float>() << std::endl;
    }
    return 0;
}
