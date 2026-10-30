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

// Declaration for the GPU kernel launcher
void launch_gpu_implementation(
    void* output,           // output: pointer to float16 (fp16) scalar [1 element]
    void* predictions,      // input: pointer to float16 tensor [batch_size, 4096]
    void* targets,          // input: pointer to float16 tensor [batch_size, 4096]
    int64_t batch_size,     // batch size (128)
    int64_t input_dim       // input dimension (4096)
);

int main() {
    // Use CUDA device
    torch::Device device(torch::kCUDA);

    // Set dtype to float16
    using scalar_t = at::Half;

    // Model parameters
    constexpr int64_t batch_size = 128;
    constexpr int64_t input_dim = 4096;

    // Generate random inputs (on GPU), dtype = fp16
    torch::Tensor predictions = torch::randn({batch_size, input_dim}, torch::TensorOptions().dtype(torch::kFloat16).device(device));
    torch::Tensor targets = torch::randn({batch_size, input_dim}, torch::TensorOptions().dtype(torch::kFloat16).device(device));

    // Reference output using PyTorch implementation
    torch::Tensor ref_output = torch::mean((predictions - targets).pow(2));

    // Allocate output tensor for kernel (on GPU)
    torch::Tensor output = torch::empty({1}, torch::TensorOptions().dtype(torch::kFloat16).device(device));

    // Launch custom CUDA implementation
    launch_gpu_implementation(
        output.data_ptr<scalar_t>(),
        predictions.data_ptr<scalar_t>(),
        targets.data_ptr<scalar_t>(),
        batch_size,
        input_dim
    );

    // Move outputs to float32 for allclose comparison (torch::allclose does not support fp16 well)
    torch::Tensor ref_output_f32 = ref_output.to(torch::kFloat32);
    torch::Tensor output_f32 = output.to(torch::kFloat32);

    // Use rtol=1e-1, atol=1e-1 for fp16
    bool is_close = torch::allclose(
        output_f32, ref_output_f32,
        /*rtol=*/1e-1, /*atol=*/1e-1
    );

    if (is_close) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
        std::cout << "Reference output: " << ref_output_f32.item<float>() << std::endl;
        std::cout << "Kernel output:    " << output_f32.item<float>() << std::endl;
    }

    return 0;
}
