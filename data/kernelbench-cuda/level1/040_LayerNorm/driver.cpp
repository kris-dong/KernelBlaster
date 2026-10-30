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
    void* weight, 
    void* bias, 
    int64_t batch_size, 
    int64_t features, 
    int64_t dim1, 
    int64_t dim2
); // declaration only

int main() {
    // Set the device to CUDA and use float16 (fp16)
    torch::Device device(torch::kCUDA);
    using scalar_t = at::Half;

    // Model parameters
    constexpr int64_t batch_size = 16;
    constexpr int64_t features = 64;
    constexpr int64_t dim1 = 256;
    constexpr int64_t dim2 = 256;

    // Normalized shape for LayerNorm
    std::vector<int64_t> normalized_shape = {features, dim1, dim2};

    // Generate random input tensor on GPU, fp16
    auto x = torch::randn({batch_size, features, dim1, dim2}, torch::TensorOptions().dtype(torch::kFloat16).device(device));

    // Create LayerNorm as shared_ptr and move to CUDA, fp16
    auto layer_norm = std::make_shared<torch::nn::LayerNormImpl>(
        torch::nn::LayerNormOptions(normalized_shape).elementwise_affine(true)
    );
    layer_norm->to(device, torch::kFloat16);

    // Get weight and bias pointers (ensure they're contiguous and fp16)
    auto weight = layer_norm->weight.detach().clone().contiguous().to(torch::kFloat16).to(device);
    auto bias = layer_norm->bias.detach().clone().contiguous().to(torch::kFloat16).to(device);

    // Forward pass using libtorch (reference)
    auto ref_output = layer_norm->forward(x);

    // Allocate output tensor for CUDA kernel
    auto output = torch::empty_like(ref_output);

    // Launch your CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        batch_size,
        features,
        dim1,
        dim2
    );

    // Compare results using torch::allclose
    bool passed = torch::allclose(
        output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1
    );

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
