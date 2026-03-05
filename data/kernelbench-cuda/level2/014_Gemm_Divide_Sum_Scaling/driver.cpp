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
#include "cuda_model.cuh"

struct Model : torch::nn::Module {
    torch::Tensor weight;
    float scaling_factor;

    Model(int input_size, int hidden_size, float scaling_factor) {
        auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
        weight = register_parameter("weight", torch::randn({hidden_size, input_size}, options));
        this->scaling_factor = scaling_factor;
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::matmul(x, weight.t());
        x = x / 2;
        x = torch::sum(x, {1}, /*keepdim=*/true);
        x = x * scaling_factor;
        return x;
    }
};

void launch_gpu_implementation(void* output, void* input, void* weight, float scaling_factor);

int main() {
    const int batch_size = 128;
    const int input_size = 10;
    const int hidden_size = 20;
    const float scaling_factor = 1.5f;

    // Initialize model and move to CUDA
    auto model = std::make_shared<Model>(input_size, hidden_size, scaling_factor);

    // Create input tensor on CUDA
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto input_tensor = torch::randn({batch_size, input_size}, options);

    // Compute reference output
    auto reference_output = model->forward(input_tensor);

    // Prepare output tensor for CUDA kernel
    auto output_tensor = torch::empty_like(reference_output);

    // Call CUDA implementation with raw pointers and parameters
    launch_gpu_implementation(
        output_tensor.data_ptr(),
        input_tensor.data_ptr(),
        model->weight.data_ptr(),
        model->scaling_factor
    );

    // Verify results with fp16-appropriate tolerances
    bool passed = torch::allclose(reference_output, output_tensor, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
