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

struct ModelImpl : torch::nn::Module {
    torch::nn::Linear matmul{nullptr};
    torch::nn::MaxPool1d max_pool{nullptr};
    float scale_factor;

    ModelImpl(int in_features, int out_features, int kernel_size, float scale_factor_)
        : matmul(torch::nn::LinearOptions(in_features, out_features)),
          max_pool(torch::nn::MaxPool1dOptions(kernel_size)),
          scale_factor(scale_factor_) {
        register_module("matmul", matmul);
        register_module("max_pool", max_pool);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = matmul->forward(x);
        x = x.unsqueeze(1); // Add channel dim
        x = max_pool(x);
        x = x.squeeze(1);
        x = torch::sum(x, /*dim=*/1);
        x = x * scale_factor;
        return x;
    }
};
TORCH_MODULE(Model);

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, int kernel_size, float scale_factor);

int main() {
    // Model parameters
    int batch_size = 128;
    int in_features = 10;
    int out_features = 5;
    int kernel_size = 2;
    float scale_factor = 0.5f;

    // Create model and move to CUDA
    Model model(ModelImpl(in_features, out_features, kernel_size, scale_factor));
    model->to(torch::kCUDA, torch::kHalf);
    model->eval();

    // Generate input tensor on CUDA
    auto input = torch::randn({batch_size, in_features}, torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Run reference implementation
    auto reference_output = model->forward(input);

    // Prepare output tensor for GPU implementation
    auto output = torch::empty_like(reference_output);

    // Get parameters from the model
    auto weight = model->matmul->weight;
    auto bias = model->matmul->bias;

    // Call GPU implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        kernel_size,
        scale_factor
    );

    // Verify results
    bool passed = torch::allclose(reference_output, output, /*rtol=*/1e-1, /*atol=*/1e-1);
    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
