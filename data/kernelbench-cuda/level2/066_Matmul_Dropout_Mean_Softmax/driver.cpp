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
    torch::nn::Linear matmul{nullptr};
    torch::nn::Dropout dropout{nullptr};

    Model(int in_features, int out_features, double dropout_p) {
        matmul = register_module("matmul", torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features)));
        dropout = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(dropout_p)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = matmul->forward(x);
        x = dropout->forward(x);
        x = torch::mean(x, {1}, /*keepdim=*/true);
        x = torch::softmax(x, 1);
        return x;
    }
};

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, float dropout_p);

int main() {
    // Initialize parameters
    const int in_features = 100;
    const int out_features = 50;
    const float dropout_p = 0.2f;
    const int batch_size = 128;

    // Create model and move to CUDA with float16
    auto model = std::make_shared<Model>(in_features, out_features, dropout_p);
    model->to(torch::kCUDA, torch::kFloat16);

    // Generate input tensor on CUDA
    auto input = torch::randn({batch_size, in_features}, 
                 torch::dtype(torch::kFloat16).device(torch::kCUDA));

    // Run reference implementation
    auto reference_output = model->forward(input);

    // Prepare output tensor for CUDA kernel
    auto output = torch::empty_like(reference_output);

    // Get model parameters
    auto weight = model->matmul->weight;
    auto bias = model->matmul->bias;

    // Call CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        dropout_p
    );

    // Verify results with fp16 tolerance
    bool passed = torch::allclose(reference_output, output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
