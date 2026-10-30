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
    torch::nn::Linear linear{nullptr};
    float divisor;

    Model(int in_features, int out_features, float divisor_) : divisor(divisor_) {
        linear = register_module("linear", torch::nn::Linear(in_features, out_features));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = linear->forward(x);
        x = torch::relu(x);
        x = x / divisor;
        return x;
    }
};

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, float divisor, int batch_size, int in_features, int out_features);

int main() {
    const int batch_size = 128;
    const int in_features = 1024;
    const int out_features = 512;
    const float divisor = 2.0f;

    // Create and configure model
    auto model = std::make_shared<Model>(in_features, out_features, divisor);
    model->to(torch::kCUDA);
    model->to(torch::kFloat16);

    // Create input tensor
    torch::Tensor input = torch::randn({batch_size, in_features}, 
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));

    // Reference implementation
    torch::Tensor reference_output = model->forward(input);

    // Get model parameters
    torch::Tensor weight = model->linear->weight;
    torch::Tensor bias = model->linear->bias;

    // Allocate CUDA output tensor
    torch::Tensor cuda_output = torch::empty_like(reference_output);

    // Get raw pointers
    void* input_ptr = input.data_ptr();
    void* output_ptr = cuda_output.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* bias_ptr = bias.data_ptr();

    // Launch CUDA implementation
    launch_gpu_implementation(output_ptr, input_ptr, weight_ptr, bias_ptr, 
                             divisor, batch_size, in_features, out_features);

    // Verify results
    bool passed = torch::allclose(reference_output, cuda_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
