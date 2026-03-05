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

struct ModelImpl : torch::nn::Module {
    torch::nn::Linear linear{nullptr};

    ModelImpl(int input_size, int hidden_size) {
        linear = register_module("linear", torch::nn::Linear(input_size, hidden_size));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::sigmoid(linear->forward(x));
        x = torch::sum(x, 1, /*keepdim=*/true);
        return x;
    }
};
TORCH_MODULE(Model);

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, int batch_size, int input_size, int hidden_size);

int main() {
    const int batch_size = 128;
    const int input_size = 10;
    const int hidden_size = 20;

    // Create and configure model
    auto model = Model(input_size, hidden_size);
    model->to(torch::Device(torch::kCUDA), torch::kHalf);

    // Create input tensor on CUDA with FP16
    auto input_tensor = torch::randn({batch_size, input_size}, 
                                   torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Run reference implementation
    auto reference_output = model->forward(input_tensor);

    // Get model weights and biases
    auto weight_tensor = model->linear->weight;
    auto bias_tensor = model->linear->bias;

    // Prepare output tensor for CUDA kernel
    auto cuda_output = torch::empty_like(reference_output);

    // Execute CUDA implementation
    launch_gpu_implementation(
        cuda_output.data_ptr(),
        input_tensor.data_ptr(),
        weight_tensor.data_ptr(),
        bias_tensor.data_ptr(),
        batch_size,
        input_size,
        hidden_size
    );

    // Verify results with relaxed tolerances for FP16
    bool passed = torch::allclose(cuda_output, reference_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
