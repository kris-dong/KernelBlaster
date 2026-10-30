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
    
    Model(int64_t in_features, int64_t out_features) {
        linear = register_module("linear", torch::nn::Linear(in_features, out_features));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x = linear->forward(x);
        x = torch::logsumexp(x, 1, true);
        x = torch::leaky_relu(x, 0.01);
        x = torch::leaky_relu(x, 0.01);
        x = torch::gelu(x);
        x = torch::gelu(x);
        return x;
    }
};

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias);

int main() {
    const int batch_size = 128;
    const int in_features = 1024;
    const int out_features = 512;

    // Setup input tensor on GPU with fp16
    auto input = torch::randn({batch_size, in_features}, 
                torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Create and configure model
    auto model = std::make_shared<Model>(in_features, out_features);
    model->to(torch::kCUDA, torch::kHalf);

    // Run reference implementation
    auto ref_output = model->forward(input);

    // Allocate CUDA output tensor
    auto cuda_output = torch::empty_like(ref_output);

    // Get parameter pointers
    void* weight_ptr = model->linear->weight.data_ptr();
    void* bias_ptr = model->linear->bias.data_ptr();

    // Launch GPU implementation
    launch_gpu_implementation(cuda_output.data_ptr(), 
                            input.data_ptr(), 
                            weight_ptr, 
                            bias_ptr);

    // Verify results with fp16-appropriate tolerances
    if (torch::allclose(ref_output, cuda_output, /*rtol=*/1e-1, /*atol=*/1e-1)) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
