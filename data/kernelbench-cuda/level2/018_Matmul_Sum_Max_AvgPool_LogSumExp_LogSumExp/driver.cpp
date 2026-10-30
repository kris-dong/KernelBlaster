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

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, int batch_size, int in_features, int out_features);

struct Model : torch::nn::Module {
    torch::nn::Linear linear{nullptr};

    Model(int64_t in_features, int64_t out_features) {
        linear = register_module("linear", torch::nn::Linear(in_features, out_features));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = linear->forward(x);
        x = torch::sum(x, 1, /*keepdim=*/true);
        x = std::get<0>(torch::max(x, 1, /*keepdim=*/true));
        x = torch::mean(x, 1, /*keepdim=*/true);
        x = torch::logsumexp(x, 1, /*keepdim=*/true);
        x = torch::logsumexp(x, 1, /*keepdim=*/true);
        return x;
    }
};

int main() {
    const int batch_size = 128;
    const int in_features = 10;
    const int out_features = 5;

    // Create and configure model
    auto model = std::make_shared<Model>(in_features, out_features);
    model->to(torch::kCUDA, torch::kHalf);

    // Create input tensor on CUDA with fp16
    auto input = torch::randn({batch_size, in_features}, 
                            torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Run reference implementation
    auto reference_output = model->forward(input);

    // Get model parameters
    auto weight = model->linear->weight;
    auto bias = model->linear->bias;

    // Allocate and initialize output tensor with NaNs to detect uninitialized memory
    auto output = torch::full_like(reference_output, std::numeric_limits<float>::quiet_NaN());

    // Call CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        batch_size,
        in_features,
        out_features
    );

    // Verify results with relaxed tolerances for fp16
    bool passed = torch::allclose(output, reference_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
