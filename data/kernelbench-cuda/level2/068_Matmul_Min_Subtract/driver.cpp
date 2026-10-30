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

// Declaration must come before main()
void launch_gpu_implementation(
    void* output,
    void* input,
    void* weight,
    void* bias,
    void* constant,
    int batch_size,
    int in_features,
    int out_features
);

struct Model : torch::nn::Module {
    torch::nn::Linear linear{nullptr};
    torch::Tensor constant;

    Model(int in_features, int out_features, float constant_val) {
        linear = register_module("linear", torch::nn::Linear(in_features, out_features));
        constant = register_parameter("constant", torch::tensor(constant_val, torch::kHalf));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = linear->forward(x);
        x = torch::min(x, constant);
        x = x - constant;
        return x;
    }
};

int main() {
    // Setup parameters
    const int batch_size = 128;
    const int in_features = 10;
    const int out_features = 5;
    const float constant = 2.0f;
    
    // Create model and move to GPU
    Model model(in_features, out_features, constant);
    model.to(torch::kCUDA, torch::kHalf);
    
    // Create input tensor on GPU
    torch::Tensor input = torch::randn({batch_size, in_features}, 
                                     torch::dtype(torch::kHalf).device(torch::kCUDA));
    
    // Get reference output
    torch::Tensor ref_output = model.forward(input);
    
    // Allocate output tensor for GPU implementation
    torch::Tensor gpu_output = torch::empty_like(ref_output);
    
    // Get parameter pointers
    auto weight_ptr = model.linear->weight.data_ptr<torch::Half>();
    auto bias_ptr = model.linear->bias.data_ptr<torch::Half>();
    auto constant_ptr = model.constant.data_ptr<torch::Half>();
    
    // Launch GPU implementation
    launch_gpu_implementation(
        gpu_output.data_ptr(),
        input.data_ptr(),
        weight_ptr,
        bias_ptr,
        constant_ptr,
        batch_size,
        in_features,
        out_features
    );
    
    // Verify results
    bool passed = torch::allclose(gpu_output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;
    
    return 0;
}
