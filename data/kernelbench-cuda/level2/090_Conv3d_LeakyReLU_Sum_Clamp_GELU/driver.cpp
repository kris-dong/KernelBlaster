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

// Declaration must come before usage
void launch_gpu_implementation(
    void* output,
    void* input,
    void* conv_weight,
    void* conv_bias,
    void* sum_tensor
);

struct CppModel : torch::nn::Module {
    torch::nn::Conv3d conv;
    torch::Tensor sum_tensor;

    CppModel(int in_channels, int out_channels, int kernel_size, std::vector<int64_t> sum_shape)
        : conv(register_module("conv", torch::nn::Conv3d(
            torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
        ))) {
        sum_tensor = register_parameter("sum_tensor", 
            torch::randn(sum_shape, torch::dtype(torch::kHalf).device(torch::kCUDA)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv->forward(x);
        x = torch::leaky_relu(x, 0.2);
        x = x + sum_tensor;
        x = torch::clamp(x, -1.0, 1.0);
        x = torch::gelu(x);
        return x;
    }
};

int main() {
    // Configuration constants
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int depth = 16, height = 32, width = 32;
    const int kernel_size = 3;
    const std::vector<int64_t> sum_shape = {out_channels, 1, 1, 1};
    
    // Create model and move to CUDA
    CppModel model(in_channels, out_channels, kernel_size, sum_shape);
    model.to(torch::kHalf);
    model.to(torch::kCUDA);
    
    // Generate input tensor
    torch::Tensor input = torch::randn({batch_size, in_channels, depth, height, width}, 
        torch::dtype(torch::kHalf).device(torch::kCUDA));
    
    // Run reference implementation
    torch::Tensor ref_output = model.forward(input.clone());
    
    // Prepare CUDA implementation inputs/output
    torch::Tensor cuda_output = torch::empty_like(ref_output);
    
    // Get parameter pointers
    auto conv_weight = model.conv->weight.data_ptr();
    auto conv_bias = model.conv->bias.data_ptr();
    auto sum_tensor_ptr = model.sum_tensor.data_ptr();
    
    // Launch CUDA implementation
    launch_gpu_implementation(
        cuda_output.data_ptr(),
        input.data_ptr(),
        conv_weight,
        conv_bias,
        sum_tensor_ptr
    );
    
    // Verify results
    bool passed = torch::allclose(cuda_output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;
    
    return 0;
}
