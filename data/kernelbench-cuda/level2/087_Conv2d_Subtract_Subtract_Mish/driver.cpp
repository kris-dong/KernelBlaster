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
    torch::nn::Conv2d conv{nullptr};
    torch::Tensor subtract_1, subtract_2;

    Model(int in_channels, int out_channels, int kernel_size, float sub1, float sub2) {
        conv = register_module("conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).padding(kernel_size / 2)));
        subtract_1 = register_buffer("subtract_1", torch::tensor(sub1, torch::dtype(torch::kHalf)));
        subtract_2 = register_buffer("subtract_2", torch::tensor(sub2, torch::dtype(torch::kHalf)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv(x);
        x = x - subtract_1;
        x = x - subtract_2;
        x = torch::mish(x);
        return x;
    }
};

void launch_gpu_implementation(
    void* output,
    void* input,
    void* conv_weight,
    void* conv_bias,
    void* subtract_1,
    void* subtract_2,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size
);

int main() {
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int height = 32, width = 32;
    const int kernel_size = 3;
    const float subtract_value_1 = 0.5f;
    const float subtract_value_2 = 0.2f;

    auto opts = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);
    
    // Create input tensor
    torch::Tensor input_tensor = torch::randn({batch_size, in_channels, height, width}, opts);
    
    // Create and configure model
    Model model(in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2);
    model.to(torch::kCUDA, torch::kHalf);
    
    // Get reference output
    torch::Tensor reference_output = model.forward(input_tensor);
    
    // Prepare GPU output tensor
    torch::Tensor output_gpu = torch::zeros_like(reference_output);
    
    // Get parameter pointers
    auto conv_weight = model.conv->weight.data_ptr();
    auto conv_bias = model.conv->bias.data_ptr();
    auto sub1_ptr = model.subtract_1.data_ptr();
    auto sub2_ptr = model.subtract_2.data_ptr();
    
    // Launch custom implementation
    launch_gpu_implementation(
        output_gpu.data_ptr(),
        input_tensor.data_ptr(),
        conv_weight,
        conv_bias,
        sub1_ptr,
        sub2_ptr,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size
    );
    
    // Verify results
    bool passed = torch::allclose(output_gpu, reference_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
