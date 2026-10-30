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
    torch::nn::ConvTranspose3d conv_transpose{nullptr};
    
    Model(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
          int64_t stride, int64_t padding, int64_t output_padding, bool bias)
        : conv_transpose(torch::nn::ConvTranspose3dOptions(in_channels, out_channels, kernel_size)
                             .stride(stride)
                             .padding(padding)
                             .output_padding(output_padding)
                             .bias(bias)) {
        register_module("conv_transpose", conv_transpose);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv_transpose(x);
        x = torch::softmax(x, 1);
        x = torch::sigmoid(x);
        return x;
    }
};

void launch_gpu_implementation(void* output, void* input, 
                               int in_channels, int out_channels,
                               int kernel_size, int stride,
                               int padding, int output_padding,
                               void* weight, void* bias);

int main() {
    // Initialize parameters from Python equivalent
    const int batch_size = 16;
    const int in_channels = 32;
    const int out_channels = 64;
    const int D = 16, H = 32, W = 32;
    const int kernel_size = 3;
    const int stride = 2;
    const int padding = 1;
    const int output_padding = 1;
    const bool bias = true;

    // Create input tensor on GPU with float16
    auto input = torch::randn({batch_size, in_channels, D, H, W}, 
                             torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Create and configure model
    Model model(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias);
    model.to(torch::kCUDA, torch::kHalf);

    // Run reference implementation
    auto ref_output = model.forward(input);

    // Prepare CUDA output tensor
    auto cuda_output = torch::empty_like(ref_output);

    // Get model parameters
    auto weight = model.conv_transpose->weight;
    auto bias_tensor = model.conv_transpose->bias;

    // Launch CUDA kernel implementation
    launch_gpu_implementation(
        cuda_output.data_ptr(),
        input.data_ptr(),
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        weight.data_ptr(),
        bias_tensor.defined() ? bias_tensor.data_ptr() : nullptr
    );

    // Verify results with relaxed tolerances for fp16
    bool passed = torch::allclose(ref_output, cuda_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
