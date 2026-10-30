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
    torch::Tensor bias;

    Model(int64_t in_channels, int64_t out_channels, int64_t kernel_size, 
          int64_t stride, int64_t padding, const std::vector<int64_t>& bias_shape) {
        conv_transpose = register_module("conv_transpose", 
            torch::nn::ConvTranspose3d(torch::nn::ConvTranspose3dOptions(in_channels, out_channels, kernel_size)
                .stride(stride).padding(padding)));
        bias = register_parameter("bias", torch::randn(bias_shape, torch::dtype(torch::kHalf).device(torch::kCUDA)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv_transpose->forward(x);
        x = torch::logsumexp(x, {1}, /*keepdim=*/true);
        x = x * torch::sigmoid(x + 3) / 6;
        x = x - bias;
        x = torch::clamp(x, -1, 1);
        x = std::get<0>(torch::max(x, 1, true));
        return x;
    }
};

void launch_gpu_implementation(void* output, void* input, 
    void* conv_weight, void* conv_bias, void* model_bias,
    int in_channels, int out_channels, 
    int kernel_size, int stride, int padding, 
    const std::vector<int64_t>& bias_shape);

int main() {
    // Parameters from Python code
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int depth = 16, height = 32, width = 32;
    const int kernel_size = 3;
    const int stride = 2;
    const int padding = 1;
    const std::vector<int64_t> bias_shape = {out_channels, 1, 1, 1};

    // Create input tensor
    auto input = torch::randn({batch_size, in_channels, depth, height, width}, 
        torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Initialize model
    Model model(in_channels, out_channels, kernel_size, stride, padding, bias_shape);
    model.to(torch::kCUDA, torch::kHalf);

    // Run reference forward pass
    auto ref_output = model.forward(input);

    // Get parameter pointers
    auto conv_weight = model.conv_transpose->weight;
    auto conv_bias = model.conv_transpose->bias;
    auto model_bias = model.bias;

    // Prepare output tensor for CUDA kernel
    auto output = torch::empty_like(ref_output);

    // Call CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        conv_weight.data_ptr(),
        conv_bias.data_ptr(),
        model_bias.data_ptr(),
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias_shape
    );

    // Verify results
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
