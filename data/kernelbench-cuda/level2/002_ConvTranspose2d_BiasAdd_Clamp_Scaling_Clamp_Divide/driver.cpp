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
    torch::nn::ConvTranspose2d conv_transpose{nullptr};
    torch::Tensor bias;
    float scaling_factor;

    Model(int64_t in_channels, int64_t out_channels, int64_t kernel_size, 
          int64_t stride, int64_t padding, int64_t output_padding,
          std::vector<int64_t> bias_shape, float scaling_factor_)
        : conv_transpose(torch::nn::ConvTranspose2dOptions(in_channels, out_channels, kernel_size)
                            .stride(stride)
                            .padding(padding)
                            .output_padding(output_padding)
                            .bias(true)),
          scaling_factor(scaling_factor_) {
        register_module("conv_transpose", conv_transpose);
        bias = register_parameter("bias", torch::randn(bias_shape, torch::dtype(torch::kFloat16).device(torch::kCUDA)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv_transpose->forward(x);
        x = x + bias;
        x = torch::clamp(x, 0.0, 1.0);
        x = x * scaling_factor;
        x = torch::clamp(x, 0.0, 1.0);
        x = x / scaling_factor;
        return x;
    }
};

void launch_gpu_implementation(void* output, void* input, void* conv_weight, 
                              void* conv_bias, void* model_bias, float scaling_factor);

int main() {
    torch::ScalarType dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);
    torch::set_default_dtype(c10::scalarTypeToTypeMeta(dtype));

    // Initialize model parameters
    const int64_t in_channels = 3, out_channels = 16, kernel_size = 3;
    const int64_t stride = 2, padding = 1, output_padding = 1;
    const std::vector<int64_t> bias_shape = {16, 1, 1};
    const float scaling_factor = 2.0f;

    // Create and move model to CUDA
    Model model(in_channels, out_channels, kernel_size, stride, padding, 
               output_padding, bias_shape, scaling_factor);
    model.to(device, dtype);

    // Create input tensor
    auto input = torch::randn({128, in_channels, 32, 32}, 
                             torch::dtype(dtype).device(device));

    // Run reference implementation
    auto reference_output = model.forward(input);

    // Prepare output tensor for CUDA kernel
    auto output = torch::empty_like(reference_output);

    // Get raw pointers for kernel launch
    auto input_ptr = input.data_ptr();
    auto output_ptr = output.data_ptr();
    auto conv_weight_ptr = model.conv_transpose->weight.data_ptr();
    auto conv_bias_ptr = model.conv_transpose->bias.data_ptr();
    auto model_bias_ptr = model.bias.data_ptr();

    // Launch CUDA kernel
    launch_gpu_implementation(output_ptr, input_ptr, conv_weight_ptr, 
                             conv_bias_ptr, model_bias_ptr, scaling_factor);

    // Verify results with relaxed tolerances for fp16
    bool passed = torch::allclose(reference_output, output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
