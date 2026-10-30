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
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include "cuda_model.cuh"

struct Model : torch::nn::Module {
    torch::nn::ConvTranspose3d conv_transpose{nullptr};
    torch::Tensor scale1, bias, scale2;

    Model(int64_t in_channels, int64_t out_channels, int64_t kernel_size, 
          int64_t stride, int64_t padding, float scale1_val, float scale2_val,
          std::vector<int64_t> bias_shape) {
        conv_transpose = register_module("conv_transpose", torch::nn::ConvTranspose3d(
            torch::nn::ConvTranspose3dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
        ));
        scale1 = register_parameter("scale1", torch::tensor(scale1_val, torch::dtype(torch::kFloat16)));
        scale2 = register_parameter("scale2", torch::tensor(scale2_val, torch::dtype(torch::kFloat16)));
        bias = register_parameter("bias", torch::randn(bias_shape, torch::dtype(torch::kFloat16)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv_transpose->forward(x);
        x = x * scale1;
        x = torch::avg_pool3d(x, /*kernel_size=*/2);
        x = x + bias;
        x = x * scale2;
        return x;
    }
};

void launch_gpu_implementation(
    void* output, void* input,
    const void* conv_weight, const void* conv_bias,
    const void* scale1, const void* model_bias, const void* scale2,
    int in_channels, int out_channels,
    int kernel_size, int stride, int padding,
    int batch_size, int input_depth, int input_height, int input_width
);

int main() {
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int depth = 16, height = 32, width = 32;
    const int kernel_size = 3;
    const int stride = 2;
    const int padding = 1;
    const float scale1_val = 0.5f;
    const float scale2_val = 1.0f;
    const std::vector<int64_t> bias_shape = {out_channels, 1, 1, 1};

    // Create and configure model
    auto model = std::make_shared<Model>(in_channels, out_channels, kernel_size, stride, padding,
                                        scale1_val, scale2_val, bias_shape);
    model->to(torch::kCUDA, torch::kFloat16);

    // Create input tensor on CUDA
    auto input = torch::randn({batch_size, in_channels, depth, height, width}, 
                             torch::dtype(torch::kFloat16).device(torch::kCUDA));

    // Run reference implementation
    auto ref_output = model->forward(input);

    // Get parameter pointers
    auto conv_weight = model->conv_transpose->weight;
    auto conv_bias = model->conv_transpose->bias;
    auto scale1_tensor = model->scale1;
    auto model_bias_tensor = model->bias;
    auto scale2_tensor = model->scale2;

    // Allocate output tensor
    auto output = torch::empty_like(ref_output);

    // Call CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        conv_weight.data_ptr(),
        conv_bias.data_ptr(),
        scale1_tensor.data_ptr(),
        model_bias_tensor.data_ptr(),
        scale2_tensor.data_ptr(),
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        batch_size,
        depth,
        height,
        width
    );

    // Verify results
    bool passed = torch::allclose(ref_output, output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
