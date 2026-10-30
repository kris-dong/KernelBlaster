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
    torch::nn::Conv3d conv{nullptr};
    float divisor;
    torch::nn::MaxPool3d max_pool{nullptr};
    torch::nn::AdaptiveAvgPool3d global_avg_pool{nullptr};
    torch::Tensor bias;
    int sum_dim;

    Model(int in_channels, int out_channels, 
          std::vector<int64_t> kernel_size, 
          float divisor,
          std::vector<int64_t> pool_size,
          std::vector<int64_t> bias_shape,
          int sum_dim) {
        conv = register_module("conv", torch::nn::Conv3d(
            torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)));
        this->divisor = divisor;
        max_pool = register_module("max_pool", torch::nn::MaxPool3d(pool_size));
        global_avg_pool = register_module("global_avg_pool", 
            torch::nn::AdaptiveAvgPool3d(std::vector<int64_t>{1, 1, 1}));
        bias = register_parameter("bias", 
            torch::randn(bias_shape, torch::dtype(torch::kHalf).device(torch::kCUDA)));
        this->sum_dim = sum_dim;
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv->forward(x);
        x = x.div(divisor);
        x = max_pool->forward(x);
        x = global_avg_pool->forward(x);
        x = x + bias;
        x = torch::sum(x, sum_dim);
        return x;
    }
};

void launch_gpu_implementation(
    void* output, void* input,
    void* conv_weight, void* conv_bias, void* model_bias,
    float divisor,
    int kernel_d, int kernel_h, int kernel_w,
    int pool_d, int pool_h, int pool_w,
    int sum_dim,
    int in_channels, int out_channels,
    int batch_size, int depth, int height, int width
);

int main() {
    // Configuration parameters
    int batch_size = 128;
    int in_channels = 3;
    int out_channels = 16;
    int depth = 16, height = 32, width = 32;
    std::vector<int64_t> kernel_size = {3, 3, 3};
    float divisor = 2.0f;
    std::vector<int64_t> pool_size = {2, 2, 2};
    std::vector<int64_t> bias_shape = {out_channels, 1, 1, 1};
    int sum_dim = 1;

    // Create model and move to CUDA
    auto model = std::make_shared<Model>(
        in_channels, out_channels, kernel_size, divisor,
        pool_size, bias_shape, sum_dim
    );
    model->to(torch::kCUDA, torch::kHalf);

    // Generate input tensor
    auto input = torch::randn({batch_size, in_channels, depth, height, width}, 
                            torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Run reference implementation
    auto reference_output = model->forward(input.clone());

    // Prepare output tensor
    auto output = torch::empty_like(reference_output);

    // Get raw pointers for model parameters
    void* conv_weight_ptr = model->conv->weight.data_ptr();
    void* conv_bias_ptr = model->conv->bias.data_ptr();
    void* model_bias_ptr = model->bias.data_ptr();

    // Launch CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        conv_weight_ptr, conv_bias_ptr, model_bias_ptr,
        divisor,
        kernel_size[0], kernel_size[1], kernel_size[2],
        pool_size[0], pool_size[1], pool_size[2],
        sum_dim,
        in_channels, out_channels,
        batch_size, depth, height, width
    );

    // Verify results
    bool is_allclose = torch::allclose(
        reference_output, output,
        /*rtol=*/1e-1, /*atol=*/1e-1
    );
    std::cout << (is_allclose ? "passed" : "failed") << std::endl;

    return 0;
}
