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
    torch::nn::GroupNorm group_norm{nullptr};
    torch::Tensor scale;
    torch::nn::MaxPool2d maxpool{nullptr};
    float clamp_min, clamp_max;

    Model(int in_channels, int out_channels, int kernel_size, int num_groups,
          std::vector<int64_t> scale_shape, int maxpool_kernel_size,
          float clamp_min, float clamp_max)
        : clamp_min(clamp_min), clamp_max(clamp_max) {
        conv = register_module("conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)));
        group_norm = register_module("group_norm", torch::nn::GroupNorm(
            num_groups, out_channels));
        scale = register_parameter("scale", torch::ones(scale_shape));
        maxpool = register_module("maxpool", torch::nn::MaxPool2d(
            torch::nn::MaxPool2dOptions(maxpool_kernel_size)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv->forward(x);
        x = group_norm->forward(x);
        x = x * scale;
        x = maxpool->forward(x);
        x = torch::clamp(x, clamp_min, clamp_max);
        return x;
    }
};

void launch_gpu_implementation(
    void* output, void* input,
    int in_channels, int out_channels, int kernel_size,
    int num_groups, const std::vector<int64_t>& scale_shape,
    int maxpool_kernel_size, float clamp_min, float clamp_max,
    void* conv_weight, void* conv_bias,
    void* group_norm_weight, void* group_norm_bias,
    void* scale
);

int main() {
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int height = 32, width = 32;
    const int kernel_size = 3;
    const int num_groups = 8;
    const std::vector<int64_t> scale_shape = {out_channels, 1, 1};
    const int maxpool_kernel_size = 2;
    const float clamp_min = 0.0f, clamp_max = 1.0f;

    auto model = std::make_shared<Model>(
        in_channels, out_channels, kernel_size,
        num_groups, scale_shape, maxpool_kernel_size,
        clamp_min, clamp_max
    );
    model->to(torch::kCUDA, torch::kFloat16);

    auto input = torch::randn({batch_size, in_channels, height, width}, 
                             torch::dtype(torch::kFloat16).device(torch::kCUDA));
    auto ref_output = model->forward(input);
    auto output_cuda = torch::empty_like(ref_output);

    launch_gpu_implementation(
        output_cuda.data_ptr(),
        input.data_ptr(),
        in_channels,
        out_channels,
        kernel_size,
        num_groups,
        scale_shape,
        maxpool_kernel_size,
        clamp_min,
        clamp_max,
        model->conv->weight.data_ptr(),
        model->conv->bias.data_ptr(),
        model->group_norm->weight.data_ptr(),
        model->group_norm->bias.data_ptr(),
        model->scale.data_ptr()
    );

    bool passed = torch::allclose(ref_output, output_cuda, 1e-1, 1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
