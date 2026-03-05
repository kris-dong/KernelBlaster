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
    torch::Tensor multiplier;
    torch::nn::InstanceNorm3d instance_norm{nullptr};
    float clamp_min, clamp_max;

    Model(int64_t in_channels, int64_t out_channels, int64_t kernel_size, 
          std::vector<int64_t> multiplier_shape, float clamp_min_, float clamp_max_)
        : clamp_min(clamp_min_), clamp_max(clamp_max_) {
        conv = register_module("conv", torch::nn::Conv3d(
            torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)));
        multiplier = register_parameter("multiplier", 
            torch::randn(multiplier_shape, torch::dtype(torch::kFloat16)));
        instance_norm = register_module("instance_norm", 
            torch::nn::InstanceNorm3d(out_channels));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv->forward(x);
        x = x * multiplier;
        x = instance_norm->forward(x);
        x = torch::clamp(x, clamp_min, clamp_max);
        x = x * multiplier;
        x = std::get<0>(torch::max(x, 1));
        return x;
    }
};

void launch_gpu_implementation(void* output, void* input, 
                               void* conv_weight, void* conv_bias, void* multiplier,
                               float clamp_min, float clamp_max);

int main() {
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int depth = 16, height = 32, width = 32;
    const int kernel_size = 3;
    const std::vector<int64_t> multiplier_shape = {out_channels, 1, 1, 1};
    const float clamp_min = -1.0f, clamp_max = 1.0f;

    auto model = std::make_shared<Model>(in_channels, out_channels, kernel_size,
                                        multiplier_shape, clamp_min, clamp_max);
    model->to(torch::kHalf);
    model->to(torch::kCUDA);

    auto input = torch::randn({batch_size, in_channels, depth, height, width},
                             torch::dtype(torch::kFloat16).device(torch::kCUDA));

    auto ref_output = model->forward(input);
    auto output = torch::empty_like(ref_output);

    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        model->conv->weight.data_ptr(),
        model->conv->bias.data_ptr(),
        model->multiplier.data_ptr(),
        clamp_min,
        clamp_max
    );

    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
