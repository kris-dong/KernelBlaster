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

struct Model {
    torch::nn::ConvTranspose3d conv_transpose{nullptr};
    torch::nn::MaxPool3d maxpool{nullptr};
    torch::nn::AdaptiveAvgPool3d global_avg_pool{nullptr};
    const float scale;
    const float clamp_min = 0.0f;
    const float clamp_max = 1.0f;

    Model(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
          int64_t stride, int64_t padding, float scale_val, int64_t pool_ks)
        : conv_transpose(torch::nn::ConvTranspose3dOptions(in_channels, out_channels, kernel_size)
                            .stride(stride).padding(padding)),
          maxpool(torch::nn::MaxPool3dOptions(pool_ks)),
          global_avg_pool(torch::nn::AdaptiveAvgPool3dOptions({1, 1, 1})),
          scale(scale_val) {
        conv_transpose->to(torch::kCUDA, torch::kHalf);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv_transpose->forward(x);
        x = x * scale;
        x = maxpool->forward(x);
        x = global_avg_pool->forward(x);
        return torch::clamp(x, clamp_min, clamp_max);
    }
};

void launch_gpu_implementation(
    void* output, void* input,
    void* conv_weight, void* conv_bias,
    int in_channels, int out_channels, int kernel_size,
    int stride, int padding, float scale,
    int maxpool_kernel_size, float clamp_min, float clamp_max
);

int main() {
    const int batch_size = 128;
    const int in_channels = 3, out_channels = 16;
    const int depth = 16, height = 32, width = 32;
    const int kernel_size = 3, stride = 2, padding = 1;
    const float scale = 0.5f;
    const int maxpool_ks = 2;

    Model model(in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_ks);
    auto input = torch::randn({batch_size, in_channels, depth, height, width},
                             torch::device(torch::kCUDA).dtype(torch::kHalf));

    auto ref_output = model.forward(input);
    
    // Initialize output with invalid values (outside clamp range)
    auto output = torch::full_like(ref_output, 2.0f);

    auto conv_weight = model.conv_transpose->weight.data_ptr();
    auto conv_bias = model.conv_transpose->bias.data_ptr();

    launch_gpu_implementation(
        output.data_ptr(), input.data_ptr(),
        conv_weight, conv_bias,
        in_channels, out_channels, kernel_size,
        stride, padding, scale,
        maxpool_ks, model.clamp_min, model.clamp_max
    );

    // Verify numerical accuracy
    bool passed = torch::allclose(ref_output, output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
