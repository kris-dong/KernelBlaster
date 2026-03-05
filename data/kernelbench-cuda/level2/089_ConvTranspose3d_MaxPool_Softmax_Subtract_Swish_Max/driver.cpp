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
    torch::nn::MaxPool3d max_pool{nullptr};
    torch::Tensor subtract;

    Model(int in_channels, int out_channels, int kernel_size, int stride, int padding, int output_padding,
          int pool_kernel_size, int pool_stride, int pool_padding) {
        conv_transpose = register_module("conv_transpose", torch::nn::ConvTranspose3d(
            torch::nn::ConvTranspose3dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .output_padding(output_padding)));
        
        max_pool = register_module("max_pool", torch::nn::MaxPool3d(
            torch::nn::MaxPool3dOptions({pool_kernel_size, pool_kernel_size, pool_kernel_size})
                .stride({pool_stride, pool_stride, pool_stride})
                .padding({pool_padding, pool_padding, pool_padding})));
        
        subtract = register_parameter("subtract", torch::randn({out_channels}, torch::dtype(torch::kHalf)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv_transpose->forward(x);
        x = max_pool->forward(x);
        x = torch::softmax(x, 1);
        x = x - subtract.view({1, -1, 1, 1, 1});
        x = x * torch::sigmoid(x);
        x = std::get<0>(x.max(1));
        return x;
    }
};

void launch_gpu_implementation(
    void* output, void* input,
    void* conv_weight, void* conv_bias, void* subtract_param,
    int in_channels, int out_channels, int kernel_size, int stride, int padding, int output_padding,
    int pool_kernel_size, int pool_stride, int pool_padding
);

int main() {
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int depth = 16, height = 32, width = 32;
    const int kernel_size = 3;
    const int stride = 2;
    const int padding = 1;
    const int output_padding = 1;
    const int pool_kernel_size = 2;
    const int pool_stride = 2;
    const int pool_padding = 0;

    auto input = torch::randn({batch_size, in_channels, depth, height, width}, 
                            torch::dtype(torch::kHalf).device(torch::kCUDA));

    Model model(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                pool_kernel_size, pool_stride, pool_padding);
    
    // Convert both device and dtype first
    model.to(torch::kCUDA);
    model.to(torch::kHalf);  // Convert all parameters to half precision

    auto reference_output = model.forward(input);

    auto conv_weight = model.conv_transpose->weight;
    auto conv_bias = model.conv_transpose->bias;
    auto subtract_param = model.subtract;

    auto output = torch::empty_like(reference_output);

    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        conv_weight.data_ptr(),
        conv_bias.data_ptr(),
        subtract_param.data_ptr(),
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        pool_kernel_size,
        pool_stride,
        pool_padding
    );

    bool passed = torch::allclose(output, reference_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
