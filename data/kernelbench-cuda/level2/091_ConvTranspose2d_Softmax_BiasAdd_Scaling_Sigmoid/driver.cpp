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
#include "cuda_model.cuh"
#include <torch/torch.h>

struct Model : torch::nn::Module {
    torch::nn::ConvTranspose2d conv_transpose{nullptr};
    torch::Tensor model_bias;
    float scaling_factor;

    Model(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, 
          int64_t padding, int64_t output_padding, std::vector<int64_t> bias_shape, float scaling_factor)
        : scaling_factor(scaling_factor) {
        conv_transpose = register_module("conv_transpose", torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .output_padding(output_padding)
        ));
        model_bias = register_parameter("bias", torch::randn(bias_shape, torch::dtype(torch::kFloat16)));
        conv_transpose->weight.set_data(conv_transpose->weight.data().to(torch::kFloat16));
        conv_transpose->bias.set_data(conv_transpose->bias.data().to(torch::kFloat16));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv_transpose->forward(x);
        x = torch::softmax(x, 1);
        x = x + model_bias;
        x = x * scaling_factor;
        x = torch::sigmoid(x);
        return x;
    }
};

void launch_gpu_implementation(void* output, void* input, void* conv_weight, void* conv_bias,
                               void* model_bias, float scaling_factor, int64_t in_channels,
                               int64_t out_channels, int64_t kernel_size, int64_t stride,
                               int64_t padding, int64_t output_padding, int64_t batch_size,
                               int64_t input_height, int64_t input_width);

int main() {
    int64_t batch_size = 128;
    int64_t in_channels = 32;
    int64_t out_channels = 64;
    int64_t height = 16, width = 16;
    int64_t kernel_size = 4;
    int64_t stride = 2;
    int64_t padding = 1;
    int64_t output_padding = 1;
    std::vector<int64_t> bias_shape = {out_channels, 1, 1};
    float scaling_factor = 2.0f;

    Model model(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor);
    model.to(torch::kCUDA);

    auto input = torch::randn({batch_size, in_channels, height, width}, 
                              torch::dtype(torch::kFloat16).device(torch::kCUDA));
    auto ref_output = model.forward(input);

    auto output = torch::zeros_like(ref_output);

    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        model.conv_transpose->weight.data_ptr(),
        model.conv_transpose->bias.data_ptr(),
        model.model_bias.data_ptr(),
        scaling_factor,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        batch_size,
        height,
        width
    );

    bool is_close = torch::allclose(ref_output, output, 1e-1, 1e-1);
    std::cout << (is_close ? "passed" : "failed") << std::endl;

    return 0;
}
