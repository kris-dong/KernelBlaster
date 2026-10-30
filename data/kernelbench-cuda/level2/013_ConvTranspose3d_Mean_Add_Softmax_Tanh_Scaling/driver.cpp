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

struct TorchModel : torch::nn::Module {
    torch::nn::ConvTranspose3d conv_transpose{nullptr};
    torch::Tensor bias;
    float scaling_factor;

    TorchModel(int in_channels, int out_channels, int kernel_size, int stride, int padding, 
              std::vector<int64_t> bias_shape, float scaling_factor) {
        conv_transpose = register_module("conv_transpose", 
            torch::nn::ConvTranspose3d(torch::nn::ConvTranspose3dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)));
        bias = register_parameter("bias", torch::randn(bias_shape));
        this->scaling_factor = scaling_factor;
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv_transpose->forward(x);
        x = torch::mean(x, /*dim=*/1, /*keepdim=*/true);
        x = x + bias;
        x = torch::softmax(x, /*dim=*/1);
        x = torch::tanh(x);
        x = x * scaling_factor;
        return x;
    }
};

void launch_gpu_implementation(
    void* output, void* input,
    void* conv_weight, void* conv_bias, void* model_bias,
    int in_channels, int out_channels, int kernel_size, int stride, int padding,
    float scaling_factor,
    int batch_size, int input_depth, int input_height, int input_width
);

int main() {
    const int batch_size = 16;
    const int in_channels = 8;
    const int out_channels = 16;
    const int depth = 16, height = 32, width = 32;
    const int kernel_size = 3;
    const int stride = 2;
    const int padding = 1;
    const std::vector<int64_t> bias_shape = {1, 1, 1, 1, 1};
    const float scaling_factor = 2.0f;

    // Create input tensor on CUDA in fp16
    auto input = torch::randn({batch_size, in_channels, depth, height, width}, 
                            torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Create and configure model
    TorchModel model(in_channels, out_channels, kernel_size, stride, padding, bias_shape, scaling_factor);
    model.to(torch::kCUDA, torch::kHalf);

    // Reference forward pass
    auto ref_output = model.forward(input.clone());

    // Get output dimensions for CUDA allocation
    auto output_shape = ref_output.sizes();
    torch::Tensor cuda_output = torch::empty(output_shape, 
                                           torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Get parameter pointers
    auto conv_weight = model.conv_transpose->weight;
    auto conv_bias = model.conv_transpose->bias;
    auto model_bias = model.bias;

    // Launch CUDA implementation
    launch_gpu_implementation(
        cuda_output.data_ptr(),
        input.data_ptr(),
        conv_weight.data_ptr(),
        conv_bias.data_ptr(),
        model_bias.data_ptr(),
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        scaling_factor,
        batch_size,
        depth,
        height,
        width
    );

    // Verify results
    bool passed = torch::allclose(cuda_output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
