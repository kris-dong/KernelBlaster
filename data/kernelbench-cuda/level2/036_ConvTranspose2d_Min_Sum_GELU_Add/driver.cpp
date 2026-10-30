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

void launch_gpu_implementation(
    void* output, void* input,
    void* conv_weight, void* conv_bias, void* model_bias,
    int in_channels, int out_channels,
    int kernel_size, int stride, int padding, int output_padding,
    int bias_c, int bias_h, int bias_w
);

int main() {
    // Initialize parameters from Python spec
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int height = 32, width = 32;
    const int kernel_size = 3;
    const int stride = 2;
    const int padding = 1;
    const int output_padding = 1;
    const std::vector<int64_t> bias_shape = {out_channels, 1, 1};

    // Create input tensor on GPU with fp16
    auto input = torch::randn({batch_size, in_channels, height, width}, 
                             torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Create reference model components
    auto conv_transpose = torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .output_padding(output_padding)
    );
    conv_transpose->to(torch::kCUDA);
    conv_transpose->weight.set_data(conv_transpose->weight.data().to(torch::kHalf));
    conv_transpose->bias.set_data(conv_transpose->bias.data().to(torch::kHalf));
    
    auto model_bias = torch::randn(bias_shape, 
                                  torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Run reference implementation
    auto x = conv_transpose->forward(input);
    auto min_result = torch::min(x, 1, /*keepdim=*/true);  // Returns tuple<values, indices>
    x = std::get<0>(min_result);  // Extract values tensor
    x = torch::sum(x, 2, /*keepdim=*/true);
    x = torch::gelu(x);
    auto output_ref = x + model_bias;

    // Allocate output tensor for CUDA kernel
    auto output = torch::empty_like(output_ref);

    // Get raw pointers for GPU implementation
    launch_gpu_implementation(
        output.data_ptr(), input.data_ptr(),
        conv_transpose->weight.data_ptr(), conv_transpose->bias.data_ptr(), model_bias.data_ptr(),
        in_channels, out_channels,
        kernel_size, stride, padding, output_padding,
        bias_shape[0], bias_shape[1], bias_shape[2]
    );

    // Verify results
    bool is_close = torch::allclose(output_ref, output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (is_close ? "passed" : "failed") << std::endl;

    return 0;
}
