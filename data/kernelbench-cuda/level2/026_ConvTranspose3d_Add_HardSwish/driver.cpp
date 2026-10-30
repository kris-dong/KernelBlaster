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
    void* output,
    void* input,
    void* add_input,
    const void* conv_weight,
    const void* conv_bias,
    const void* model_bias,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    const std::vector<int64_t>& bias_shape
);

int main() {
    torch::manual_seed(42);
    
    // Initialize parameters from Python's get_init_inputs()
    const int in_channels = 32;
    const int out_channels = 64;
    const int kernel_size = 3;
    const int stride = 2;
    const int padding = 1;
    const int output_padding = 1;
    const std::vector<int64_t> bias_shape{out_channels, 1, 1, 1, 1};
    const int batch_size = 128;

    // Create convtranspose3d layer
    auto conv_transpose = torch::nn::ConvTranspose3d(
        torch::nn::ConvTranspose3dOptions(in_channels, out_channels, kernel_size)
        .stride(stride)
        .padding(padding)
        .output_padding(output_padding)
        .bias(true)
    );
    conv_transpose->to(torch::kCUDA, torch::kHalf);

    // Create model bias parameter
    auto model_bias = torch::randn(bias_shape, torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Create input tensors
    auto input = std::vector<int64_t>{batch_size, in_channels, 16, 16, 16};
    auto add_input_shape = std::vector<int64_t>{batch_size, out_channels, 32, 32, 32};
    auto x = torch::randn(input, torch::dtype(torch::kHalf).device(torch::kCUDA));
    auto add_input = torch::randn(add_input_shape, torch::dtype(torch::kHalf).device(torch::kCUDA));

    // LibTorch reference implementation
    auto conv_out = conv_transpose->forward(x);
    auto added = conv_out + add_input;
    auto output_ref = added * at::hardswish(added);  // Fixed namespace

    // Allocate output tensor for CUDA implementation
    auto output_cuda = torch::empty_like(output_ref);

    // Call CUDA implementation
    launch_gpu_implementation(
        output_cuda.data_ptr(),
        x.data_ptr(),
        add_input.data_ptr(),
        conv_transpose->weight.data_ptr(),
        conv_transpose->bias.data_ptr(),
        model_bias.data_ptr(),
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape
    );

    // Verify results
    if (torch::allclose(output_cuda, output_ref, /*rtol=*/1e-1, /*atol=*/1e-1)) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
