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
#include <vector>
#include "cuda_model.cuh"

void launch_gpu_implementation(
    void* output, void* input,
    void* conv_weight, void* conv_bias, void* model_bias,
    int in_channels, int out_channels,
    int kernel_size, int stride, int padding, int output_padding,
    int64_t bias_dim0, int64_t bias_dim1, int64_t bias_dim2, int64_t bias_dim3
);

int main() {
    // Model configuration
    const int batch_size = 16;
    const int in_channels = 32;
    const int out_channels = 64;
    const int kernel_size = 3;
    const int stride = 2;
    const int padding = 1;
    const int output_padding = 1;
    const std::vector<int64_t> bias_shape = {out_channels, 1, 1, 1};
    
    // Create input tensor on GPU
    auto input = torch::randn({batch_size, in_channels, 16, 32, 32}, 
                             torch::dtype(torch::kFloat16).device(torch::kCUDA));

    // Create reference model components
    auto conv_options = torch::nn::ConvTranspose3dOptions(in_channels, out_channels, kernel_size)
                        .stride(stride)
                        .padding(padding)
                        .output_padding(output_padding)
                        .bias(true);
    
    auto conv_transpose = torch::nn::ConvTranspose3d(conv_options);
    conv_transpose->to(torch::kCUDA, torch::kFloat16);
    
    auto model_bias = torch::randn(bias_shape, 
                                  torch::dtype(torch::kFloat16).device(torch::kCUDA));
    
    // Run reference implementation
    auto x = conv_transpose->forward(input);
    auto original_x = x.clone().detach();
    x = x + model_bias;
    x = x + original_x;
    x = x * original_x;
    auto reference_output = x + original_x;

    // Prepare CUDA kernel input/output
    auto cuda_output = torch::empty_like(reference_output, 
                                       torch::dtype(torch::kFloat16).device(torch::kCUDA));

    // Get raw pointers for kernel launch
    launch_gpu_implementation(
        cuda_output.data_ptr(), input.data_ptr(),
        conv_transpose->weight.data_ptr(), conv_transpose->bias.data_ptr(), model_bias.data_ptr(),
        in_channels, out_channels,
        kernel_size, stride, padding, output_padding,
        bias_shape[0], bias_shape[1], bias_shape[2], bias_shape[3]
    );

    // Verify results
    bool passed = torch::allclose(cuda_output, reference_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
