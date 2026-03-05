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
    torch::nn::ConvTranspose3d conv_transpose;
    torch::Tensor multiplier;
    torch::nn::LeakyReLU leaky_relu;
    torch::nn::MaxPool3d max_pool;

    Model(int in_channels, int out_channels, int kernel_size, int stride, 
          int padding, int output_padding, std::vector<int64_t> multiplier_shape)
        : conv_transpose(torch::nn::ConvTranspose3dOptions(in_channels, out_channels, kernel_size)
                         .stride(stride)
                         .padding(padding)
                         .output_padding(output_padding)),
          leaky_relu(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
          max_pool(torch::nn::MaxPool3dOptions(2)) 
    {
        register_parameter("multiplier", multiplier = torch::randn(multiplier_shape));
        register_module("conv_transpose", conv_transpose);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv_transpose(x);
        x = leaky_relu(x);
        x = x * multiplier;
        x = leaky_relu(x);
        x = max_pool(x);
        return x;
    }
};

void launch_gpu_implementation(void* output, void* input, 
                              void* conv_weight, void* conv_bias,
                              void* multiplier,
                              int kernel_size, int stride, 
                              int padding, int output_padding,
                              float negative_slope, int pool_kernel);

int main() {
    // Setup parameters
    const int batch_size = 16;
    const int in_channels = 16, out_channels = 32;
    const int depth = 16, height = 32, width = 32;
    const int kernel_size = 3, stride = 2, padding = 1, output_padding = 1;
    const std::vector<int64_t> multiplier_shape = {out_channels, 1, 1, 1};
    
    // Create model and move to CUDA
    Model model(in_channels, out_channels, kernel_size, stride, padding, 
               output_padding, multiplier_shape);
    model.to(torch::kCUDA, torch::kFloat16);
    
    // Generate input tensor on CUDA
    auto input = torch::randn({batch_size, in_channels, depth, height, width}, 
                             torch::dtype(torch::kFloat16).device(torch::kCUDA));
    
    // Run reference implementation
    auto ref_output = model.forward(input);
    
    // Prepare CUDA implementation inputs
    auto conv_weight = model.conv_transpose->weight;
    auto conv_bias = model.conv_transpose->bias;
    auto multiplier = model.multiplier;
    
    // Create output tensor for CUDA implementation
    auto cuda_output = torch::empty_like(ref_output);
    
    // Launch CUDA implementation
    launch_gpu_implementation(cuda_output.data_ptr(), input.data_ptr(),
                             conv_weight.data_ptr(), conv_bias.data_ptr(),
                             multiplier.data_ptr(),
                             kernel_size, stride, padding, output_padding,
                             0.2f, 2);
    
    // Verify results with fp16-appropriate tolerances
    bool passed = torch::allclose(cuda_output, ref_output, 
                                 /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;
    
    return 0;
}
