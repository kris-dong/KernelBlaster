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
#include <iostream>
#include "cuda_model.cuh"

// Declaration for GPU implementation.
// You may change void* to at::Half* if your CUDA kernel expects it.
void launch_gpu_implementation(
    void* output, 
    void* input, 
    void* weight, 
    void* bias, 
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int depth, 
    int height, 
    int width, 
    int kernel_size, 
    int stride, 
    int padding, 
    int output_padding, 
    int groups, 
    bool has_bias
);

int main() {
    using namespace torch;
    using torch::indexing::Slice;

    // Set default dtype to float16 if supported by your libtorch version
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Model parameters
    int batch_size = 16;
    int in_channels = 3;
    int out_channels = 64;
    int kernel_size = 3;
    int depth = 32;
    int height = 32;
    int width = 32;
    int stride = 1;
    int padding = 0;
    int output_padding = 0;
    int groups = 1;
    bool has_bias = false;

    // Create input tensor (CUDA, fp16)
    auto input = torch::randn(
        {batch_size, in_channels, depth, height, width},
        torch::TensorOptions().dtype(dtype).device(device)
    );

    // Create ConvTranspose3d layer and move to CUDA and fp16
    auto conv = torch::nn::ConvTranspose3d(
        torch::nn::ConvTranspose3dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .output_padding(output_padding)
            .groups(groups)
            .bias(has_bias)
    );
    conv->to(device, dtype);
    conv->eval();

    // Clone weights and bias to ensure correct memory layout and device
    auto weight = conv->weight.detach().clone().to(device, dtype).contiguous();
    torch::Tensor bias;
    if (has_bias) {
        bias = conv->bias.detach().clone().to(device, dtype).contiguous();
    } else {
        bias = torch::zeros({out_channels}, torch::TensorOptions().dtype(dtype).device(device));
    }

    // Compute reference output (no grad, eval mode)
    at::NoGradGuard no_grad;
    auto ref_output = conv->forward(input);

    // Allocate output tensor for CUDA implementation
    auto output = torch::empty_like(ref_output);

    // Call CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(), 
        input.data_ptr(), 
        weight.data_ptr(), 
        bias.data_ptr(),
        batch_size, 
        in_channels, 
        out_channels, 
        depth, 
        height, 
        width, 
        kernel_size, 
        stride, 
        padding, 
        output_padding, 
        groups, 
        has_bias
    );

    // Use rtol/atol=1e-1 for fp16
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }
    return 0;
}
