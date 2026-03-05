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
// Pass pointers to weight and (optional) bias to ensure identical computation.
void launch_gpu_implementation(
    void* output,                  // Output tensor (GPU memory, fp16)
    void* input,                   // Input tensor (GPU memory, fp16)
    void* weight,                  // Weight tensor (GPU memory, fp16)
    void* bias,                    // Bias tensor (GPU memory, fp16), can be nullptr if bias is not used
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
    int dilation,
    int groups
);

int main() {
    // Parameters
    const int batch_size = 16;
    const int in_channels = 32;
    const int out_channels = 16;
    const int kernel_size = 3;
    const int depth = 16;
    const int height = 32;
    const int width = 64;
    const int stride = 1;
    const int padding = 0;
    const int output_padding = 0;
    const int dilation = 1;
    const int groups = 1;
    const bool bias = false;

    // Set default dtype to Half (fp16)
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Input tensor (random, fp16, GPU)
    auto input = torch::randn({batch_size, in_channels, depth, height, width}, torch::TensorOptions().dtype(dtype).device(device));

    // Create ConvTranspose3d module
    torch::nn::ConvTranspose3dOptions conv_options(in_channels, out_channels, {kernel_size, kernel_size, kernel_size});
    conv_options.stride(stride).padding(padding).output_padding(output_padding).dilation(dilation).groups(groups).bias(bias);

    torch::nn::ConvTranspose3d conv_transpose3d(conv_options);
    conv_transpose3d->to(device, dtype);

    // Run reference implementation (libtorch)
    auto ref_output = conv_transpose3d->forward(input);

    // Get weight and bias pointers (ensure on GPU and fp16)
    auto weight = conv_transpose3d->weight.detach();
    TORCH_CHECK(weight.device().is_cuda(), "Weight must be on CUDA");
    TORCH_CHECK(weight.dtype() == dtype, "Weight must be fp16");
    void* weight_ptr = weight.data_ptr();

    // For bias
    void* bias_ptr = nullptr;
    if (bias) {
        auto bias_tensor = conv_transpose3d->bias.detach();
        TORCH_CHECK(bias_tensor.device().is_cuda(), "Bias must be on CUDA");
        TORCH_CHECK(bias_tensor.dtype() == dtype, "Bias must be fp16");
        bias_ptr = bias_tensor.data_ptr();
    }

    // Allocate output tensor for custom CUDA kernel (same shape/dtype/device as ref_output)
    auto output = torch::empty_like(ref_output);

    // Launch custom CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        weight_ptr,
        bias_ptr,
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
        dilation,
        groups
    );

    // Compare outputs
    double rtol = 1e-1, atol = 1e-1; // fp16 tolerance
    bool passed = torch::allclose(output, ref_output, rtol, atol);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
