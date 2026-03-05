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
#include <iostream>

void launch_gpu_implementation(
    void* output, void* input,
    void* weight, void* bias,
    int batch_size, int in_channels, int out_channels,
    int input_length, int kernel_size, int stride, int padding, int dilation, bool has_bias
);

int main() {
    // Set device and dtype
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model and input parameters (from get_init_inputs and test code)
    int batch_size = 16;
    int in_channels = 32;
    int out_channels = 64;
    int kernel_size = 3;
    int stride = 2;
    int padding = 1;
    int dilation = 2;
    bool has_bias = false;
    int input_length = 128;

    // Create input tensor
    torch::Tensor input = torch::randn({batch_size, in_channels, input_length}, torch::TensorOptions().dtype(dtype).device(device));

    // Create ConvTranspose1d layer with matching parameters
    torch::nn::ConvTranspose1d conv(
        torch::nn::ConvTranspose1dOptions(in_channels, out_channels, kernel_size)
            .stride(stride).padding(padding).dilation(dilation).bias(has_bias)
    );
    conv->to(device, dtype);

    // Get weight and (optional) bias pointers
    torch::Tensor weight = conv->weight.detach().clone().to(device, dtype);
    torch::Tensor bias;
    if (has_bias) {
        bias = conv->bias.detach().clone().to(device, dtype);
    }

    // Reference output using libtorch
    torch::NoGradGuard no_grad;
    conv->weight.set_data(weight);
    if (has_bias) {
        conv->bias.set_data(bias);
    }
    torch::Tensor ref_output = conv->forward(input);

    // Prepare output tensor for CUDA implementation
    torch::Tensor output = torch::empty_like(ref_output);

    // Call the GPU implementation (to be implemented elsewhere)
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        weight.data_ptr(),
        has_bias ? bias.data_ptr() : nullptr,
        batch_size, in_channels, out_channels,
        input_length, kernel_size, stride, padding, dilation, has_bias
    );

    // Compare outputs using torch::allclose (rtol/atol = 1e-1 for fp16)
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
