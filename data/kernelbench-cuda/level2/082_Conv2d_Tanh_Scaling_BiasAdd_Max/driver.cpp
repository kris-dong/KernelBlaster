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
    float scaling_factor, int pool_kernel_size,
    int in_channels, int out_channels, int kernel_size,
    int batch_size, int height, int width
);

int main() {
    // Initialize model parameters
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int height = 32, width = 32;
    const int kernel_size = 3;
    const float scaling_factor = 2.0f;
    const std::vector<int64_t> bias_shape{out_channels, 1, 1};
    const int pool_kernel_size = 2;

    // Create input tensor on GPU
    auto input = torch::randn({batch_size, in_channels, height, width}, 
                            torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Initialize reference model components
    auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).bias(true));
    conv->to(torch::kCUDA, torch::kHalf);
    auto model_bias = torch::randn(bias_shape, torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Run reference implementation
    auto ref_output = conv->forward(input);
    ref_output = torch::tanh(ref_output);
    ref_output = ref_output * scaling_factor;
    ref_output = ref_output + model_bias;
    ref_output = torch::max_pool2d(ref_output, pool_kernel_size);

    // Prepare output tensor for CUDA kernel
    auto gpu_output = torch::empty_like(ref_output);

    // Get raw pointers for kernel launch
    void* input_ptr = input.data_ptr();
    void* output_ptr = gpu_output.data_ptr();
    void* conv_weight_ptr = conv->weight.data_ptr();
    void* conv_bias_ptr = conv->bias.data_ptr();
    void* model_bias_ptr = model_bias.data_ptr();

    // Launch CUDA implementation
    launch_gpu_implementation(
        output_ptr, input_ptr,
        conv_weight_ptr, conv_bias_ptr, model_bias_ptr,
        scaling_factor, pool_kernel_size,
        in_channels, out_channels, kernel_size,
        batch_size, height, width
    );

    // Verify results
    bool passed = torch::allclose(ref_output, gpu_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
