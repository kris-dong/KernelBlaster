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

void launch_gpu_implementation(void* output, void* input, void* conv_weight, void* conv_bias, void* model_bias,
                               int batch_size, int in_channels, int out_channels, int height, int width,
                               int kernel_size, int output_h, int output_w);

int main() {
    // Initialize parameters from Python spec
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int height = 32, width = 32;
    const int kernel_size = 3;
    const std::vector<int64_t> bias_shape{out_channels, 1, 1};
    const int output_h = height - kernel_size + 1;
    const int output_w = width - kernel_size + 1;

    // Create and configure Conv2d module
    auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size));
    conv->to(torch::kCUDA, torch::kHalf);
    
    // Create input tensor and model parameters
    auto input = torch::randn({batch_size, in_channels, height, width}, 
                             torch::dtype(torch::kHalf).device(torch::kCUDA));
    auto model_bias = torch::randn(bias_shape, 
                                  torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Reference implementation
    auto ref_output = conv->forward(input);
    ref_output = torch::relu(ref_output);
    ref_output = ref_output + model_bias;

    // Allocate GPU memory for CUDA implementation output
    auto cuda_output = torch::empty_like(ref_output);

    // Get raw pointers for kernel launch
    launch_gpu_implementation(
        cuda_output.data_ptr(),         // Output tensor
        input.data_ptr(),               // Input tensor
        conv->weight.data_ptr(),        // Conv weight
        conv->bias.data_ptr(),          // Conv bias
        model_bias.data_ptr(),          // Model bias
        batch_size, in_channels, out_channels,
        height, width, kernel_size,
        output_h, output_w
    );

    // Verify results with fp16 tolerance
    bool passed = torch::allclose(ref_output, cuda_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
