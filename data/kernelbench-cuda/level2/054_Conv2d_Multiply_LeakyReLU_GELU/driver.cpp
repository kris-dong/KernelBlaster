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

void launch_gpu_implementation(void* output, void* input, void* conv_weight, void* conv_bias, void* multiplier, 
                              int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size);

int main() {
    // Setup parameters and device
    torch::ScalarType dtype = torch::kHalf;
    torch::Device device(torch::kCUDA);
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int height = 32, width = 32;
    const int kernel_size = 3;
    const std::vector<int64_t> multiplier_shape = {out_channels, 1, 1};

    // Create input tensor
    auto input = torch::randn({batch_size, in_channels, height, width}, 
                            torch::dtype(dtype).device(device));

    // Initialize model components
    auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).bias(true));
    conv->to(device, dtype);
    
    auto multiplier = torch::randn(multiplier_shape, 
                                 torch::dtype(dtype).device(device));

    // Reference implementation
    auto ref_output = conv->forward(input);
    ref_output = ref_output * multiplier;
    ref_output = torch::leaky_relu(ref_output, 0.01);
    ref_output = torch::gelu(ref_output);

    // Prepare output tensor for CUDA implementation
    auto output = torch::empty_like(ref_output);

    // Get raw pointers for CUDA launch
    launch_gpu_implementation(
        output.data_ptr(), 
        input.data_ptr(),
        conv->weight.data_ptr(),
        conv->bias.data_ptr(),
        multiplier.data_ptr(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size
    );

    // Verify results with fp16 tolerance
    bool is_close = torch::allclose(ref_output, output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (is_close ? "passed" : "failed") << std::endl;

    return 0;
}
