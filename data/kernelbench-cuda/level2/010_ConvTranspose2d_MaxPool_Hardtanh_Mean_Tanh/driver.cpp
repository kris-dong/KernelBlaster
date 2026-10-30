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

void launch_gpu_implementation(void* output, void* input, 
                              void* conv_weight, void* conv_bias,
                              int in_channels, int out_channels,
                              int kernel_size, int stride, int padding,
                              int maxpool_kernel_size, int maxpool_stride,
                              float hardtanh_min, float hardtanh_max,
                              int batch_size, int input_height, int input_width);

int main() {
    // Setup parameters from Python implementation
    const int batch_size = 128;
    const int in_channels = 32;
    const int out_channels = 64;
    const int height = 16, width = 16;
    const int kernel_size = 4;
    const int stride = 2;
    const int padding = 1;
    const int maxpool_kernel_size = 2;
    const int maxpool_stride = 2;
    const float hardtanh_min = -1.0f;
    const float hardtanh_max = 1.0f;

    // Create model and move to CUDA with FP16
    auto conv_transpose = torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding));
    conv_transpose->to(torch::kCUDA, torch::kHalf);

    // Create input tensor on CUDA with FP16
    auto input = torch::randn({batch_size, in_channels, height, width}, 
                             torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Reference implementation
    auto x = conv_transpose->forward(input);
    x = torch::max_pool2d(x, {maxpool_kernel_size, maxpool_kernel_size}, {maxpool_stride, maxpool_stride});
    x = torch::hardtanh(x, hardtanh_min, hardtanh_max);
    x = torch::mean(x, {2, 3}, /*keepdim=*/true);
    auto output_ref = torch::tanh(x);

    // Get parameter pointers
    auto conv_weight_ptr = conv_transpose->weight.data_ptr();
    auto conv_bias_ptr = conv_transpose->bias.data_ptr();

    // Allocate CUDA output tensor
    auto output_cuda = torch::empty_like(output_ref);

    // Call GPU implementation
    launch_gpu_implementation(
        output_cuda.data_ptr(),
        input.data_ptr(),
        conv_weight_ptr,
        conv_bias_ptr,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        maxpool_kernel_size,
        maxpool_stride,
        hardtanh_min,
        hardtanh_max,
        batch_size,
        height,
        width
    );

    // Verify results with FP16 tolerance
    if (torch::allclose(output_ref, output_cuda, /*rtol=*/1e-1, /*atol=*/1e-1)) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
