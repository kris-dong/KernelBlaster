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

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias,
                               int in_channels, int out_channels, int kernel_size,
                               int stride, int padding, float min_value, float divisor);

int main() {
    // Configuration parameters
    const int batch_size = 16;
    const int in_channels = 32;
    const int out_channels = 16;
    const int depth = 16, height = 32, width = 32;
    const int kernel_size = 3;
    const int stride = 2;
    const int padding = 1;
    const float min_value = -1.0f;
    const float divisor = 2.0f;

    // Create input tensor on GPU with FP16
    auto input = torch::randn({batch_size, in_channels, depth, height, width},
                             torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Create reference model components
    auto conv = torch::nn::ConvTranspose3d(
        torch::nn::ConvTranspose3dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
    );
    conv->to(torch::kCUDA, torch::kHalf);

    // Forward pass for reference implementation
    auto ref_output = conv->forward(input);
    ref_output = torch::clamp(ref_output, min_value);
    ref_output = ref_output / divisor;

    // Prepare output tensor for CUDA implementation
    auto output = torch::empty_like(ref_output, torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Get raw pointers for GPU memory
    auto weight_ptr = conv->weight.data_ptr();
    auto bias_ptr = conv->bias.defined() ? conv->bias.data_ptr() : nullptr;

    // Launch custom GPU implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        weight_ptr,
        bias_ptr,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        min_value,
        divisor
    );

    // Verify results with relaxed tolerances for FP16
    if (torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1)) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
