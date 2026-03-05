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
                              int64_t in_channels, int64_t out_channels,
                              int64_t kernel_size, float subtract_value,
                              int64_t pool_kernel_size,
                              int64_t batch_size, int64_t height, int64_t width);

int main() {
    // Initialize parameters
    const int64_t batch_size = 128;
    const int64_t in_channels = 3;
    const int64_t out_channels = 16;
    const int64_t height = 32, width = 32;
    const int64_t kernel_size = 3;
    const float subtract_value = 0.5f;
    const int64_t pool_kernel_size = 2;

    // Create model and move to CUDA
    torch::nn::Conv2d conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).stride(1));
    conv->to(torch::kCUDA, torch::kHalf);

    // Create input tensor on GPU with FP16
    auto input = torch::randn({batch_size, in_channels, height, width}, 
                             torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Run reference implementation
    auto ref_output = conv->forward(input);
    ref_output = ref_output - subtract_value;
    ref_output = torch::hardswish(ref_output);
    ref_output = torch::max_pool2d(ref_output, pool_kernel_size);
    ref_output = torch::mish(ref_output);

    // Prepare output tensor
    auto output = torch::empty_like(ref_output);

    // Get model parameters with explicit type casting
    auto conv_weight = static_cast<void*>(conv->weight.data_ptr<c10::Half>());
    auto conv_bias = static_cast<void*>(conv->bias.data_ptr<c10::Half>());
    auto input_ptr = static_cast<void*>(input.data_ptr<c10::Half>());
    auto output_ptr = static_cast<void*>(output.data_ptr<c10::Half>());

    // Call GPU implementation
    launch_gpu_implementation(
        output_ptr,
        input_ptr,
        conv_weight,
        conv_bias,
        in_channels,
        out_channels,
        kernel_size,
        subtract_value,
        pool_kernel_size,
        batch_size,
        height,
        width
    );

    // Verify results with FP16 tolerance
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
