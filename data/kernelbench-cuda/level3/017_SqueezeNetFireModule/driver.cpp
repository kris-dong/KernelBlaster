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
#include <cuda_runtime.h>
#include <iostream>
#include "cuda_model.cuh"

// Declaration only. Do not implement here.
void launch_gpu_implementation(
    void* output,
    const void* input,
    const void* squeeze_weight,
    const void* squeeze_bias,
    const void* expand1x1_weight,
    const void* expand1x1_bias,
    const void* expand3x3_weight,
    const void* expand3x3_bias,
    int64_t batch_size,
    int64_t in_channels,
    int64_t height,
    int64_t width,
    int64_t squeeze_channels,
    int64_t expand1x1_channels,
    int64_t expand3x3_channels,
    // kernel sizes
    int64_t squeeze_kernel_h, int64_t squeeze_kernel_w,
    int64_t expand1x1_kernel_h, int64_t expand1x1_kernel_w,
    int64_t expand3x3_kernel_h, int64_t expand3x3_kernel_w,
    // padding for 3x3 conv
    int64_t expand3x3_padding_h, int64_t expand3x3_padding_w,
    // stride (all convs use stride=1)
    int64_t stride_h, int64_t stride_w,
    // dilation (all convs use dilation=1)
    int64_t dilation_h, int64_t dilation_w
);

int main() {
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available. Exiting.\n";
        return 0;
    }

    // Use fp16 (half) dtype for all tensors as per torch.set_default_dtype(torch.float16)
    auto device = torch::Device(torch::kCUDA);
    auto dtype = torch::kHalf;

    // Test configuration (mirrors the provided Python)
    int64_t batch_size = 10;
    int64_t num_input_features = 3;
    int64_t height = 224, width = 224;
    int64_t squeeze_channels = 6;
    int64_t expand1x1_channels = 64;
    int64_t expand3x3_channels = 64;

    // Create input on GPU with half dtype
    auto input = torch::randn({batch_size, num_input_features, height, width},
                              torch::TensorOptions().device(device).dtype(dtype));

    // Build the model layers on GPU with half dtype
    auto squeeze = torch::nn::Conv2d(torch::nn::Conv2dOptions(num_input_features, squeeze_channels, /*kernel_size=*/1).bias(true));
    auto expand1x1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(squeeze_channels, expand1x1_channels, /*kernel_size=*/1).bias(true));
    auto expand3x3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(squeeze_channels, expand3x3_channels, /*kernel_size=*/3).padding(1).bias(true));
    auto relu_inplace = torch::nn::ReLU(torch::nn::ReLUOptions(true));

    squeeze->to(device, dtype);
    expand1x1->to(device, dtype);
    expand3x3->to(device, dtype);
    relu_inplace->to(device);

    // Reference forward (libtorch) on GPU
    torch::NoGradGuard no_grad;
    auto x = squeeze->forward(input);
    x = relu_inplace->forward(x);
    auto y1 = expand1x1->forward(x);
    y1 = relu_inplace->forward(y1);
    auto y3 = expand3x3->forward(x);
    y3 = relu_inplace->forward(y3);
    auto ref_output = torch::cat({y1, y3}, /*dim=*/1);

    // Allocate output tensor for GPU implementation (zeros_like to avoid uninitialized memory)
    auto gpu_output = torch::zeros_like(ref_output);

    // Collect raw device pointers to parameters (all fp16 on device)
    const void* input_ptr = static_cast<const void*>(input.data_ptr<c10::Half>());
    void* output_ptr = static_cast<void*>(gpu_output.data_ptr<c10::Half>());

    const void* squeeze_w_ptr = static_cast<const void*>(squeeze->weight.data_ptr<c10::Half>());
    const void* squeeze_b_ptr = static_cast<const void*>(squeeze->bias.defined() ? squeeze->bias.data_ptr<c10::Half>() : nullptr);

    const void* expand1x1_w_ptr = static_cast<const void*>(expand1x1->weight.data_ptr<c10::Half>());
    const void* expand1x1_b_ptr = static_cast<const void*>(expand1x1->bias.defined() ? expand1x1->bias.data_ptr<c10::Half>() : nullptr);

    const void* expand3x3_w_ptr = static_cast<const void*>(expand3x3->weight.data_ptr<c10::Half>());
    const void* expand3x3_b_ptr = static_cast<const void*>(expand3x3->bias.defined() ? expand3x3->bias.data_ptr<c10::Half>() : nullptr);

    // Convolution parameters
    int64_t squeeze_kernel_h = 1, squeeze_kernel_w = 1;
    int64_t expand1x1_kernel_h = 1, expand1x1_kernel_w = 1;
    int64_t expand3x3_kernel_h = 3, expand3x3_kernel_w = 3;
    int64_t expand3x3_padding_h = 1, expand3x3_padding_w = 1;
    int64_t stride_h = 1, stride_w = 1;
    int64_t dilation_h = 1, dilation_w = 1;

    // Launch user-provided GPU implementation
    launch_gpu_implementation(
        output_ptr,
        input_ptr,
        squeeze_w_ptr,
        squeeze_b_ptr,
        expand1x1_w_ptr,
        expand1x1_b_ptr,
        expand3x3_w_ptr,
        expand3x3_b_ptr,
        batch_size,
        num_input_features,
        height,
        width,
        squeeze_channels,
        expand1x1_channels,
        expand3x3_channels,
        squeeze_kernel_h, squeeze_kernel_w,
        expand1x1_kernel_h, expand1x1_kernel_w,
        expand3x3_kernel_h, expand3x3_kernel_w,
        expand3x3_padding_h, expand3x3_padding_w,
        stride_h, stride_w,
        dilation_h, dilation_w
    );

    // Validate results with torch::allclose
    // fp16 => rtol=1e-1, atol=1e-1
    bool ok = torch::allclose(ref_output, gpu_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    if (ok) {
        std::cout << "passed\n";
    } else {
        std::cout << "failed\n";
    }

    return 0; // Always return 0 per instructions
}
