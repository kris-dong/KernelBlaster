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
    void* conv_weight, void* conv_bias,
    void* bn_weight, void* bn_bias,
    int in_channels, int out_channels,
    int kernel_size, int stride, int padding,
    int batch_size, int input_depth, int input_height, int input_width,
    int output_depth, int output_height, int output_width
);

int main() {
    const int batch_size = 16;
    const int in_channels = 16;
    const int out_channels = 32;
    const int depth = 16, height = 32, width = 32;
    const int kernel_size = 3;
    const int stride = 2;
    const int padding = 1;

    // Create input tensor
    auto input = torch::randn({batch_size, in_channels, depth, height, width}, 
                            torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Initialize model components
    auto conv_transpose = torch::nn::ConvTranspose3d(
        torch::nn::ConvTranspose3dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .bias(true)
    );
    conv_transpose->to(torch::kCUDA, torch::kHalf);
    
    auto batch_norm = torch::nn::BatchNorm3d(out_channels);
    batch_norm->to(torch::kCUDA, torch::kHalf);

    // Manually initialize parameters
    conv_transpose->weight.data().normal_();
    conv_transpose->bias.data().normal_();
    batch_norm->weight.data().normal_();
    batch_norm->bias.data().normal_();

    // Run reference implementation
    auto x = conv_transpose->forward(input);
    x = batch_norm->forward(x);
    x = x - torch::mean(x, {2, 3, 4}, true);
    auto reference_output = x;

    // Get parameter pointers with explicit template parameters
    auto conv_weight_ptr = static_cast<void*>(conv_transpose->weight.data_ptr<torch::Half>());
    auto conv_bias_ptr = static_cast<void*>(conv_transpose->bias.data_ptr<torch::Half>());
    auto bn_weight_ptr = static_cast<void*>(batch_norm->weight.data_ptr<torch::Half>());
    auto bn_bias_ptr = static_cast<void*>(batch_norm->bias.data_ptr<torch::Half>());

    // Prepare output tensor
    auto cuda_output = torch::empty_like(reference_output);

    // Get spatial dimensions from reference output
    const int output_depth = reference_output.size(2);
    const int output_height = reference_output.size(3);
    const int output_width = reference_output.size(4);

    // Launch CUDA implementation with properly cast pointers
    launch_gpu_implementation(
        static_cast<void*>(cuda_output.data_ptr<torch::Half>()),
        static_cast<void*>(input.data_ptr<torch::Half>()),
        conv_weight_ptr,
        conv_bias_ptr,
        bn_weight_ptr,
        bn_bias_ptr,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        batch_size,
        depth,
        height,
        width,
        output_depth,
        output_height,
        output_width
    );

    // Verify results
    bool passed = torch::allclose(cuda_output, reference_output, 1e-1, 1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
