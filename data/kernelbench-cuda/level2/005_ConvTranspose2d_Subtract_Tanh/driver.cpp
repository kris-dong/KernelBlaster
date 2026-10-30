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
    int64_t batch_size, int64_t in_channels, int64_t out_channels,
    int64_t input_height, int64_t input_width,
    int64_t kernel_size, int64_t stride, int64_t padding, int64_t output_padding,
    int64_t output_height, int64_t output_width
);

int main() {
    // Setup device and dtype
    torch::ScalarType dtype = torch::kHalf;
    torch::Device device(torch::kCUDA);
    
    // Model parameters from Python code
    const int64_t batch_size = 128;
    const int64_t in_channels = 32;
    const int64_t out_channels = 16;
    const int64_t height = 16, width = 16;
    const int64_t kernel_size = 4;
    const std::vector<int64_t> bias_shape{out_channels, 1, 1};
    const int64_t stride = 2;
    const int64_t padding = 1;
    const int64_t output_padding = 1;

    // Create reference model
    auto conv = torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .output_padding(output_padding)
    );
    auto model_bias = torch::randn(bias_shape, torch::dtype(dtype).device(device));
    
    // Move components to GPU and set dtype
    conv->to(device, dtype);
    model_bias = model_bias.to(device).to(dtype);

    // Create input tensor
    auto input = torch::randn({batch_size, in_channels, height, width}, 
                             torch::dtype(dtype).device(device));

    // Run reference implementation
    auto conv_output = conv->forward(input);
    auto sub_bias = conv_output - model_bias;
    auto ref_output = torch::tanh(sub_bias);

    // Allocate output tensor for CUDA implementation
    auto cuda_output = torch::empty_like(ref_output);

    // Get tensor data pointers
    auto input_ptr = input.data_ptr();
    auto output_ptr = cuda_output.data_ptr();
    auto conv_weight_ptr = conv->weight.data_ptr();
    auto conv_bias_ptr = conv->bias.data_ptr();
    auto model_bias_ptr = model_bias.data_ptr();

    // Get output dimensions from reference
    const int64_t output_height = ref_output.size(2);
    const int64_t output_width = ref_output.size(3);

    // Launch CUDA implementation
    launch_gpu_implementation(
        output_ptr, input_ptr,
        conv_weight_ptr, conv_bias_ptr, model_bias_ptr,
        batch_size, in_channels, out_channels,
        height, width,
        kernel_size, stride, padding, output_padding,
        output_height, output_width
    );

    // Verify results
    if (torch::allclose(ref_output, cuda_output, /*rtol=*/1e-1, /*atol=*/1e-1)) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
