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

void launch_gpu_implementation(void* output, void* input, void* conv_weight, void* conv_bias, float divide_by);

int main() {
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int height = 32, width = 32;
    const int kernel_size = 3;
    const float divide_by = 2.0f;

    // Create input tensor on CUDA with FP16
    auto input_tensor = torch::randn({batch_size, in_channels, height, width}, 
                                   torch::dtype(torch::kFloat16).device(torch::kCUDA));

    // Create reference model components
    auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                  .padding(kernel_size / 2));
    auto instance_norm = torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(out_channels)
                                  .eps(1e-5).affine(false));

    // Move parameters to CUDA and set FP16
    conv->to(torch::kCUDA, torch::kFloat16);
    instance_norm->to(torch::kCUDA, torch::kFloat16);

    // Capture initialized parameters
    auto conv_weight = conv->weight.clone().detach();
    auto conv_bias = conv->bias.clone().detach();

    // Run reference implementation
    auto ref_output = conv->forward(input_tensor);
    ref_output = instance_norm(ref_output);
    ref_output = ref_output / divide_by;

    // Prepare output tensor for CUDA implementation
    auto custom_output = torch::empty_like(ref_output);

    // Call CUDA implementation with captured parameters
    launch_gpu_implementation(custom_output.data_ptr(),
                            input_tensor.data_ptr(),
                            conv_weight.data_ptr(),
                            conv_bias.data_ptr(),
                            divide_by);

    // Verify results with relaxed tolerances for FP16
    bool passed = torch::allclose(custom_output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
