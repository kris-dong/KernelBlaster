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

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, int batch_size, int in_channels, int out_channels, int depth, int height, int width, int kernel_size);

int main() {
    // Setup parameters
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int depth = 16, height = 32, width = 32;
    const int kernel_size = 3;

    // Create input tensor on CUDA with FP16
    auto input = torch::randn({batch_size, in_channels, depth, height, width}, 
                             torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Create reference model with fixed seed for deterministic results
    torch::manual_seed(42);
    auto conv = torch::nn::Conv3d(torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size).bias(true));
    conv->to(torch::kCUDA, torch::kHalf);
    conv->eval();

    // Get parameters as pointers
    auto weight = conv->weight;
    auto bias = conv->bias;

    // Run reference implementation
    auto x = conv->forward(input);
    x = torch::hardswish(x);
    x = torch::relu(x);
    x = torch::softmax(x, 1);
    x = torch::mean(x, {2, 3, 4});
    auto reference_output = x.contiguous();

    // Allocate and initialize CUDA output tensor with NaNs to ensure failure
    auto output_cuda = torch::full_like(reference_output, NAN);

    // Call CUDA implementation
    launch_gpu_implementation(
        output_cuda.data_ptr(),
        input.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        batch_size,
        in_channels,
        out_channels,
        depth,
        height,
        width,
        kernel_size
    );

    // Verify results with explicit check for NaN values
    bool has_nan = torch::isnan(output_cuda).any().item().toBool();
    bool passed = !has_nan && torch::allclose(reference_output, output_cuda, /*rtol=*/1e-1, /*atol=*/1e-1);
    
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
