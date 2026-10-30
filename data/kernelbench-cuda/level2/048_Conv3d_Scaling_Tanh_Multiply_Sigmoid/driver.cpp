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

struct ModelImpl : torch::nn::Module {
    torch::nn::Conv3d conv{nullptr};
    torch::Tensor scaling_factor, bias;

    ModelImpl(int64_t in_channels, int64_t out_channels, 
             int64_t kernel_size, int64_t /*scaling_factor_unused*/, 
             std::vector<int64_t> bias_shape) {
        conv = register_module("conv", torch::nn::Conv3d(
            torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)));
        
        scaling_factor = register_parameter("scaling_factor", 
            torch::randn(bias_shape, torch::dtype(torch::kFloat16)));
        bias = register_parameter("bias",
            torch::randn(bias_shape, torch::dtype(torch::kFloat16)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv->forward(x);
        x = x * scaling_factor;
        x = torch::tanh(x);
        x = x * bias;
        x = torch::sigmoid(x);
        return x;
    }
};
TORCH_MODULE(Model);

void launch_gpu_implementation(void* output, void* input, 
                              void* conv_weight, void* conv_bias,
                              void* scaling_factor, void* bias);

int main() {
    // Setup parameters
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int depth = 16, height = 32, width = 32;
    const int kernel_size = 3;
    const std::vector<int64_t> bias_shape{out_channels, 1, 1, 1};

    // Create model and move to CUDA
    Model model(in_channels, out_channels, kernel_size, 0, bias_shape);
    model->to(torch::kCUDA, torch::kHalf);

    // Create input tensor on CUDA
    auto input = torch::randn({batch_size, in_channels, depth, height, width}, 
                             torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Run reference implementation
    auto ref_output = model->forward(input);

    // Get parameter pointers
    auto conv_weight = model->conv->weight.data_ptr<torch::Half>();
    auto conv_bias = model->conv->bias.data_ptr<torch::Half>();
    auto scaling_factor_ptr = model->scaling_factor.data_ptr<torch::Half>();
    auto bias_ptr = model->bias.data_ptr<torch::Half>();

    // Allocate output tensor for CUDA implementation
    auto cuda_output = torch::empty_like(ref_output);

    // Launch CUDA implementation
    launch_gpu_implementation(
        cuda_output.data_ptr<torch::Half>(),
        input.data_ptr<torch::Half>(),
        conv_weight,
        conv_bias,
        scaling_factor_ptr,
        bias_ptr
    );

    // Verify results with fp16-appropriate tolerances
    if (torch::allclose(cuda_output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1)) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
