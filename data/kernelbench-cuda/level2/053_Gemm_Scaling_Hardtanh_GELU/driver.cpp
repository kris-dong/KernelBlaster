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

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, float scaling_factor, float hardtanh_min, float hardtanh_max, int batch_size, int in_features, int out_features);

int main() {
    // Configuration parameters
    const int batch_size = 128;
    const int in_features = 1024;
    const int out_features = 512;
    const float scaling_factor = 0.5f;
    const float hardtanh_min = -2.0f;
    const float hardtanh_max = 2.0f;
    const float atol = 1e-1;
    const float rtol = 1e-1;

    // Create input tensor on GPU with fp16
    auto input_tensor = torch::randn({batch_size, in_features}, 
        torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Create reference model
    auto gemm = torch::nn::Linear(in_features, out_features);
    gemm->to(torch::kCUDA, torch::kHalf);
    
    // Get model parameters
    auto gemm_weight = gemm->weight.detach().clone();
    auto gemm_bias = gemm->bias.detach().clone();

    // Build reference computation
    auto reference_output = gemm->forward(input_tensor);
    reference_output = reference_output * scaling_factor;
    reference_output = torch::hardtanh(reference_output, hardtanh_min, hardtanh_max);
    reference_output = torch::gelu(reference_output);

    // Prepare custom implementation output tensor
    auto custom_output = torch::empty_like(reference_output);

    // Get raw pointers for GPU implementation
    auto input_data = input_tensor.data_ptr<torch::Half>();
    auto weight_data = gemm_weight.data_ptr<torch::Half>();
    auto bias_data = gemm_bias.data_ptr<torch::Half>();
    auto output_data = custom_output.data_ptr<torch::Half>();

    // Launch custom GPU implementation
    launch_gpu_implementation(
        static_cast<void*>(output_data),
        static_cast<void*>(input_data),
        static_cast<void*>(weight_data),
        static_cast<void*>(bias_data),
        scaling_factor,
        hardtanh_min,
        hardtanh_max,
        batch_size,
        in_features,
        out_features
    );

    // Verify results
    bool passed = torch::allclose(custom_output, reference_output, rtol, atol);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
