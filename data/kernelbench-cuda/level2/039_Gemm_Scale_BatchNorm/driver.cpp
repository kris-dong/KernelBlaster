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
    void* gemm_weight, void* gemm_bias,
    void* scale,
    void* bn_weight, void* bn_bias,
    void* bn_running_mean, void* bn_running_var,
    float bn_eps,
    int64_t batch_size, int64_t in_features, int64_t out_features
);

int main() {
    // Model configuration
    const int64_t batch_size = 128;
    const int64_t in_features = 1024;
    const int64_t out_features = 512;
    const std::vector<int64_t> scale_shape = {out_features};
    const float eps = 1e-5;
    const float momentum = 0.1;

    // Initialize model components
    auto gemm = torch::nn::Linear(in_features, out_features);
    gemm->to(torch::kCUDA, torch::kFloat16);
    
    auto scale = torch::randn(scale_shape, 
        torch::dtype(torch::kFloat16).device(torch::kCUDA));
    
    auto bn_options = torch::nn::BatchNorm1dOptions(out_features)
                        .eps(eps).momentum(momentum);
    auto bn = torch::nn::BatchNorm1d(bn_options);
    bn->to(torch::kCUDA, torch::kFloat16);
    bn->eval();

    // Create input tensor
    auto input = torch::randn({batch_size, in_features}, 
        torch::dtype(torch::kFloat16).device(torch::kCUDA));

    // Reference implementation
    auto x = gemm->forward(input);
    x = x * scale;
    x = bn->forward(x);
    auto reference_output = x;

    // Prepare CUDA output tensor
    auto output = torch::empty_like(reference_output);

    // Get parameter pointers
    auto gemm_weight = gemm->weight;
    auto gemm_bias = gemm->bias;
    auto bn_weight = bn->weight;
    auto bn_bias = bn->bias;
    auto bn_running_mean = bn->running_mean;
    auto bn_running_var = bn->running_var;

    // Execute CUDA kernel
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        gemm_weight.data_ptr(),
        gemm_bias.data_ptr(),
        scale.data_ptr(),
        bn_weight.data_ptr(),
        bn_bias.data_ptr(),
        bn_running_mean.data_ptr(),
        bn_running_var.data_ptr(),
        eps,
        batch_size,
        in_features,
        out_features
    );

    // Verify results
    bool passed = torch::allclose(reference_output, output, 1e-1, 1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
