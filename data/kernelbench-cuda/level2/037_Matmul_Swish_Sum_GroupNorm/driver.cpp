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
                               void* matmul_weight, void* matmul_bias,
                               void* extra_bias,
                               void* gn_weight, void* gn_bias,
                               int64_t in_features, int64_t out_features,
                               int64_t num_groups, int64_t batch_size);

int main() {
    // Initialize parameters from Python spec
    const int64_t batch_size = 128;
    const int64_t in_features = 512;
    const int64_t out_features = 1024;
    const int64_t num_groups = 32;
    const std::vector<int64_t> bias_shape = {out_features};

    // Create model components on CUDA with float16
    auto matmul = torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features));
    auto extra_bias = torch::randn(bias_shape, torch::dtype(torch::kHalf).device(torch::kCUDA));
    auto group_norm = torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, out_features));

    // Convert modules to CUDA and half precision
    matmul->to(torch::kCUDA, torch::kHalf);
    group_norm->to(torch::kCUDA, torch::kHalf);

    // Create input tensor on GPU
    auto input = torch::randn({batch_size, in_features}, 
                             torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Reference implementation
    auto x = matmul->forward(input.clone());
    x = torch::sigmoid(x) * x;  // Swish activation
    x += extra_bias;
    x = group_norm->forward(x);
    auto reference_output = x.contiguous();

    // Allocate output tensor for CUDA implementation
    auto output = torch::empty_like(reference_output);

    // Get raw pointers for all parameters
    auto matmul_weight_ptr = matmul->weight.data_ptr();
    auto matmul_bias_ptr = matmul->bias.data_ptr();
    auto gn_weight_ptr = group_norm->weight.data_ptr();
    auto gn_bias_ptr = group_norm->bias.data_ptr();

    // Launch CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        matmul_weight_ptr,
        matmul_bias_ptr,
        extra_bias.data_ptr(),
        gn_weight_ptr,
        gn_bias_ptr,
        in_features,
        out_features,
        num_groups,
        batch_size
    );

    // Verify results with fp16 thresholds
    bool passed = torch::allclose(reference_output, output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
