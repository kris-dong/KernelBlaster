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

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, 
                              int batch_size, int in_features, int out_features, int max_dim);

int main() {
    // Setup parameters
    const int batch_size = 128;
    const int in_features = 512;
    const int out_features = 1024;
    const int max_dim = 1;
    const float atol = 1e-1, rtol = 1e-1;  // fp16 requires higher tolerance

    // Create model and move to CUDA
    torch::nn::Linear gemm(in_features, out_features);
    gemm->to(torch::kCUDA, torch::kFloat16);

    // Generate input tensor on GPU
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    torch::Tensor input = torch::randn({batch_size, in_features}, options);

    // Run reference implementation
    torch::Tensor x = gemm(input);
    x = std::get<0>(x.max(max_dim, true));
    x = x - x.mean(1, true);
    torch::Tensor ref_output = torch::gelu(x);

    // Prepare CUDA kernel inputs/output
    torch::Tensor cuda_output = torch::empty_like(ref_output);
    
    // Get raw pointers for model parameters
    auto weight_ptr = gemm->weight.data_ptr<torch::Half>();
    auto bias_ptr = gemm->bias.data_ptr<torch::Half>();

    // Launch CUDA implementation
    launch_gpu_implementation(cuda_output.data_ptr<torch::Half>(),
                             input.data_ptr<torch::Half>(),
                             weight_ptr,
                             bias_ptr,
                             batch_size,
                             in_features,
                             out_features,
                             max_dim);

    // Compare results
    bool passed = torch::allclose(cuda_output, ref_output, rtol, atol);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
