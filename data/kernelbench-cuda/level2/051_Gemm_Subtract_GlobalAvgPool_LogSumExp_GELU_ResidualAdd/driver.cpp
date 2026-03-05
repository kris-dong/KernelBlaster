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

void launch_gpu_implementation(void* output, void* input, void* gemm_weight, void* gemm_bias, void* subtract_param, 
                              int batch_size, int in_features, int out_features);

int main() {
    const int batch_size = 128;
    const int in_features = 1024;
    const int out_features = 512;
    const bool use_bias = true;

    // Create model and move to CUDA
    torch::nn::Linear gemm(torch::nn::LinearOptions(in_features, out_features).bias(use_bias));
    gemm->to(torch::kCUDA, torch::kFloat16);
    torch::Tensor subtract = torch::randn({out_features}, torch::dtype(torch::kFloat16).device(torch::kCUDA));

    // Initialize input tensor
    torch::Tensor input = torch::randn({batch_size, in_features}, torch::dtype(torch::kFloat16).device(torch::kCUDA));

    // Reference implementation
    torch::Tensor original_input = input.clone();
    torch::Tensor x = gemm->forward(input);
    x = x - subtract;
    x = torch::mean(x, {1}, /*keepdim=*/true);
    x = torch::logsumexp(x, 1, /*keepdim=*/true);
    x = torch::gelu(x);
    torch::Tensor ref_output = x + original_input;

    // Prepare output tensor for CUDA kernel
    torch::Tensor output = torch::empty_like(ref_output);

    // Get raw pointers for GPU memory
    void* output_ptr = output.data_ptr();
    void* input_ptr = input.data_ptr();
    void* weight_ptr = gemm->weight.data_ptr();
    void* bias_ptr = gemm->bias.data_ptr();
    void* subtract_ptr = subtract.data_ptr();

    // Launch CUDA implementation
    launch_gpu_implementation(output_ptr, input_ptr, weight_ptr, bias_ptr, subtract_ptr,
                             batch_size, in_features, out_features);

    // Verify results
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
