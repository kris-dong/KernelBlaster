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
                               const void* fc_weight, const void* fc_bias,
                               const void* gn_weight, const void* gn_bias,
                               int batch_size, int input_size, int hidden_size,
                               int num_groups, float negative_slope, float eps);

int main() {
    // Configuration parameters
    const int batch_size = 128;
    const int input_size = 512;
    const int hidden_size = 256;
    const int num_groups = 8;
    const float negative_slope = 0.01f;
    const float eps = 1e-5f;

    // Create input tensor on GPU
    auto input = torch::randn({batch_size, input_size}, 
                            torch::dtype(torch::kFloat16).device(torch::kCUDA));

    // Create reference model components
    auto fc = torch::nn::Linear(input_size, hidden_size);
    auto gn_options = torch::nn::GroupNormOptions(num_groups, hidden_size).eps(eps);
    auto gn = torch::nn::GroupNorm(gn_options);

    // Move parameters to GPU and convert to FP16
    fc->to(torch::kCUDA, torch::kFloat16);
    gn->to(torch::kCUDA, torch::kFloat16);

    // Run reference implementation
    auto fc_out = fc->forward(input);
    auto gn_out = gn->forward(fc_out);
    auto lrelu_out = torch::leaky_relu(gn_out, negative_slope);
    auto output_ref = lrelu_out + lrelu_out;

    // Create output tensor for CUDA implementation
    auto output = torch::empty_like(output_ref);

    // Get raw pointers for GPU implementation
    void* input_ptr = input.data_ptr();
    void* output_ptr = output.data_ptr();
    void* fc_weight_ptr = fc->weight.data_ptr();
    void* fc_bias_ptr = fc->bias.data_ptr();
    void* gn_weight_ptr = gn->weight.data_ptr();
    void* gn_bias_ptr = gn->bias.data_ptr();

    // Launch CUDA implementation
    launch_gpu_implementation(output_ptr, input_ptr,
                            fc_weight_ptr, fc_bias_ptr,
                            gn_weight_ptr, gn_bias_ptr,
                            batch_size, input_size, hidden_size,
                            num_groups, negative_slope, eps);

    // Verify results with FP16 thresholds
    bool is_close = torch::allclose(output_ref, output, 
                                  /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (is_close ? "passed" : "failed") << std::endl;

    return 0;
}
