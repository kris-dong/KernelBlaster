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
                               void* linear1_weight, void* linear1_bias,
                               void* linear2_weight, void* linear2_bias,
                               int64_t input_size, int64_t hidden_size, int64_t output_size);

int main() {
    // Setup model parameters
    const int64_t batch_size = 128;
    const int64_t input_size = 10;
    const int64_t hidden_size = 20;
    const int64_t output_size = 5;
    const float rtol = 1e-1;  // For fp16
    const float atol = 1e-1;

    // Create reference model
    torch::nn::Linear linear1(input_size, hidden_size);
    torch::nn::Linear linear2(hidden_size, output_size);
    linear1->to(torch::kCUDA, torch::kHalf);
    linear2->to(torch::kCUDA, torch::kHalf);

    // Create input tensor
    auto input = torch::randn({batch_size, input_size}, 
                        torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Run reference implementation
    auto x = linear1->forward(input);
    x = torch::sigmoid(x);
    x = torch::sum(x, /*dim=*/1);
    x = torch::logsumexp(x, /*dim=*/0);
    auto reference_output = x.unsqueeze(0);  // Add dim for output comparison

    // Prepare output tensor for GPU implementation
    auto kernel_output = torch::empty_like(reference_output);

    // Get parameter pointers
    auto linear1_w = linear1->weight.data_ptr<c10::Half>();
    auto linear1_b = linear1->bias.data_ptr<c10::Half>();
    auto linear2_w = linear2->weight.data_ptr<c10::Half>();
    auto linear2_b = linear2->bias.data_ptr<c10::Half>();

    // Call GPU implementation
    launch_gpu_implementation(
        kernel_output.data_ptr(),
        input.data_ptr(),
        linear1_w,
        linear1_b,
        linear2_w,
        linear2_b,
        input_size,
        hidden_size,
        output_size
    );

    // Verify results
    if (torch::allclose(kernel_output, reference_output, rtol, atol)) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
