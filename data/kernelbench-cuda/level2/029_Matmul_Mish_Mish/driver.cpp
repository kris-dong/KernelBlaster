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

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, int batch_size, int in_features, int out_features);

int main() {
    // Setup parameters
    const int batch_size = 128;
    const int in_features = 10;
    const int out_features = 20;
    const float atol = 1e-1;  // FP16 requires higher tolerance
    const float rtol = 1e-1;

    // Create model and move to CUDA with FP16
    torch::nn::Linear linear(in_features, out_features);
    linear->to(torch::kCUDA, torch::kHalf);
    
    // Generate random input (FP16 on CUDA)
    auto input = torch::randn({batch_size, in_features}, 
                torch::dtype(torch::kHalf).device(torch::kCUDA));
    
    // Run reference implementation
    auto ref_output = linear->forward(input);
    ref_output = torch::mish(ref_output);
    ref_output = torch::mish(ref_output);

    // Get model parameters
    auto weight = linear->weight;
    auto bias = linear->bias;

    // Prepare output tensor for CUDA implementation
    auto output = torch::empty_like(ref_output);

    // Call CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        weight.data_ptr(),
        bias.defined() ? bias.data_ptr() : nullptr,
        batch_size,
        in_features,
        out_features
    );

    // Verify results
    if (torch::allclose(output, ref_output, rtol, atol)) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
