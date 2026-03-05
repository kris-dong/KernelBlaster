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
    // Setup dimensions and device
    const int batch_size = 128;
    const int in_features = 100;
    const int out_features = 10;
    const torch::Device device(torch::kCUDA);
    const c10::ScalarType dtype = torch::kHalf;

    // Create reference model
    torch::nn::Linear linear(in_features, out_features);
    linear->to(device, dtype, true);
    linear->weight.data().normal_(0, 0.02);
    linear->bias.data().normal_(0, 0.02);

    // Create input tensor
    torch::Tensor input = torch::randn({batch_size, in_features}, 
                                     torch::dtype(dtype).device(device));
    
    // Run reference implementation
    torch::Tensor output_ref = linear->forward(input);
    output_ref = torch::gelu(output_ref);
    output_ref = torch::softmax(output_ref, 1);

    // Prepare CUDA implementation output tensor
    torch::Tensor output_cuda = torch::empty_like(output_ref);
    
    // Get raw pointers for GPU implementation
    void* input_ptr = input.data_ptr();
    void* output_ptr = output_cuda.data_ptr();
    void* weight_ptr = linear->weight.data_ptr();
    void* bias_ptr = linear->bias.data_ptr();

    // Launch CUDA implementation
    launch_gpu_implementation(output_ptr, input_ptr, weight_ptr, bias_ptr,
                            batch_size, in_features, out_features);

    // Verify results
    const float rtol = 1e-1;  // Using relaxed tolerance for FP16
    const float atol = 1e-1;
    if (torch::allclose(output_cuda, output_ref, rtol, atol)) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
