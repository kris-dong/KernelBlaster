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

// Declaration for CUDA implementation
void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, float scaling_factor);

int main() {
    // Configure FP16 datatype
    torch::ScalarType dtype = torch::kHalf;
    torch::Device device(torch::kCUDA);
    constexpr float rtol = 1e-1, atol = 1e-1;  // Reduced tolerance for FP16
    
    // Model parameters
    int64_t batch_size = 128, input_size = 1024, hidden_size = 512;
    float scaling_factor = 2.0f;

    // Create model and move to CUDA
    struct Model : torch::nn::Module {
        torch::nn::Linear gemm{nullptr};
        float scaling_factor;
        
        Model(int64_t in, int64_t out, float scale)
            : gemm(register_module("gemm", torch::nn::Linear(in, out))),
              scaling_factor(scale) {}
        
        torch::Tensor forward(torch::Tensor x) {
            auto original_x = gemm->forward(x);
            auto x_sig = torch::sigmoid(original_x);
            auto x_scaled = x_sig * scaling_factor;
            return x_scaled + original_x;
        }
    };
    
    auto model = std::make_shared<Model>(input_size, hidden_size, scaling_factor);
    model->to(device, dtype, /*non_blocking=*/false);
    
    // Generate input tensor
    auto input = torch::randn({batch_size, input_size}, 
        torch::dtype(dtype).device(device));
    
    // Run reference implementation
    auto reference_output = model->forward(input);
    
    // Get model parameters
    auto weight = model->gemm->weight.contiguous();
    auto bias = model->gemm->bias.contiguous();
    
    // Prepare output tensor for CUDA implementation
    auto gpu_output = torch::empty_like(reference_output);
    
    // Launch CUDA kernel
    launch_gpu_implementation(
        gpu_output.data_ptr(),
        input.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        scaling_factor
    );
    
    // Verify results
    bool passed = torch::allclose(gpu_output, reference_output, rtol, atol);
    std::cout << (passed ? "passed" : "failed") << std::endl;
    
    return 0;
}
