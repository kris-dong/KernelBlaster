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
    void* output, 
    void* input,
    const void* matmul_weight,
    const void* matmul_bias,
    int64_t in_features,
    int64_t out_features,
    int64_t pool_kernel_size,
    float scale_factor,
    int64_t batch_size
);

int main() {
    // Configuration parameters
    const int64_t batch_size = 128;
    const int64_t in_features = 512;
    const int64_t out_features = 256;
    const int64_t pool_kernel_size = 4;
    const float scale_factor = 2.0f;
    const at::ScalarType dtype = torch::kFloat16;
    const float tolerance = 1e-1;  // For fp16/bf16
    
    // Create and initialize model
    torch::nn::Linear matmul{in_features, out_features};
    matmul->to(torch::kCUDA, dtype, /*non_blocking=*/false);
    // Initialize with default initialization for consistency with PyTorch
    
    // Create input tensor on GPU
    auto input = torch::randn({batch_size, in_features}, 
                            torch::dtype(dtype).device(torch::kCUDA));
    
    // Run libtorch reference implementation
    auto x = matmul(input);
    x = torch::avg_pool1d(x.unsqueeze(1), /*kernel_size=*/pool_kernel_size).squeeze(1);
    x = torch::gelu(x);
    x = x * scale_factor;
    auto ref_output = std::get<0>(x.max(/*dim=*/1, /*keepdim=*/false));
    
    // Allocate GPU memory for CUDA implementation output
    auto cuda_output = torch::empty_like(ref_output);
    
    // Get raw pointers to model parameters
    auto matmul_weight_ptr = matmul->weight.data_ptr();
    auto matmul_bias_ptr = matmul->bias.data_ptr();
    
    // Call CUDA implementation
    launch_gpu_implementation(
        cuda_output.data_ptr(),
        input.data_ptr(),
        matmul_weight_ptr,
        matmul_bias_ptr,
        in_features,
        out_features,
        pool_kernel_size,
        scale_factor,
        batch_size
    );
    
    // Compare results
    bool passed = torch::allclose(
        ref_output,
        cuda_output,
        /*rtol=*/tolerance,
        /*atol=*/tolerance
    );
    
    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }
    
    return 0;
}
