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
                              void* matmul_weight, void* matmul_bias, void* add_value,
                              int64_t batch_size, int64_t in_features, int64_t out_features);

int main() {
    // Configuration parameters
    const int64_t batch_size = 128;
    const int64_t in_features = 1024;
    const int64_t out_features = 512;
    const torch::Dtype dtype = torch::kHalf;
    const c10::Device device(torch::kCUDA);
    
    // Create options with correct dtype and device
    auto options = torch::TensorOptions().dtype(dtype).device(device);

    // Create input tensor
    auto input_tensor = torch::randn({batch_size, in_features}, options);

    // Initialize model parameters
    // Matmul layer parameters
    auto matmul_weight = torch::empty({out_features, in_features}, options);
    auto matmul_bias = torch::empty({out_features}, options);
    
    // Kaiming initialization with fixed nonlinearity type
    torch::nn::init::kaiming_uniform_(
        matmul_weight,
        std::sqrt(5.0),
        torch::kFanIn,
        torch::kLeakyReLU  // Using enum instead of string
    );
    
    // Bias initialization
    float bias_bound = 1.0f / std::sqrt(in_features);
    matmul_bias.uniform_(-bias_bound, bias_bound);

    // Add value parameter
    auto add_value = torch::randn({out_features}, options);

    // LibTorch reference implementation
    auto x = input_tensor.matmul(matmul_weight.transpose(0, 1)) + matmul_bias;
    x = x + add_value;
    x = x * torch::sigmoid(x);  // Swish
    x = torch::tanh(x);
    x = torch::gelu(x);
    x = torch::clamp(x, -1, 1);  // Hardtanh
    auto libtorch_result = x;

    // Initialize GPU tensor with obviously wrong values
    auto gpu_output = torch::full_like(libtorch_result, 1234.0);  // Will fail if kernel doesn't write

    // Get raw pointers for CUDA implementation
    launch_gpu_implementation(
        gpu_output.data_ptr(),
        input_tensor.data_ptr(),
        matmul_weight.data_ptr(),
        matmul_bias.data_ptr(),
        add_value.data_ptr(),
        batch_size,
        in_features,
        out_features
    );

    // Verify results
    bool passed = torch::allclose(
        libtorch_result,
        gpu_output,
        /*rtol=*/1e-1,
        /*atol=*/1e-1,
        /*equal_nan=*/false
    );

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
