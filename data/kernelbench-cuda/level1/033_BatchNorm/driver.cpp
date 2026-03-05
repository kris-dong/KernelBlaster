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
#include <iostream>
#include "cuda_model.cuh"

// Declaration for the GPU implementation.
// Pass pointers to the parameter tensors (weight, bias, running_mean, running_var) for exact matching.
void launch_gpu_implementation(
    void* output,               // output tensor (fp16, GPU)
    void* input,                // input tensor (fp16, GPU)
    void* weight,               // BatchNorm weight (gamma) (fp16, GPU)
    void* bias,                 // BatchNorm bias (beta) (fp16, GPU)
    void* running_mean,         // running mean (fp16, GPU)
    void* running_var,          // running var (fp16, GPU)
    int64_t batch_size,
    int64_t num_features,
    int64_t dim1,
    int64_t dim2
);

int main() {
    // Set up parameters
    using scalar_t = at::Half; // fp16

    const int64_t batch_size = 16;
    const int64_t features = 64;
    const int64_t dim1 = 256;
    const int64_t dim2 = 256;

    // Set default dtype globally to float16 (fp16)
    torch::Dtype dtype = torch::kFloat16;

    // Allocate input tensor on CUDA in fp16
    torch::Tensor input = torch::randn({batch_size, features, dim1, dim2}, torch::TensorOptions().dtype(dtype).device(torch::kCUDA));

    // Instantiate BatchNorm2d module using torch::nn
    torch::nn::BatchNorm2d bn(features);
    bn->to(torch::kCUDA, dtype);

    // Set to eval mode to use running stats only (to match typical inference behavior)
    bn->eval();

    // Save pointers to the parameter tensors (weights, bias, running_mean, running_var)
    torch::Tensor weight = bn->weight.detach().clone().to(torch::kCUDA, dtype);
    torch::Tensor bias = bn->bias.detach().clone().to(torch::kCUDA, dtype);
    torch::Tensor running_mean = bn->running_mean.detach().clone().to(torch::kCUDA, dtype);
    torch::Tensor running_var = bn->running_var.detach().clone().to(torch::kCUDA, dtype);

    // Copy parameters into module to ensure reference and kernel see the same values
    bn->weight.set_data(weight);
    bn->bias.set_data(bias);
    bn->running_mean.set_data(running_mean);
    bn->running_var.set_data(running_var);

    // Reference output (libtorch)
    torch::Tensor ref_output = bn->forward(input);

    // Allocate output tensor for GPU implementation
    torch::Tensor gpu_output = torch::empty_like(ref_output, torch::TensorOptions().dtype(dtype).device(torch::kCUDA));

    // Call the kernel: pass raw pointers (void*) to the data
    launch_gpu_implementation(
        gpu_output.data_ptr(),
        input.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        running_mean.data_ptr(),
        running_var.data_ptr(),
        batch_size,
        features,
        dim1,
        dim2
    );

    // Compare outputs using torch::allclose with rtol=1e-1, atol=1e-1 for fp16
    bool passed = torch::allclose(gpu_output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
