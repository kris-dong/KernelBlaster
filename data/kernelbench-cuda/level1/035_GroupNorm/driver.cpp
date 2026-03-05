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
#include "cuda_model.cuh"
#include <torch/torch.h>
#include <iostream>

// Declaration of the GPU kernel launcher
void launch_gpu_implementation(
    void* output,                       // Output tensor (fp16, GPU memory)
    void* input,                        // Input tensor (fp16, GPU memory)
    void* weight,                       // GroupNorm weight parameter (fp16, GPU memory)
    void* bias,                         // GroupNorm bias parameter (fp16, GPU memory)
    int batch_size,
    int num_features,
    int num_groups,
    int dim1,
    int dim2
);

int main() {
    // Set default dtype to Half (fp16)
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Model and input parameters
    int batch_size = 16;
    int num_features = 64;
    int num_groups = 8;
    int dim1 = 256;
    int dim2 = 256;

    // Generate input tensor on GPU
    torch::Tensor input = torch::randn({batch_size, num_features, dim1, dim2}, torch::TensorOptions().dtype(dtype).device(device));

    // Instantiate the reference GroupNorm layer and move to GPU
    torch::nn::GroupNorm gn(torch::nn::GroupNormOptions(num_groups, num_features));
    gn->to(device, dtype);

    // Forward pass (reference output)
    torch::Tensor ref_output = gn->forward(input);

    // Prepare output tensor for GPU kernel
    torch::Tensor gpu_output = torch::empty_like(ref_output, torch::TensorOptions().dtype(dtype).device(device));

    // Extract pointers to the parameters (weight, bias)
    torch::Tensor weight = gn->weight.detach(); // shape: [num_features], fp16, CUDA
    torch::Tensor bias = gn->bias.detach();     // shape: [num_features], fp16, CUDA

    // Launch the GPU implementation
    launch_gpu_implementation(
        gpu_output.data_ptr(),         // output (void*)
        input.data_ptr(),              // input (void*)
        weight.data_ptr(),             // weight (void*)
        bias.data_ptr(),               // bias (void*)
        batch_size,
        num_features,
        num_groups,
        dim1,
        dim2
    );

    // Compare outputs (fp16: use rtol=1e-1, atol=1e-1)
    if (torch::allclose(gpu_output, ref_output, 1e-1, 1e-1)) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
