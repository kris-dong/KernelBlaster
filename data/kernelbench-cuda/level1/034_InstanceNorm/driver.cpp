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
#include <vector>
#include "cuda_model.cuh"

// Declaration for the CUDA implementation
void launch_gpu_implementation(
    void* output,
    void* input,
    void* weight,
    void* bias,
    int batch_size,
    int num_features,
    int height,
    int width
);

int main() {
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    int batch_size = 16;
    int num_features = 64;
    int height = 256;
    int width = 256;

    // Make sure affine=True so weight and bias exist
    torch::nn::InstanceNorm2dOptions inorm_options(num_features);
    inorm_options.affine(true);
    auto inorm = torch::nn::InstanceNorm2d(inorm_options);
    inorm->to(device, dtype);

    // Input
    torch::Tensor input = torch::randn({batch_size, num_features, height, width},
        torch::TensorOptions().dtype(dtype).device(device));

    // Reference output
    torch::Tensor ref_output = inorm->forward(input);

    // CUDA output tensor
    torch::Tensor cuda_output = torch::empty_like(ref_output);

    // Raw pointers for CUDA kernel
    void* input_ptr = input.data_ptr<at::Half>();
    void* output_ptr = cuda_output.data_ptr<at::Half>();
    void* weight_ptr = nullptr;
    void* bias_ptr = nullptr;

    // Get weight and bias pointers (affine guaranteed true above)
    weight_ptr = inorm->weight.data_ptr<at::Half>();
    bias_ptr = inorm->bias.data_ptr<at::Half>();

    // Call CUDA kernel
    launch_gpu_implementation(
        output_ptr,
        input_ptr,
        weight_ptr,
        bias_ptr,
        batch_size,
        num_features,
        height,
        width
    );

    // Compare outputs with fp16 tolerances
    bool passed = torch::allclose(cuda_output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
