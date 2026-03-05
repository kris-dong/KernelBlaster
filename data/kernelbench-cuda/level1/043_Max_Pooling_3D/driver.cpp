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

// Declaration for the custom CUDA implementation (to be implemented elsewhere).
void launch_gpu_implementation(
    void* output,                   // Output tensor (GPU memory, fp16)
    void* input,                    // Input tensor (GPU memory, fp16)
    int batch_size,
    int channels,
    int dim1,
    int dim2,
    int dim3,
    int kernel_size,
    int stride,
    int padding,
    int dilation
);

int main() {
    // Set device to CUDA
    torch::Device device(torch::kCUDA);

    // Use fp16 for all tensors
    using dtype = at::Half;

    // Model and input parameters
    int batch_size = 16;
    int channels = 32;
    int dim1 = 64;
    int dim2 = 64;
    int dim3 = 64;
    int kernel_size = 3;
    int stride = 2;
    int padding = 1;
    int dilation = 3;

    // Create random input tensor on GPU (fp16)
    auto input = torch::randn({batch_size, channels, dim1, dim2, dim3}, torch::TensorOptions().dtype(torch::kFloat16).device(device));

    // Instantiate reference MaxPool3d layer
    torch::nn::MaxPool3d maxpool(
        torch::nn::MaxPool3dOptions(kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .ceil_mode(false)
    );

    // Move the module to GPU and convert to fp16
    maxpool->to(device, torch::kFloat16);

    // Reference output using libtorch
    auto ref_output = maxpool->forward(input);

    // Allocate output tensor for CUDA implementation
    auto cuda_output = torch::empty_like(ref_output);

    // Launch the CUDA implementation
    launch_gpu_implementation(
        cuda_output.data_ptr(),
        input.data_ptr(),
        batch_size,
        channels,
        dim1,
        dim2,
        dim3,
        kernel_size,
        stride,
        padding,
        dilation
    );

    // Compare outputs using torch::allclose with rtol/atol = 1e-1 for fp16
    bool passed = torch::allclose(ref_output, cuda_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
