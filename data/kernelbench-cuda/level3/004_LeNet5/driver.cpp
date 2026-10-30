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
#include <cuda_runtime.h>
#include <iostream>

// Declaration only. Do not implement here.
void launch_gpu_implementation(
    void* output,
    const void* input,
    const void* conv1_weight,
    const void* conv1_bias,
    const void* conv2_weight,
    const void* conv2_bias,
    const void* fc1_weight,
    const void* fc1_bias,
    const void* fc2_weight,
    const void* fc2_bias,
    const void* fc3_weight,
    const void* fc3_bias,
    int64_t batch_size,
    int64_t in_channels,
    int64_t in_h,
    int64_t in_w,
    // Conv1 params
    int64_t conv1_out_channels,
    int64_t conv1_kernel_h,
    int64_t conv1_kernel_w,
    int64_t conv1_stride_h,
    int64_t conv1_stride_w,
    int64_t conv1_pad_h,
    int64_t conv1_pad_w,
    // Pool params
    int64_t pool_kernel_h,
    int64_t pool_kernel_w,
    int64_t pool_stride_h,
    int64_t pool_stride_w,
    // Conv2 params
    int64_t conv2_out_channels,
    int64_t conv2_kernel_h,
    int64_t conv2_kernel_w,
    int64_t conv2_stride_h,
    int64_t conv2_stride_w,
    int64_t conv2_pad_h,
    int64_t conv2_pad_w,
    // Linear params
    int64_t fc1_in_features,
    int64_t fc1_out_features,
    int64_t fc2_out_features,
    int64_t fc3_out_features
);

int main() {
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available. Exiting." << std::endl;
        return 0;
    }

    namespace F = torch::nn::functional;

    // Use fp16 for all tensors
    const auto device = torch::Device(torch::kCUDA, 0);
    const auto dtype = torch::kFloat16;
    auto opts = torch::TensorOptions().dtype(dtype).device(device);

    // Model hyperparameters (LeNet-5 style)
    int64_t batch_size = 1;
    int64_t num_classes = 10;

    // Input dimensions
    int64_t in_channels = 1;
    int64_t in_h = 32;
    int64_t in_w = 32;

    // Conv1
    int64_t conv1_out_channels = 6;
    int64_t conv1_kernel_h = 5, conv1_kernel_w = 5;
    int64_t conv1_stride_h = 1, conv1_stride_w = 1;
    int64_t conv1_pad_h = 0, conv1_pad_w = 0;

    // Pool
    int64_t pool_kernel_h = 2, pool_kernel_w = 2;
    int64_t pool_stride_h = 2, pool_stride_w = 2;

    // Conv2
    int64_t conv2_out_channels = 16;
    int64_t conv2_kernel_h = 5, conv2_kernel_w = 5;
    int64_t conv2_stride_h = 1, conv2_stride_w = 1;
    int64_t conv2_pad_h = 0, conv2_pad_w = 0;

    // Derived dims after conv/pool:
    // After conv1: (32 - 5 + 2*0)/1 + 1 = 28
    // After pool: 28/2 = 14
    // After conv2: (14 - 5 + 2*0)/1 + 1 = 10
    // After pool: 10/2 = 5
    // Flatten size: 16 * 5 * 5 = 400
    int64_t fc1_in_features = 16 * 5 * 5;
    int64_t fc1_out_features = 120;
    int64_t fc2_out_features = 84;
    int64_t fc3_out_features = num_classes;

    // Random input on GPU, fp16
    auto input = torch::randn({batch_size, in_channels, in_h, in_w}, opts);

    // Initialize weights and biases on GPU with fp16
    auto conv1_w = torch::randn({conv1_out_channels, in_channels, conv1_kernel_h, conv1_kernel_w}, opts);
    auto conv1_b = torch::randn({conv1_out_channels}, opts);
    auto conv2_w = torch::randn({conv2_out_channels, conv1_out_channels, conv2_kernel_h, conv2_kernel_w}, opts);
    auto conv2_b = torch::randn({conv2_out_channels}, opts);

    auto fc1_w = torch::randn({fc1_out_features, fc1_in_features}, opts);
    auto fc1_b = torch::randn({fc1_out_features}, opts);

    auto fc2_w = torch::randn({fc2_out_features, fc1_out_features}, opts);
    auto fc2_b = torch::randn({fc2_out_features}, opts);

    auto fc3_w = torch::randn({fc3_out_features, fc2_out_features}, opts);
    auto fc3_b = torch::randn({fc3_out_features}, opts);

    // Reference implementation using libtorch (GPU, fp16)
    auto x = F::conv2d(
        input,
        conv1_w,
        F::Conv2dFuncOptions()
            .bias(conv1_b)
            .stride({conv1_stride_h, conv1_stride_w})
            .padding({conv1_pad_h, conv1_pad_w})
    );
    x = F::relu(x);
    x = F::max_pool2d(x, F::MaxPool2dFuncOptions({pool_kernel_h, pool_kernel_w})
                                .stride({pool_stride_h, pool_stride_w}));

    x = F::conv2d(
        x,
        conv2_w,
        F::Conv2dFuncOptions()
            .bias(conv2_b)
            .stride({conv2_stride_h, conv2_stride_w})
            .padding({conv2_pad_h, conv2_pad_w})
    );
    x = F::relu(x);
    x = F::max_pool2d(x, F::MaxPool2dFuncOptions({pool_kernel_h, pool_kernel_w})
                                .stride({pool_stride_h, pool_stride_w}));

    x = x.view({batch_size, fc1_in_features});
    x = F::linear(x, fc1_w, fc1_b);
    x = F::relu(x);
    x = F::linear(x, fc2_w, fc2_b);
    x = F::relu(x);
    auto output_ref = F::linear(x, fc3_w, fc3_b);

    // Output tensor for GPU implementation (initialized to zeros to avoid uninitialized memory issues)
    auto output_gpu = torch::zeros_like(output_ref);

    // Launch user-provided CUDA implementation
    launch_gpu_implementation(
        output_gpu.data_ptr(),
        input.data_ptr(),
        conv1_w.data_ptr(),
        conv1_b.data_ptr(),
        conv2_w.data_ptr(),
        conv2_b.data_ptr(),
        fc1_w.data_ptr(),
        fc1_b.data_ptr(),
        fc2_w.data_ptr(),
        fc2_b.data_ptr(),
        fc3_w.data_ptr(),
        fc3_b.data_ptr(),
        batch_size,
        in_channels,
        in_h,
        in_w,
        // Conv1
        conv1_out_channels,
        conv1_kernel_h,
        conv1_kernel_w,
        conv1_stride_h,
        conv1_stride_w,
        conv1_pad_h,
        conv1_pad_w,
        // Pool
        pool_kernel_h,
        pool_kernel_w,
        pool_stride_h,
        pool_stride_w,
        // Conv2
        conv2_out_channels,
        conv2_kernel_h,
        conv2_kernel_w,
        conv2_stride_h,
        conv2_stride_w,
        conv2_pad_h,
        conv2_pad_w,
        // Linear
        fc1_in_features,
        fc1_out_features,
        fc2_out_features,
        fc3_out_features
    );

    cudaDeviceSynchronize();

    // Validation with tolerance for fp16/bf16
    double rtol = 1e-1;
    double atol = 1e-1;
    bool ok = torch::allclose(output_gpu, output_ref, rtol, atol);

    if (ok) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
