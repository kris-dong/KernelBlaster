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
#include <cuda_runtime.h>
#include <iostream>
#include "cuda_model.cuh"

// Declaration only. Do not implement here.
void launch_gpu_implementation(
    void* output,
    const void* input,
    int64_t batch,
    int64_t in_channels,
    int64_t in_h,
    int64_t in_w,
    int64_t out_channels,
    int64_t hidden_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    bool use_residual,
    // expand conv params
    const void* expand_conv_weight,
    // expand BN params
    const void* expand_bn_weight,
    const void* expand_bn_bias,
    const void* expand_bn_running_mean,
    const void* expand_bn_running_var,
    double expand_bn_eps,
    // depthwise conv params
    const void* depthwise_conv_weight,
    const void* depthwise_bn_weight,
    const void* depthwise_bn_bias,
    const void* depthwise_bn_running_mean,
    const void* depthwise_bn_running_var,
    double depthwise_bn_eps,
    // project conv params
    const void* project_conv_weight,
    const void* project_bn_weight,
    const void* project_bn_bias,
    const void* project_bn_running_mean,
    const void* project_bn_running_var,
    double project_bn_eps
);

int main() {
    // Hyperparameters from the Python code
    const int64_t batch_size = 10;
    const int64_t in_channels = 112;
    const int64_t out_channels = 192;
    const int64_t kernel_size = 5;
    const int64_t stride = 2;
    const int64_t expand_ratio = 6;
    const int64_t hidden_dim = in_channels * expand_ratio;
    const int64_t padding = (kernel_size - 1) / 2;
    const int64_t H = 224;
    const int64_t W = 224;

    const bool use_residual = (stride == 1 && in_channels == out_channels);

    torch::Device device(torch::kCUDA);
    auto dtype = torch::kHalf; // torch.set_default_dtype(torch.float16)

    // Build MBConv block components in libtorch (CUDA + fp16)
    auto expand_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, hidden_dim, 1)
                                         .stride(1).padding(0).bias(false));
    auto expand_bn   = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(hidden_dim));
    auto relu6_1     = torch::nn::ReLU6(torch::nn::ReLU6Options(true));

    auto depthwise_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(hidden_dim, hidden_dim, kernel_size)
                                            .stride(stride).padding(padding).groups(hidden_dim).bias(false));
    auto depthwise_bn   = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(hidden_dim));
    auto relu6_2        = torch::nn::ReLU6(torch::nn::ReLU6Options(true));

    auto project_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(hidden_dim, out_channels, 1)
                                          .stride(1).padding(0).bias(false));
    auto project_bn   = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels));

    // Move modules to GPU and half precision
    expand_conv->to(device, dtype);
    expand_bn->to(device, dtype);
    relu6_1->to(device);

    depthwise_conv->to(device, dtype);
    depthwise_bn->to(device, dtype);
    relu6_2->to(device);

    project_conv->to(device, dtype);
    project_bn->to(device, dtype);

    // Use eval mode so BatchNorm uses running stats (deterministic behavior for testing)
    expand_bn->eval();
    depthwise_bn->eval();
    project_bn->eval();

    // Input tensor on GPU in fp16
    auto options = torch::TensorOptions().device(device).dtype(dtype);
    auto input = torch::randn({batch_size, in_channels, H, W}, options);

    // Reference forward (libtorch)
    auto x = input;
    x = expand_conv->forward(x);
    x = expand_bn->forward(x);
    x = relu6_1->forward(x);

    x = depthwise_conv->forward(x);
    x = depthwise_bn->forward(x);
    x = relu6_2->forward(x);

    x = project_conv->forward(x);
    x = project_bn->forward(x);

    if (use_residual) {
        x = x + input;
    }
    auto ref_output = x.contiguous();

    // Prepare output buffer for GPU implementation (initialized to zeros)
    auto gpu_output = torch::zeros_like(ref_output);

    // Gather parameter tensors (ensure contiguous and on CUDA)
    auto expand_w = expand_conv->weight.contiguous();
    auto expand_bn_weight = expand_bn->weight.contiguous();
    auto expand_bn_bias = expand_bn->bias.contiguous();
    auto expand_bn_running_mean = expand_bn->running_mean.contiguous();
    auto expand_bn_running_var = expand_bn->running_var.contiguous();
    double expand_bn_eps = expand_bn->options.eps();

    auto depthwise_w = depthwise_conv->weight.contiguous();
    auto depthwise_bn_weight = depthwise_bn->weight.contiguous();
    auto depthwise_bn_bias = depthwise_bn->bias.contiguous();
    auto depthwise_bn_running_mean = depthwise_bn->running_mean.contiguous();
    auto depthwise_bn_running_var = depthwise_bn->running_var.contiguous();
    double depthwise_bn_eps = depthwise_bn->options.eps();

    auto project_w = project_conv->weight.contiguous();
    auto project_bn_weight = project_bn->weight.contiguous();
    auto project_bn_bias = project_bn->bias.contiguous();
    auto project_bn_running_mean = project_bn->running_mean.contiguous();
    auto project_bn_running_var = project_bn->running_var.contiguous();
    double project_bn_eps = project_bn->options.eps();

    // Launch user-implemented CUDA kernel (declaration only above)
    launch_gpu_implementation(
        gpu_output.data_ptr(),
        input.data_ptr(),
        batch_size,
        in_channels,
        H,
        W,
        out_channels,
        hidden_dim,
        kernel_size,
        stride,
        padding,
        use_residual,
        // expand conv
        expand_w.data_ptr(),
        // expand BN
        expand_bn_weight.data_ptr(),
        expand_bn_bias.data_ptr(),
        expand_bn_running_mean.data_ptr(),
        expand_bn_running_var.data_ptr(),
        expand_bn_eps,
        // depthwise conv
        depthwise_w.data_ptr(),
        depthwise_bn_weight.data_ptr(),
        depthwise_bn_bias.data_ptr(),
        depthwise_bn_running_mean.data_ptr(),
        depthwise_bn_running_var.data_ptr(),
        depthwise_bn_eps,
        // project conv
        project_w.data_ptr(),
        project_bn_weight.data_ptr(),
        project_bn_bias.data_ptr(),
        project_bn_running_mean.data_ptr(),
        project_bn_running_var.data_ptr(),
        project_bn_eps
    );

    // Compare results
    double rtol = (ref_output.dtype() == torch::kFloat16 || ref_output.dtype() == torch::kBFloat16) ? 1e-1 : 1e-3;
    double atol = rtol;

    bool ok = torch::allclose(ref_output, gpu_output, rtol, atol);
    if (ok) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0; // Always exit 0
}
