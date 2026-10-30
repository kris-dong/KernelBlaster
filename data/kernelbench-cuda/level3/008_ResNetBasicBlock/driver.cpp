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
#include <cuda_runtime.h>
#include "cuda_model.cuh"

// Declaration only. Do not implement here.
void launch_gpu_implementation(
    void* output,
    const void* input,
    int64_t N, int64_t C_in, int64_t H, int64_t W,
    int64_t C_out,
    // conv1 (3x3)
    const void* conv1_weight,
    int64_t conv1_stride,
    int64_t conv1_padding,
    int64_t conv1_kernel,
    // bn1
    const void* bn1_weight,
    const void* bn1_bias,
    const void* bn1_running_mean,
    const void* bn1_running_var,
    double bn1_eps,
    // conv2 (3x3)
    const void* conv2_weight,
    int64_t conv2_stride,
    int64_t conv2_padding,
    int64_t conv2_kernel,
    // bn2
    const void* bn2_weight,
    const void* bn2_bias,
    const void* bn2_running_mean,
    const void* bn2_running_var,
    double bn2_eps,
    // downsample conv (1x1)
    const void* downsample_conv_weight,
    int64_t downsample_conv_stride,
    int64_t downsample_conv_padding,
    int64_t downsample_conv_kernel,
    // downsample bn
    const void* downsample_bn_weight,
    const void* downsample_bn_bias,
    const void* downsample_bn_running_mean,
    const void* downsample_bn_running_var,
    double downsample_bn_eps
);

struct ModelImpl : torch::nn::Module {
    // Layers
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::ReLU relu{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::BatchNorm2d bn2{nullptr};

    // Downsample path (conv + bn)
    torch::nn::Conv2d downsample_conv{nullptr};
    torch::nn::BatchNorm2d downsample_bn{nullptr};

    int64_t stride_;

    ModelImpl(int64_t in_channels, int64_t out_channels, int64_t stride = 1)
        : conv1(torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1).bias(false)),
          bn1(torch::nn::BatchNorm2dOptions(out_channels)),
          relu(torch::nn::ReLUOptions(true)),
          conv2(torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(1).bias(false)),
          bn2(torch::nn::BatchNorm2dOptions(out_channels)),
          downsample_conv(torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride).bias(false)),
          downsample_bn(torch::nn::BatchNorm2dOptions(out_channels)),
          stride_(stride) {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("relu", relu);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        register_module("downsample_conv", downsample_conv);
        register_module("downsample_bn", downsample_bn);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto identity = x;

        auto out = conv1->forward(x);
        out = bn1->forward(out);
        out = relu->forward(out);

        out = conv2->forward(out);
        out = bn2->forward(out);

        identity = downsample_bn->forward(downsample_conv->forward(x));

        out = out + identity;
        out = relu->forward(out);
        return out;
    }
};
TORCH_MODULE(Model);

int main() {
    if (!torch::cuda::is_available()) {
        std::cout << "CUDA is not available. This test expects a CUDA device." << std::endl;
        // Still exit with code 0 as requested
        return 0;
    }

    // Problem configuration
    const int64_t in_channels = 3;
    const int64_t out_channels = 64;
    const int64_t stride = 1;
    const int64_t batch_size = 10;
    const int64_t H = 224;
    const int64_t W = 224;

    torch::Device device(torch::kCUDA);
    auto dtype = torch::kHalf; // torch.set_default_dtype(torch.float16)

    // Build model and move to GPU with fp16
    Model model(in_channels, out_channels, stride);
    model->to(device, dtype);
    model->eval(); // Use running stats for BatchNorm (inference mode)

    // Create input on GPU with fp16
    auto input = torch::randn({batch_size, in_channels, H, W}, torch::TensorOptions().device(device).dtype(dtype));

    // Reference output using libtorch
    auto ref_output = model->forward(input);

    // Prepare output buffer for GPU implementation (initialized to zeros)
    auto gpu_output = torch::zeros_like(ref_output);

    // Extract pointers to parameters/buffers (all on GPU, fp16)
    // conv1
    void* conv1_weight = static_cast<void*>(model->conv1->weight.data_ptr<at::Half>());
    // bn1
    void* bn1_weight = static_cast<void*>(model->bn1->weight.data_ptr<at::Half>());
    void* bn1_bias   = static_cast<void*>(model->bn1->bias.data_ptr<at::Half>());
    void* bn1_running_mean = static_cast<void*>(model->bn1->running_mean.data_ptr<at::Half>());
    void* bn1_running_var  = static_cast<void*>(model->bn1->running_var.data_ptr<at::Half>());
    double bn1_eps = model->bn1->options.eps();

    // conv2
    void* conv2_weight = static_cast<void*>(model->conv2->weight.data_ptr<at::Half>());
    // bn2
    void* bn2_weight = static_cast<void*>(model->bn2->weight.data_ptr<at::Half>());
    void* bn2_bias   = static_cast<void*>(model->bn2->bias.data_ptr<at::Half>());
    void* bn2_running_mean = static_cast<void*>(model->bn2->running_mean.data_ptr<at::Half>());
    void* bn2_running_var  = static_cast<void*>(model->bn2->running_var.data_ptr<at::Half>());
    double bn2_eps = model->bn2->options.eps();

    // downsample conv and bn
    void* downsample_conv_weight = static_cast<void*>(model->downsample_conv->weight.data_ptr<at::Half>());
    void* downsample_bn_weight   = static_cast<void*>(model->downsample_bn->weight.data_ptr<at::Half>());
    void* downsample_bn_bias     = static_cast<void*>(model->downsample_bn->bias.data_ptr<at::Half>());
    void* downsample_bn_running_mean = static_cast<void*>(model->downsample_bn->running_mean.data_ptr<at::Half>());
    void* downsample_bn_running_var  = static_cast<void*>(model->downsample_bn->running_var.data_ptr<at::Half>());
    double downsample_bn_eps = model->downsample_bn->options.eps();

    // Launch the user-provided GPU implementation
    launch_gpu_implementation(
        static_cast<void*>(gpu_output.data_ptr<at::Half>()),
        static_cast<const void*>(input.data_ptr<at::Half>()),
        /*N*/ batch_size, /*C_in*/ in_channels, /*H*/ H, /*W*/ W,
        /*C_out*/ out_channels,
        // conv1 (3x3)
        conv1_weight,
        /*conv1_stride*/ stride,
        /*conv1_padding*/ 1,
        /*conv1_kernel*/ 3,
        // bn1
        bn1_weight,
        bn1_bias,
        bn1_running_mean,
        bn1_running_var,
        bn1_eps,
        // conv2 (3x3)
        conv2_weight,
        /*conv2_stride*/ 1,
        /*conv2_padding*/ 1,
        /*conv2_kernel*/ 3,
        // bn2
        bn2_weight,
        bn2_bias,
        bn2_running_mean,
        bn2_running_var,
        bn2_eps,
        // downsample conv (1x1)
        downsample_conv_weight,
        /*downsample_conv_stride*/ stride,
        /*downsample_conv_padding*/ 0,
        /*downsample_conv_kernel*/ 1,
        // downsample bn
        downsample_bn_weight,
        downsample_bn_bias,
        downsample_bn_running_mean,
        downsample_bn_running_var,
        downsample_bn_eps
    );

    // Compare outputs using torch::allclose with fp16 tolerances (rtol=1e-1, atol=1e-1)
    bool ok = torch::allclose(ref_output, gpu_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    if (ok) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    // Always exit with return code 0
    return 0;
}
