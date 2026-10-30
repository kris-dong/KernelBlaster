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
    void* gemm_weight,
    void* gemm_bias,
    void* scale,
    void* bn_weight,
    void* bn_bias,
    void* bn_running_mean,
    void* bn_running_var,
    float bn_eps,
    int batch_size,
    int in_features,
    int out_features
);

struct ModelImpl : torch::nn::Module {
    torch::nn::Linear gemm{nullptr};
    torch::Tensor scale;
    torch::nn::BatchNorm1d bn{nullptr};

    ModelImpl(int in_feat, int out_feat, std::vector<int64_t> scale_shape, float eps = 1e-5, float momentum = 0.1)
        : gemm(register_module("gemm", torch::nn::Linear(in_feat, out_feat))),
          bn(register_module("bn", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(out_feat).eps(eps).momentum(momentum)))) {
        scale = register_parameter("scale", torch::randn(scale_shape));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = gemm(x);
        x = x * scale;
        x = bn(x);
        return x;
    }
};

TORCH_MODULE(Model);

int main() {
    const int batch_size = 128;
    const int in_features = 1024;
    const int out_features = 512;
    const std::vector<int64_t> scale_shape = {out_features};
    const float eps = 1e-5;
    const float momentum = 0.1;

    // Create and configure model
    Model model(in_features, out_features, scale_shape, eps, momentum);
    model->to(torch::kCUDA, torch::kFloat16);

    // Create input tensor
    auto input = torch::randn({batch_size, in_features}, 
        torch::dtype(torch::kFloat16).device(torch::kCUDA));

    // Reference forward pass
    auto ref_output = model->forward(input.clone());

    // Get parameter pointers
    auto gemm_weight = model->gemm->weight.data_ptr();
    auto gemm_bias = model->gemm->bias.data_ptr();
    auto scale = model->scale.data_ptr();
    auto bn_weight = model->bn->weight.data_ptr();
    auto bn_bias = model->bn->bias.data_ptr();
    auto bn_running_mean = model->bn->running_mean.data_ptr();
    auto bn_running_var = model->bn->running_var.data_ptr();
    float bn_eps = model->bn->options.eps();

    // Allocate output tensor
    auto output = torch::empty_like(ref_output);

    // Launch CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        gemm_weight,
        gemm_bias,
        scale,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        bn_eps,
        batch_size,
        in_features,
        out_features
    );

    // Verify results
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
