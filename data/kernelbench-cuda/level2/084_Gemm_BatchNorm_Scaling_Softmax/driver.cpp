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

struct Model : torch::nn::Module {
    torch::nn::Linear gemm{nullptr};
    torch::nn::BatchNorm1d bn{nullptr};
    torch::Tensor scale;

    Model(int64_t in_features, int64_t out_features, double bn_eps, double bn_momentum, const std::vector<int64_t>& scale_shape) {
        gemm = register_module("gemm", torch::nn::Linear(in_features, out_features));
        bn = register_module("bn", torch::nn::BatchNorm1d(
            torch::nn::BatchNorm1dOptions(out_features)
            .eps(bn_eps)
            .momentum(bn_momentum)));
        scale = register_parameter("scale", torch::ones(scale_shape));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = gemm->forward(x);
        x = bn->forward(x);
        x = scale * x;
        x = torch::softmax(x, 1);
        return x;
    }
};

void launch_gpu_implementation(
    void* output, void* input,
    void* gemm_weight, void* gemm_bias,
    void* bn_weight, void* bn_bias,
    void* bn_running_mean, void* bn_running_var,
    void* scale,
    int batch_size, int in_features, int out_features,
    float bn_eps
);

int main() {
    // Initialize parameters
    const int batch_size = 128;
    const int in_features = 1024;
    const int out_features = 512;
    const float bn_eps = 1e-5f;
    const float bn_momentum = 0.1f;
    const std::vector<int64_t> scale_shape = {1};

    // Create and configure model
    auto model = std::make_shared<Model>(in_features, out_features, bn_eps, bn_momentum, scale_shape);
    model->to(torch::kCUDA, torch::kHalf);

    // Create input tensor
    auto input = torch::randn({batch_size, in_features}, 
                            torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Run reference implementation
    auto ref_output = model->forward(input);

    // Get parameter pointers
    auto gemm_weight = model->gemm->weight;
    auto gemm_bias = model->gemm->bias;
    auto bn_weight = model->bn->weight;
    auto bn_bias = model->bn->bias;
    auto bn_running_mean = model->bn->running_mean;
    auto bn_running_var = model->bn->running_var;
    auto scale = model->scale;

    // Allocate output tensor
    auto output = torch::empty_like(ref_output);

    // Launch CUDA kernel
    launch_gpu_implementation(
        output.data_ptr(), input.data_ptr(),
        gemm_weight.data_ptr(), gemm_bias.data_ptr(),
        bn_weight.data_ptr(), bn_bias.data_ptr(),
        bn_running_mean.data_ptr(), bn_running_var.data_ptr(),
        scale.data_ptr(),
        batch_size, in_features, out_features,
        bn_eps
    );

    // Verify results
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
