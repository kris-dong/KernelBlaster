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
    torch::nn::Linear matmul{nullptr};
    torch::nn::BatchNorm1d bn{nullptr};
    torch::Tensor bias;
    float divide_value;

    Model(int in_features, int out_features, float bn_eps, float bn_momentum, std::vector<int64_t> bias_shape, float divide_value)
        : matmul(torch::nn::LinearOptions(in_features, out_features)),
          bn(torch::nn::BatchNorm1dOptions(out_features).eps(bn_eps).momentum(bn_momentum)),
          divide_value(divide_value) {
        register_module("matmul", matmul);
        register_module("bn", bn);
        bias = register_parameter("bias", torch::randn(bias_shape, torch::dtype(torch::kHalf).device(torch::kCUDA)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = matmul(x);
        x = bn(x);
        x = x + bias;
        x = x / divide_value;
        x = x * torch::sigmoid(x);
        return x;
    }
};

void launch_gpu_implementation(
    void* output, void* input,
    void* matmul_weight, void* matmul_bias,
    void* bn_weight, void* bn_bias, void* bn_running_mean, void* bn_running_var,
    void* custom_bias,
    int in_features, int out_features,
    float bn_eps, float bn_momentum,
    const int64_t* bias_shape, int64_t bias_shape_len,
    float divide_value
);

int main() {
    const int batch_size = 128;
    const int in_features = 1024;
    const int out_features = 512;
    const float bn_eps = 1e-5;
    const float bn_momentum = 0.1;
    const std::vector<int64_t> bias_shape = {1};
    const float divide_value = 1.0f;

    auto model = std::make_shared<Model>(in_features, out_features, bn_eps, bn_momentum, bias_shape, divide_value);
    model->to(torch::kCUDA, torch::kHalf);
    model->eval();

    auto input = torch::randn({batch_size, in_features}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    auto output_ref = model->forward(input.clone());
    auto output = torch::empty_like(output_ref);

    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        model->matmul->weight.data_ptr(),
        model->matmul->bias.data_ptr(),
        model->bn->weight.data_ptr(),
        model->bn->bias.data_ptr(),
        model->bn->running_mean.data_ptr(),
        model->bn->running_var.data_ptr(),
        model->bias.data_ptr(),
        in_features,
        out_features,
        bn_eps,
        bn_momentum,
        bias_shape.data(),
        bias_shape.size(),
        divide_value
    );

    bool passed = torch::allclose(output_ref, output, /*rtol=*/1e-1, /*atol=*/1e-1);
    std::cout << (passed ? "passed" : "failed") << std::endl;

    return 0;
}
