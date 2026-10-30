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
    torch::nn::GroupNorm group_norm{nullptr};
    torch::Tensor multiply_weight;

    Model(int64_t in_features, int64_t out_features, int64_t num_groups, const std::vector<int64_t>& multiply_weight_shape) {
        gemm = register_module("gemm", torch::nn::Linear(in_features, out_features));
        group_norm = register_module("group_norm", torch::nn::GroupNorm(num_groups, out_features));
        multiply_weight = register_parameter("multiply_weight", 
            torch::randn(multiply_weight_shape, torch::dtype(torch::kHalf).device(torch::kCUDA)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = gemm->forward(x);
        x = group_norm->forward(x);
        x = x * torch::sigmoid(x);
        x = x * multiply_weight;
        x = x * torch::sigmoid(x);
        return x;
    }
};

void launch_gpu_implementation(
    void* output, void* input,
    void* gemm_weight, void* gemm_bias,
    void* group_norm_weight, void* group_norm_bias,
    void* multiply_weight,
    int num_groups,
    int batch_size, int in_features, int out_features
);

int main() {
    const int batch_size = 128;
    const int in_features = 512;
    const int out_features = 1024;
    const int num_groups = 16;
    const std::vector<int64_t> multiply_weight_shape = {out_features};

    // Create model and move to CUDA with half precision
    auto model = std::make_shared<Model>(in_features, out_features, num_groups, multiply_weight_shape);
    model->to(torch::kCUDA, torch::kHalf);

    // Create input tensor
    auto input = torch::randn({batch_size, in_features}, 
                    torch::device(torch::kCUDA).dtype(torch::kHalf));
    
    // Run reference implementation
    auto ref_output = model->forward(input);

    // Get parameter pointers
    auto gemm_weight = model->gemm->weight;
    auto gemm_bias = model->gemm->bias;
    auto group_norm_weight = model->group_norm->weight;
    auto group_norm_bias = model->group_norm->bias;
    auto multiply_weight_param = model->multiply_weight;

    // Allocate output tensor
    auto output = torch::empty_like(ref_output);

    // Launch CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        gemm_weight.data_ptr(),
        gemm_bias.data_ptr(),
        group_norm_weight.data_ptr(),
        group_norm_bias.data_ptr(),
        multiply_weight_param.data_ptr(),
        num_groups,
        batch_size,
        in_features,
        out_features
    );

    // Verify results with fp16 tolerances
    if (torch::allclose(ref_output, output, /*rtol=*/1e-1, /*atol=*/1e-1)) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
