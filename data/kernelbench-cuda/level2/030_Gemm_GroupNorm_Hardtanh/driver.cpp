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

struct CppModel : torch::nn::Module {
    torch::nn::Linear gemm{nullptr};
    torch::nn::GroupNorm group_norm{nullptr};
    float hardtanh_min, hardtanh_max;

    CppModel(int in_features, int out_features, int num_groups, float hmin, float hmax) :
        gemm(torch::nn::LinearOptions(in_features, out_features)),
        group_norm(torch::nn::GroupNormOptions(num_groups, out_features)),
        hardtanh_min(hmin),
        hardtanh_max(hmax) {
        register_module("gemm", gemm);
        register_module("group_norm", group_norm);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = gemm->forward(x);
        x = group_norm->forward(x);
        x = torch::hardtanh_(x, hardtanh_min, hardtanh_max);
        return x;
    }
};

void launch_gpu_implementation(void* output, void* input, 
                              void* gemm_weight, void* gemm_bias,
                              void* gn_weight, void* gn_bias,
                              int64_t batch_size, int64_t in_features, int64_t out_features,
                              int num_groups, float hardtanh_min, float hardtanh_max);

int main() {
    const int batch_size = 128;
    const int in_features = 1024;
    const int out_features = 512;
    const int num_groups = 8;
    const float hardtanh_min = -2.0f;
    const float hardtanh_max = 2.0f;
    const float atol = 1e-1, rtol = 1e-1;

    // Create model and convert parameters to FP16 before moving to CUDA
    CppModel model(in_features, out_features, num_groups, hardtanh_min, hardtanh_max);
    model.to(torch::kFloat16);  // Convert parameters to FP16 first
    model.to(torch::kCUDA);     // Then move to CUDA device

    // Generate FP16 CUDA input
    torch::Tensor input = torch::randn({batch_size, in_features}, 
                                     torch::dtype(torch::kFloat16).device(torch::kCUDA));

    // Reference forward pass
    torch::Tensor ref_output = model.forward(input);

    // Get parameter pointers
    auto gemm_weight = model.gemm->weight.data_ptr();
    auto gemm_bias = model.gemm->bias.data_ptr();
    auto gn_weight = model.group_norm->weight.data_ptr();
    auto gn_bias = model.group_norm->bias.data_ptr();

    // Allocate output buffer
    torch::Tensor gpu_output = torch::empty_like(ref_output);

    // Launch kernel
    launch_gpu_implementation(gpu_output.data_ptr(), input.data_ptr(),
                             gemm_weight, gemm_bias,
                             gn_weight, gn_bias,
                             batch_size, in_features, out_features,
                             num_groups, hardtanh_min, hardtanh_max);

    // Verify results
    if (torch::allclose(gpu_output, ref_output, rtol, atol)) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
