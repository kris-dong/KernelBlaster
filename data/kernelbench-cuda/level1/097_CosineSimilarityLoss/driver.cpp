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

// Declaration for the CUDA implementation
void launch_gpu_implementation(
    void* output,
    void* predictions,
    void* targets,
    int64_t batch_size,
    int64_t input_dim
);

int main() {
    torch::Device device(torch::kCUDA);

    using scalar_t = at::Half;
    const float rtol = 1e-1f;
    const float atol = 1e-1f;

    constexpr int64_t batch_size = 128;
    constexpr int64_t input_dim = 4096;

    auto predictions = torch::randn({batch_size, input_dim}, torch::dtype(torch::kFloat16).device(device));
    auto targets     = torch::randn({batch_size, input_dim}, torch::dtype(torch::kFloat16).device(device));

    // Reference PyTorch implementation (Cosine Similarity Loss)
    auto cosine_sim = torch::nn::functional::cosine_similarity(
        predictions,
        targets,
        torch::nn::functional::CosineSimilarityFuncOptions().dim(1)
    );
    auto ref_output = torch::mean(1 - cosine_sim).reshape({1}); // shape [1], float16, CUDA

    // Allocate and zero output tensor for CUDA kernel
    auto output = torch::zeros({1}, torch::dtype(torch::kFloat16).device(device));

    // Launch CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),
        predictions.data_ptr(),
        targets.data_ptr(),
        batch_size,
        input_dim
    );

    // Defensive: check for NaN/Inf in kernel output
    bool has_nan = torch::isnan(output.to(torch::kFloat32)).any().item<bool>();
    bool has_inf = torch::isinf(output.to(torch::kFloat32)).any().item<bool>();

    // Compare the outputs using torch::allclose (convert to float32 for robust comparison)
    bool numerics_ok = torch::allclose(
        output.to(torch::kFloat32),
        ref_output.to(torch::kFloat32),
        rtol,
        atol
    );

    if (!has_nan && !has_inf && numerics_ok) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
        std::cout << "Reference output: " << ref_output.to(torch::kFloat32).item<float>() << std::endl;
        std::cout << "CUDA output: " << output.to(torch::kFloat32).item<float>() << std::endl;
        if (has_nan) std::cout << "CUDA output contains NaN!" << std::endl;
        if (has_inf) std::cout << "CUDA output contains Inf!" << std::endl;
    }

    return 0;
}
