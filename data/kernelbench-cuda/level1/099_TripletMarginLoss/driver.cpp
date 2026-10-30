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

void launch_gpu_implementation(void* output, void* anchor, void* positive, void* negative, float margin, int64_t batch_size, int64_t input_dim);

// Helper to select rtol/atol based on dtype
void select_fp_tolerances(const at::Tensor& t, double& rtol, double& atol) {
    if (t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16) {
        rtol = 1e-1;
        atol = 1e-1;
    } else {
        rtol = 1e-3;
        atol = 1e-3;
    }
}

int main() {
    // Set up parameters
    const int64_t batch_size = 128;
    const int64_t input_dim = 4096;
    const float margin = 1.0f;

    // Set default dtype and device
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Generate input tensors on GPU
    auto anchor   = torch::randn({batch_size, input_dim}, torch::TensorOptions().dtype(dtype).device(device));
    auto positive = torch::randn({batch_size, input_dim}, torch::TensorOptions().dtype(dtype).device(device));
    auto negative = torch::randn({batch_size, input_dim}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference output using libtorch
    auto loss_fn = torch::nn::TripletMarginLoss(torch::nn::TripletMarginLossOptions().margin(margin));
    auto ref_output = loss_fn(anchor, positive, negative);

    // Allocate output tensor for CUDA kernel
    auto output = torch::empty_like(ref_output);

    // Launch CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(), 
        anchor.data_ptr(), 
        positive.data_ptr(), 
        negative.data_ptr(),
        margin,
        batch_size,
        input_dim
    );

    // Compare outputs
    double rtol, atol;
    select_fp_tolerances(ref_output, rtol, atol);
    if (torch::allclose(output, ref_output, rtol, atol)) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }
    return 0;
}
