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

// Declaration for the GPU implementation
void launch_gpu_implementation(void* output, void* input, int64_t batch_size, int64_t dim);

// Helper function for L1 normalization (reference implementation)
torch::Tensor l1_normalize(const torch::Tensor& x) {
    // x: [batch_size, dim]
    // L1 norm along dim=1
    return x / torch::sum(torch::abs(x), /*dim=*/1, /*keepdim=*/true);
}

int main() {
    // Set the default dtype to float16 for all tensors
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device(torch::kCUDA);

    // Model input dimensions
    const int64_t batch_size = 16;
    const int64_t dim = 16384;

    // Generate input tensor on GPU with float16 dtype
    torch::Tensor x = torch::randn({batch_size, dim}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference output using libtorch (L1 normalization)
    torch::Tensor ref_output = l1_normalize(x);

    // Allocate tensor for GPU implementation output
    torch::Tensor gpu_output = torch::empty_like(ref_output);

    // Call the GPU implementation (pass pointers to device data)
    launch_gpu_implementation(
        gpu_output.data_ptr(),         // output
        x.data_ptr(),                  // input
        batch_size,
        dim
    );

    // Compare outputs using torch::allclose (rtol, atol depend on dtype)
    double rtol = 1e-1;
    double atol = 1e-1;

    bool passed = torch::allclose(gpu_output, ref_output, rtol, atol);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
