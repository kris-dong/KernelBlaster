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
#include "cuda_model.cuh"

void launch_gpu_implementation(
    void* output,           // float16* output, shape: (batch_size, dim)
    void* input,            // float16* input, shape: (batch_size, dim)
    int64_t batch_size,
    int64_t dim
);

int main() {
    torch::Device device(torch::kCUDA);

    const int64_t batch_size = 16;
    const int64_t dim = 16384;

    // Use float16 for all tensors
    torch::Tensor input = torch::randn({batch_size, dim}, torch::TensorOptions().dtype(torch::kFloat16).device(device));
    torch::Tensor ref_output = torch::softmax(input, 1);

    // Fill output with a known wrong value to ensure false positive is impossible
    torch::Tensor gpu_output = torch::full({batch_size, dim}, 12345.0f, torch::TensorOptions().dtype(torch::kFloat16).device(device));

    launch_gpu_implementation(
        gpu_output.data_ptr<at::Half>(),
        input.data_ptr<at::Half>(),
        batch_size,
        dim
    );

    torch::cuda::synchronize();

    // For fp16, use rtol=1e-1, atol=1e-1
    bool is_close = torch::allclose(gpu_output, ref_output, 1e-1, 1e-1);

    if (is_close) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
        // Optionally print a small slice for debugging
        std::cout << "Reference (first row, first 8): " << ref_output[0].slice(0,0,8) << std::endl;
        std::cout << "GPU Output (first row, first 8): " << gpu_output[0].slice(0,0,8) << std::endl;
    }

    return 0;
}
