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
#include "cuda_model.cuh"
#include <torch/torch.h>
#include <iostream>

// Declaration of the CUDA kernel launcher.
// x, mask, and output are pointers to GPU memory (float16).
// dim is the dimension along which to compute the masked cumsum.
void launch_gpu_implementation(
    void* output,
    void* x,
    void* mask,
    int64_t dim,
    int64_t batch_size,
    int64_t input_size
);

int main() {
    // Set device and dtype
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = torch::kFloat16;

    // Model parameters
    const int64_t batch_size = 128;
    const int64_t input_size = 4000;
    const int64_t dim = 1;

    // Prepare random inputs (on GPU)
    torch::Tensor x = torch::randn({batch_size, input_size}, torch::TensorOptions().dtype(dtype).device(device));
    torch::Tensor mask = torch::randint(0, 2, {batch_size, input_size}, torch::TensorOptions().dtype(torch::kBool).device(device));

    // Reference output using libtorch (masked cumsum)
    torch::Tensor masked_x = x * mask;
    torch::Tensor ref_output = torch::cumsum(masked_x, dim);

    // Allocate output tensor for CUDA kernel
    torch::Tensor output = torch::empty_like(ref_output);

    // Launch CUDA implementation
    launch_gpu_implementation(
        output.data_ptr(),
        x.data_ptr(),
        mask.data_ptr(),
        dim,
        batch_size,
        input_size
    );

    // Compare outputs
    bool passed = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
