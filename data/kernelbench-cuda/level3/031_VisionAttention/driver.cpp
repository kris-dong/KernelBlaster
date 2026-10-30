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
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "cuda_model.cuh"

// Declaration only - to be implemented elsewhere.
// Pointers are device pointers (CUDA), with dtype = torch::kHalf (fp16).
void launch_gpu_implementation(
    void* output,            // [B, C, H, W] fp16
    void* input,             // [B, C, H, W] fp16
    void* in_proj_weight,    // [3*E, E] fp16
    void* in_proj_bias,      // [3*E] fp16
    void* out_proj_weight,   // [E, E] fp16
    void* out_proj_bias,     // [E] fp16
    void* ln_weight,         // [E] fp16 (LayerNorm gamma)
    void* ln_bias,           // [E] fp16 (LayerNorm beta)
    int64_t batch_size,
    int64_t channels,        // == embed_dim
    int64_t height,
    int64_t width,
    int64_t embed_dim,
    int64_t num_heads
);

int main() {
    try {
        if (!torch::cuda::is_available()) {
            std::cout << "CUDA is not available, but proceeding as requested (test expects CUDA tensors)." << std::endl;
        }

        // Problem parameters from the provided Python snippet
        const int64_t embed_dim = 128;
        const int64_t num_heads = 4;
        const int64_t batch_size = 2;
        const int64_t num_channels = embed_dim;
        const int64_t image_height = 128;
        const int64_t image_width = 128;

        const auto device = torch::kCUDA;
        const auto dtype = torch::kHalf; // torch.set_default_dtype(torch.float16)

        // Create the modules: MultiheadAttention and LayerNorm
        auto mha = torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(embed_dim, num_heads));
        auto ln = torch::nn::LayerNorm(
            torch::nn::LayerNormOptions(std::vector<int64_t>({embed_dim})));

        // Move modules to CUDA and fp16
        mha->to(device, dtype);
        ln->to(device, dtype);

        // Create input tensor on GPU with fp16 dtype
        auto options = torch::TensorOptions().device(device).dtype(dtype);
        auto x = torch::randn({batch_size, num_channels, image_height, image_width}, options);

        // Reference implementation (libtorch) to match the given Python model
        // Reshape: (B, C, H, W) -> (S, B, E) where S = H*W, E = embed_dim
        const int64_t seq_len = image_height * image_width;
        auto x_seq = x.view({batch_size, num_channels, seq_len}).permute({2, 0, 1}); // (S, B, E)

        // MultiheadAttention: self-attention
        auto attn_out_tuple = mha->forward(x_seq, x_seq, x_seq);
        auto attn_output = std::get<0>(attn_out_tuple); // (S, B, E)

        // Residual + LayerNorm over last dimension (E)
        auto x_res = attn_output + x_seq;
        auto x_norm = ln->forward(x_res); // (S, B, E)

        // Reshape back to (B, C, H, W)
        auto ref_output = x_norm.permute({1, 2, 0}).contiguous().view({batch_size, num_channels, image_height, image_width});

        // Allocate output tensor for GPU implementation (initialized to zeros to avoid uninitialized memory issues)
        auto gpu_output = torch::zeros_like(ref_output);

        // Collect parameter tensors to pass as device pointers
        auto in_proj_weight = mha->in_proj_weight;             // [3*E, E]
        auto in_proj_bias   = mha->in_proj_bias;               // [3*E]
        auto out_proj_weight = mha->out_proj->weight;          // [E, E]
        auto out_proj_bias   = mha->out_proj->bias;            // [E]
        auto ln_weight = ln->weight;                           // [E]
        auto ln_bias   = ln->bias;                             // [E]

        // Ensure parameters are on CUDA and correct dtype
        in_proj_weight = in_proj_weight.to(options);
        in_proj_bias   = in_proj_bias.to(options);
        out_proj_weight = out_proj_weight.to(options);
        out_proj_bias   = out_proj_bias.to(options);
        ln_weight = ln_weight.to(options);
        ln_bias   = ln_bias.to(options);

        // Launch user-implemented CUDA kernel via declaration
        launch_gpu_implementation(
            gpu_output.data_ptr(),
            x.data_ptr(),
            in_proj_weight.data_ptr(),
            in_proj_bias.data_ptr(),
            out_proj_weight.data_ptr(),
            out_proj_bias.data_ptr(),
            ln_weight.data_ptr(),
            ln_bias.data_ptr(),
            batch_size,
            num_channels,
            image_height,
            image_width,
            embed_dim,
            num_heads
        );

        // Validate results using torch::allclose with fp16 tolerances (rtol=1e-1, atol=1e-1)
        bool ok = torch::allclose(ref_output, gpu_output, /*rtol=*/1e-1, /*atol=*/1e-1);
        std::cout << (ok ? "passed" : "failed") << std::endl;

    } catch (const c10::Error& e) {
        std::cout << "failed" << std::endl;
        std::cerr << "Exception: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "failed" << std::endl;
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    // Always exit with return code 0 as requested
    return 0;
}
