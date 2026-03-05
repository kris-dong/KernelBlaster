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
#include <cmath>
#include <limits>
#include <cuda_runtime.h>
#include "cuda_model.cuh"

// Declaration only. Implement this in a separate .cu/.cuh as needed.
void launch_gpu_implementation(
    void* output,
    const void* input,
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t n_head,
    const void* c_attn_weight,
    const void* c_attn_bias,
    const void* c_proj_weight,
    const void* c_proj_bias,
    const void* attn_bias,    // causal mask buffer of shape (1, 1, max_seqlen, max_seqlen)
    int64_t max_seqlen,
    float attn_pdrop,
    float resid_pdrop
);

int main() {
    using namespace torch::indexing;

    // Problem parameters (from the provided Python code)
    const int64_t batch_size = 128;
    const int64_t max_seqlen = 1024;
    const int64_t seq_len = 512;
    const int64_t n_embd = 768;
    const int64_t n_head = 8;
    const float attn_pdrop = 0.0f;
    const float resid_pdrop = 0.0f;

    // Use fp16 (half) dtype for all tensors, on GPU
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);

    // Create model components (Linear layers) on GPU with fp16
    auto c_attn = torch::nn::Linear(torch::nn::LinearOptions(n_embd, 3 * n_embd).bias(true));
    auto c_proj = torch::nn::Linear(torch::nn::LinearOptions(n_embd, n_embd).bias(true));

    c_attn->to(torch::kCUDA, torch::kHalf);
    c_proj->to(torch::kCUDA, torch::kHalf);

    // Dropout modules (p=0.0, so they will be no-ops; keep for structural parity)
    auto attn_dropout = torch::nn::Dropout(torch::nn::DropoutOptions(attn_pdrop));
    auto resid_dropout = torch::nn::Dropout(torch::nn::DropoutOptions(resid_pdrop));
    attn_dropout->to(torch::kCUDA);
    resid_dropout->to(torch::kCUDA);
    attn_dropout->eval();
    resid_dropout->eval();

    // Causal bias mask buffer: shape (1, 1, max_seqlen, max_seqlen)
    auto bias = torch::tril(torch::ones({max_seqlen, max_seqlen}, options)).view({1, 1, max_seqlen, max_seqlen});

    // Input tensor on GPU
    auto x = torch::randn({batch_size, seq_len, n_embd}, options);

    // Reference implementation using libtorch
    // 1) Compute q, k, v
    auto x_proj = c_attn->forward(x); // [B, T, 3C]
    auto splits = x_proj.split(n_embd, /*dim=*/2);
    auto q = splits[0];
    auto k = splits[1];
    auto v = splits[2];

    const int64_t hs = n_embd / n_head;

    k = k.view({batch_size, seq_len, n_head, hs}).transpose(1, 2); // [B, nh, T, hs]
    q = q.view({batch_size, seq_len, n_head, hs}).transpose(1, 2); // [B, nh, T, hs]
    v = v.view({batch_size, seq_len, n_head, hs}).transpose(1, 2); // [B, nh, T, hs]

    // 2) Attention weights
    auto scale = 1.0f / std::sqrt(static_cast<float>(hs));
    auto att = torch::matmul(q, k.transpose(-2, -1)) * scale; // [B, nh, T, T]

    // 3) Apply causal mask
    auto causal = bias.index({Slice(), Slice(), Slice(None, seq_len), Slice(None, seq_len)}); // [1,1,T,T]
    att = att.masked_fill(causal.eq(0), -std::numeric_limits<float>::infinity());

    // 4) Softmax and dropout
    att = torch::softmax(att, /*dim=*/-1);
    att = attn_dropout->forward(att);

    // 5) Attention output
    auto y = torch::matmul(att, v); // [B, nh, T, hs]
    y = y.transpose(1, 2).contiguous().view({batch_size, seq_len, n_embd}); // [B, T, C]

    // 6) Output projection and residual dropout
    auto ref_output = resid_dropout->forward(c_proj->forward(y)).contiguous();

    // Prepare output tensor for GPU implementation (zero-initialized to avoid uninitialized memory)
    auto gpu_output = torch::zeros_like(ref_output, options);

    // Gather raw pointers to parameters and tensors for the GPU implementation
    void* out_ptr = gpu_output.data_ptr();
    const void* in_ptr = x.contiguous().data_ptr();
    const void* c_attn_w_ptr = c_attn->weight.contiguous().data_ptr();
    const void* c_attn_b_ptr = c_attn->bias.contiguous().data_ptr();
    const void* c_proj_w_ptr = c_proj->weight.contiguous().data_ptr();
    const void* c_proj_b_ptr = c_proj->bias.contiguous().data_ptr();
    const void* bias_ptr = bias.contiguous().data_ptr();

    // Launch the user-provided GPU implementation (to be implemented separately)
    launch_gpu_implementation(
        out_ptr,
        in_ptr,
        batch_size,
        seq_len,
        n_embd,
        n_head,
        c_attn_w_ptr,
        c_attn_b_ptr,
        c_proj_w_ptr,
        c_proj_b_ptr,
        bias_ptr,
        max_seqlen,
        attn_pdrop,
        resid_pdrop
    );

    // Validate results with torch::allclose
    // Since dtype is fp16, use rtol=1e-1, atol=1e-1
    const double rtol = 1e-1;
    const double atol = 1e-1;
    bool ok = torch::allclose(gpu_output, ref_output, rtol, atol);

    if (ok) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    // Always return 0
    return 0;
}
