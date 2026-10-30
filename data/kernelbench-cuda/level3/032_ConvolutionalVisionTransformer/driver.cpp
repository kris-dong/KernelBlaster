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
#include <cuda_runtime.h>
#include "cuda_model.cuh"

// Declaration only. Do not provide an implementation here.
void launch_gpu_implementation(
    void* output,                       // [B, num_classes], dtype = fp16
    void* input,                        // [B, C, H, W], dtype = fp16
    // conv1
    void* conv1_weight,                 // [embed_dim, in_channels, patch, patch], fp16
    void* conv1_bias,                   // [embed_dim], fp16
    // linear projection
    void* linear_proj_weight,           // [embed_dim, embed_dim*(H/patch)*(W/patch)], fp16
    void* linear_proj_bias,             // [embed_dim], fp16
    // cls token
    void* cls_token,                    // [1, 1, embed_dim], fp16
    // Transformer layers parameter pointer arrays (size = num_layers)
    void** attn_in_proj_weight,         // each [3*embed_dim, embed_dim], fp16
    void** attn_in_proj_bias,           // each [3*embed_dim], fp16
    void** attn_out_proj_weight,        // each [embed_dim, embed_dim], fp16
    void** attn_out_proj_bias,          // each [embed_dim], fp16
    void** ff1_weight,                  // each [ff_dim, embed_dim], fp16
    void** ff1_bias,                    // each [ff_dim], fp16
    void** ff2_weight,                  // each [embed_dim, ff_dim], fp16
    void** ff2_bias,                    // each [embed_dim], fp16
    void** norm1_weight,                // each [embed_dim], fp16
    void** norm1_bias,                  // each [embed_dim], fp16
    void** norm2_weight,                // each [embed_dim], fp16
    void** norm2_bias,                  // each [embed_dim], fp16
    // final classifier
    void* fc_weight,                    // [num_classes, embed_dim], fp16
    void* fc_bias,                      // [num_classes], fp16
    // problem sizes / hyperparameters
    int64_t batch_size,
    int64_t in_channels,
    int64_t image_h,
    int64_t image_w,
    int64_t patch_size,
    int64_t embed_dim,
    int64_t num_heads,
    int64_t num_layers,
    int64_t ff_dim,
    int64_t num_classes
);

int main() {
    // Hyperparameters mirroring the Python snippet
    const int64_t batch_size = 10;
    const int64_t image_size = 32;
    const int64_t embed_dim = 128;
    const int64_t in_channels = 3;
    const int64_t num_heads = 4;
    const int64_t num_classes = 1000;
    const int64_t patch_size = 4;     // default from the Python model
    const int64_t num_layers = 6;     // default from the Python model
    const float mlp_ratio = 4.0f;
    const int64_t ff_dim = static_cast<int64_t>(embed_dim * mlp_ratio);

    torch::Device device(torch::kCUDA);
    auto dtype = torch::kHalf; // torch.set_default_dtype(torch.float16)

    // Create input on GPU with fp16
    auto input = torch::randn({batch_size, in_channels, image_size, image_size},
                              torch::TensorOptions().device(device).dtype(dtype));

    // Build the model components in LibTorch (all on GPU, dtype fp16)
    // conv1: Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    auto conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, embed_dim, patch_size).stride(patch_size).bias(true));
    conv1->to(device, dtype);

    // linear_proj: Linear(embed_dim * (32/patch_size)^2, embed_dim)
    const int64_t patches_h = image_size / patch_size;
    const int64_t patches_w = image_size / patch_size;
    const int64_t flattened_size = embed_dim * patches_h * patches_w;
    auto linear_proj = torch::nn::Linear(torch::nn::LinearOptions(flattened_size, embed_dim).bias(true));
    linear_proj->to(device, dtype);

    // Transformer layers list (older libtorch may not support batch_first; use S,B,E)
    std::vector<torch::nn::TransformerEncoderLayer> layers;
    layers.reserve(num_layers);
    for (int64_t i = 0; i < num_layers; ++i) {
        auto opts = torch::nn::TransformerEncoderLayerOptions(embed_dim, num_heads)
                        .dim_feedforward(ff_dim)
                        .dropout(0.0);
        auto layer = torch::nn::TransformerEncoderLayer(opts);
        layer->to(device, dtype);
        layers.push_back(layer);
    }

    // cls_token parameter: initialized to zeros like Python
    auto cls_token = torch::zeros({1, 1, embed_dim}, torch::TensorOptions().device(device).dtype(dtype));

    // final classifier
    auto fc_out = torch::nn::Linear(torch::nn::LinearOptions(embed_dim, num_classes).bias(true));
    fc_out->to(device, dtype);

    // Reference forward pass (LibTorch)
    using namespace torch::indexing;

    // x = conv1(x)
    auto x = conv1->forward(input);
    // x = flatten(x)
    x = x.flatten(1);
    // x = linear_proj(x)
    x = linear_proj->forward(x);
    // Add cls token
    auto cls_tokens = cls_token.expand({x.size(0), 1, x.size(1)}); // (B, 1, E)
    x = torch::cat({cls_tokens, x.unsqueeze(1)}, 1);               // (B, 2, E)

    // Transformer layers expect (S, B, E) for older libtorch without batch_first
    x = x.transpose(0, 1); // (S=2, B, E)
    for (auto& layer : layers) {
        x = layer->forward(x);
    }
    x = x.transpose(0, 1); // back to (B, S=2, E)

    // x = x[:, 0]
    x = x.index({Slice(), 0});
    // x = fc_out(x)
    auto ref_output = fc_out->forward(x).contiguous();

    // Prepare output tensor for GPU implementation
    auto gpu_output = torch::zeros_like(ref_output); // initialize with zeros

    // Collect parameter pointers for kernel call
    // Conv1
    void* conv1_w_ptr = conv1->weight.data_ptr();
    void* conv1_b_ptr = conv1->bias.defined() ? conv1->bias.data_ptr() : nullptr;

    // Linear projection
    void* lin_proj_w_ptr = linear_proj->weight.data_ptr();
    void* lin_proj_b_ptr = linear_proj->bias.defined() ? linear_proj->bias.data_ptr() : nullptr;

    // cls token
    void* cls_token_ptr = cls_token.data_ptr();

    // Transformer layers params as arrays of pointers
    std::vector<void*> attn_in_proj_weight_ptrs(num_layers);
    std::vector<void*> attn_in_proj_bias_ptrs(num_layers);
    std::vector<void*> attn_out_proj_weight_ptrs(num_layers);
    std::vector<void*> attn_out_proj_bias_ptrs(num_layers);
    std::vector<void*> ff1_weight_ptrs(num_layers);
    std::vector<void*> ff1_bias_ptrs(num_layers);
    std::vector<void*> ff2_weight_ptrs(num_layers);
    std::vector<void*> ff2_bias_ptrs(num_layers);
    std::vector<void*> norm1_weight_ptrs(num_layers);
    std::vector<void*> norm1_bias_ptrs(num_layers);
    std::vector<void*> norm2_weight_ptrs(num_layers);
    std::vector<void*> norm2_bias_ptrs(num_layers);

    for (int64_t i = 0; i < num_layers; ++i) {
        auto& layer = layers[i];
        // Self-attention parameters
        auto self_attn = layer->self_attn;
        attn_in_proj_weight_ptrs[i] = self_attn->in_proj_weight.data_ptr();
        attn_in_proj_bias_ptrs[i]   = self_attn->in_proj_bias.defined() ? self_attn->in_proj_bias.data_ptr() : nullptr;
        attn_out_proj_weight_ptrs[i]= self_attn->out_proj->weight.data_ptr();
        attn_out_proj_bias_ptrs[i]  = self_attn->out_proj->bias.defined() ? self_attn->out_proj->bias.data_ptr() : nullptr;

        // Feedforward layers
        ff1_weight_ptrs[i] = layer->linear1->weight.data_ptr();
        ff1_bias_ptrs[i]   = layer->linear1->bias.defined() ? layer->linear1->bias.data_ptr() : nullptr;
        ff2_weight_ptrs[i] = layer->linear2->weight.data_ptr();
        ff2_bias_ptrs[i]   = layer->linear2->bias.defined() ? layer->linear2->bias.data_ptr() : nullptr;

        // LayerNorms
        norm1_weight_ptrs[i] = layer->norm1->weight.defined() ? layer->norm1->weight.data_ptr() : nullptr;
        norm1_bias_ptrs[i]   = layer->norm1->bias.defined()   ? layer->norm1->bias.data_ptr()   : nullptr;
        norm2_weight_ptrs[i] = layer->norm2->weight.defined() ? layer->norm2->weight.data_ptr() : nullptr;
        norm2_bias_ptrs[i]   = layer->norm2->bias.defined()   ? layer->norm2->bias.data_ptr()   : nullptr;
    }

    // Final classifier
    void* fc_w_ptr = fc_out->weight.data_ptr();
    void* fc_b_ptr = fc_out->bias.defined() ? fc_out->bias.data_ptr() : nullptr;

    // Launch the CUDA implementation
    launch_gpu_implementation(
        gpu_output.data_ptr(),
        input.data_ptr(),
        // conv1
        conv1_w_ptr,
        conv1_b_ptr,
        // linear projection
        lin_proj_w_ptr,
        lin_proj_b_ptr,
        // cls token
        cls_token_ptr,
        // transformer layers arrays
        attn_in_proj_weight_ptrs.data(),
        attn_in_proj_bias_ptrs.data(),
        attn_out_proj_weight_ptrs.data(),
        attn_out_proj_bias_ptrs.data(),
        ff1_weight_ptrs.data(),
        ff1_bias_ptrs.data(),
        ff2_weight_ptrs.data(),
        ff2_bias_ptrs.data(),
        norm1_weight_ptrs.data(),
        norm1_bias_ptrs.data(),
        norm2_weight_ptrs.data(),
        norm2_bias_ptrs.data(),
        // final classifier
        fc_w_ptr,
        fc_b_ptr,
        // sizes
        batch_size,
        in_channels,
        image_size,
        image_size,
        patch_size,
        embed_dim,
        num_heads,
        num_layers,
        ff_dim,
        num_classes
    );

    // Validate results: fp16 -> rtol=1e-1, atol=1e-1
    bool ok = torch::allclose(gpu_output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1, /*equal_nan=*/false);
    if (ok) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0; // Always exit with code 0
}
