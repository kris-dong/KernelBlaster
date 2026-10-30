# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Local LLM wrapper for running models locally with quantization support.

This module provides support for running LLMs locally, particularly
Qwen2.5-Coder-32B-Instruct with 4-bit quantization using bitsandbytes.
"""
import os
import time
import asyncio
from typing import List, Dict, Optional
from copy import deepcopy
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from loguru import logger

from .query import LLMResponse, to_dict_recursive

__all__ = [
    "generate_code_local",
    "generate_code_local_batch",
    "get_local_model",
    "is_local_model",
]

# Global model cache to avoid reloading
_model_cache: Dict[str, tuple] = {}  # model_name -> (tokenizer, model)


def is_local_model(model_name: str) -> bool:
    """Check if a model name indicates it should be run locally."""
    if not model_name:
        return False
    local_model_patterns = [
        "qwen2.5-coder-32b-instruct",
        "qwen2.5-coder-32b",
        "qwen/qwen2.5-coder-32b-instruct",
        "local-qwen",
    ]
    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in local_model_patterns)


def get_local_model(
    model_name: str,
    use_4bit: bool = True,
    device_map: str = None,
    trust_remote_code: bool = True,
):
    """
    Load a local model with optional 4-bit quantization.
    
    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g., "Qwen/Qwen2.5-Coder-32B-Instruct")
    use_4bit : bool, default True
        Whether to use 4-bit quantization via bitsandbytes
    device_map : str | None, default None
        Device mapping strategy for model loading. Options:
        - None: Use LOCAL_LLM_DEVICE_MAP env var or "auto"
        - "auto": Automatically distribute across available GPUs
        - "single": Force single GPU (cuda:0)
        - "balanced": Balance across all GPUs
        - "balanced_low_0": Balance with first GPU getting less
        - Dict: Custom device mapping
    trust_remote_code : bool, default True
        Whether to trust remote code from HuggingFace
        
    Returns
    -------
    tuple
        (tokenizer, model) tuple
    """
    # Determine device_map from parameter or environment variable
    if device_map is None:
        device_map = os.getenv("LOCAL_LLM_DEVICE_MAP", "auto")
    
    # Handle special "single" option to force single GPU
    if device_map == "single":
        device_map = "cuda:0"
        logger.info("Using single GPU (cuda:0) for model loading")
    # Check cache first
    cache_key = f"{model_name}_{use_4bit}"
    if cache_key in _model_cache:
        logger.info(f"Using cached model: {cache_key}")
        return _model_cache[cache_key]
    
    logger.info(f"Loading local model: {model_name} (4-bit: {use_4bit}, device_map: {device_map})")
    
    # Determine actual model path
    model_lower = model_name.lower()
    if "qwen2.5-coder-32b" in model_lower:
        # Use the official HuggingFace model name
        actual_model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
    elif model_name.startswith("Qwen/") or "/" in model_name:
        # Already a full HuggingFace path
        actual_model_name = model_name
    else:
        actual_model_name = model_name
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        actual_model_name,
        trust_remote_code=trust_remote_code,
    )
    
    # Configure quantization if requested
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Using 4-bit quantization (NF4)")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        actual_model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16 if not use_4bit else None,
    )
    
    # Cache the model
    _model_cache[cache_key] = (tokenizer, model)
    logger.info(f"Model loaded and cached: {cache_key}")
    
    return tokenizer, model


def format_messages_for_qwen(messages: List[Dict]) -> str:
    """
    Format messages in the chat format expected by Qwen models.
    
    Qwen models use a specific chat template format. This function
    converts OpenAI-style messages to Qwen format.
    """
    # Extract system message if present
    system_message = None
    conversation_messages = []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            system_message = content
        else:
            conversation_messages.append({"role": role, "content": content})
    
    # Use tokenizer's chat template if available, otherwise format manually
    # We'll use the tokenizer's apply_chat_template method
    return conversation_messages, system_message


async def generate_code_local(
    messages: List[Dict],
    n_tasks: int,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_4bit: bool = True,
) -> LLMResponse:
    """
    Generate code using a local model.
    
    Parameters
    ----------
    messages : List[Dict]
        Chat messages in OpenAI format
    n_tasks : int
        Number of completions to generate
    model : str
        Model name/identifier
    max_tokens : int, default 4096
        Maximum tokens to generate
    temperature : float, default 0.7
        Sampling temperature
    top_p : float, default 0.9
        Nucleus sampling parameter
    use_4bit : bool, default True
        Whether to use 4-bit quantization
        
    Returns
    -------
    LLMResponse
        Response object with generations
    """
    start_time = time.time()
    
    try:
        # Load model (will use cache if already loaded)
        # Allow device_map to be configured via environment variable
        device_map = os.getenv("LOCAL_LLM_DEVICE_MAP", None)
        tokenizer, model_obj = get_local_model(
            model,
            use_4bit=use_4bit,
            device_map=device_map,
        )
        
        # Format messages for Qwen
        conversation_messages, system_message = format_messages_for_qwen(messages)
        
        # Apply chat template
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            # Use tokenizer's chat template
            if system_message:
                # Prepend system message to first user message if needed
                if conversation_messages and conversation_messages[0]["role"] == "user":
                    conversation_messages[0]["content"] = (
                        system_message + "\n\n" + conversation_messages[0]["content"]
                    )
            
            prompt = tokenizer.apply_chat_template(
                conversation_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: manual formatting
            prompt_parts = []
            if system_message:
                prompt_parts.append(f"System: {system_message}\n\n")
            for msg in conversation_messages:
                role = msg["role"].capitalize()
                prompt_parts.append(f"{role}: {msg['content']}\n\n")
            prompt_parts.append("Assistant: ")
            prompt = "".join(prompt_parts)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model_obj.device)
        
        # Generate multiple completions using batched inference
        # Run in executor to avoid blocking the async event loop
        loop = asyncio.get_event_loop()
        
        def generate_batched():
            """Synchronous batched generation function to run in executor."""
            with torch.no_grad():
                # Use batched generation when n_tasks > 1
                # This is more efficient than generating sequentially as it processes
                # all sequences in parallel during the forward pass
                if n_tasks > 1:
                    # Expand inputs to batch size n_tasks
                    # Repeat the input_ids and attention_mask for each task
                    batch_input_ids = inputs["input_ids"].repeat(n_tasks, 1)
                    if "attention_mask" in inputs:
                        batch_attention_mask = inputs["attention_mask"].repeat(n_tasks, 1)
                        batch_inputs = {
                            "input_ids": batch_input_ids,
                            "attention_mask": batch_attention_mask,
                        }
                    else:
                        batch_inputs = {"input_ids": batch_input_ids}
                    
                    logger.info(f"Generating {n_tasks} completions in batch with local model")
                    generated_ids = model_obj.generate(
                        **batch_inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                else:
                    # Single generation
                    logger.info(f"Generating single completion with local model")
                    generated_ids = model_obj.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    # Add batch dimension for consistent processing
                    if generated_ids.dim() == 1:
                        generated_ids = generated_ids.unsqueeze(0)
            
            # Decode all generated sequences
            input_length = inputs["input_ids"].shape[1]
            outputs = []
            batch_size = generated_ids.shape[0]
            for i in range(min(batch_size, n_tasks)):
                generated_text = tokenizer.decode(
                    generated_ids[i][input_length:],
                    skip_special_tokens=True,
                )
                outputs.append(generated_text)
            
            return outputs
        
        # Run batched generation in thread pool to avoid blocking
        outputs = await loop.run_in_executor(None, generate_batched)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Estimate token usage (rough approximation)
        total_input_tokens = inputs["input_ids"].shape[1]
        total_output_tokens = sum(
            len(tokenizer.encode(output, add_special_tokens=False))
            for output in outputs
        )
        
        usage_info = {
            "prompt_tokens": total_input_tokens,
            "completion_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }
        
        return LLMResponse(
            deepcopy(messages),
            outputs,
            usage_info,
            model,
            n_tasks,
            elapsed_time,
        )
        
    except Exception as e:
        logger.error(f"Error generating code with local model: {e}")
        raise


async def generate_code_local_batch(
    prompts: List[List[Dict]],
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_4bit: bool = True,
) -> List[LLMResponse]:
    """
    Generate code for multiple different prompts in a single batched forward pass.
    
    This is more efficient than calling generate_code_local multiple times as it
    processes all prompts in parallel during a single forward pass.
    
    Parameters
    ----------
    prompts : List[List[Dict]]
        List of chat message lists (one per prompt)
    model : str
        Model name/identifier
    max_tokens : int, default 4096
        Maximum tokens to generate per prompt
    temperature : float, default 0.7
        Sampling temperature
    top_p : float, default 0.9
        Nucleus sampling parameter
    use_4bit : bool, default True
        Whether to use 4-bit quantization
        
    Returns
    -------
    List[LLMResponse]
        List of response objects, one per prompt
    """
    start_time = time.time()
    
    try:
        # Load model (will use cache if already loaded)
        tokenizer, model_obj = get_local_model(
            model,
            use_4bit=use_4bit,
        )
        
        # Format all prompts
        formatted_prompts = []
        for messages in prompts:
            conversation_messages, system_message = format_messages_for_qwen(messages)
            
            # Apply chat template
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                if system_message:
                    if conversation_messages and conversation_messages[0]["role"] == "user":
                        conversation_messages[0]["content"] = (
                            system_message + "\n\n" + conversation_messages[0]["content"]
                        )
                
                prompt = tokenizer.apply_chat_template(
                    conversation_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                # Fallback: manual formatting
                prompt_parts = []
                if system_message:
                    prompt_parts.append(f"System: {system_message}\n\n")
                for msg in conversation_messages:
                    role = msg["role"].capitalize()
                    prompt_parts.append(f"{role}: {msg['content']}\n\n")
                prompt_parts.append("Assistant: ")
                prompt = "".join(prompt_parts)
            
            formatted_prompts.append(prompt)
        
        # Tokenize all prompts
        tokenized = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,  # Reasonable max length
        ).to(model_obj.device)
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        def generate_batched():
            """Synchronous batched generation function."""
            with torch.no_grad():
                logger.info(
                    f"Generating batch of {len(prompts)} different prompts with local model"
                )
                generated_ids = model_obj.generate(
                    **tokenized,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            # Decode each sequence independently
            input_lengths = tokenized["attention_mask"].sum(dim=1).tolist()
            outputs = []
            
            for i, (gen_ids, input_len) in enumerate(zip(generated_ids, input_lengths)):
                generated_text = tokenizer.decode(
                    gen_ids[input_len:],
                    skip_special_tokens=True,
                )
                outputs.append(generated_text)
            
            return outputs, input_lengths
        
        # Run batched generation
        outputs, input_lengths = await loop.run_in_executor(None, generate_batched)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Create LLMResponse for each prompt
        responses = []
        for i, (prompt_messages, output) in enumerate(zip(prompts, outputs)):
            # Estimate token usage
            input_tokens = input_lengths[i]
            output_tokens = len(tokenizer.encode(output, add_special_tokens=False))
            
            usage_info = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
            
            response = LLMResponse(
                deepcopy(prompt_messages),
                [output],
                usage_info,
                model,
                1,  # n_tasks per prompt
                elapsed_time / len(prompts),  # Average time per prompt
            )
            responses.append(response)
        
        return responses
        
    except Exception as e:
        logger.error(f"Error generating batched code with local model: {e}")
        raise
