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
from copy import deepcopy
from typing import List, Tuple
import openai
import os
import re
import time
import asyncio
import json
import random
from loguru import logger
from dataclasses import dataclass, asdict

from ...config import config

__all__ = [
    "generate_code_openai",
    "generate_code",
    "generate_code_retry",
    "convert_messages_to_string",
    "LLMResponse",
    "trim_messages",
    "to_dict_recursive",
    "extract_code_from_response",
]

# Lazy import to avoid loading heavy dependencies if not using local models
_local_llm_module = None

TIMEOUT = 1800
# Rough estimation of the maximum tokens as tokenization is expensive
MAX_CHARACTERS_TOTAL = 340_000

# Single OpenAI-compatible client configured via OPENAI_API_KEY.
oai_client = None
if os.getenv("OPENAI_API_KEY"):
    oai_client = openai.AsyncOpenAI(
        base_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=TIMEOUT,
    )

# Perflab client configured via PERFLAB_KEY (fallback option).
perflab_client = None
if os.getenv("PERFLAB_KEY"):
    perflab_client = openai.AsyncAzureOpenAI(
        azure_endpoint="https://llm-proxy.perflab.nvidia.com",
        api_version="2024-12-01-preview",
        api_key=os.getenv("PERFLAB_KEY"),
        timeout=None,
    )

# Bedrock client configured via AWS_BEARER_TOKEN_BEDROCK (or standard AWS
# creds). Uses the Converse API — one code path handles both Anthropic
# ``anthropic.claude-*`` and Meta ``meta.llama*`` model IDs (as well as
# their region-prefixed inference-profile variants ``us.meta.llama*``,
# ``us.anthropic.claude-*``). ``asyncio.to_thread`` is used to bridge
# boto3's sync API into the async pipeline — a native async Bedrock SDK
# doesn't exist yet.
bedrock_client = None
if os.getenv("AWS_BEARER_TOKEN_BEDROCK") or os.getenv("AWS_ACCESS_KEY_ID"):
    try:
        import boto3  # type: ignore
        bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-west-2"),
        )
    except Exception:
        bedrock_client = None


class TrimError(Exception):
    pass


class NoResponseError(Exception):
    pass


def trim_messages(messages, logger):
    total_characters = sum(
        len(message["content"]) if message["content"] is not None else 0
        for message in messages
    )
    while total_characters > MAX_CHARACTERS_TOTAL and len(messages) >= 3:
        logger.warning(
            f"Trimming messages. Total characters: {total_characters}. Max characters: {MAX_CHARACTERS_TOTAL}"
        )
        # Keep the first system message and the first user message
        # Poping two messages at a time to ensure alternative roles are kept
        messages.pop(2)  # Remove the first assistant
        if len(messages) >= 3:
            messages.pop(2)  # Remove the second user message

        total_characters = sum(
            len(message["content"]) if message["content"] is not None else 0
            for message in messages
        )
        logger.warning(
            f"Trimmed messages. Total characters: {total_characters}. Max characters: {MAX_CHARACTERS_TOTAL}"
        )

    if total_characters > MAX_CHARACTERS_TOTAL:
        logger.error(
            f"Failed to trim messages. Total characters: {total_characters}. Max characters: {MAX_CHARACTERS_TOTAL}"
        )
        raise TrimError(
            f"Failed to trim messages. Total characters: {total_characters}. Max characters: {MAX_CHARACTERS_TOTAL}"
        )

    return messages


def to_dict_recursive(obj):
    obj = dict(obj)
    for k, v in obj.items():
        obj[k] = to_dict_recursive(v) if hasattr(v, "__dict__") else v
    return obj


def extract_code_from_response(response_text, tag="cpp") -> str | None:
    # First remove any content inside <think></think> tags if they exist
    # to avoid picking up any code content within the reasoning process.
    # Deepseek-R1 introduced the <think> start tag into the chat template itself
    # so it is no longer returned during generation.
    cleaned_text = re.sub(r".*?</think>", "", response_text, flags=re.DOTALL)
    code_blocks = re.findall(f"```{tag}\n(.*?)```", cleaned_text, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[0]


@dataclass
class LLMResponse:
    input_messages: list[dict]
    generations: list[str]
    usage: dict
    model: str
    num_tasks: int
    elapsed_time: int

    def __str__(self):
        return str(asdict(self))

    @property
    def response(self) -> str:
        """Return the first generation for backward-compatibility with legacy
        code that accessed *response* instead of *generations[0]*.
        """
        return self.generations[0] if self.generations else ""


def process_messages(messages: List[dict], model: str) -> List[dict]:
    messages = deepcopy(messages)
    if re.search(r".*-nemotron-.*-thinking", model.lower()):
        # Enable thinking mode for nemotron models by using special system prompt
        formatted_messages = messages
        if messages[0]["role"] == "system":
            formatted_messages[0]["content"] = "detailed thinking on"
            formatted_messages[1]["content"] = (
                messages[0]["content"] + "\n\n" + messages[1]["content"]
            )
        return formatted_messages

    if "o1" in model.lower():
        formatted_messages = []
        sys_message = ""
        for message in messages:
            if message["role"] == "system":
                sys_message += message["content"] + "\n\n"
            else:
                formatted_messages.append(deepcopy(message))
        if len(sys_message) > 0:
            formatted_messages.insert(0, {"role": "user", "content": sys_message})
        return formatted_messages

    return messages


def _is_bedrock_model(model: str) -> bool:
    """Detect Bedrock model IDs / inference-profile IDs.

    Matches ``anthropic.claude-*``, ``meta.llama*``, ``us.meta.llama*``,
    ``us.anthropic.claude-*``, and other Bedrock-native prefixes.
    Cross-region inference profiles use a ``<region>.<vendor>.<model>``
    shape; the ``.`` separator + vendor prefix is the reliable signal.
    """
    low = model.lower()
    for prefix in (
        "anthropic.claude", "us.anthropic.claude",
        "eu.anthropic.claude", "apac.anthropic.claude",
        "meta.llama", "us.meta.llama",
        "eu.meta.llama", "apac.meta.llama",
        "mistral.", "us.mistral.",
        "amazon.nova", "us.amazon.nova",
        "cohere.command", "ai21.jamba",
        "moonshotai.", "moonshot.", "us.moonshotai.", "us.moonshot.",
    ):
        if low.startswith(prefix):
            return True
    return False


async def generate_code_bedrock(
    messages, n_tasks: int, model: str
) -> LLMResponse:
    """Generate via Bedrock's Converse API.

    One code path for Anthropic Claude, Meta Llama, Mistral, Nova, etc.
    Converse accepts a normalised ``[{role, content:[{text}]}]`` message
    shape; we translate from the OpenAI ``[{role, content: str}]`` shape
    here.

    ``n_tasks`` > 1 issues that many independent Converse calls (Converse
    has no ``n`` parameter). Sequential for simplicity — the RL loop
    already parallelises at the trajectory level, and only NUM_PARALLEL_
    GENERATIONS_PER_ATTEMPT-tier callers ever pass ``n_tasks > 1``.
    """
    if bedrock_client is None:
        raise RuntimeError(
            "Bedrock model requested but no bedrock-runtime client is "
            "configured — set AWS_BEARER_TOKEN_BEDROCK (or standard AWS "
            "credentials) + AWS_REGION."
        )

    # OpenAI messages → Converse messages. Converse takes the system
    # prompt separately; user/assistant messages are lists of content
    # blocks. Empty content blocks are illegal — drop them.
    system_blocks: list[dict] = []
    conv_messages: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            if content:
                system_blocks.append({"text": content})
            continue
        if not content:
            continue
        conv_messages.append({"role": role, "content": [{"text": content}]})

    # Bedrock rejects an assistant-terminated message list — drop a
    # trailing assistant turn if the caller left one on the tail.
    while conv_messages and conv_messages[-1]["role"] == "assistant":
        conv_messages.pop()
    if not conv_messages:
        raise RuntimeError("generate_code_bedrock: no user messages after conversion")

    max_tokens_env = os.getenv("BEDROCK_MAX_TOKENS") or os.getenv("ANTHROPIC_MAX_TOKENS")
    try:
        max_tokens = int(max_tokens_env) if max_tokens_env else 8192
    except ValueError:
        max_tokens = 8192

    inference_config: dict = {"maxTokens": max_tokens}
    temperature_env = os.getenv("BEDROCK_TEMPERATURE")
    if temperature_env:
        try:
            inference_config["temperature"] = float(temperature_env)
        except ValueError:
            pass

    kwargs: dict = {
        "modelId": model,
        "messages": conv_messages,
        "inferenceConfig": inference_config,
    }
    if system_blocks:
        kwargs["system"] = system_blocks

    outputs: list[str] = []
    total_input = 0
    total_output = 0
    start = time.time()
    for _ in range(n_tasks):
        # boto3 client is sync; hop off the event loop.
        resp = await asyncio.to_thread(bedrock_client.converse, **kwargs)
        content = resp["output"]["message"].get("content") or []
        text = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and "text" in block
        )
        outputs.append(text)
        usage = resp.get("usage") or {}
        total_input += int(usage.get("inputTokens") or 0)
        total_output += int(usage.get("outputTokens") or 0)
    end = time.time()

    return LLMResponse(
        deepcopy(messages),
        outputs,
        {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
        },
        model,
        n_tasks,
        end - start,
    )


async def generate_code_openai(client, messages, n_tasks, model: str) -> LLMResponse:
    args = {
        "model": model,
        "messages": messages,
        "stream": config.STREAM
        and config.STREAM.lower() in ("true", "1", "yes", "y", "on"),
        "n": n_tasks,
    }

    if re.search(r"claude-.*-thinking", model.lower()):
        args["extra_body"] = {"thinking": {"type": "enabled", "budget_tokens": 16384}}
        args["model"] = model.replace("-thinking", "")
    elif re.search(r"deepseek-r1", model.lower()):
        # thinking mode should be enabled by default
        args["top_p"] = 0.95
        args["max_tokens"] = 12288
        args["enable_thinking"] = True
    elif re.search("qwen", model.lower()) or re.search("kevin", model.lower()):
        args["top_p"] = 0.9
        args["enable_thinking"] = True
        args["max_tokens"] = 12288
    elif re.search(r".*-nemotron-.*-thinking", model.lower()):
        args["model"] = model.replace("-thinking", "")
        args["max_tokens"] = 12288

    start = time.time()

    if not args.get("stream", False):
        # Handle non-streaming response
        response = await client.chat.completions.create(**args)
        outputs = [choice.message.content for choice in response.choices]

        end = time.time()
        return LLMResponse(
            deepcopy(messages),
            outputs,
            to_dict_recursive(response.usage),
            model,
            n_tasks,
            end - start,
        )
    else:
        # Handle streaming response
        outputs = [""] * n_tasks
        usage_info = None

        try:
            stream = await client.chat.completions.create(**args)
            async for chunk in stream:
                # Check if usage information is available in this chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_info = to_dict_recursive(chunk.usage)

                for i, choice in enumerate(chunk.choices):
                    if i >= len(outputs):
                        # Just in case more choices are returned than requested
                        continue

                    if (
                        hasattr(choice, "delta")
                        and hasattr(choice.delta, "content")
                        and choice.delta.content
                    ):
                        outputs[i] += choice.delta.content
                    elif (
                        hasattr(choice, "message")
                        and hasattr(choice.message, "content")
                        and choice.message.content
                    ):
                        # Some clients might return full messages even in streaming mode
                        outputs[i] = choice.message.content

            # Get final timing
            end = time.time()

            # If we didn't get usage info from the stream, estimate it
            if not usage_info:
                usage_info = {
                    "completion_tokens": sum(
                        len(output.split()) * 4 // 3 for output in outputs
                    ),
                    "prompt_tokens": sum(
                        len(message["content"].split()) * 4 // 3 for message in messages
                    ),
                    "total_tokens": 0,
                }
                usage_info["total_tokens"] = (
                    usage_info["completion_tokens"] + usage_info["prompt_tokens"]
                )

            return LLMResponse(
                deepcopy(messages),
                outputs,
                usage_info,
                model,
                n_tasks,
                end - start,
            )
        except Exception as e:
            logger.error(f"Error during streaming response: {e}")
            raise


async def generate_code(messages, n_tasks=1, model=None) -> LLMResponse:
    """Main function to generate code using either OpenAI REST interface or local models.

    Supports:
    - OpenAI-compatible clients configured via OPENAI_API_KEY
    - Local models (e.g., Qwen2.5-Coder-32B-Instruct) when model name indicates local usage
    - Automatic batching for local models via batch queue
    """
    # Check if this is a local model
    global _local_llm_module
    if _local_llm_module is None:
        try:
            from . import local_llm as _local_llm_module
        except ImportError:
            _local_llm_module = False  # Mark as unavailable
    
    # For local models, use batch queue if enabled
    if _local_llm_module and _local_llm_module.is_local_model(model):
        # Check if batching is enabled
        batch_enabled = os.getenv("LLM_BATCH_ENABLED", "true").lower() == "true"
        batch_only_local = os.getenv("LLM_BATCH_ONLY_LOCAL", "true").lower() == "true"
        
        if batch_enabled and batch_only_local:
            # Use batch queue for local models
            from .batch_queue import get_batch_queue
            batch_queue = get_batch_queue()
            return await batch_queue.submit(
                messages=messages,
                model=model,
                n_tasks=n_tasks,
                max_tokens=4096,
                temperature=0.7,
                top_p=0.9,
                use_4bit=True,
            )
        else:
            # Direct call (batching disabled)
            logger.info(f"Using local model (direct): {model}")
            return await _local_llm_module.generate_code_local(
                messages,
                n_tasks,
                model,
                max_tokens=4096,
                temperature=0.7,
                top_p=0.9,
                use_4bit=True,
            )
    
    # Bedrock: matches Anthropic + Meta + Mistral + Nova etc. (including
    # ``us.<vendor>...`` cross-region inference profiles) whenever the
    # bedrock-runtime client is configured. Preferred over OpenAI when
    # the model ID is clearly Bedrock-flavored — this is what lets the
    # heterogeneous flow use e.g. Llama 4 Scout for MODEL_CODEGEN_SIMPLE
    # and Llama 4 Maverick for MODEL_CODEGEN_HARD.
    if bedrock_client is not None and _is_bedrock_model(model):
        return await generate_code_bedrock(messages, n_tasks, model)

    # Fall back to OpenAI-compatible client
    # Prefer OpenAI if configured (this is the most common local dev path).
    client = oai_client

    # Fall back to Perflab proxy only if configured.
    if client is None and perflab_client is not None:
        client = perflab_client

    if client is None:
        raise RuntimeError(
            "No LLM client configured. Set OPENAI_API_KEY or PERFLAB_KEY, "
            "AWS_BEARER_TOKEN_BEDROCK / AWS credentials for Bedrock models, "
            "or use a local model."
        )

    return await generate_code_openai(client, messages, n_tasks, model)


async def generate_code_retry(
    messages,
    model,
    logger=None,
    n_tasks: int = 1,
    max_retries: int | None = config.LLM_MAX_RETRIES,
    **kwargs,
) -> LLMResponse:
    """Robust wrapper around generate_code that automatically retries and
    supports optional arguments for backward-compatibility (e.g. *system_prompt*,
    *max_attempts*).

    Parameters
    ----------
    messages : list[dict]
        Chat completion messages (same format as OpenAI/other providers).
    model : str
        LLM model name.
    logger : logging.Logger | None, optional
        Logger for debug/warning messages. A default logger will be created if
        ``None`` is supplied so that legacy callers that did not pass a logger
        continue to work.
    n_tasks : int, default 1
        Number of completions to request.
    max_retries : int | None, optional
        Retry attempts before giving up. If ``None`` the call will keep retrying
        indefinitely.
    **kwargs : Any
        Silently-ignored extra keyword arguments. Currently the following keys
        are interpreted:

        - ``system_prompt`` (str): will be prepended to *messages* as a
          ``{"role": "system", "content": system_prompt}`` entry if provided.
        - ``max_attempts`` (int): alias for *max_retries* kept for historical
          reasons.
    """

    # ------------------------------------------------------------------
    # Backwards-compatibility handling of newly introduced kwargs
    # ------------------------------------------------------------------
    system_prompt: str | None = kwargs.pop("system_prompt", None)
    max_attempts = kwargs.pop("max_attempts", None)
    if max_attempts is not None:
        max_retries = max_attempts

    # If any other unexpected kwargs remain, just ignore them but emit a trace.
    if kwargs:
        import warnings as _warnings
        _warnings.warn(
            f"[generate_code_retry] Ignoring unrecognised kwargs: {list(kwargs.keys())}",
            RuntimeWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Ensure we always have a logger instance.
    # ------------------------------------------------------------------
    if logger is None:
        import logging as _logging
        logger = _logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Prepend system prompt if provided (LLM APIs expect it as first message)
    # ------------------------------------------------------------------
    if system_prompt:
        # Avoid mutating caller-provided list in-place
        messages = [{"role": "system", "content": system_prompt}] + list(messages)

    response = None
    tries = 0
    while max_retries is None or tries < max_retries:
        delay = 5
        try:
            response = await generate_code(messages, n_tasks=n_tasks, model=model)
            break
        except openai.RateLimitError as e:
            # Exponential backoff with jitter to smooth out concurrent requests
            delay = (2**tries) + random.uniform(0, 60)
            logger.warning(
                f"Rate limit error on attempt {tries}/{max_retries}. "
                f"Retrying in {delay:.2f} seconds: {e}"
            )
        except openai.AuthenticationError as e:
            logger.error(
                f"Failed to generate code due to 401 error. Trying again ({e})"
            )
        except openai.APITimeoutError as e:
            logger.error(
                f"Failed to generate code due to timeout error. Trying again ({e})"
            )
        except openai.InternalServerError as e:
            logger.error(
                f"Failed to generate code due to internal server error. Trying again ({e})"
            )
        except openai.APIConnectionError as e:
            logger.error(
                f"Failed to generate code due to connection error. Trying again ({e})"
            )
        except NoResponseError as e:
            logger.error(f"No response from LLM. Response: {e}")
        except Exception as e:
            if "timeout" in str(e).lower() or "time-out" in str(e).lower():
                delay = (2**tries) + random.uniform(0, 60)
                logger.error(
                    f"Failed to generate code due to timeout. Trying again (attempt {tries}/{max_retries}, error: {e})"
                )
            elif "502" in str(e) or "bad gateway" in str(e).lower():
                logger.error(
                    f"Failed to generate code due to bad gateway. Trying again (attempt {tries}/{max_retries}, error: {e})"
                )
            elif "rate_limited" in str(e).lower() or "rate limit" in str(e).lower():
                delay = (2**tries) + random.uniform(0, 60)
                logger.warning(
                    f"Failed to generate code due to rate limit. Retrying in {delay:.2f} seconds (attempt {tries}/{max_retries}, error: {e})"
                )
            elif "invalid client" in str(e).lower():
                logger.error(
                    f"Failed to generate code due to invalid client bug. Trying again (attempt {tries}/{max_retries}, error: {e})"
                )
            elif "inference connection error" in str(e).lower():
                logger.error(
                    f"Failed to generate code due to inference connection error. Trying again (attempt {tries}/{max_retries}, error: {e})"
                )
            elif "device or resource busy" in str(e).lower():
                delay = (2**tries) + random.uniform(0, 60)
                tries += 1
                logger.warning(
                    f"Device or resource busy. Retrying in {delay:.2f} seconds: {e}"
                )
                continue
            else:
                logger.error(
                    f"Failed to generate code due to unknown error. Trying again (attempt {tries}/{max_retries}, error: {e})"
                )
            await asyncio.sleep(delay)
            tries += 1
    if not response:
        logger.error("Failed to generate code exceeding max retries.")
        raise RuntimeError("Exceeded max retries")
    return response


def convert_messages_to_string(
    messages: list[dict], response: str = None, usage: dict = None
):
    def to_string(role, content):
        r = "# " + "=" * 20 + role.upper() + "=" * 20 + "\n"
        r += content
        return r

    string = ""
    for message in messages:
        string += "\n" + to_string(message["role"], message["content"])
    if response:
        string += "\n" + to_string("assistant", response)
    if usage:
        string += "\n" + to_string("usage", json.dumps(usage, indent=2))
    return string
