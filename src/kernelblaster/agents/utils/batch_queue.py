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
Batch queue for aggregating LLM requests from multiple trajectories/agents.

This module provides batching middleware that collects requests over a time window
and batches them together for more efficient processing, especially for local models.
"""
import asyncio
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from copy import deepcopy
import os

from loguru import logger

from .local_llm import is_local_model, generate_code_local, generate_code_local_batch
from .query import LLMResponse

__all__ = ["LLMBatchQueue", "get_batch_queue"]


@dataclass
class QueuedRequest:
    """A single LLM request waiting to be batched."""
    messages: List[Dict]
    model: str
    n_tasks: int
    max_tokens: int
    temperature: float
    top_p: float
    future: asyncio.Future
    timestamp: float
    use_4bit: bool = True


class LLMBatchQueue:
    """
    Queue that batches LLM requests for local models.
    
    Collects requests over a time window and processes them in batches
    for improved GPU utilization and reduced latency.
    """
    
    def __init__(
        self,
        window_ms: int = 100,
        max_batch_size: int = 8,
        enabled: bool = True,
    ):
        """
        Initialize the batch queue.
        
        Parameters
        ----------
        window_ms : int, default 100
            Maximum time to wait (in milliseconds) before processing a batch
        max_batch_size : int, default 8
            Maximum number of requests to include in a single batch
        enabled : bool, default True
            Whether batching is enabled (can be disabled via env var)
        """
        self.window_ms = window_ms / 1000.0  # Convert to seconds
        self.max_batch_size = max_batch_size
        self.enabled = enabled and (os.getenv("LLM_BATCH_ENABLED", "true").lower() == "true")
        
        self.queue: List[QueuedRequest] = []
        self.lock = asyncio.Lock()
        self.processing = False
        self._pending_timer: Optional[asyncio.Task] = None
        
        if self.enabled:
            logger.info(
                f"LLM batch queue enabled: window={window_ms}ms, max_batch={max_batch_size}"
            )
        else:
            logger.info("LLM batch queue disabled")
    
    async def submit(
        self,
        messages: List[Dict],
        model: str,
        n_tasks: int = 1,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_4bit: bool = True,
    ) -> LLMResponse:
        """
        Submit a request to the batch queue.
        
        For local models, requests are batched together. For API models,
        requests are passed through immediately (no batching).
        
        Parameters
        ----------
        messages : List[Dict]
            Chat messages in OpenAI format
        model : str
            Model name/identifier
        n_tasks : int, default 1
            Number of completions to generate
        max_tokens : int, default 4096
            Maximum tokens to generate
        temperature : float, default 0.7
            Sampling temperature
        top_p : float, default 0.9
            Nucleus sampling parameter
        use_4bit : bool, default True
            Whether to use 4-bit quantization (for local models)
            
        Returns
        -------
        LLMResponse
            Response object with generations
        """
        # If batching is disabled or not a local model, process immediately
        if not self.enabled or not is_local_model(model):
            # For API models or when disabled, use direct call
            from .query import generate_code
            return await generate_code(messages, n_tasks, model)
        
        # Create future for async result
        future = asyncio.Future()
        request = QueuedRequest(
            messages=deepcopy(messages),
            model=model,
            n_tasks=n_tasks,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            future=future,
            timestamp=time.time(),
            use_4bit=use_4bit,
        )
        
        async with self.lock:
            self.queue.append(request)
            
            # Process immediately if batch is full
            if len(self.queue) >= self.max_batch_size:
                if not self.processing:
                    asyncio.create_task(self._process_batch())
            # Otherwise, schedule processing after window delay
            elif not self.processing and self._pending_timer is None:
                self._pending_timer = asyncio.create_task(self._process_batch_after_delay())
        
        # Wait for result
        return await future
    
    async def _process_batch_after_delay(self):
        """Process batch after window delay."""
        await asyncio.sleep(self.window_ms)
        await self._process_batch()
    
    async def _process_batch(self):
        """Process a batch of queued requests."""
        # Extract batch from queue
        async with self.lock:
            if self.processing or not self.queue:
                if self._pending_timer:
                    self._pending_timer = None
                return
            
            self.processing = True
            self._pending_timer = None
            
            # Take up to max_batch_size requests
            batch = self.queue[:self.max_batch_size]
            self.queue = self.queue[self.max_batch_size:]
            
            batch_size = len(batch)
            model = batch[0].model  # Assume same model for batch
        
        try:
            logger.info(f"Processing batch of {batch_size} requests for model {model}")
            
            # Group requests by parameters (same model, temperature, top_p, etc.)
            # For now, we'll process all together if they have compatible parameters
            # In the future, we could group more intelligently
            
            # Extract prompts from batch
            prompts = [req.messages for req in batch]
            
            # Group requests by compatible parameters
            # For true batching, we need same model, temperature, top_p, max_tokens, use_4bit
            # For now, we'll batch all together if they're the same model
            # (could be made smarter to group by parameters)
            
            first_req = batch[0]
            
            # Check if all requests have compatible parameters for true batching
            compatible = all(
                req.model == first_req.model
                and req.temperature == first_req.temperature
                and req.top_p == first_req.top_p
                and req.max_tokens == first_req.max_tokens
                and req.use_4bit == first_req.use_4bit
                and req.n_tasks == 1  # Only batch single-task requests for now
                for req in batch
            )
            
            if compatible and len(batch) > 1:
                # True batching: single forward pass for all prompts
                prompts = [req.messages for req in batch]
                responses = await generate_code_local_batch(
                    prompts,
                    first_req.model,
                    max_tokens=first_req.max_tokens,
                    temperature=first_req.temperature,
                    top_p=first_req.top_p,
                    use_4bit=first_req.use_4bit,
                )
            else:
                # Fallback: process individually but in parallel
                # (for requests with different parameters or n_tasks > 1)
                tasks = []
                for req in batch:
                    task = asyncio.create_task(
                        generate_code_local(
                            req.messages,
                            req.n_tasks,
                            req.model,
                            max_tokens=req.max_tokens,
                            temperature=req.temperature,
                            top_p=req.top_p,
                            use_4bit=req.use_4bit,
                        )
                    )
                    tasks.append(task)
                
                # Wait for all to complete
                responses = await asyncio.gather(*tasks)
            
            # Route responses back to futures
            for req, response in zip(batch, responses):
                if not req.future.done():
                    req.future.set_result(response)
            
            logger.info(
                f"Completed batch of {batch_size} requests in "
                f"{sum(r.elapsed_time for r in responses) / batch_size:.2f}s average"
            )
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Route errors back to futures
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
        finally:
            async with self.lock:
                self.processing = False
                # Process remaining queue if there are more requests
                if self.queue:
                    if len(self.queue) >= self.max_batch_size:
                        asyncio.create_task(self._process_batch())
                    else:
                        self._pending_timer = asyncio.create_task(
                            self._process_batch_after_delay()
                        )
    
    async def flush(self):
        """Force process all pending requests immediately."""
        async with self.lock:
            if self.queue and not self.processing:
                await self._process_batch()


# Global queue instance
_global_batch_queue: Optional[LLMBatchQueue] = None


def get_batch_queue() -> LLMBatchQueue:
    """Get or create the global batch queue instance."""
    global _global_batch_queue
    
    if _global_batch_queue is None:
        window_ms = int(os.getenv("LLM_BATCH_WINDOW_MS", "100"))
        max_batch = int(os.getenv("LLM_BATCH_MAX_SIZE", "8"))
        enabled = os.getenv("LLM_BATCH_ENABLED", "true").lower() == "true"
        
        _global_batch_queue = LLMBatchQueue(
            window_ms=window_ms,
            max_batch_size=max_batch,
            enabled=enabled,
        )
    
    return _global_batch_queue
