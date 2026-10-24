# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keyless LLM providers for development and tests.

Two model-name prefixes are routed here by ``query.generate_code``:

``claude-code:<model>``
    Runs the Claude Code CLI headless (``claude -p``) with tools disabled,
    so it behaves as a plain chat completion billed to the logged-in
    Claude Code account. ``<model>`` is passed to ``--model`` (``sonnet``,
    ``opus``, a full model id, ...). Stand-in until API-key access is set
    up; switch by changing the model name only.

``stub:<name>``
    Calls a Python function registered with :func:`register_stub`. Unit
    and integration tests use this to script responses deterministically.

Env:
    KB_CLAUDE_CLI            path to the ``claude`` binary (default: on PATH)
    KB_CLAUDE_CLI_PARALLEL   max concurrent CLI calls (default 4)
    KB_CLAUDE_CLI_TIMEOUT_S  per-call timeout (default 600)
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
from typing import Awaitable, Callable, Union

CLAUDE_CODE_PREFIX = "claude-code:"
STUB_PREFIX = "stub:"

StubFn = Callable[[list], Union[str, Awaitable[str]]]
_stubs: dict[str, StubFn] = {}
_cli_sem: asyncio.Semaphore | None = None


def is_agentic_model(model: str | None) -> bool:
    return bool(model) and (model.startswith(CLAUDE_CODE_PREFIX) or model.startswith(STUB_PREFIX))


def register_stub(name: str, fn: StubFn) -> None:
    """Route ``stub:<name>`` to ``fn(messages) -> str`` (sync or async)."""
    _stubs[name] = fn


def unregister_stub(name: str) -> None:
    _stubs.pop(name, None)


def _split_messages(messages: list) -> tuple[str, str]:
    """System text, and the rest as one transcript (the CLI takes one prompt)."""
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
    turns = [m for m in messages if m.get("role") != "system"]
    if len(turns) == 1:
        return system, turns[0]["content"]
    body = "\n\n".join(f"[{m.get('role', 'user').upper()}]\n{m['content']}" for m in turns)
    return system, body


async def _claude_cli_once(messages: list, model: str) -> tuple[str, dict]:
    global _cli_sem
    if _cli_sem is None:
        _cli_sem = asyncio.Semaphore(int(os.getenv("KB_CLAUDE_CLI_PARALLEL", "4")))
    system, prompt = _split_messages(messages)
    cmd = [os.getenv("KB_CLAUDE_CLI", "claude"), "-p", "--model", model, "--tools", "",
           "--no-session-persistence", "--output-format", "json"]
    if system:
        cmd += ["--system-prompt", system]
    timeout = float(os.getenv("KB_CLAUDE_CLI_TIMEOUT_S", "600"))
    async with _cli_sem:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"claude CLI timed out after {timeout:.0f}s")
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI exit {proc.returncode}: {err.decode()[-500:]}")
    res = json.loads(out.decode())
    if res.get("is_error"):
        raise RuntimeError(f"claude CLI error: {str(res.get('result'))[:500]}")
    u = res.get("usage", {})
    usage = {
        "prompt_tokens": u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
        + u.get("cache_creation_input_tokens", 0),
        "completion_tokens": u.get("output_tokens", 0),
        "cost_usd": res.get("total_cost_usd", 0.0),
    }
    return res.get("result", ""), usage


async def generate_code_agentic(messages: list, n_tasks: int, model: str):
    from .query import LLMResponse

    start = time.time()
    if model.startswith(STUB_PREFIX):
        name = model[len(STUB_PREFIX):]
        if name not in _stubs:
            raise RuntimeError(f"no stub registered for {model!r}")
        gens = []
        for _ in range(n_tasks):
            r = _stubs[name](messages)
            gens.append(await r if inspect.isawaitable(r) else r)
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
    else:
        outs = await asyncio.gather(*[
            _claude_cli_once(messages, model[len(CLAUDE_CODE_PREFIX):]) for _ in range(n_tasks)])
        gens = [o[0] for o in outs]
        usage = {k: sum(o[1].get(k, 0) for o in outs)
                 for k in ("prompt_tokens", "completion_tokens", "cost_usd")}
    return LLMResponse(input_messages=messages, generations=gens, usage=usage, model=model,
                       num_tasks=n_tasks, elapsed_time=int(time.time() - start))
