# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Adaptive token-budget for LLM codegen calls.

Extracted from ``opt_ncu_rl_optimized``. Fully backend-agnostic —
reads ``BEDROCK_MAX_TOKENS`` / ``ANTHROPIC_MAX_TOKENS`` env vars fresh
on every LLM query (the query helpers in ``agents/utils/query.py`` do
this), so a bump here takes effect on the next call.

Two heuristics detect truncation:

1. Odd number of ``` fences in the response — the model got cut mid
   code-block.
2. ``output_tokens`` within 5% of the current cap — almost certainly
   hit the limit.

When either fires and there's a higher tier to bump to, both env vars
get updated under a process-global lock (safe from any task/thread).
"""
from __future__ import annotations

import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_TOKEN_BUDGET_TIERS: List[int] = [16384, 32768, 65536]
_TOKEN_BUDGET_LOCK = threading.Lock()


def current_max_tokens() -> int:
    """Read the effective ``BEDROCK_MAX_TOKENS`` /
    ``ANTHROPIC_MAX_TOKENS`` cap. Falls back to 16384 when unset."""
    raw = (
        os.getenv("BEDROCK_MAX_TOKENS")
        or os.getenv("ANTHROPIC_MAX_TOKENS")
        or "16384"
    )
    try:
        return int(raw)
    except ValueError:
        return 16384


def _looks_truncated(
    text: str,
    usage: Optional[Dict[str, Any]],
    current_cap: int,
) -> Tuple[bool, str]:
    """Return ``(truncated, reason)``. Both heuristics documented at
    the module docstring."""
    if text:
        fences = len(re.findall(r"```", text))
        if fences > 0 and fences % 2 == 1:
            return True, "odd number of ``` fences (unclosed code block)"
    if usage:
        out = usage.get("output_tokens") or 0
        if out and current_cap and out >= int(current_cap * 0.95):
            return True, f"output_tokens={out} ≥ 95% of cap={current_cap}"
    return False, ""


def maybe_bump_token_budget(
    response_text: str,
    usage: Optional[Dict[str, Any]],
    logger=None,
    tiers: Optional[List[int]] = None,
) -> bool:
    """If the response looks truncated, raise the cap one tier.

    Returns True when a bump happened. Process-global; safe to call
    from any task. Passing ``tiers`` lets callers substitute a
    workload-specific schedule (the default suits Bedrock / Anthropic
    Sonnet-family models).
    """
    tiers = tiers if tiers is not None else DEFAULT_TOKEN_BUDGET_TIERS
    cur = current_max_tokens()
    truncated, reason = _looks_truncated(response_text or "", usage, cur)
    if not truncated:
        return False
    with _TOKEN_BUDGET_LOCK:
        cur = current_max_tokens()  # re-read under lock
        next_tier = next((t for t in tiers if t > cur), None)
        if next_tier is None:
            if logger:
                logger.warning(
                    f"Codegen response looks truncated ({reason}) but max_tokens "
                    f"is already at top tier ({cur}); cannot bump further."
                )
            return False
        os.environ["BEDROCK_MAX_TOKENS"] = str(next_tier)
        os.environ["ANTHROPIC_MAX_TOKENS"] = str(next_tier)
        if logger:
            logger.warning(
                f"Codegen looks truncated ({reason}); raised max_tokens "
                f"{cur} → {next_tier} for all subsequent LLM calls."
            )
    return True


__all__ = [
    "DEFAULT_TOKEN_BUDGET_TIERS",
    "current_max_tokens",
    "maybe_bump_token_budget",
]
