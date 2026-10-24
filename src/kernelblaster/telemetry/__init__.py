# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-loop telemetry: spans, LLM token accounting, JSONL events, reports.

See ``docs/telemetry.md``. Stdlib only; ``KB_TELEMETRY=0`` turns it off.

    from kernelblaster import telemetry as tel

    with tel.span("problem", collect=True, problem=name) as sp:
        ...
        with tel.attrs(role="fix"):
            await generate_code_retry(...)       # recorded as an llm.fix span + llm event
    result["telemetry"] = tel.full_summary(sp)
"""
from .core import (NOSPAN, Collector, LLMStats, Span, attrs, close, configure, count, ctx_summary, current,
                   emit, enabled, end_laps, epoch, flush, lap, record_span, run_id, set_enabled, sink, span,
                   summary_of, traced, union_length, within)
from .evaluator import instrument_class, purpose_of
from .llm import backoff, llm_traced, normalize_usage
from .report import by_category, category, derive, full_summary, rollup

__all__ = [
    "NOSPAN", "Collector", "LLMStats", "Span", "attrs", "backoff", "by_category", "category", "close", "configure",
    "count", "ctx_summary", "current", "derive", "emit", "enabled", "end_laps", "epoch", "flush", "full_summary",
    "instrument_class", "lap", "llm_traced", "normalize_usage", "purpose_of", "record_span", "run_id",
    "rollup", "set_enabled", "sink", "span", "summary_of", "traced", "union_length", "within",
]
