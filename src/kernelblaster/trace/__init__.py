# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full trace store for replaying / auditing the agent loop (docs/trace.md).

Every LLM call (exact messages, model, parameters, raw response, stop reason,
request id), every candidate source (winning and losing), every build
(command, compiler output), driver run (raw MB_* output), NCU pass (raw CSV)
and every step's decision, as JSONL records with content-addressed blobs under
``<out>/trace``; ids are stable (run / problem / rollout / step / attempt) and
link to telemetry span ids, the relay's request id, outcome ids and
result.json ``steps[]``.

    from kernelblaster import trace
    trace.configure(out / "trace", level="full")    # or KB_TRACE=full|meta|off
    with trace.scope(problem=p, rollout=r, step=s):
        ...
    trace.close()

Levels: ``full`` (texts stored, gzip, deduplicated), ``meta`` (hashes and sizes
only), ``off`` (nothing; the default outside ``scripts/run_multislot.py``).
Traces contain prompts and generated code; secrets are scrubbed (see
``core.Scrubber``). Stdlib only.
"""
from .core import (LEVELS, REDACTED, Scrubber, Store, base_key, close, configure, current, enabled, flush, key,
                   level, put, put_json, read_ref, record, scope, set_attrs, store, telemetry_span_id)
from .hooks import (attempt, in_llm, instrument_evaluator, link, llm_recorded, llm_replayed, on_extract, output,
                    request_id, sha, step_done, step_input, tool)


__all__ = [
    "LEVELS", "REDACTED", "Scrubber", "Store", "attempt", "base_key", "close", "configure", "current", "enabled",
    "flush", "in_llm", "instrument_evaluator", "key", "level", "link", "llm_recorded", "llm_replayed", "on_extract",
    "output",
    "put", "put_json", "read_ref", "record", "request_id", "scope", "set_attrs", "sha", "step_done",
    "step_input", "store", "telemetry_span_id", "tool",
]
