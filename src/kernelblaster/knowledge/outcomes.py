# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Outcome records: what happened when a technique was applied to a kernel.

Every optimisation attempt (successful or not) appends one record. They are
the evidence the semantic update pass reads and the claims cite, so they are
append-only and never rewritten.
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

# status values
IMPROVED = "improved"
NEUTRAL = "neutral"            # within +-NEUTRAL_PCT of the parent
REGRESSED = "regressed"
COMPILE_ERROR = "compile_error"
WRONG_OUTPUT = "wrong_output"
RUNTIME_ERROR = "runtime_error"
STATUSES = (IMPROVED, NEUTRAL, REGRESSED, COMPILE_ERROR, WRONG_OUTPUT, RUNTIME_ERROR)
NEUTRAL_PCT = 2.0

# op classes (ModelBlaster tags each KernelSpec with one; KernelBench
# problems without a tag use OTHER)
OP_CLASSES = ("gemm", "conv", "reduction", "elementwise", "data_movement", "attention", "scan", "recurrent", "other")


def classify_speedup(speedup: Optional[float]) -> str:
    if speedup is None:
        return NEUTRAL
    pct = (speedup - 1.0) * 100.0
    if pct > NEUTRAL_PCT:
        return IMPROVED
    if pct < -NEUTRAL_PCT:
        return REGRESSED
    return NEUTRAL


@dataclass
class OutcomeRecord:
    technique: str
    status: str                              # one of STATUSES
    target: str = "cuda"                     # backend name
    arch: Optional[str] = None               # "sm_75", "rvv-vlen256", ...
    problem_id: Optional[str] = None
    run_id: Optional[str] = None
    slot: Optional[str] = None               # None for single-file problems
    op_kind: Optional[str] = None            # MB op ("conv2d", "matmul_tb", ...)
    op_class: str = "other"                  # one of OP_CLASSES
    shape_bucket: Optional[str] = None       # see impl_store.shape_bucket
    dtype: Optional[str] = None
    state_before: Optional[str] = None       # technique-db state name
    state_after: Optional[str] = None
    metric_before: Optional[float] = None    # slot time (or problem time), lower is better
    metric_after: Optional[float] = None
    speedup: Optional[float] = None          # metric_before / metric_after
    metrics_delta: dict = field(default_factory=dict)   # profiler metric -> after - before
    error: Optional[str] = None              # truncated compiler / verify text
    diff_summary: Optional[str] = None       # short description of the code change
    impl_id: Optional[str] = None            # impl_store record this produced (verified only)
    parent_impl_id: Optional[str] = None
    # what status / reward were measured on: "e2e_graph" / "e2e_eager" (the deployed
    # partition time; metric_* are then the slot's attributed GPU times, speedup the
    # slot's), "slot_events" (the slot's event-timed time); None: records written
    # before the field existed (slot_events)
    timing_basis: Optional[str] = None
    e2e_before: Optional[float] = None       # deployed partition time before / with the change, us
    e2e_after: Optional[float] = None
    partition_effect: Optional[float] = None  # fraction of the deployed partition time saved (> 0 faster)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @classmethod
    def from_json(cls, d: dict) -> "OutcomeRecord":
        """A stored record; fields this version does not know are dropped (newer writers)."""
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"bad outcome status {self.status!r}")
        if self.error and len(self.error) > 2000:
            self.error = self.error[:1000] + "\n...\n" + self.error[-1000:]


class OutcomeLog:
    """Append-only JSONL of :class:`OutcomeRecord` (one file, thread-safe)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, rec: OutcomeRecord) -> OutcomeRecord:
        line = json.dumps(asdict(rec), sort_keys=True)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return rec

    def __iter__(self) -> Iterator[OutcomeRecord]:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield OutcomeRecord.from_json(json.loads(line))

    def get(self, ids) -> list[OutcomeRecord]:
        want = set(ids)
        return [r for r in self if r.id in want]

    def since(self, cursor: int) -> tuple[list[OutcomeRecord], int]:
        """Records after the first ``cursor`` lines, and the new cursor."""
        recs = list(self)
        return recs[cursor:], len(recs)
