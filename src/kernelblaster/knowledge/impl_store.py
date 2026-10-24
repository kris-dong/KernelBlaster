# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Implementation knowledgebase: verified kernel sources, shared across problems.

Every slot implementation that passes the driver's partition and isolated
checks is stored (code content-addressed, one JSONL index line per record)
with what it implements (target, arch, op, signature, shapes, op class), how
fast it was, which techniques produced it and what it was derived from.

Retrieval for a slot (see :meth:`ImplStore.retrieve`):
  seed     same op + signature + shape bucket on this arch: start from it
  adapt    same op + signature, other shapes: start from it, say what changed
  examples best records of the same op class (other ops): few-shot context

ModelBlaster's write-back reads :meth:`ImplStore.winners`.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _mag(v) -> str:
    """Order-of-magnitude bucket (powers of 4) of a positive size."""
    v = max(int(v), 1)
    return f"{4 ** int(math.log(v, 4)):d}"


# shape keys that set a kernel's work, per op class; others ignored for bucketing
_BUCKET_KEYS = {
    "gemm": ("M", "N", "K", "batch"),
    "conv": ("N", "IC", "OC", "OH", "OW", "KH", "KW", "G"),
    "attention": ("BH", "S", "D"),
}


def shape_bucket(op_class: str, shape: dict) -> str:
    """Coarse shape key: winners usually transfer within a bucket.

    GEMM/conv/attention use their work dimensions; everything else the total
    element count (the shape dict's ``n`` or the product of its dims).
    """
    keys = _BUCKET_KEYS.get(op_class)
    if keys:
        parts = [f"{k}{_mag(shape[k])}" for k in keys if k in shape]
        if parts:
            return op_class + ":" + ",".join(parts)
    n = shape.get("n")
    if n is None:
        ints = [v for v in shape.values() if isinstance(v, int) and v > 0]
        n = math.prod(ints) if ints else 1
    return f"{op_class}:n{_mag(n)}"


def signature_sha(signature: str) -> str:
    return hashlib.sha1(" ".join(signature.split()).encode()).hexdigest()[:12]


@dataclass
class ImplRecord:
    target: str
    arch: str
    op: str
    op_class: str
    signature_sha: str
    code_sha: str
    shapes: list                                 # instance shape dicts it was verified on
    buckets: list                                # their shape_bucket()s
    slot_us: float                               # summed over the instances, on `arch`
    baseline_us: Optional[float] = None          # its parent's time (the step it improved on)
    speedup: Optional[float] = None              # baseline_us / slot_us (vs parent)
    seed_us: Optional[float] = None              # the problem's seed (hand kernel) time
    speedup_vs_seed: Optional[float] = None      # seed_us / slot_us -- what write-back filters on
    techniques: list = field(default_factory=list)   # lineage, oldest first
    parent_id: Optional[str] = None
    problem_id: Optional[str] = None
    run_id: Optional[str] = None
    slot: Optional[str] = None
    max_rel_err: Optional[float] = None
    notes: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


class ImplStore:
    """``<root>/index.jsonl`` + ``<root>/code/<sha>.cu``. Append-only; thread-safe."""

    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "code").mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"
        self._lock = threading.Lock()

    # ---- write
    def add(self, code: str, **kw) -> ImplRecord:
        sha = hashlib.sha1(code.encode()).hexdigest()
        p = self.root / "code" / f"{sha}.cu"
        if not p.exists():
            p.write_text(code)
        rec = ImplRecord(code_sha=sha, **kw)
        with self._lock, open(self.index_path, "a") as f:
            f.write(json.dumps(asdict(rec), sort_keys=True) + "\n")
        return rec

    # ---- read
    def records(self) -> list[ImplRecord]:
        if not self.index_path.exists():
            return []
        out = []
        for line in self.index_path.read_text().splitlines():
            if line.strip():
                out.append(ImplRecord(**json.loads(line)))
        return out

    def code(self, rec: ImplRecord) -> str:
        return (self.root / "code" / f"{rec.code_sha}.cu").read_text()

    def get(self, rec_id: str) -> Optional[ImplRecord]:
        return next((r for r in self.records() if r.id == rec_id), None)

    def retrieve(self, *, target: str, arch: str, op: str, op_class: str, signature: str,
                 shapes: list, n_examples: int = 2) -> dict:
        """-> {"seed": rec|None, "adapt": rec|None, "examples": [rec, ...]}"""
        sig = signature_sha(signature)
        want = {shape_bucket(op_class, s) for s in shapes}
        recs = [r for r in self.records() if r.target == target]
        same_op = [r for r in recs if r.op == op and r.signature_sha == sig]
        exact = [r for r in same_op if r.arch == arch and want <= set(r.buckets)]
        seed = min(exact, key=lambda r: r.slot_us / max(len(r.shapes), 1), default=None)
        others = [r for r in same_op if r is not seed]
        adapt = None if seed else max(others, key=lambda r: (r.arch == arch, r.speedup_vs_seed or 0), default=None)
        ex = [r for r in recs if r.op_class == op_class and r.op != op and (r.speedup_vs_seed or 0) > 1.05]
        best_per_op = {}
        for r in sorted(ex, key=lambda r: -(r.speedup_vs_seed or 0)):
            best_per_op.setdefault(r.op, r)
        return {"seed": seed, "adapt": adapt, "examples": list(best_per_op.values())[:n_examples]}

    def winners(self, *, target: str = "cuda", arch: Optional[str] = None) -> list[ImplRecord]:
        """Fastest verified record per (op, signature, arch, shape set)."""
        best: dict = {}
        for r in self.records():
            if r.target != target or (arch and r.arch != arch):
                continue
            k = (r.op, r.signature_sha, r.arch, json.dumps(r.shapes, sort_keys=True))
            if k not in best or r.slot_us < best[k].slot_us:
                best[k] = r
        return list(best.values())

    def export_winners(self, out: Path, *, target: str = "cuda", arch: Optional[str] = None,
                       min_speedup: float = 1.05) -> Path:
        """Write ``winners.json`` + sources for ModelBlaster's ingest (records
        at least ``min_speedup`` over their problem's seed; seeds excluded)."""
        out = Path(out)
        (out / "code").mkdir(parents=True, exist_ok=True)
        rows = []
        for r in self.winners(target=target, arch=arch):
            if r.speedup_vs_seed is None or r.speedup_vs_seed < min_speedup:
                continue
            (out / "code" / f"{r.code_sha}.cu").write_text(self.code(r))
            rows.append({**asdict(r), "file": f"code/{r.code_sha}.cu"})
        (out / "winners.json").write_text(json.dumps({"format": "kb-winners/1", "winners": rows}, indent=1))
        return out / "winners.json"
