# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experiment recording pipeline (X-145): what multi-turn codegen probes teach, recorded, shared and reused.

A multi-turn codegen step (``agents/experiment_tool.py``, ``KERNELBLASTER_MULTITURN=1``) runs standalone CUDA probe
programs and reads their output before it submits the slot kernel. ``multiturn.jsonl`` keeps one summary line per step
(unchanged). This module keeps the experiments themselves and turns them into shareable knowledge. Everything is
opt-in; with the flags off nothing here runs and nothing is written.

Flags (env)::

    KERNELBLASTER_EXPERIMENT_RECORD=1   write <out>/experiments.jsonl (+ <out>/experiments/blobs/<sha>.cu)
    KERNELBLASTER_EXPERIMENT_INGEST=1   also, after every committed step, ingest into the run's knowledge dir
                                        (<kdir>/experiments/{records,probes}.jsonl + claims of kind "experiment")
    KERNELBLASTER_EXPERIMENT_PROMPT=N   show up to N prior facts / probe templates for the slot's op class and arch
                                        at the start of a multi-turn step (cap KERNELBLASTER_EXPERIMENT_PROMPT_CHARS,
                                        default 800); 0 / unset = off

Pieces:

* :func:`parse_measurements` -- tolerant ``key: value`` / ``name <n> us`` / ``<n> ns`` parser.
* :class:`ExperimentRecorder` -- append-only ``experiments.jsonl`` with three line types: ``experiment`` (one per
  program run), ``step`` (prompt-size accounting) and ``outcome`` (what the step it fed did).
* :func:`finalize` -- join experiment + outcome lines into finished records (one dict per experiment).
* :func:`facts_from_records` / :func:`ingest` -- deterministic facts (claims of kind ``experiment`` scoped to
  ``(target, arch, op class)``), probe templates (diagnostic probes, not part of the technique menu) and links.
* :func:`merge_experiments` / :func:`seed_experiments` -- the global store side (``Policy.experiments`` /
  ``Policy.seed_experiments``; tables ``gk_experiments`` / ``gk_probes`` are created only then).
* :func:`prompt_text` -- the opt-in prompt section.
* :func:`import_legacy` -- ``multiturn.jsonl`` + ``result.json`` of older runs (no sources / purposes).
* CLI: ``python -m kernelblaster.knowledge.v2.experiments {report,ingest,show,import-legacy}``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

SCHEMA = "kb-experiment/1"
ENV_RECORD = "KERNELBLASTER_EXPERIMENT_RECORD"
ENV_INGEST = "KERNELBLASTER_EXPERIMENT_INGEST"
ENV_PROMPT = "KERNELBLASTER_EXPERIMENT_PROMPT"
ENV_PROMPT_CHARS = "KERNELBLASTER_EXPERIMENT_PROMPT_CHARS"
MAX_PROGRAM = 64 * 1024                 # the tool's own cap
OUT_TAIL = 4000                         # output tail kept per record (the tool's cap)
MAX_TEXT = 1200                         # purpose / conclusion text cap
GLOBAL_OUT_TAIL = 1500                  # output tail kept in the global store
MAX_OBS = 24                            # per-fact observations kept
EXPERIMENT = "experiment"               # claim kind

_ON = ("1", "on", "true", "yes")


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in _ON


def record_enabled() -> bool:
    return _flag(ENV_RECORD) or _flag(ENV_INGEST)


def ingest_enabled() -> bool:
    return _flag(ENV_INGEST)


def prompt_n() -> int:
    try:
        v = os.environ.get(ENV_PROMPT, "0").strip().lower()
        return 0 if v in ("", "off", "false", "no") else (3 if v in _ON and v != "1" else int(v))
    except ValueError:
        return 0


def prompt_chars() -> int:
    try:
        return int(os.environ.get(ENV_PROMPT_CHARS, "800") or 800)
    except ValueError:
        return 800


def sha1(*parts) -> str:
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode()).hexdigest()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# measurement parser
# ---------------------------------------------------------------------------------------------------------------

_UNITS = {"us": "us", "µs": "us", "usec": "us", "ms": "ms", "ns": "ns", "s": "s", "sec": "s", "cycles": "cycles",
          "cycle": "cycles", "clk": "cycles", "clks": "cycles", "gb/s": "GB/s", "gflops": "GFLOPS",
          "tflops": "TFLOPS", "%": "%", "x": "x", "bytes": "B", "kb": "KB", "mb": "MB", "regs": "regs"}
_TIME_US = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}
_TOK = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])|[^\W\d][\w.\-/%µ]*|[:=|,;()\[\]]|%")
_NUM = re.compile(r"^[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$")


def _is_num(t: str) -> bool:
    return bool(_NUM.match(t))


def _unit_of(t: Optional[str]) -> Optional[str]:
    return _UNITS.get(t.lower()) if t else None


def parse_measurements(text: str, *, max_items: int = 200) -> list[dict]:
    """Tolerant parse of a program's output into ``[{name, value, unit, line, group, inferred_unit?}]``.

    Handles ``label: 836.9 us``, ``label = 12``, ``median 2087.9 us``, ``mode 1 median 32.5 us`` (a bare integer
    followed by ``:`` or by a word and then a number is part of the label), several pairs per line separated by
    ``|`` / ``,`` (``blk0 wait 1988 mma 1078 gate 446 | ... 416 ns``: a unit-less number takes the unit of the next
    unit-bearing number of the line, flagged ``inferred_unit``). Unlabelled unit-less numbers are ignored. The raw
    output is kept next to this in the record, so a miss here loses nothing."""
    out: list[dict] = []
    for li, line in enumerate((text or "").splitlines()):
        toks = _TOK.findall(line)
        label: list[str] = []
        items: list[dict] = []
        group = 0
        i = 0
        while i < len(toks):
            t = toks[i]
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if t in ("|", ",", ";", "(", ")", "[", "]"):
                label = []
                group += 1
                i += 1
                continue
            if t in (":", "="):
                i += 1
                continue
            if _is_num(t):
                nxt2 = toks[i + 2] if i + 2 < len(toks) else None
                is_int = re.match(r"^[-+]?\d+$", t) is not None
                unit = _unit_of(nxt) if nxt and not _is_num(nxt) else None
                index_like = (is_int and label and unit is None and
                              (nxt in (":", "=") or (abs(float(t)) <= 16 and nxt and not _is_num(nxt) and
                                                     nxt not in "|,;()[]" and nxt2 is not None and
                                                     (_is_num(nxt2) or nxt2 in (":", "=")))))
                if index_like:
                    label.append(t)
                    i += 1
                    continue
                if label or unit:
                    items.append({"name": " ".join(label[-4:]) or "_", "value": float(t), "unit": unit, "line": li,
                                  "group": group})
                    label = []
                    i += 2 if unit else 1
                    continue
                i += 1
                continue
            if _unit_of(t) and items and items[-1].get("unit") is None and i >= 1 and _is_num(toks[i - 1]) and not label:
                items[-1]["unit"] = _unit_of(t)                      # "416 ns" after a skipped unlabeled number
                i += 1
                continue
            label.append(t)
            i += 1
        # unit-less numbers take the unit of the next unit-bearing item of the line (flagged)
        nxt_unit = None
        for it in reversed(items):
            if it["unit"]:
                nxt_unit = it["unit"] if it["unit"] in ("ns", "us", "ms", "cycles") else None
            elif nxt_unit and it["name"] != "_":
                it["unit"], it["inferred_unit"] = nxt_unit, True
        out.extend(items)
        if len(out) >= max_items:
            break
    return out[:max_items]


def to_us(m: dict) -> Optional[float]:
    f = _TIME_US.get(m.get("unit") or "")
    return None if f is None else m["value"] * f


def _stem(name: str) -> str:
    """Name without index-like tokens (``blk0 wait`` -> ``wait``, ``v0 it1`` -> ``""``): grouping key."""
    toks = [t for t in name.split() if not re.search(r"\d", t)]
    return " ".join(toks)


# ---------------------------------------------------------------------------------------------------------------
# program analysis: probe kind, parameter extraction, reference overlap
# ---------------------------------------------------------------------------------------------------------------

def classify_probe(program: Optional[str], out: str = "", ms: Optional[list] = None) -> str:
    """The question a probe asks, from simple signals in the source (or, for legacy records, the output)."""
    p = program or ""
    ms = ms if ms is not None else parse_measurements(out)
    comps = defaultdict(set)
    for m in ms:
        if m["unit"] in ("ns", "us", "cycles") and _stem(m["name"]):
            comps[(m["line"], m.get("group"), m["unit"])].add(_stem(m["name"]))
    phase_out = any(len(v) >= 3 for v in comps.values())
    if "clock64" in p or "clock()" in p or "globaltimer" in p or (not p and phase_out):
        return "phase_breakdown"
    if "cudaOccupancy" in p:
        return "occupancy"
    if re.search(r"__nanosleep|spin|atomicAdd\s*\(\s*&?\s*flag|volatile\s+int\s*\*\s*flag|__threadfence", p):
        return "sync_cost"
    names = Counter(re.sub(r"\d+", "#", m["name"]) for m in ms if m["unit"] in ("us", "ms", "ns"))
    if (re.search(r"variant|mode", p) or any(c >= 2 for c in names.values())) and (
            "cudaEventElapsedTime" in p or not p):
        return "variant_timing"
    if "cudaEventElapsedTime" in p or "cudaEventRecord" in p:
        return "timing"
    return "unknown" if not p else "other"


_DEF_RE = re.compile(r"^[ \t]*#\s*define\s+([A-Za-z_]\w*)\s+(-?\d+)[uUlL]*\s*(?://.*)?$", re.M)
_CONST_RE = re.compile(r"^[ \t]*(?:static\s+)?(?:constexpr|const)\s+(?:unsigned\s+|long\s+)*(?:int|size_t|long|unsigned|"
                       r"int64_t|uint32_t|int32_t)\s+([A-Za-z_]\w*)\s*=\s*(-?\d+)[uUlL]*\s*;", re.M)


def extract_params(program: str, cap: int = 24) -> tuple[str, dict]:
    """Simple generalisation: integer ``#define`` / ``constexpr`` / ``const int`` constants become named
    parameters. Returns ``(template, params)``; the template has each value replaced by ``@@NAME@@``. The template
    sha identifies the probe independent of the shape it was run at."""
    params: dict = {}
    tmpl = program

    def sub(m):
        name = m.group(1)
        if len(params) >= cap or name in params:
            return m.group(0)
        params[name] = int(m.group(2))
        a, b = m.span(2)
        return m.group(0)[: a - m.start()] + f"@@{name}@@" + m.group(0)[b - m.start():]
    tmpl = _DEF_RE.sub(sub, tmpl)
    tmpl = _CONST_RE.sub(sub, tmpl)
    return tmpl, params


_KEYWORDS = frozenset("""int float double void char bool half unsigned long short const static struct class template
typename if else for while do return include define printf cudaEvent_t cudaEventCreate cudaEventRecord
cudaEventSynchronize cudaEventElapsedTime cudaMalloc cudaMemcpy cudaFree cudaDeviceSynchronize main blockIdx threadIdx
blockDim gridDim __global__ __device__ __shared__ __syncthreads size_t nullptr true false auto std namespace using
constexpr uint32_t uint64_t int64_t cuda_fp16 cuda_runtime cstdio cstdlib cstdint vector cmath algorithm""".split())


def _idents(src: str) -> set:
    src = re.sub(r"//[^\n]*|/\*.*?\*/", " ", src or "", flags=re.S)
    return {w for w in re.findall(r"[A-Za-z_]\w{3,}", src) if w not in _KEYWORDS}


def reference_overlap(seed_src: Optional[str], final_src: Optional[str], program: Optional[str]) -> Optional[dict]:
    """Best effort: do the identifiers the step ADDED to the slot source (final minus seed) reuse names from the
    experiment program? ``{shared, new, score}`` (``score = shared / new``); None when a source is missing."""
    if not (seed_src and final_src and program):
        return None
    new = _idents(final_src) - _idents(seed_src)
    shared = new & _idents(program)
    return {"new": len(new), "shared": len(shared), "score": round(len(shared) / max(len(new), 1), 3),
            "names": sorted(shared)[:8]}


def purpose_of(text: str) -> str:
    """The text before the first code fence of a reply: the model's stated purpose for its experiment(s)."""
    head = re.split(r"```", text or "", maxsplit=1)[0]
    return " ".join(head.split())[:MAX_TEXT]


def split_purposes(text: str) -> list[str]:
    """One purpose per ```experiment block: the prose between the previous fence (or the start) and the block."""
    out, pos = [], 0
    for m in re.finditer(r"```experiment[^\n]*\n.*?```", text or "", re.S):
        if not m.group(0).strip("`\n "):
            continue
        out.append(" ".join(text[pos:m.start()].split())[:MAX_TEXT])
        pos = m.end()
    return out


# ---------------------------------------------------------------------------------------------------------------
# recorder (in-run, append-only)
# ---------------------------------------------------------------------------------------------------------------

def _jl(path: Path, obj: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _status(r: dict) -> str:
    st = r.get("stage")
    if st == "refused":
        return "refused"
    if st == "compile":
        return "compile_failed"
    if r.get("timed_out"):
        return "timed_out"
    return "ok" if r.get("ok") else "run_failed"


class ExperimentRecorder:
    """``<out>/experiments.jsonl`` + ``<out>/experiments/blobs/<sha256>.cu`` for one run."""

    def __init__(self, out: Path, *, run_id: str = "", problem: str = "", arch: str = "", target: str = "cuda",
                 dtype: Optional[str] = None):
        self.out = Path(out)
        self.path = self.out / "experiments.jsonl"
        self.blobs = self.out / "experiments" / "blobs"
        self.ctx = {"run_id": run_id, "problem": problem, "arch": arch, "target": target, "dtype": dtype}
        self.n_records = 0
        self.n_steps = 0
        self.n_outcomes = 0
        self.extra_chars = 0
        self._ids: set = set()

    def _blob(self, program: str) -> str:
        sha = sha256(program)
        self.blobs.mkdir(parents=True, exist_ok=True)
        p = self.blobs / f"{sha}.cu"
        if not p.exists():
            tmp = p.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(program)
            os.replace(tmp, p)
        return sha

    def record_step(self, key: str, log, *, slot: str, technique: str, op: str = "", op_class: str = "",
                    shape_bucket: str = "", model: str = "", role: str = "", rationale: str = "",
                    extra_prompt: str = "", fact_ids: Iterable = ()) -> list[str]:
        """One line per experiment of the step (``log.records`` from ``experiment_tool.converse``) + one ``step``
        line. Returns the experiment ids. Ids are deterministic, so a resumed step that reruns its probes does not
        add a second record (readers keep the last line per id)."""
        self.out.mkdir(parents=True, exist_ok=True)
        m = re.match(r"r(\d+)s(\d+)", key or "")
        ridx, step = (int(m.group(1)), int(m.group(2))) if m else (None, None)
        ids = []
        for r in getattr(log, "records", []):
            prog = r.get("program") or ""
            psha = self._blob(prog) if prog else None
            ms = parse_measurements(r.get("out") or "")
            eid = sha1(self.ctx["run_id"], key, r.get("turn"), r.get("idx"), psha)[:16]
            rec = {"type": "experiment", "schema": SCHEMA, "id": eid, **self.ctx, "step_key": key, "ridx": ridx,
                   "step": step, "turn": r.get("turn"), "idx": r.get("idx"), "slot": slot, "technique": technique,
                   "op_kind": op, "op_class": op_class, "shape_bucket": shape_bucket,
                   "gpu": {"device": r.get("device"), "arch": self.ctx["arch"]}, "model": model, "role": role,
                   "purpose": r.get("purpose") or None, "conclusion": r.get("conclusion") or None,
                   "program_sha": psha, "program_bytes": len(prog.encode()) if prog else 0,
                   "status": _status(r), "exit": r.get("exit"), "wall_s": r.get("wall_s"),
                   "out_tail": (r.get("out") or "")[-OUT_TAIL:], "measurements": ms,
                   "probe_kind": classify_probe(prog, r.get("out") or "", ms), "ts": round(time.time(), 1)}
            _jl(self.path, rec)
            ids.append(eid)
            if eid not in self._ids:
                self._ids.add(eid)
                self.n_records += 1
        self.extra_chars += len(extra_prompt)
        self.n_steps += 1
        _jl(self.path, {"type": "step", "schema": SCHEMA, "run_id": self.ctx["run_id"], "step_key": key,
                        "slot": slot, "technique": technique, "turns": getattr(log, "turns", None),
                        "runs": getattr(log, "runs", None), "n_experiments": len(ids),
                        "rationale": (rationale or "")[:MAX_TEXT] or None,
                        "prompt_extra_chars": len(extra_prompt), "prompt_extra_ids": list(fact_ids)})
        return ids

    def record_outcome(self, key: str, *, status: str, speedup=None, reward=None, e2e_gain=None, slot_us_before=None,
                       slot_us_after=None, impl_id=None, outcome_id=None, overlap=None) -> None:
        """What the step the experiments fed did (a line per step, also for steps that ran no experiment: cheap and
        lets a report compare)."""
        self.n_outcomes += 1
        _jl(self.path, {"type": "outcome", "schema": SCHEMA, "run_id": self.ctx["run_id"], "step_key": key,
                        "status": status, "speedup": speedup, "reward": reward, "e2e_gain": e2e_gain,
                        "slot_us_before": slot_us_before, "slot_us_after": slot_us_after, "impl_id": impl_id,
                        "outcome_id": outcome_id, "final_overlap": overlap})

    def pointer(self) -> dict:
        """The result.json block (only written when recording is enabled)."""
        return {"records": self.n_records, "steps": self.n_steps, "path": "experiments.jsonl",
                "blobs": "experiments/blobs", "prompt_extra_chars": self.extra_chars}


# ---------------------------------------------------------------------------------------------------------------
# finalize: join lines into records
# ---------------------------------------------------------------------------------------------------------------

def finalize(lines: list[dict], *, gpu: Optional[dict] = None, blob_dir: Optional[Path] = None) -> list[dict]:
    """Experiment lines joined with the outcome line of their step. Later lines with the same id win. ``gpu``
    (result.json ``gpu``) fills the GPU name / clocks. Records are ordered by (ridx, step, turn, idx)."""
    exps: dict = {}
    outs: dict = {}
    for ln in lines:
        t = ln.get("type")
        if t == "experiment":
            exps[ln["id"]] = ln
        elif t == "outcome":
            outs[(ln.get("run_id"), ln.get("step_key"))] = ln
    res = []
    for e in sorted(exps.values(), key=lambda x: (x.get("ridx") if x.get("ridx") is not None else -1,
                                                  x.get("step") if x.get("step") is not None else -1,
                                                  x.get("turn") or 0, x.get("idx") or 0, x["id"])):
        r = dict(e)
        o = outs.get((e.get("run_id"), e.get("step_key")))
        r["outcome"] = ({k: o.get(k) for k in ("status", "speedup", "reward", "e2e_gain", "slot_us_before",
                                               "slot_us_after", "impl_id", "outcome_id", "final_overlap")}
                        if o else None)
        g = dict(r.get("gpu") or {})
        if gpu:
            g.update({k: v for k, v in gpu.items() if k in ("name", "arch", "label", "index", "sm_mhz", "mem_mhz",
                                                            "clocks") and v is not None})
        r["gpu"] = g
        if blob_dir and r.get("program_sha") and not (Path(blob_dir) / f"{r['program_sha']}.cu").exists():
            r.setdefault("missing", []).append("program_blob")
        res.append(r)
    return res


def load_program(rec: dict, *dirs: Path) -> Optional[str]:
    sha = rec.get("program_sha")
    if not sha:
        return None
    for d in dirs:
        p = Path(d) / f"{sha}.cu"
        if p.exists():
            return p.read_text(errors="replace")
    return None


# ---------------------------------------------------------------------------------------------------------------
# facts
# ---------------------------------------------------------------------------------------------------------------

_VARIANT_WORDS = frozenset(("variant", "version", "config", "impl", "candidate", "option", "kernel", "layout", "v"))

def _observations(rec: dict) -> list[dict]:
    """Candidate facts of one experiment: ``{fact_type, subject, technique, values, unit, text}``."""
    ms = rec.get("measurements") or []
    cls = rec.get("op_class") or "kernel"
    out = []
    # phase profile: >= 3 named components of one time unit in one `|`-separated group of an output line, averaged over lines
    lines = defaultdict(lambda: defaultdict(list))
    for m in ms:
        s = _stem(m["name"])
        if m.get("unit") in ("ns", "us", "cycles") and s and s != "_":
            lines[(m["line"], m.get("group"), m["unit"])][s].append(m["value"])
    prof = defaultdict(lambda: defaultdict(list))
    for (li, grp, unit), comps in lines.items():
        if len(comps) >= 3:
            for k, v in comps.items():
                prof[unit][k].append(sum(v) / len(v))
    for unit, comps in prof.items():
        vals = {k: statistics.median(v) for k, v in comps.items()}
        tot = sum(vals.values())
        if tot <= 0:
            continue
        order = sorted(vals, key=lambda k: -vals[k])
        dom = order[0]
        if vals[dom] / tot < 0.4:
            continue
        out.append({"fact_type": "phase_profile", "subject": dom, "technique": None, "unit": unit,
                    "values": {k: round(vals[k], 3) for k in order},
                    "text": f"{cls} kernels: per-step cost is {dom}-dominated in standalone probes "
                            f"(order: {' > '.join(order[:4])})"})
        break
    # variant spread: >= 2 labelled variants (same stem, different index) in us / ms
    var = defaultdict(dict)
    for m in ms:
        us = to_us(m)
        if us is None or m.get("unit") not in ("us", "ms"):
            continue
        key = re.sub(r"\d+(?:\.\d+)?", "#", m["name"])
        if "#" in key and key.split()[0] in _VARIANT_WORDS:
            var[key][m["name"]] = us
    for key, d in var.items():
        if len(d) >= 2 and min(d.values()) > 0:
            lo, hi = min(d.values()), max(d.values())
            if hi / lo >= 1.5 and rec.get("technique"):
                tech = rec["technique"]
                out.append({"fact_type": "variant_spread", "subject": tech, "technique": tech, "unit": "us",
                            "values": {k: round(v, 3) for k, v in sorted(d.items())},
                            "text": f"probing {tech} on {cls}: the compared variants differed by more than 1.5x"})
                break
    return out



def fact_key(arch: str, op_class: str, fact_type: str, subject: str, technique: Optional[str]) -> str:
    return sha1("kb-exp-fact", arch, op_class, fact_type, subject, technique or "")[:12]


def facts_from_records(records: list[dict]) -> dict:
    """Aggregate experiment observations into facts keyed by (arch, op class, type, subject, technique).
    ``{key: {..., obs: {exp_id: values}, support: [ids], contradict: [ids]}}``.

    Confidence rule: one independent experiment = one distinct ``(run_id, probe template)``; the experiments of a
    fact's key that share that pair count once. An experiment of the same ``(arch, op class, fact type)`` with
    another subject contradicts the fact (e.g. ``mma``-dominated vs ``wait``-dominated)."""
    facts: dict = {}
    by_group: dict = defaultdict(list)
    for r in records:
        if r.get("status") != "ok" or not r.get("arch") and not (r.get("gpu") or {}).get("arch"):
            continue
        arch = r.get("arch") or (r.get("gpu") or {}).get("arch")
        for ob in _observations(r):
            k = fact_key(arch, r.get("op_class") or "", ob["fact_type"], ob["subject"], ob["technique"])
            f = facts.setdefault(k, {"key": k, "arch": arch, "op_class": r.get("op_class") or None,
                                     "target": r.get("target") or "cuda", "fact_type": ob["fact_type"],
                                     "subject": ob["subject"], "technique": ob["technique"], "text": ob["text"],
                                     "unit": ob["unit"], "obs": {}, "support": [], "indep": set(), "problems": set(),
                                     "runs": set(), "shapes": set(), "gpus": set(), "dtype": r.get("dtype")})
            f["obs"][r["id"]] = {"values": ob["values"], "problem": r.get("problem"), "run": r.get("run_id"),
                                 "shape": r.get("shape_bucket"), "step_key": r.get("step_key")}
            ind = (r.get("run_id"), r.get("program_sha") or r.get("probe_kind") or r["id"])
            if ind not in f["indep"]:
                f["indep"].add(ind)
                f["support"].append(r["id"])
            f["problems"].add(r.get("problem") or "")
            f["runs"].add(r.get("run_id") or "")
            if r.get("shape_bucket"):
                f["shapes"].add(r["shape_bucket"])
            g = (r.get("gpu") or {}).get("name")
            if g:
                f["gpus"].add(g)
            by_group[(arch, r.get("op_class") or "", ob["fact_type"], ob["technique"])].append((r["id"], k, ob["subject"]))
    for f in facts.values():
        grp = by_group[(f["arch"], f["op_class"] or "", f["fact_type"], f["technique"])]
        f["contradict"] = [i for i, k, s in grp if s != f["subject"] and i not in f["obs"]]
    return facts


def _summary(f: dict) -> str:
    """Numeric rendering from the observations (median per component over experiments)."""
    comp = defaultdict(list)
    for o in f["obs"].values():
        for k, v in o["values"].items():
            comp[k].append(v)
    unit = f.get("unit") or ""
    parts = [f"{k} ~{statistics.median(v):.4g}{unit}" for k, v in sorted(comp.items(), key=lambda kv: -statistics.median(kv[1]))]
    return ", ".join(parts[:6]) + f" (median of {len(f['obs'])} probe(s))"


def claim_id(key: str) -> str:
    return "cx_" + key


def fact_claim_data(f: dict) -> dict:
    return {"evidence_kind": "experiment", "fact_type": f["fact_type"], "subject": f["subject"], "unit": f.get("unit"),
            "summary": _summary(f), "obs": dict(list(f["obs"].items())[-MAX_OBS:]), "n_obs": len(f["obs"]),
            "problems": sorted(f["problems"])[:12], "runs": sorted(f["runs"])[:12], "shapes": sorted(f["shapes"])[:12],
            "gpus": sorted(f["gpus"])[:4], "experiment_ids": f["support"][-MAX_OBS:]}


def merge_fact_data(a: dict, b: dict) -> dict:
    """Union of two experiment claims' ``data`` (global merge / consolidation): observations by experiment id."""
    a, b = dict(a or {}), dict(b or {})
    obs = dict(a.get("obs") or {})
    obs.update(b.get("obs") or {})
    a["obs"] = dict(list(obs.items())[-MAX_OBS:])
    a["n_obs"] = max(int(a.get("n_obs", 0)), int(b.get("n_obs", 0)), len(obs))
    for k in ("problems", "runs", "shapes", "gpus"):
        a[k] = sorted(set(a.get(k) or []) | set(b.get(k) or []))[:12]
    a["experiment_ids"] = list(dict.fromkeys((a.get("experiment_ids") or []) + (b.get("experiment_ids") or [])))[-MAX_OBS:]
    comp = defaultdict(list)
    for o in a["obs"].values():
        for k, v in (o.get("values") or {}).items():
            comp[k].append(v)
    if comp:
        unit = a.get("unit") or ""
        a["summary"] = ", ".join(f"{k} ~{statistics.median(v):.4g}{unit}" for k, v in
                                 sorted(comp.items(), key=lambda kv: -statistics.median(kv[1]))[:6]) + \
            f" (median of {len(a['obs'])} probe(s))"
    return a


def ingest_facts(claims, records: list[dict]) -> dict:
    """Add / update the facts of ``records`` in a ClaimStore (idempotent: ids are deterministic, evidence ids
    de-duplicate). Returns counts."""
    from ..claims import Claim
    facts = facts_from_records(records)
    added = updated = 0
    # agreement across ingests: an experiment that found another dominant component than an existing fact of the
    # same (arch, op class, type, technique) contradicts it, and that fact's experiments contradict the new one
    existing = [c for c in claims.claims.values() if c.kind == EXPERIMENT and c.status == "active"]
    for k, f in sorted(facts.items()):
        for c in existing:
            if (c.arch, c.op_class, c.technique, (c.data or {}).get("fact_type")) == \
                    (f["arch"], f["op_class"], f["technique"], f["fact_type"]) and c.id != claim_id(k):
                f["contradict"] = list(dict.fromkeys(f["contradict"] + list((c.data or {}).get("experiment_ids") or [])))
                c.add_contradiction(f["support"])
    for k, f in sorted(facts.items()):
        cid = claim_id(k)
        data = fact_claim_data(f)
        cur = claims.claims.get(cid)
        if cur is None:
            c = Claim(kind=EXPERIMENT, text=f["text"], target=f["target"], arch=f["arch"], op_class=f["op_class"],
                      technique=f["technique"], data=data, source="experiment", id=cid)
            c.add_support(f["support"])
            c.add_contradiction(f["contradict"])
            claims.claims[cid] = c
            claims._touch(c)
            added += 1
        else:
            before = (cur.support, cur.contradict, json.dumps(cur.data, sort_keys=True))
            cur.add_support(f["support"])
            cur.add_contradiction(f["contradict"])
            cur.data = merge_fact_data(cur.data, data)
            if cur.status != "active":
                cur.status = "active"
            if before != (cur.support, cur.contradict, json.dumps(cur.data, sort_keys=True)):
                claims._touch(cur)
                updated += 1
    claims.save()
    return {"facts": len(facts), "added": added, "updated": updated}


# ---------------------------------------------------------------------------------------------------------------
# probe templates
# ---------------------------------------------------------------------------------------------------------------

def probes_from_records(records: list[dict], blob_dirs: Iterable[Path] = ()) -> list[dict]:
    """Diagnostic probe templates keyed by (op class, question, template sha) with usage statistics. Records with
    no program (legacy) only count under a source-less ``kind`` entry (``template_sha`` null, no source)."""
    blob_dirs = [Path(d) for d in blob_dirs]
    book: dict = {}
    for r in records:
        prog = load_program(r, *blob_dirs)
        tsha, params, tmpl_sha = None, {}, None
        if prog:
            tmpl, params = extract_params(prog)
            tsha = sha256(tmpl)
        q = r.get("probe_kind") or "unknown"
        key = (r.get("op_class") or "", q, tsha)
        pid = "probe_" + sha1(*key)[:12]
        p = book.setdefault(pid, {"id": pid, "kind": "diagnostic_probe", "op_class": r.get("op_class") or None,
                                  "question": q, "template_sha": tsha, "template_blob": tsha,
                                  "params": params, "title": None, "arch": r.get("arch"), "uses": 0, "runs_ok": 0,
                                  "runs_failed": 0, "next_improved": 0, "next_neutral": 0, "next_regressed": 0,
                                  "next_error": 0, "problems": set(), "exp_ids": [], "programs": set(),
                                  "legacy_only": prog is None,
                                  "ontology": {"id": f"diag:{r.get('op_class') or '*'}:{q}", "kind": "diagnostic_probe",
                                               "optional": True, "in_technique_menu": False}})
        if r["id"] in p["exp_ids"]:
            continue
        p["exp_ids"].append(r["id"])
        p["uses"] += 1
        p["runs_ok" if r.get("status") == "ok" else "runs_failed"] += 1
        p["problems"].add(r.get("problem") or "")
        if r.get("program_sha"):
            p["programs"].add(r["program_sha"])
        if not p["title"] and r.get("purpose"):
            p["title"] = r["purpose"][:160]
        if tsha and tsha not in p.get("_src", ()):
            p.setdefault("_src", {})[tsha] = prog
        o = r.get("outcome") or {}
        st = o.get("status")
        key2 = {"improved": "next_improved", "neutral": "next_neutral", "regressed": "next_regressed"}.get(st)
        if st in ("compile_error", "runtime_error", "wrong_output"):
            key2 = "next_error"
        if key2:
            p[key2] += 1
    out = []
    for p in book.values():
        p["problems"] = sorted(p["problems"])[:12]
        p["programs"] = sorted(p["programs"])[:8]
        p["exp_ids"] = p["exp_ids"][-MAX_OBS:]
        p.pop("_src", None)
        out.append(p)
    return sorted(out, key=lambda p: (-p["uses"], p["id"]))


def write_run_artifacts(out_dir: Path, kdir: Path, *, gpu: Optional[dict] = None) -> dict:
    """``<out>/experiments.jsonl`` -> ``<kdir>/experiments/{records,probes}.jsonl`` + blobs (idempotent rewrite)."""
    out_dir, kdir = Path(out_dir), Path(kdir)
    lines = read_jsonl(out_dir / "experiments.jsonl")
    blob_src = out_dir / "experiments" / "blobs"
    recs = finalize(lines, gpu=gpu, blob_dir=blob_src)
    ed = kdir / "experiments"
    (ed / "blobs").mkdir(parents=True, exist_ok=True)
    for r in recs:
        sha = r.get("program_sha")
        if sha and (blob_src / f"{sha}.cu").exists() and not (ed / "blobs" / f"{sha}.cu").exists():
            shutil.copyfile(blob_src / f"{sha}.cu", ed / "blobs" / f"{sha}.cu")
    probes = probes_from_records(recs, [ed / "blobs"])
    _write_jsonl(ed / "records.jsonl", recs)
    _write_jsonl(ed / "probes.jsonl", probes)
    return {"records": recs, "probes": probes}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text("".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in rows))
    os.replace(tmp, path)


def ingest(out_dir: Path, kdir: Path, claims=None, *, gpu: Optional[dict] = None) -> dict:
    """Run-level ingest: records / probes into ``<kdir>/experiments`` and facts into the run's claim store
    (``claims``: a ClaimStore, or None to open ``<kdir>/knowledge.db`` through KnowledgeV2). Idempotent."""
    art = write_run_artifacts(out_dir, kdir, gpu=gpu)
    rep = {"records": len(art["records"]), "probes": len(art["probes"])}
    if claims is None:
        from . import KnowledgeV2
        claims = KnowledgeV2(Path(kdir), seed_mined=False).claims
    rep.update(ingest_facts(claims, art["records"]))
    return rep


# ---------------------------------------------------------------------------------------------------------------
# links: experiments that preceded a winner
# ---------------------------------------------------------------------------------------------------------------

class ExperimentLinks:
    """impl id -> experiments that fed the step that produced it (and, for ``children``, steps run on top of it)."""

    def __init__(self, records: Iterable[dict] = ()):
        self.by_impl: dict = defaultdict(list)
        for r in records:
            o = r.get("outcome") or {}
            if o.get("impl_id"):
                self.by_impl[o["impl_id"]].append(r)

    @classmethod
    def from_db(cls, db_path, *, part: Optional[str] = None) -> Optional["ExperimentLinks"]:
        """Records of a global snapshot's ``gk_experiments`` or of ``<kdir>/experiments/records.jsonl`` next to a run
        db; None when neither exists (old stores)."""
        import sqlite3
        p = Path(db_path)
        recs = []
        fs = [p.parent / "experiments" / n for n in ("records.jsonl", "seed_records.jsonl")]
        if any(f.exists() for f in fs):
            seen = {}
            for f in fs:
                for r in read_jsonl(f):
                    seen.setdefault(r["id"], r)
            recs = list(seen.values())
        else:
            try:
                c = sqlite3.connect(f"file:{p.resolve()}?mode=ro", uri=True, timeout=10)
                try:
                    if c.execute("SELECT 1 FROM sqlite_master WHERE name='gk_experiments'").fetchone():
                        q = "SELECT rec FROM gk_experiments" + (" WHERE part=?" if part else "")
                        recs = [json.loads(x[0]) for x in c.execute(q, (part,) if part else ())]
                finally:
                    c.close()
            except sqlite3.Error:
                return None
        return cls(recs) if recs else None

    def for_impl(self, impl_id: str, k: int = 3) -> list[dict]:
        out = []
        for r in self.by_impl.get(impl_id, [])[:k]:
            out.append({"id": r["id"], "probe_kind": r.get("probe_kind"), "technique": r.get("technique"),
                        "purpose": (r.get("purpose") or "")[:120] or None,
                        "measurements": [f"{m['name']}={m['value']:g}{m.get('unit') or ''}"
                                         for m in (r.get("measurements") or [])[:4]]})
        return out


# ---------------------------------------------------------------------------------------------------------------
# retrieval / prompt
# ---------------------------------------------------------------------------------------------------------------

def facts_for(claims, *, arch: Optional[str], op_class: Optional[str], technique: Optional[str] = None,
              k: int = 3) -> list:
    """Active experiment claims for ``arch`` / ``op_class`` (most support first, then fewest contradictions)."""
    out = []
    for c in claims.claims.values():
        if c.kind != EXPERIMENT or c.status != "active" or c.support <= c.contradict:
            continue
        if (c.arch and arch and c.arch != arch) or (c.op_class and op_class and c.op_class != op_class):
            continue
        if c.technique and technique and c.technique != technique:
            continue
        out.append(c)
    out.sort(key=lambda c: (-(c.support - c.contradict), 0 if c.technique == technique else 1, c.id))
    return out[:k]


def read_probes(*dirs: Path) -> list[dict]:
    rows: dict = {}
    for d in dirs:
        for f in ("seed_probes.jsonl", "probes.jsonl"):
            for p in read_jsonl(Path(d) / "experiments" / f):
                if p["id"] not in rows or p.get("uses", 0) >= rows[p["id"]].get("uses", 0):
                    rows[p["id"]] = p
    return list(rows.values())


def prompt_text(claims, kdir: Optional[Path], *, arch: str, op_class: str, technique: Optional[str] = None,
                n: Optional[int] = None, max_chars: Optional[int] = None) -> tuple[str, list]:
    """The opt-in prompt section: up to ``n`` facts / probe templates for the slot's class and arch, at most
    ``max_chars``. Returns (text, ids shown); ("", []) when there is nothing. Pure function of its inputs."""
    n = prompt_n() if n is None else n
    max_chars = prompt_chars() if max_chars is None else max_chars
    if n <= 0:
        return "", []
    items = []
    for c in facts_for(claims, arch=arch, op_class=op_class, technique=technique, k=n):
        d = c.data or {}
        items.append((c.id, f"- [{d.get('n_obs', c.support)} probe(s), +{c.support}/-{c.contradict}] {c.text}; "
                            f"{d.get('summary', '')}".rstrip("; ")))
    if kdir is not None:
        probes = [p for p in read_probes(Path(kdir)) if (p.get("op_class") in (op_class, None)) and
                  (not p.get("arch") or p.get("arch") == arch)]
        probes.sort(key=lambda p: (-(p.get("next_improved", 0) + 0.5 * p.get("next_neutral", 0)), -p.get("uses", 0), p["id"]))
        for p in probes:
            if not p.get("template_sha"):                 # legacy entries have no source: nothing to reuse
                continue
            if len(items) >= n:
                break
            used = f"used {p.get('uses', 0)}x, next step improved {p.get('next_improved', 0)}x"
            items.append((p["id"], f"- probe template '{p.get('question')}' ({used})" +
                          (f": {p['title']}" if p.get("title") else "")))
    items = items[:n]
    if not items:
        return "", []
    head = "Experiments already run for this kind of kernel on this GPU (skip redundant probes):\n"
    text, ids = head, []
    for i, ln in items:
        if len(text) + len(ln) + 1 > max_chars:
            break
        text += ln + "\n"
        ids.append(i)
    return (text.rstrip(), ids) if ids else ("", [])


# ---------------------------------------------------------------------------------------------------------------
# global store (Policy.experiments / seed_experiments)
# ---------------------------------------------------------------------------------------------------------------

GK_EXP_SCHEMA = """
CREATE TABLE IF NOT EXISTS gk_experiments (id TEXT PRIMARY KEY, part TEXT, problem TEXT, run_uid TEXT, ts TEXT,
    rec TEXT);
CREATE INDEX IF NOT EXISTS gk_experiments_pp ON gk_experiments (part, problem);
CREATE TABLE IF NOT EXISTS gk_probes (part TEXT, id TEXT, data TEXT, prov TEXT, PRIMARY KEY (part, id));
"""


def _slim(r: dict) -> dict:
    s = dict(r)
    s["out_tail"] = (s.get("out_tail") or "")[-GLOBAL_OUT_TAIL:]
    return s


def merge_experiments(g, store, kdir, *, info, part: str, seq: int, pol) -> dict:
    """Merge ``<kdir>/experiments/records.jsonl`` into ``gk_experiments`` (insert-or-replace by id: a re-merge of
    the same records is a no-op; an outcome that arrives later updates the row), copy new program blobs to
    ``<store>/experiment_code/`` and rebuild the touched probes. Called inside the merge transaction."""
    recs = read_jsonl(Path(kdir) / "experiments" / "records.jsonl") if kdir else []
    if not recs:
        return {"records": 0}
    for stmt in GK_EXP_SCHEMA.split(";"):          # not executescript: it would commit the merge transaction
        if stmt.strip():
            g.execute(stmt)
    code = Path(store.root) / "experiment_code"
    code.mkdir(parents=True, exist_ok=True)
    new = 0
    for r in recs:
        row = g.execute("SELECT rec FROM gk_experiments WHERE id=?", (r["id"],)).fetchone()
        s = _slim(r)
        s.pop("missing", None)
        if row is None:
            new += 1
        elif json.loads(row[0]) == s:
            continue
        g.execute("INSERT OR REPLACE INTO gk_experiments VALUES (?,?,?,?,?,?)",
                  (r["id"], part, info.problem, info.run_uid, r.get("ts") and str(r["ts"]),
                   json.dumps(s, sort_keys=True, default=str)))
        sha = r.get("program_sha")
        src = Path(kdir) / "experiments" / "blobs" / f"{sha}.cu" if sha else None
        if src and src.exists() and not (code / f"{sha}.cu").exists():
            tmp = (code / f"{sha}.cu").with_suffix(f".tmp{os.getpid()}")
            shutil.copyfile(src, tmp)
            os.replace(tmp, code / f"{sha}.cu")
    cap = int(getattr(pol, "max_experiments", 4000))
    n = g.execute("SELECT COUNT(*) FROM gk_experiments WHERE part=?", (part,)).fetchone()[0]
    dropped = 0
    if n > cap:
        for (i,) in g.execute("SELECT id FROM gk_experiments WHERE part=? ORDER BY ts, id LIMIT ?",
                              (part, n - cap)).fetchall():
            g.execute("DELETE FROM gk_experiments WHERE id=?", (i,))
            dropped += 1
    allrecs = [json.loads(x[0]) for x in g.execute("SELECT rec FROM gk_experiments WHERE part=?", (part,))]
    probes = probes_from_records(allrecs, [code])
    g.execute("DELETE FROM gk_probes WHERE part=?", (part,))
    for p in probes:
        g.execute("INSERT INTO gk_probes VALUES (?,?,?,?)", (part, p["id"], json.dumps(p, sort_keys=True),
                                                            json.dumps(info.prov(), default=str)))
    return {"records": len(recs), "new": new, "dropped": dropped, "probes": len(probes)}


def seed_experiments(g, store, part: str, out_dir: Path, *, max_probes: int = 60, max_records: int = 400) -> dict:
    """Seed side: the partition's probes and the slim records (links) into ``<out>/experiments/``."""
    names = {r[0] for r in g.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "gk_probes" not in names:
        return {"probes": 0, "records": 0}
    ed = Path(out_dir) / "experiments"
    ed.mkdir(parents=True, exist_ok=True)
    probes = [json.loads(d) for (d,) in g.execute("SELECT data FROM gk_probes WHERE part=?", (part,))]
    probes.sort(key=lambda p: (-p.get("uses", 0), p["id"]))
    _write_jsonl(ed / "seed_probes.jsonl", probes[:max_probes])
    recs = [json.loads(d) for (d,) in g.execute("SELECT rec FROM gk_experiments WHERE part=? ORDER BY ts DESC, id LIMIT ?",
                                                (part, max_records))]
    _write_jsonl(ed / "seed_records.jsonl", recs)
    return {"probes": min(len(probes), max_probes), "records": len(recs)}


# ---------------------------------------------------------------------------------------------------------------
# legacy import (multiturn.jsonl + result.json, no sources / purposes)
# ---------------------------------------------------------------------------------------------------------------

_CLASS_GUESS = (("gru", "recurrent"), ("lstm", "recurrent"), ("rnn", "recurrent"), ("conv", "conv"),
                ("gemm", "gemm"), ("matmul", "gemm"), ("attention", "attention"), ("softmax", "reduction"),
                ("layernorm", "reduction"), ("norm", "reduction"))


def guess_op_class(slot: str) -> str:
    s = (slot or "").lower().split("__")[0]
    for k, v in _CLASS_GUESS:
        if k in s:
            return v
    return re.sub(r"_(f16|f32|fp16|fp32|bf16)$", "", s) or "unknown"


def import_legacy(run_dir: Path) -> list[dict]:
    """Records from an older run dir: ``multiturn.jsonl`` items (``out_head`` = first 300 characters of the output)
    joined with ``result.json`` (arch, GPU, model, step outcomes). Missing: program, purpose, conclusion. The ids
    are deterministic, so importing twice gives the same records."""
    rd = Path(run_dir)
    mt = read_jsonl(rd / "multiturn.jsonl")
    res = {}
    if (rd / "result.json").exists():
        try:
            res = json.loads((rd / "result.json").read_text())
        except ValueError:
            res = {}
    steps = {(s.get("rollout"), s.get("step")): s for s in res.get("steps") or []}
    arch = res.get("arch") or ""
    gpu = {k: v for k, v in (res.get("gpu") or {}).items() if k in ("name", "arch", "label", "index")}
    run_id = res.get("run_id") or rd.name
    problem = res.get("problem") or rd.name
    recs = []
    for st in mt:
        key = st.get("key") or ""
        m = re.match(r"r(\d+)s(\d+)", key)
        ridx, step = (int(m.group(1)), int(m.group(2))) if m else (None, None)
        sd = steps.get((ridx, step)) or {}
        ctx = sd.get("context") or {}
        out = {"status": sd.get("status"), "speedup": (sd["slot_us_before"] / sd["slot_us_after"]
                                                       if sd.get("slot_us_after") else None),
               "reward": sd.get("reward"), "e2e_gain": (ctx.get("e2e") or {}).get("gain"),
               "slot_us_before": sd.get("slot_us_before"), "slot_us_after": sd.get("slot_us_after"),
               "impl_id": None, "outcome_id": None, "final_overlap": None} if sd else None
        for i, it in enumerate(st.get("items") or []):
            txt = it.get("out_head") or ""
            ms = parse_measurements(txt)
            status = _status({"stage": it.get("stage"), "ok": it.get("exit") == 0, "timed_out": False})
            recs.append({
                "type": "experiment", "schema": SCHEMA, "id": sha1("legacy", run_id, key, it.get("turn"), i)[:16],
                "run_id": run_id, "problem": problem, "arch": arch, "target": res.get("target") or "cuda",
                "dtype": None, "step_key": key, "ridx": ridx, "step": step, "turn": it.get("turn"), "idx": i,
                "slot": st.get("slot"), "technique": st.get("technique"), "op_kind": None,
                "op_class": guess_op_class(st.get("slot") or ""), "shape_bucket": None, "gpu": dict(gpu),
                "model": ctx.get("model") or res.get("model"), "role": ctx.get("model_role"), "purpose": None,
                "conclusion": None, "program_sha": None, "program_bytes": 0, "status": status,
                "exit": it.get("exit"), "wall_s": it.get("wall_s"), "out_tail": txt, "measurements": ms,
                "probe_kind": classify_probe(None, txt, ms), "ts": None, "outcome": out,
                "missing": ["program", "purpose", "conclusion", "shape_bucket", "op_kind", "full_output"],
                "legacy": True})
    return recs


# ---------------------------------------------------------------------------------------------------------------
# reports / CLI
# ---------------------------------------------------------------------------------------------------------------

def load_records(path: Path, *, part: Optional[str] = None) -> tuple[list[dict], dict]:
    """Records of a run dir (experiments.jsonl, else legacy multiturn.jsonl), a knowledge dir
    (``experiments/records.jsonl``) or a global store root. Returns (records, info)."""
    p = Path(path)
    info = {"path": str(p)}
    if (p / "current").exists() and (p / "snapshots").exists():
        import sqlite3
        vid = (p / "current").read_text().strip()
        c = sqlite3.connect(f"file:{p / 'snapshots' / (vid + '.db')}?mode=ro", uri=True)
        try:
            if not c.execute("SELECT 1 FROM sqlite_master WHERE name='gk_experiments'").fetchone():
                return [], dict(info, kind="store", version=vid, note="no gk_experiments table (flag never on)")
            q = "SELECT rec FROM gk_experiments" + (" WHERE part=?" if part else "")
            recs = [json.loads(x[0]) for x in c.execute(q, (part,) if part else ())]
        finally:
            c.close()
        return recs, dict(info, kind="store", version=vid)
    if (p / "experiments" / "records.jsonl").exists():
        return read_jsonl(p / "experiments" / "records.jsonl"), dict(info, kind="knowledge-dir")
    if (p / "experiments.jsonl").exists():
        gpu = None
        if (p / "result.json").exists():
            try:
                gpu = json.loads((p / "result.json").read_text()).get("gpu")
            except ValueError:
                pass
        return finalize(read_jsonl(p / "experiments.jsonl"), gpu=gpu), dict(info, kind="run")
    if (p / "multiturn.jsonl").exists():
        return import_legacy(p), dict(info, kind="legacy")
    return [], dict(info, kind="none")


def report(records: list[dict], claims=None) -> dict:
    facts = facts_from_records(records)
    probes = probes_from_records(records)
    stat = Counter(r.get("status") for r in records)
    pk = Counter(r.get("probe_kind") for r in records)
    oc = Counter((r.get("outcome") or {}).get("status") or "no_outcome" for r in records)
    missing = Counter(m for r in records for m in r.get("missing") or [])
    nm = sum(1 for r in records if r.get("measurements"))
    return {"experiments": len(records), "steps": len({(r.get("run_id"), r.get("step_key")) for r in records}),
            "status": dict(stat), "probe_kinds": dict(pk), "step_outcomes": dict(oc),
            "with_measurements": nm, "missing_fields": dict(missing),
            "facts": [{"key": k, "type": f["fact_type"], "subject": f["subject"], "arch": f["arch"],
                       "op_class": f["op_class"], "technique": f["technique"], "text": f["text"],
                       "summary": _summary(f), "independent": len(f["support"]), "contradicting": len(f["contradict"])}
                      for k, f in sorted(facts.items())],
            "probes": [{k: p[k] for k in ("id", "op_class", "question", "uses", "next_improved", "next_neutral",
                                          "next_regressed", "next_error", "legacy_only")} for p in probes]}


def _fmt(rep: dict) -> str:
    L = [f"experiments: {rep['experiments']} in {rep['steps']} step(s); status {rep['status']}",
         f"probe kinds: {rep['probe_kinds']}", f"step outcomes: {rep['step_outcomes']}",
         f"with parsed measurements: {rep['with_measurements']}"]
    if rep["missing_fields"]:
        L.append(f"missing fields (legacy): {rep['missing_fields']}")
    L.append(f"facts ({len(rep['facts'])}):")
    for f in rep["facts"]:
        L.append(f"  [{f['type']}] {f['arch']} {f['op_class']}"
                 + (f" / {f['technique']}" if f["technique"] else "") + f": {f['text']}")
        L.append(f"      {f['summary']}; independent={f['independent']} contradicting={f['contradicting']}")
    L.append(f"probe templates ({len(rep['probes'])}):")
    for p in rep["probes"]:
        L.append(f"  {p['id']} {p['op_class']} / {p['question']}: used {p['uses']}x, next step improved "
                 f"{p['next_improved']} neutral {p['next_neutral']} regressed {p['next_regressed']} error "
                 f"{p['next_error']}" + (" (no source: legacy)" if p["legacy_only"] else ""))
    return "\n".join(L)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kernelblaster.knowledge.v2.experiments", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="summarize a run dir, a knowledge dir or a global store")
    r.add_argument("path")
    r.add_argument("--part", default=None, help="store partition, e.g. 'cuda:sm_89|fp16'")
    r.add_argument("--json", action="store_true")
    i = sub.add_parser("ingest", help="run dir -> <kdir>/experiments/ + experiment claims in <kdir>/knowledge.db")
    i.add_argument("run_dir")
    i.add_argument("--kdir", required=True)
    i.add_argument("--legacy", action="store_true", help="the run dir has only multiturn.jsonl / result.json")
    s = sub.add_parser("show", help="print one experiment record (with its program) by id prefix")
    s.add_argument("path")
    s.add_argument("id")
    s.add_argument("--part", default=None)
    lg = sub.add_parser("import-legacy", help="legacy multiturn.jsonl -> records.jsonl (+ report); same as ingest --legacy")
    lg.add_argument("run_dir")
    lg.add_argument("--kdir", default=None, help="write <kdir>/experiments/ and claims (default: report only)")
    a = ap.parse_args(argv)
    if a.cmd == "report":
        recs, info = load_records(Path(a.path), part=a.part)
        rep = report(recs)
        rep["source"] = info
        print(json.dumps(rep, indent=1, default=str) if a.json else f"source: {info}\n" + _fmt(rep))
        return 0
    if a.cmd in ("ingest", "import-legacy"):
        rd = Path(a.run_dir)
        legacy = a.cmd == "import-legacy" or a.legacy or not (rd / "experiments.jsonl").exists()
        kdir = getattr(a, "kdir", None)
        if legacy:
            recs = import_legacy(rd)
            rep = report(recs)
            if kdir:
                ed = Path(kdir) / "experiments"
                ed.mkdir(parents=True, exist_ok=True)
                _write_jsonl(ed / "records.jsonl", recs)
                _write_jsonl(ed / "probes.jsonl", probes_from_records(recs))
                from . import KnowledgeV2
                rep["ingest"] = ingest_facts(KnowledgeV2(Path(kdir), seed_mined=False).claims, recs)
            print(_fmt(rep) + (f"\ningest: {rep['ingest']}" if kdir else ""))
            return 0
        gpu = None
        if (rd / "result.json").exists():
            try:
                gpu = json.loads((rd / "result.json").read_text()).get("gpu")
            except ValueError:
                pass
        print(json.dumps(ingest(rd, Path(a.kdir), gpu=gpu), indent=1))
        return 0
    if a.cmd == "show":
        recs, _ = load_records(Path(a.path), part=a.part)
        hit = [x for x in recs if x["id"].startswith(a.id)]
        if not hit:
            print("no such experiment", file=sys.stderr)
            return 1
        x = hit[0]
        prog = None
        for d in (Path(a.path) / "experiments" / "blobs", Path(a.path) / "blobs", Path(a.path) / "experiment_code"):
            prog = prog or load_program(x, d)
        print(json.dumps(x, indent=1, default=str))
        if prog:
            print("---- program ----\n" + prog)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
