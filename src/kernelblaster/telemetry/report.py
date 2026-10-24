# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derived metrics, categories, and the ``telemetry.jsonl`` report
(``scripts/telemetry_report.py``)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from .core import Collector, LLMStats, Span

# span name -> category (first match; prefixes end with ".")
CATEGORIES = [
    ("llm.backoff", "llm_backoff"),
    ("llm.", "llm_wait"),
    ("build.queue", "build_queue"),
    ("gpu.queue", "gpu_queue"),
    ("build", "compile"),
    ("ncu.", "profile_ncu"),
    ("driver.check_", "checks"),
    ("driver.time_", "timing"),
    ("driver.graph_build", "timing"),
    ("driver.", "driver_startup"),
    ("run.exec", "process_overhead"),
    ("run.stage", "transfer"),
    ("run", "run_other"),
    ("eval", "eval_overhead"),
    ("knowledge.", "knowledge"),
    ("reflect", "knowledge"),
    ("upkeep", "knowledge"),
    ("io.", "io"),
    ("setup.", "setup"),
    ("prompt.", "prompt"),
    ("extract", "extract"),
    ("mb.kb_wait", "kb_wait"),
    ("mb.", "mb"),
]

CATEGORY_HELP = {
    "llm_wait": "waiting for LLM responses (codegen, fix, reflect, upkeep)",
    "llm_backoff": "retry back-off sleeps after LLM errors / throttling",
    "build_queue": "waiting for a compile slot (build limiter)",
    "gpu_queue": "waiting for a GPU lease",
    "compile": "nvcc / link / build-cache lookups",
    "profile_ncu": "NCU profiling runs",
    "checks": "driver correctness checks (partition + isolated ops)",
    "timing": "driver timing loops (partition, ops, graph replay)",
    "driver_startup": "driver start-up: process start, CUDA init, allocation",
    "process_overhead": "process spawn, cross-process GPU lock wait, teardown",
    "transfer": "copying binaries / data to a remote host",
    "run_other": "run bookkeeping (lease, infra retries)",
    "eval_overhead": "evaluator bookkeeping (parse, work dirs, server round trips)",
    "knowledge": "knowledge lookup / record / reflection / upkeep (non-LLM part)",
    "io": "file I/O (work dirs, result files)",
    "setup": "run start-up: knowledge stores, loading problems",
    "prompt": "prompt building",
    "extract": "code extraction from responses",
    "agent": "agent loop bookkeeping not covered by a finer span",
    "kb_wait": "ModelBlaster waiting on KernelBlaster runs",
    "mb": "ModelBlaster's own phases",
}


def category(name: str) -> str:
    """Category of a span name: a pattern ending in "." or "_" matches by prefix, any
    other pattern matches itself and its dotted children."""
    for pat, cat in CATEGORIES:
        if pat.endswith((".", "_")):
            if name.startswith(pat):
                return cat
        elif name == pat or name.startswith(pat + "."):
            return cat
    return "agent"


def by_category(time_s: dict) -> dict:
    out: dict[str, float] = defaultdict(float)
    for k, v in time_s.items():
        out[category(k)] += v
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def derive(s: Optional[dict]) -> dict:
    """Derived metrics from a collector summary (``Collector.summary``)."""
    if not s:
        return {}
    wall = s.get("wall_s") or 0.0
    c = s.get("counters", {})
    ev = s.get("evals", {})
    llm = s.get("llm", {}).get("total", {})
    steps, wins = c.get("step", 0), c.get("win", 0)
    cands = sum(ev.get(k, {}).get("n", 0) for k in ("candidate", "fix"))
    cats = by_category(s.get("time_s", {}))
    usd = llm.get("cost_usd") or llm.get("est_usd") or 0.0
    toks = (llm.get("input_tokens") or 0) + (llm.get("output_tokens") or 0)
    gpu = s.get("gpu_busy_s", {})
    hits, miss = c.get("build.cache_hit", 0), c.get("build.cache_miss", 0)
    d = {
        "steps": steps,
        "accepted_wins": wins,
        "steps_per_hour": round(steps / (wall / 3600.0), 2) if wall and steps else None,
        "candidates_per_min": round(cands / (wall / 60.0), 3) if wall and cands else None,
        "llm_wait_frac": round(cats.get("llm_wait", 0.0) / wall, 4) if wall else None,
        "gpu_busy_s": round(sum(gpu.values()), 3),
        "gpu_idle_frac": (round(1.0 - sum(gpu.values()) / (wall * max(len(gpu), 1)), 4) if wall else None),
        "compile_cache_hit_rate": round(hits / (hits + miss), 4) if hits + miss else None,
        "usd": round(usd, 5),
        "usd_is_estimate": not llm.get("cost_usd"),
        "usd_per_win": round(usd / wins, 5) if wins else None,
        "tokens_per_win": round(toks / wins, 1) if wins else None,
        "truncation_rate": round(llm.get("truncated", 0) / llm["calls"], 4) if llm.get("calls") else None,
        "mean_llm_latency_s": round(llm["latency_s"] / llm["calls"], 3) if llm.get("calls") else None,
        "time_by_category_s": {k: round(v, 3) for k, v in cats.items()},
        "time_by_category_pct": {k: round(100.0 * v / wall, 1) for k, v in cats.items()} if wall else {},
    }
    return d


def full_summary(sp) -> Optional[dict]:
    """Collector summary + derived metrics of a ``collect=True`` span."""
    c = getattr(sp, "collector", None)
    if c is None:
        return None
    s = c.summary()
    s["derived"] = derive(s)
    return s


def rollup(summaries: Iterable[Optional[dict]]) -> dict:
    """Sum several collector summaries (e.g. ModelBlaster's KernelBlaster runs):
    wall clocks, time by category, LLM totals and per role/model, evals,
    counters; derived ratios recomputed over the sums."""
    out = {"runs": 0, "wall_s": 0.0, "time_by_category_s": defaultdict(float), "evals": {},
           "counters": defaultdict(float), "gpu_busy_s": 0.0}
    llm = LLMStats()
    for s in summaries:
        if not s:
            continue
        out["runs"] += 1
        out["wall_s"] += s.get("wall_s") or 0.0
        for k, v in by_category(s.get("time_s", {})).items():
            out["time_by_category_s"][k] += v
        for k, v in (s.get("counters") or {}).items():
            out["counters"][k] += v
        for k, v in (s.get("evals") or {}).items():
            e = out["evals"].setdefault(k, {"n": 0, "s": 0.0, "failed": 0})
            for f in e:
                e[f] += v.get(f, 0)
        out["gpu_busy_s"] += sum((s.get("gpu_busy_s") or {}).values())
        part = LLMStats()
        lj = s.get("llm") or {}
        for k in part.total:
            part.total[k] = (lj.get("total") or {}).get(k, 0) or 0
        for b in lj.get("by_role_model") or []:
            part.by[f"{b['role']} {b['model']}"] = {k: b.get(k, 0) for k in ("role", "model", *part.total)}
        llm.merge(part)
    wall = out["wall_s"]
    cats = dict(sorted(out["time_by_category_s"].items(), key=lambda kv: -kv[1]))
    t = llm.total
    usd = t["cost_usd"] or t["est_usd"]
    wins = out["counters"].get("win", 0)
    out.update(time_by_category_s={k: round(v, 3) for k, v in cats.items()},
               time_by_category_pct={k: round(100.0 * v / wall, 1) for k, v in cats.items()} if wall else {},
               counters={k: round(v, 3) for k, v in sorted(out["counters"].items())},
               wall_s=round(wall, 3), gpu_busy_s=round(out["gpu_busy_s"], 3), llm=llm.to_json(),
               derived={"steps": out["counters"].get("step", 0), "accepted_wins": wins, "usd": round(usd, 5),
                        "usd_is_estimate": not t["cost_usd"],
                        "usd_per_win": round(usd / wins, 5) if wins else None,
                        "tokens_per_win": round((t["input_tokens"] + t["output_tokens"]) / wins, 1) if wins else None,
                        "llm_wait_frac": round(cats.get("llm_wait", 0.0) / wall, 4) if wall else None,
                        "truncation_rate": round(t["truncated"] / t["calls"], 4) if t["calls"] else None})
    return out


# ---------------------------------------------------------------- JSONL reports

def find_files(paths: Iterable) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(p.rglob("telemetry.jsonl"))
        elif p.exists():
            out.append(p)
    return out


def read_events(path: Path) -> list[dict]:
    evs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evs.append(json.loads(line))
            except ValueError:
                continue                         # a torn last line of a killed run
    return evs


class _Owner:
    """Stand-in for a Span when rebuilding a collector from events."""

    def __init__(self, t0, t1, name="(no span)", id=0):
        self.t0, self.t1, self.name, self.id, self.attrs = t0, t1, name, id, {}


def summarize_events(evs: list[dict], *, problem: Optional[str] = None) -> dict:
    """Rebuild one run's collector summary from its events (optionally one problem's)."""
    spans = [e for e in evs if e.get("t") == "span"]
    if problem is not None:
        spans = [e for e in spans if (e.get("attrs") or {}).get("problem") == problem]
    llms = [e for e in evs if e.get("t") in ("llm", "llm_backoff")
            and (problem is None or e.get("problem") == problem)]
    counts = [e for e in evs if e.get("t") == "count" and (problem is None or e.get("problem") == problem)]
    if not spans and not llms:
        return {}
    t0 = min([e["ts"] for e in spans] + [e["ts"] for e in llms])
    t1 = max([e["ts"] + e["dur"] for e in spans] + [e["ts"] + e.get("latency_s", 0) for e in llms])
    # a top-level run/problem span is the owner when there is one
    owner = _Owner(t0, t1)
    col = Collector.__new__(Collector)
    col.owner, col.records, col.busy, col.llm = owner, [], {}, LLMStats()
    col.counters, col.evals, col.gpu = {}, {}, []
    import threading
    col._lock = threading.Lock()
    for e in spans:
        a = e.get("attrs") or {}
        fake = Span.__new__(Span)
        fake.t0, fake.t1, fake.name, fake.id, fake.parent_id, fake.attrs = (
            e["ts"], e["ts"] + e["dur"], e["name"], e["id"], e.get("parent"), a)
        col.add_span(fake, e.get("self", 0.0))
    for e in llms:
        col.llm.add(e)
    for e in counts:
        col.add_count(e["name"], e.get("n", 1))
    s = col.summary()
    s["derived"] = derive(s)
    return s


def _fmt_s(x: float) -> str:
    return f"{x:9.1f}"


def _table(rows: list[list], header: list[str]) -> str:
    cols = [header] + [[str(c) for c in r] for r in rows]
    w = [max(len(r[i]) for r in cols) for i in range(len(header))]
    left = {0} | {i for i, h in enumerate(header) if h in ("", "model", "problem", "slot", "purpose", "stop")}
    line = lambda r: "  ".join(c.ljust(w[i]) if i in left else c.rjust(w[i]) for i, c in enumerate(r))  # noqa: E731
    return "\n".join([line(header), line(["-" * x for x in w])] + [line(r) for r in cols[1:]])


def format_summary(s: dict, *, title: str, evs: Optional[list] = None, top: int = 10) -> str:
    if not s:
        return f"== {title}: no events"
    d = s["derived"]
    llm = s["llm"]["total"]
    out = [f"== {title}",
           f"wall {s['wall_s']:.1f} s | steps {d['steps']} ({d['steps_per_hour']} /h) | accepted wins "
           f"{d['accepted_wins']} | candidates/min {d['candidates_per_min']} | LLM calls {llm['calls']} "
           f"| tokens in {llm['input_tokens']} (cache read {llm['cache_read_input_tokens']}, write "
           f"{llm['cache_write_input_tokens']}) out {llm['output_tokens']} | truncated {llm['truncated']} "
           f"| ${d['usd']:.4f}{' (est)' if d['usd_is_estimate'] else ''}"
           + (f" | $/win {d['usd_per_win']:.4f}" if d['usd_per_win'] else "")
           + f" | LLM-wait {100 * (d['llm_wait_frac'] or 0):.1f}% | GPU idle {100 * (d['gpu_idle_frac'] or 0):.1f}%"
           + (f" | build-cache hits {100 * d['compile_cache_hit_rate']:.0f}%"
              if d["compile_cache_hit_rate"] is not None else ""),
           "", "Where the wall clock goes (each instant to the innermost open span(s), split across concurrent "
               "tasks; sums to wall):"]
    wall = s["wall_s"] or 1.0
    shown = lambda v: v > 0 and v >= 5e-4 * wall  # noqa: E731
    rows = [[k, f"{v:.2f}", f"{100 * v / wall:.1f}%", CATEGORY_HELP.get(k, "")]
            for k, v in d["time_by_category_s"].items() if shown(v)]
    out.append(_table(rows, ["category", "seconds", "% wall", ""]))
    out += ["", "By span (wall attribution; busy = exclusive seconds summed over concurrent tasks):"]
    busy = s["busy_s"]
    rows = [[k, f"{v:.2f}", f"{100 * v / wall:.1f}%", f"{busy.get(k, 0):.2f}"]
            for k, v in s["time_s"].items() if shown(v)]
    out.append(_table(rows, ["span", "seconds", "% wall", "busy s"]))
    if s["evals"]:
        out += ["", "evaluate() calls by purpose:"]
        out.append(_table([[k, v["n"], f"{v['s']:.1f}", v["failed"]] for k, v in
                           sorted(s["evals"].items(), key=lambda kv: -kv[1]["s"])],
                          ["purpose", "calls", "seconds", "failed"]))
    if s["llm"]["by_role_model"]:
        out += ["", "LLM tokens by role / model:"]
        rows = [[b["role"], b["model"], b["calls"], b["errors"], b["input_tokens"], b["cache_read_input_tokens"],
                 b["cache_write_input_tokens"], b["output_tokens"], b["truncated"],
                 f"{b['latency_s'] / max(b['calls'], 1):.1f}", f"{b['latency_max_s']:.1f}",
                 f"{b['backoff_s']:.1f}", f"{(b['cost_usd'] or b['est_usd']):.4f}"]
                for b in s["llm"]["by_role_model"]]
        out.append(_table(rows, ["role", "model", "calls", "err", "in", "cache_r", "cache_w", "out", "trunc",
                                 "lat_mean", "lat_max", "backoff_s", "usd"]))
        if llm.get("relay_queue_s") or llm.get("relay_network_s"):
            out.append(f"LLM relay (docs/llm_server.md): server queue {llm['relay_queue_s']:.1f} s, "
                       f"network {llm['relay_network_s']:.1f} s over {llm['calls']} calls")
    if evs:
        tr = [e for e in evs if e.get("t") == "llm" and e.get("truncated")]
        if tr:
            out += ["", f"Truncated LLM outputs ({len(tr)}):"]
            out.append(_table([[e.get("role"), e.get("model"), e.get("output_tokens"), e.get("max_tokens", ""),
                                e.get("stop_reason", ""), e.get("problem", ""), e.get("step", ""),
                                e.get("slot", "")] for e in tr[:top * 2]],
                              ["role", "model", "out", "max", "stop", "problem", "step", "slot"]))
        leaf = [e for e in evs if e.get("t") == "span"]
        leaf.sort(key=lambda e: -e.get("self", 0.0))
        if leaf:
            out += ["", f"Slowest spans by exclusive (self) time (top {top}):"]
            out.append(_table([[e["name"], f"{e['dur']:.2f}", f"{e.get('self', 0):.2f}",
                                (e.get("attrs") or {}).get("problem", ""), (e.get("attrs") or {}).get("step", ""),
                                (e.get("attrs") or {}).get("slot", ""), (e.get("attrs") or {}).get("purpose", "")]
                               for e in leaf[:top]],
                              ["span", "dur s", "self s", "problem", "step", "slot", "purpose"]))
    return "\n".join(out)


def report(paths: Iterable, *, top: int = 10, by_problem: bool = False, as_json: bool = False) -> str:
    files = find_files(paths)
    if not files:
        return "no telemetry.jsonl found"
    res, txt = {}, []
    for f in files:
        evs = read_events(f)
        runs = defaultdict(list)
        for e in evs:
            runs[e.get("run", "?")].append(e)
        for rid, rev in runs.items():
            s = summarize_events(rev)
            key = f"{f} [{rid}]"
            res[key] = {"run": s}
            txt.append(format_summary(s, title=key, evs=rev, top=top))
            if by_problem:
                probs = sorted({(e.get("attrs") or {}).get("problem") for e in rev if e.get("t") == "span"} - {None})
                for p in probs:
                    ps = summarize_events(rev, problem=p)
                    res[key][p] = ps
                    txt.append(format_summary(ps, title=f"{key} problem {p}", evs=None, top=top))
    if len(res) > 1:
        tot = {"wall_s": 0.0, "steps": 0, "calls": 0, "in": 0, "out": 0, "usd": 0.0, "trunc": 0}
        cats: dict[str, float] = defaultdict(float)
        for v in res.values():
            s = v["run"]
            if not s:
                continue
            tot["wall_s"] += s["wall_s"]
            tot["steps"] += s["derived"]["steps"]
            t = s["llm"]["total"]
            tot["calls"] += t["calls"]
            tot["in"] += t["input_tokens"]
            tot["out"] += t["output_tokens"]
            tot["trunc"] += t["truncated"]
            tot["usd"] += s["derived"]["usd"]
            for k, x in s["derived"]["time_by_category_s"].items():
                cats[k] += x
        res["_combined"] = {**tot, "time_by_category_s": dict(cats)}
        w = tot["wall_s"] or 1.0
        txt.append("== combined over runs (sum of per-run wall clocks; a parent process's wait on a child, e.g. "
                   "ModelBlaster's kb_wait, also contains the child's time)\n"
                   f"wall {tot['wall_s']:.1f} s | steps {tot['steps']} | LLM calls {tot['calls']} | tokens in "
                   f"{tot['in']} out {tot['out']} | truncated {tot['trunc']} | ${tot['usd']:.4f}\n"
                   + _table([[k, f"{x:.1f}", f"{100 * x / w:.1f}%"] for k, x in
                             sorted(cats.items(), key=lambda kv: -kv[1]) if x > 0],
                            ["category", "seconds", "% wall"]))
    if as_json:
        return json.dumps(res, indent=1, default=str)
    return "\n\n".join(txt)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Summarise KernelBlaster / ModelBlaster telemetry.jsonl runs: time "
                                             "breakdown, tokens by role/model, truncations, slowest spans.")
    ap.add_argument("paths", nargs="+", help="telemetry.jsonl files or directories searched for them")
    ap.add_argument("--top", type=int, default=10, help="slowest spans to list")
    ap.add_argument("--by-problem", action="store_true", help="also one section per problem")
    ap.add_argument("--json", action="store_true", help="JSON instead of tables")
    a = ap.parse_args(argv)
    print(report(a.paths, top=a.top, by_problem=a.by_problem, as_json=a.json))
    return 0
