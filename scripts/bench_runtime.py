#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-candidate evaluation cost and timing agreement: persistent driver host vs one process per run.

    python scripts/bench_runtime.py PROBLEM_DIR[=WINNER_SLOTS_DIR] ... --arch 75 --reps 6 \\
        [--runtime host process] [--evaluator ssh --ssh-host user@host --remote-dir DIR --cuda-visible-devices 2] \\
        [--agree 5] --out bench.json

For every problem and runtime: one full evaluation of the base sources (the seeds, or the slot
sources in WINNER_SLOTS_DIR) warms the build caches and starts the host; then ``--reps`` new
candidates, each one slot (round robin) with a unique comment, are evaluated the way an agent step
is (``step_eval_options``; paired with the base on the host), and ``--reps`` / 2 again in full (the
confirm run). Wall time is per candidate, build included. ``--agree N`` evaluates the base N times
per runtime, interleaved, for the spread and agreement of the end-to-end, eager and attributed times.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernelblaster.knowledge.slots import LocalEvaluator, SlotProblem, SSHEvaluator, step_eval_options  # noqa: E402


def _cv(xs):
    xs = [x for x in xs if x]
    return round(100 * statistics.pstdev(xs) / statistics.mean(xs), 2) if len(xs) > 1 else None


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def _row(r, wall):
    return {"ok": r.ok, "wall_s": round(wall, 3), "run_s": r.eval_wall_s, "runtime": r.runtime,
            "e2e_us": r.e2e_us, "basis": r.e2e_basis, "eager_us": r.partition_us, "graph_us": r.partition_graph_us,
            "ratio": (r.e2e_ratio or {}).get("med"), "attr_sum_us": r.attr_sum_us, "launches": r.attr_launches,
            "phase_total_ms": None, "errors": r.errors[:2]}


async def bench(args):
    problems = []
    for spec in args.problems:
        d, _, win = spec.partition("=")
        p = SlotProblem(Path(d))
        base = p.seed_sources()
        if win:
            base.update({f.stem: f.read_text() for f in Path(win).glob("*.cu") if f.stem in base})
        problems.append((p, base))

    def make(rt):
        # "process-graph": one process per run, built for and timing the CUDA graph too
        kw = dict(work_root=Path(args.work) / rt, runtime=rt.split("-")[0], timeout_s=args.timeout,
                  graph=rt.endswith("-graph"))
        if args.evaluator == "ssh":
            return SSHEvaluator(args.ssh_host, arch=args.arch, remote_dir=f"{args.remote_dir}/{rt}",
                                ssh_key=args.ssh_key, cuda_visible_devices=args.cuda_visible_devices, **kw)
        return LocalEvaluator(args.arch, gpus=args.gpus, **kw)

    evs = {rt: make(rt) for rt in args.runtime}
    rt_of = {id(ev): rt for rt, ev in evs.items()}
    out = {"arch": args.arch, "evaluator": args.evaluator, "problems": {}}
    try:
        for p, base in problems:
            res = out["problems"][p.name] = {}
            slots = list(p.order)
            for rt, ev in evs.items():
                t = time.time()
                r = await ev.evaluate(p, base, "warm")
                rows = {"warm": _row(r, time.time() - t), "step": [], "full": []}
                for k in range(args.reps):
                    sid = slots[k % len(slots)]
                    cand = {**base, sid: base[sid] + f"\n// bench {rt} {k} {time.time()}\n"}
                    kw = step_eval_options(sid)
                    if ev.pairs_incumbent(p):
                        kw["incumbent"] = base
                    t = time.time()
                    r = await ev.evaluate(p, cand, f"step{k}", **kw)
                    rows["step"].append({"slot": sid, **_row(r, time.time() - t)})
                for k in range(max(1, args.reps // 2)):
                    sid = slots[k % len(slots)]
                    cand = {**base, sid: base[sid] + f"\n// bench full {rt} {k} {time.time()}\n"}
                    t = time.time()
                    r = await ev.evaluate(p, cand, f"full{k}", **({"incumbent": base} if ev.pairs_incumbent(p) else {}))
                    rows["full"].append({"slot": sid, **_row(r, time.time() - t)})
                rows["summary"] = {"step_wall_s": _med([x["wall_s"] for x in rows["step"]]),
                                   "step_run_s": _med([x["run_s"] for x in rows["step"]]),
                                   "full_wall_s": _med([x["wall_s"] for x in rows["full"]]),
                                   "step_ok": all(x["ok"] for x in rows["step"]),
                                   "ratio_med": _med([x["ratio"] for x in rows["step"]])}
                res[rt] = rows
                print(json.dumps({p.name: {rt: rows["summary"]}}), flush=True)
            if args.agree:
                ag = {rt: [] for rt in evs}
                for k in range(args.agree):
                    for rt, ev in evs.items():
                        r = await ev.evaluate(p, base, f"agree{k}")
                        ag[rt].append(r)
                res["agreement"] = {rt: {"e2e_us": _med([r.e2e_us for r in rs]), "e2e_cv": _cv([r.e2e_us for r in rs]),
                                         "eager_us": _med([r.partition_us for r in rs]),
                                         "eager_cv": _cv([r.partition_us for r in rs]),
                                         "graph_us": _med([r.partition_graph_us for r in rs]),
                                         "attr_sum_us": _med([r.attr_sum_us for r in rs]),
                                         "attr_slot_us": {s: _med([r.attr_slot_us.get(s) for r in rs]) for s in slots},
                                         "slot_us": {s: _med([r.slot_us.get(s) for r in rs]) for s in slots},
                                         "basis": rs[0].e2e_basis, "ok": all(r.ok for r in rs)}
                                    for rt, rs in ag.items()}
                print(json.dumps({p.name: {"agreement": {rt: {k: v for k, v in a.items() if "slot" not in k}
                                                         for rt, a in res["agreement"].items()}}}), flush=True)
    finally:
        for ev in evs.values():
            out.setdefault("host_stats", {})[rt_of[id(ev)]] = ev.host_stats()
            await ev.close()
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problems", nargs="+")
    ap.add_argument("--arch", default="75")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--agree", type=int, default=0)
    ap.add_argument("--runtime", nargs="+", default=["host", "process"], choices=["host", "process", "process-graph"])
    ap.add_argument("--evaluator", choices=["local", "ssh"], default="local")
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--ssh-host", action="append")
    ap.add_argument("--ssh-key", default=None)
    ap.add_argument("--remote-dir", default=None)
    ap.add_argument("--cuda-visible-devices", default=None)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--work", default="bench_runtime_work")
    ap.add_argument("--out", default="bench_runtime.json")
    asyncio.run(bench(ap.parse_args()))


if __name__ == "__main__":
    main()
