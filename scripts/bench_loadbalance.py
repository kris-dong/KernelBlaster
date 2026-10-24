"""Candidate throughput of the multi-slot evaluators, without an LLM.

    python scripts/bench_loadbalance.py PROBLEM_DIR [...] --evaluator local|ssh \
        [--ssh-host user@host ...] [--cuda-visible-devices auto|2,3 ...] [--remote-dir DIR] \
        --candidates 24 --concurrency 6 [--compile-jobs N] [--gpus auto|0] [--kb-root OTHER_CHECKOUT] \
        [--arch 75] [--profile-every 0] --out bench.json

A scripted candidate stream: candidate i changes one slot of problem
``i % len(problems)`` (round robin over its slots) by a unique comment, so --
like an RL step -- it recompiles one slot TU, relinks and runs; caches only
hit what a real step would hit. ``--concurrency`` candidates are in flight at
once (an agent's rollouts x problems in parallel). ``--profile-every K`` also
runs the NCU pass on every K-th candidate. Reports candidates per minute and
how the runs spread over the GPUs.

``--kb-root`` imports the evaluators from another KernelBlaster checkout (e.g.
the code before load balancing); only the keyword arguments both versions
accept are passed (``--legacy`` mimics run_multislot's old defaults: 3 compile
jobs, one ssh host, its ``--cuda-visible-devices`` as one string).
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import statistics
import sys
import time
from pathlib import Path


def _import(kb_root: Path):
    sys.path.insert(0, str(kb_root / "src"))
    sys.path.insert(0, str(kb_root))
    from kernelblaster.knowledge import slots
    return slots


def _kw(cls, **kw):
    params = inspect.signature(cls.__init__).parameters
    has_var = any(p.kind == p.VAR_KEYWORD for p in params.values())
    base = set()
    for c in cls.__mro__:
        if "__init__" in c.__dict__:
            base |= set(inspect.signature(c.__init__).parameters)
    return {k: v for k, v in kw.items() if v is not None and (k in params or (has_var and k in base))}


async def _main(args) -> dict:
    slots = _import(Path(args.kb_root).resolve())
    work = Path(args.out).resolve().parent / (Path(args.out).stem + "_work")
    problems = [slots.SlotProblem(Path(p)) for p in args.problems]
    procs = []
    if args.evaluator == "server":
        # KB compile server here (spawned unless COMPILE_SERVER_URL is set) + the exec
        # server at $GPU_SERVER_URL_<GPU> (e.g. an ssh tunnel to a remote exec server)
        import importlib.util
        spec = importlib.util.spec_from_file_location("kb_run_multislot",
                                                      Path(args.kb_root) / "scripts" / "run_multislot.py")
        rm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rm)
        gpu, procs = rm._start_servers(args.gpu, work / "servers")
        ev = slots.ServerEvaluator(gpu, work)
    elif args.evaluator == "local":
        ev = slots.LocalEvaluator(args.arch, **_kw(slots.LocalEvaluator, work_root=work,
                                                   max_parallel=args.compile_jobs, gpus=args.gpus))
    else:
        if args.legacy:
            host, cvd = args.ssh_host[0], (args.cuda_visible_devices or [None])[0]
        else:
            host = args.ssh_host
            cvd = args.cuda_visible_devices or ["auto"]
            cvd = dict(zip(host, cvd * len(host) if len(cvd) == 1 else cvd))
        ev = slots.SSHEvaluator(host, arch=args.arch, remote_dir=args.remote_dir,
                                **_kw(slots.SSHEvaluator, ssh_key=args.ssh_key, cuda_visible_devices=cvd,
                                      remote_path=args.remote_cuda_bin, work_root=work,
                                      max_parallel=args.compile_jobs))
    args._procs.extend(procs)
    # warm-up: every problem's baseline once (driver objects, first-run effects)
    t_warm = time.time()
    for p in problems:
        r = await ev.evaluate(p, p.seed_sources(), "warmup")
        if not r.ok:
            raise SystemExit(f"{p.name}: baseline failed: {r.error_text(500)}")
    t_warm = time.time() - t_warm
    gate = asyncio.Semaphore(args.concurrency)
    lat, rows = [], []

    async def cand(i: int):
        p = problems[i % len(problems)]
        sid = p.order[(i // len(problems)) % len(p.order)]
        src = p.seed_sources()
        src[sid] = src[sid] + f"\n// bench candidate {i} {time.time_ns()}\n"
        async with gate:
            t0 = time.time()
            r = await ev.evaluate(p, src, f"bench_{i}")
            prof = None
            if args.profile_every and i % args.profile_every == 0:
                by, err = await ev.profile(p, src, f"bench_{i}")
                prof = err or f"{len(by)} slots"
            dt = time.time() - t0
        lat.append(dt)
        rows.append({"i": i, "problem": p.name, "slot": sid, "ok": r.ok, "seconds": round(dt, 2),
                     "slot_us": r.slot_us.get(sid), "device": getattr(r, "device", None), "profile": prof})

    t0 = time.time()
    await asyncio.gather(*[cand(i) for i in range(args.candidates)])
    wall = time.time() - t0
    pool = getattr(ev, "_pool", None)
    jobs = getattr(ev, "_jobs", None)
    res = {
        "kb_root": str(Path(args.kb_root).resolve()), "evaluator": type(ev).__name__,
        "problems": [p.name for p in problems], "candidates": args.candidates, "concurrency": args.concurrency,
        "compile_jobs": args.compile_jobs, "gpus": args.gpus, "ssh_host": args.ssh_host,
        "cuda_visible_devices": args.cuda_visible_devices, "legacy": args.legacy,
        "warmup_s": round(t_warm, 1), "wall_s": round(wall, 1),
        "cands_per_min": round(60 * args.candidates / wall, 2),
        "ok": sum(r["ok"] for r in rows),
        "latency_median_s": round(statistics.median(lat), 2),
        "load1_start_end": None,
        "pool": pool.stats() if pool is not None else None,
        "peak_busy_gpus": pool.peak_busy if pool is not None else None,
        "compile_jobs_limiter": ({"capacity": jobs.describe(), "peak": jobs.peak,
                                  "history": [c for _, c in jobs.history]} if jobs is not None else None),
        "rows": sorted(rows, key=lambda r: r["i"]),
    }
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problems", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--evaluator", choices=["local", "ssh", "server"], default="local")
    ap.add_argument("--gpu", default="titanrtx", help="GPUType for --evaluator server")
    ap.add_argument("--arch", default="75")
    ap.add_argument("--candidates", type=int, default=24)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--compile-jobs", type=int, default=None)
    ap.add_argument("--gpus", default=None, help="local GPUs (auto|all|ids)")
    ap.add_argument("--ssh-host", action="append", default=[])
    ap.add_argument("--ssh-key", default=None)
    ap.add_argument("--cuda-visible-devices", action="append", default=[])
    ap.add_argument("--remote-dir", default=None)
    ap.add_argument("--remote-cuda-bin", default="/usr/local/cuda/bin")
    ap.add_argument("--profile-every", type=int, default=0)
    ap.add_argument("--kb-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--legacy", action="store_true", help="old SSHEvaluator API: one host, one CVD string")
    args = ap.parse_args()
    l0 = os.getloadavg()[0]
    args._procs = []
    try:
        res = asyncio.run(_main(args))
    finally:
        for p in args._procs:
            p.terminate()
    res["load1_start_end"] = [round(l0, 1), round(os.getloadavg()[0], 1)]
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
