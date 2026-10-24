#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check and time ModelBlaster problems with KernelBlaster's evaluators, outside the agent loop.

ModelBlaster's whole-model result (``pipeline/kb_model_run.py``) and its hint re-measurement
(``kb_proposals.remeasure``) use this, so their numbers come from the same evaluators -- local
GPU pool and locks, or SSH to remote GPUs, persistent driver host -- and the same GPU model as
the KernelBlaster runs that optimised the components.

    python scripts/time_problems.py JOBS.json --out OUT.json --arch 89 \\
        [--evaluator local --gpus auto|all|0,2 | --evaluator ssh --ssh-host H [--ssh-host H2] \\
         --remote-dir DIR [--cuda-visible-devices auto|ids] [--ssh-key K]] \\
        [--runtime auto|host|process] [--gpu-model 'NVIDIA L40S'] [--devices 1] [--timeout 1800]

``JOBS.json``: ``{"jobs": [{"id", "problem": DIR, "sources": {slot: text} | null (the problem's
own slot files), "incumbent": {slot: text} | null, "reps": N}], "order": "rounds" | "jobs"}``.
``rounds`` (default): rep k of every job, in order, then rep k+1 (interleaved, so a load change on
the GPU does not favour one job); ``jobs``: every rep of a job, then the next (one persistent host
at a time: big problems). A job stops at its first failing rep. A job with an ``incumbent`` is paired: on the
persistent host both are timed in the same request, interleaved rep by rep (``ref_e2e_us``,
``e2e_ratio``); with one process per run the incumbent is evaluated right after the candidate,
in the same round (``pairing: "interleaved"``).

``--gpu-model`` keeps only the selected GPUs of that model (none: the run fails with the models
it found, it never times on another model); ``--devices N`` caps the pool (default 1: every run
on one GPU). ``OUT.json`` (``kernelblaster.time_problems/v1``): every run's numbers, the GPU each
was timed on, the pool, the runtime statistics and this process's telemetry summary.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernelblaster import telemetry as tel  # noqa: E402
from kernelblaster.knowledge.slots import LocalEvaluator, SlotProblem, SSHEvaluator  # noqa: E402

FORMAT = "kernelblaster.time_problems/v1"
_DROP = ("log",)


def row(r) -> dict:
    """An EvalResult as JSON (the log only on failure, its tail)."""
    d = {k: v for k, v in asdict(r).items() if k not in _DROP}
    if not r.ok:
        d["log"] = (r.log or "")[-2000:]
    return d


def make_evaluator(a):
    kw = dict(work_root=Path(a.work), runtime=a.runtime, timeout_s=a.timeout, graph=True,
              gpu_model=a.gpu_model, max_devices=a.devices or None)
    if a.evaluator == "ssh":
        hosts = a.ssh_host or []
        if not hosts:
            raise SystemExit("--evaluator ssh needs --ssh-host")
        if not a.remote_dir:
            raise SystemExit("--evaluator ssh needs --remote-dir")
        cvd = a.cuda_visible_devices or ["auto"]
        if len(cvd) == 1:
            cvd = cvd * len(hosts)
        if len(cvd) != len(hosts):
            raise SystemExit(f"--cuda-visible-devices: give one value, or one per --ssh-host ({len(hosts)})")
        return SSHEvaluator(hosts, arch=a.arch, remote_dir=a.remote_dir, ssh_key=a.ssh_key,
                            cuda_visible_devices=dict(zip(hosts, cvd)), remote_path=a.remote_cuda_bin, **kw)
    return LocalEvaluator(a.arch, gpus=a.gpus, **kw)


async def time_jobs(ev, jobs: list[dict], order: str = "rounds") -> dict:
    probs = {j["id"]: SlotProblem(Path(j["problem"])) for j in jobs}
    out = {}
    for j in jobs:
        p = probs[j["id"]]
        paired = j.get("incumbent") is not None
        out[j["id"]] = {"id": j["id"], "problem": str(p.root), "runs": [], "ref_runs": [],
                        "paired": paired, "pairing": ("host" if ev.pairs_incumbent(p) else "interleaved")
                        if paired else None}
    live = {j["id"] for j in jobs}
    n = max([int(j.get("reps", 1)) for j in jobs] or [0])
    if order == "jobs":                      # every rep of a job, then the next job
        steps = [(k, j) for j in jobs for k in range(int(j.get("reps", 1)))]
    else:                                    # rounds: rep k of every job, then rep k + 1
        steps = [(k, j) for k in range(n) for j in jobs if k < int(j.get("reps", 1))]
    for k, j in steps:
        jid, p = j["id"], probs[j["id"]]
        if jid not in live:
            continue
        src = {**p.seed_sources(), **(j.get("sources") or {})}      # slots not given: the problem's own
        inc = j.get("incumbent")
        o = out[jid]
        host_pair = inc is not None and o["pairing"] == "host"
        r = await ev.evaluate(p, src, f"{j.get('tag', jid)}_r{k}", **({"incumbent": inc} if host_pair else {}))
        o["runs"].append(row(r))
        if inc is not None and not host_pair and r.ok:
            rr = await ev.evaluate(p, inc, f"{j.get('tag', jid)}_ref_r{k}")
            o["ref_runs"].append(row(rr))
        if not r.ok:
            live.discard(jid)
    for o in out.values():
        gpus = [x.get("gpu") for x in o["runs"] + o["ref_runs"] if x.get("gpu")]
        o["gpu"] = gpus[0] if gpus else None
        o["gpus"] = sorted({g["label"] for g in gpus})
    return out


async def main_async(a) -> int:
    spec = json.loads(Path(a.jobs).read_text())
    jobs = spec["jobs"]
    Path(a.work).mkdir(parents=True, exist_ok=True)
    res = {"format": FORMAT, "evaluator": a.evaluator, "arch": f"sm_{a.arch}", "runtime": a.runtime,
           "gpu_model": a.gpu_model, "devices": a.devices, "jobs": {}, "error": None}
    t0 = time.time()
    tel.configure(Path(a.telemetry) if a.telemetry else None)
    ev = make_evaluator(a)
    with tel.span("kb.time_problems", collect=True, jobs=len(jobs), evaluator=a.evaluator) as sp:
        try:
            pool = await ev.pool()
            res["pool"] = [{"label": d.label(), "host": d.host, "ident": d.ident, "index": d.index, "name": d.name}
                           for d in pool.devices]
            res["models"] = pool.models
            res["jobs"] = await time_jobs(ev, jobs, spec.get("order", "rounds"))
            res["pool_stats"] = pool.stats()
        except Exception as e:  # noqa: BLE001 -- reported to the caller (e.g. no GPU of the model)
            res["error"] = f"{type(e).__name__}: {e}"
        finally:
            res["host_stats"] = ev.host_stats()
            await ev.close()
    res["seconds"] = round(time.time() - t0, 2)
    summ = tel.full_summary(sp)
    if summ is not None:
        res["telemetry"] = summ
    tel.close()
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"error": res["error"], "models": res.get("models"), "seconds": res["seconds"],
                      "jobs": {k: [x.get("e2e_us") for x in v["runs"]] for k, v in res["jobs"].items()}}))
    return 1 if res["error"] else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="75")
    ap.add_argument("--evaluator", choices=["local", "ssh"], default="local")
    ap.add_argument("--gpus", default="auto")
    ap.add_argument("--ssh-host", action="append", default=[])
    ap.add_argument("--ssh-key", default=None)
    ap.add_argument("--remote-dir", default=None)
    ap.add_argument("--cuda-visible-devices", action="append", default=[])
    ap.add_argument("--remote-cuda-bin", default="/usr/local/cuda/bin")
    ap.add_argument("--runtime", choices=["auto", "host", "process"], default="auto")
    ap.add_argument("--gpu-model", default=None)
    ap.add_argument("--devices", type=int, default=1, help="GPUs in the pool at most (0: every selected one)")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--work", default="time_problems_work")
    ap.add_argument("--telemetry", default=None, help="telemetry.jsonl for this process's spans")
    sys.exit(asyncio.run(main_async(ap.parse_args())))


if __name__ == "__main__":
    main()
