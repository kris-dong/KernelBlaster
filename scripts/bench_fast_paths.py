"""Per-candidate wall time of KernelBlaster's compile / exec / profile paths.

    python scripts/bench_fast_paths.py \
        --native data/kernelbench-cuda/level1/019_ReLU data/kernelbench-cuda/level2/009_Matmul_Subtract_Multiply_ReLU \
        --multislot tests/fixtures/mb_lenet /path/to/mb_9_ResNet18 \
        --evaluators server,local --reps 3 --out bench.json

Every rep is a *new candidate*: the kernel (native) or one slot (multi-slot,
round robin) gets a unique trailing comment, so object / binary caches can
only hit what a real RL step could hit (the unchanged driver, the unchanged
slots). ``same`` rows re-evaluate the previous candidate unchanged -- what the
multi-slot agent's confirm run and the NCU profile build see.

Native rows run the CUDA agent's own ``_gather_perf_metrics``
(``agents/opt_ncu_rl_optimized.py``) against the KB compile + exec servers;
phases come from its ``perf_span`` records, the compile breakdown from the
compile server's ``[stage] finished ... elapsed=`` log lines. Without GPU
counter permission the NCU details pass fails; the row still reports how long
it took to fail.

Set ``KERNELBLASTER_FAST_COMPILE=0`` (inherited by the servers it starts) to
measure the legacy CMake path; medians over ``--reps`` are printed per phase.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import json
import os
import re
import sqlite3
import statistics
import sys
import time
import types
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

_STAGE_RE = re.compile(r"\[([A-Za-z_]+)(?::[^\]]*)?\] finished rc=\d+ elapsed=([0-9.]+)s")


def _stages(log: Path, start: int) -> dict:
    """Sum of compile-server stage times logged after byte offset ``start``."""
    out: dict = {}
    if not log.exists():
        return out
    with open(log, "rb") as f:
        f.seek(start)
        for line in f.read().decode(errors="replace").splitlines():
            m = _STAGE_RE.search(line)
            if m:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def _size(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


def _start_servers(gpu_name: str, log_dir: Path):
    from kernelblaster.config import GPUType, config
    from kernelblaster.servers.management import initialize_compiler_server, initialize_gpu_server
    import urllib.request

    gpu = GPUType(gpu_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KERNELBLASTER_COMPILE_SERVER_NUM_WORKERS", "2")
    cwd = os.getcwd()
    os.chdir(ROOT)
    procs, urls = [], []
    try:
        lf = open(log_dir / "servers.log", "a")
        if config.COMPILE_SERVER_URL is None:
            p, url = initialize_compiler_server(lf, None, log_dir / "compile_artifacts", None)
            config.set_compile_server_url(url)
            procs.append(p)
            urls.append(url)
        p2, url2 = initialize_gpu_server(lf, gpu, None)
        procs.append(p2)
        urls.append(url2)
    finally:
        os.chdir(cwd)
    procs = [p for p in procs if p]
    for url in urls:
        for _ in range(240):
            try:
                urllib.request.urlopen(url + "/health", timeout=2)
                break
            except Exception:
                time.sleep(0.5)
        else:
            for p in procs:
                p.terminate()
            raise RuntimeError(f"server at {url} did not start; see {log_dir / 'servers.log'}")
    return gpu, procs


def _spans(db: Path, since_id: int) -> tuple[dict, int]:
    if not db.exists():
        return {}, since_id
    con = sqlite3.connect(str(db))
    rows = con.execute("SELECT id, phase, duration_s FROM perf_spans WHERE id > ?", (since_id,)).fetchall()
    con.close()
    out: dict = {}
    for _, ph, d in rows:
        out[ph] = out.get(ph, 0.0) + d
    return out, max([since_id] + [r[0] for r in rows])


# ---------------------------------------------------------------- native

async def bench_native(problems, gpu, reps, work: Path, server_log: Path, db: Path) -> list[dict]:
    from loguru import logger
    from kernelblaster.agents.opt_ncu_rl_optimized import OptimizedRLNCUAgent

    rows = []
    last = 0
    for prob in problems:
        prob = Path(prob).resolve()
        seed = (prob / "init.cu").read_text()
        for rep in range(reps):
            cand = work / f"{prob.name}_r{rep}_{uuid.uuid4().hex[:6]}.cu"
            cand.write_text(seed + f"\n// bench candidate {uuid.uuid4().hex}\n")
            stub = types.SimpleNamespace(test_code_fp=prob / "driver.cpp", gpu=gpu, agent_logger=logger,
                                         ncu_timeout_s=600, problem_id=prob.name)
            for name in ("_nsys_timing", "_nsys_trace"):
                if hasattr(OptimizedRLNCUAgent, name):
                    setattr(stub, name, functools.partial(getattr(OptimizedRLNCUAgent, name), stub))
            if hasattr(OptimizedRLNCUAgent, "_nsys_timing_from_trace"):
                stub._nsys_timing_from_trace = OptimizedRLNCUAgent._nsys_timing_from_trace
            off = _size(server_log)
            t0 = time.perf_counter()
            err = None
            try:
                await OptimizedRLNCUAgent._gather_perf_metrics(stub, cand)
            except Exception as e:  # ncu without counters, or a real failure
                err = str(e)[:200]
            total = time.perf_counter() - t0
            phases, last = _spans(db, last)
            rows.append({"case": f"native/{prob.name}", "path": "server", "rep": rep, "total": total,
                         **{f"ph.{k}": v for k, v in phases.items()},
                         **{f"cs.{k}": v for k, v in _stages(server_log, off).items()},
                         "error": err})
            print(json.dumps(rows[-1]), flush=True)
    return rows


# ---------------------------------------------------------------- multi-slot

def _timed(fn, acc: dict, key: str):
    @functools.wraps(fn)
    async def w(*a, **k):
        t = time.perf_counter()
        try:
            return await fn(*a, **k)
        finally:
            acc[key] = acc.get(key, 0.0) + time.perf_counter() - t
    return w


def _make_evaluator(kind: str, gpu, work: Path, arch: str):
    from kernelblaster.knowledge.slots import LocalEvaluator, ServerEvaluator
    if kind == "local":
        return LocalEvaluator(arch=arch, work_root=work / "local")
    return ServerEvaluator(gpu, work / "server")


async def bench_multislot(problems, kinds, gpu, reps, work: Path, server_log: Path, arch: str,
                          ncu: bool) -> list[dict]:
    from kernelblaster.knowledge.slots import SlotProblem

    rows = []
    for prob in problems:
        p = SlotProblem(Path(prob))
        order = list(p.order)
        for kind in kinds:
            ev = _make_evaluator(kind, gpu, work, arch)
            acc: dict = {}
            for name in ("_build", "_run", "_compile"):
                if hasattr(ev, name):
                    setattr(ev, name, _timed(getattr(ev, name), acc, name.strip("_")))
            sources = p.seed_sources()
            for rep in range(reps):
                sid = order[rep % len(order)]
                sources = {**sources, sid: sources[sid] + f"\n// bench candidate {uuid.uuid4().hex}\n"}
                for variant in ("new", "same"):
                    acc.clear()
                    off = _size(server_log)
                    t0 = time.perf_counter()
                    res = await ev.evaluate(p, sources, f"bench_{rep}_{variant}")
                    total = time.perf_counter() - t0
                    row = {"case": f"multislot/{p.name}", "path": kind, "variant": variant, "rep": rep,
                           "slot": sid, "total": total, "ok": res.ok, "stage": res.stage,
                           **{f"ev.{k}": v for k, v in acc.items()}}
                    if kind == "server":
                        row.update({f"cs.{k}": v for k, v in _stages(server_log, off).items()})
                    if not res.ok:
                        row["error"] = res.error_text(300)
                    rows.append(row)
                    print(json.dumps(rows[-1]), flush=True)
                if ncu:
                    acc.clear()
                    t0 = time.perf_counter()
                    by, why = await ev.profile(p, sources, f"bench_{rep}_ncu")
                    rows.append({"case": f"multislot/{p.name}", "path": kind, "variant": "profile",
                                 "rep": rep, "total": time.perf_counter() - t0, "profiled": len(by),
                                 "error": why, **{f"ev.{k}": v for k, v in acc.items()}})
                    print(json.dumps(rows[-1]), flush=True)
    return rows


def summarize(rows: list[dict]) -> str:
    groups: dict = {}
    for r in rows:
        key = (r["case"], r["path"], r.get("variant", "new"))
        groups.setdefault(key, []).append(r)
    lines = []
    for (case, path, variant), rs in groups.items():
        keys = sorted({k for r in rs for k, v in r.items() if isinstance(v, float)}, key=lambda k: (k != "total", k))
        med = {k: statistics.median([r.get(k, 0.0) for r in rs]) for k in keys}
        bits = " ".join(f"{k}={v:.2f}" for k, v in med.items())
        bad = sum(1 for r in rs if r.get("error") or r.get("ok") is False)
        lines.append(f"{case:55s} {path:7s} {variant:7s} N={len(rs)} {bits}" + (f"  (errors: {bad})" if bad else ""))
    return "\n".join(lines)


async def main_async(a) -> int:
    from kernelblaster.resources import TCPClient

    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    db = work / "perf_log.sqlite"
    os.environ["KERNELBLASTER_PERF_LOG_DB"] = str(db)
    kinds = [k for k in a.evaluators.split(",") if k]
    need_servers = bool(a.native) or "server" in kinds
    procs = []
    gpu = None
    log = work / "servers" / "servers.log"
    try:
        if need_servers:
            gpu, procs = _start_servers(a.gpu, work / "servers")
        rows = []
        if a.native:
            rows += await bench_native(a.native, gpu, a.reps, work, log, db)
        if a.multislot:
            rows += await bench_multislot(a.multislot, kinds, gpu, a.reps, work, log, a.arch, a.ncu_multislot)
    finally:
        await TCPClient.close_session()
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=20)
            except Exception:
                p.kill()
    print("\n== medians (s) ==\n" + summarize(rows))
    if a.out:
        Path(a.out).write_text(json.dumps({"env": {"KERNELBLASTER_FAST_COMPILE": os.getenv("KERNELBLASTER_FAST_COMPILE")},
                                           "rows": rows}, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--native", nargs="*", default=[], help="KernelBench CUDA problem dirs (driver.cpp + init.cu)")
    ap.add_argument("--multislot", nargs="*", default=[], help="ModelBlaster-exported problem dirs")
    ap.add_argument("--evaluators", default="server,local", help="multi-slot evaluators: server,local")
    ap.add_argument("--gpu", default="titanrtx")
    ap.add_argument("--arch", default="75")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ncu-multislot", action="store_true", help="multi-slot: also time Evaluator.profile")
    ap.add_argument("--work", default=str(ROOT / "out" / "bench_fast_paths"))
    ap.add_argument("--out", default=None, help="write all rows as JSON")
    sys.exit(asyncio.run(main_async(ap.parse_args())))


if __name__ == "__main__":
    main()
