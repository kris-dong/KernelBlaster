"""Run the multi-slot optimiser on ModelBlaster-exported partition problems.

    python scripts/run_multislot.py PROBLEM_DIR_OR_ROOT [...] --out OUT \
        --model claude-code:sonnet [--reflect-model claude-code:haiku] \
        [--knowledge-dir DIR] [--arch 75] [--rollouts 4 --steps 6] \
        [--evaluator local|server|ssh|board ...] [--ncu off|best|all]

PROBLEM_DIR_OR_ROOT: a directory with manifest.json, or a root searched for
them (``pipeline/kb_export.py`` output). The knowledge directory holds the
implementation store (``impl/``), the outcome log and the semantic claims;
share it across runs so later problems start from earlier winners and
lessons. ``--model`` is any name ``generate_code`` routes: ``claude-code:<m>``
(Claude Code CLI, no key), ``stub:<name>`` (registered by a ``--plugin``
file), or an API model.

Writes OUT/<problem>/result.json and best/slots/*.cu, OUT/summary.json, and
(with --export-winners) the store's winners for ModelBlaster's kb_ingest.

OpenCL problems (manifest target "opencl", Qualcomm Adreno): ``--evaluator
board`` builds and runs on the board over ssh (``--board-host``, default
$KERNELBLASTER_ADRENO_BOARD_HOST), ``local`` on this host's OpenCL GPU,
``server`` through KernelBlaster's OpenCL compile + Adreno exec servers;
``--arch`` names the device (adreno650 default, e.g. titanrtx for the local
stand-in) and keys the store; ``--ncu best|all`` collects OpenCL event
profiles (per-kernel times, work-group limits, private memory) instead.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from kernelblaster.agents.opt_multislot import MultiSlotAgent  # noqa: E402
from kernelblaster.knowledge.claims import ClaimStore  # noqa: E402
from kernelblaster.knowledge.impl_store import ImplStore  # noqa: E402
from kernelblaster.knowledge.outcomes import OutcomeLog  # noqa: E402
from kernelblaster.knowledge.reflect import SemanticUpdater  # noqa: E402
from kernelblaster.knowledge.slots import (AdrenoEvaluator, LocalEvaluator, OpenCLLocalEvaluator,  # noqa: E402
                                           OpenCLServerEvaluator, ServerEvaluator, SlotProblem, SSHEvaluator,
                                           find_problems)


def _start_servers(gpu_name: str, log_dir: Path):
    """Spawn KernelBlaster's CUDA compile + exec servers (unless URLs are set)."""
    from kernelblaster.config import GPUType, config
    from kernelblaster.servers.management import initialize_compiler_server, initialize_gpu_server

    import urllib.request

    gpu = GPUType(gpu_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KERNELBLASTER_COMPILE_SERVER_NUM_WORKERS", "3")
    cwd = os.getcwd()
    os.chdir(ROOT)                    # servers are launched as ``python -m src.kernelblaster...``
    try:
        procs, urls = [], []
        lf = open(log_dir / "servers.log", "a")
        if config.COMPILE_SERVER_URL is None:
            proc, url = initialize_compiler_server(lf, None, log_dir / "compile_artifacts", None)
            config.set_compile_server_url(url)
            procs.append(proc)
            urls.append(url)
        p2, url2 = initialize_gpu_server(lf, gpu, None)
        procs.append(p2)
        urls.append(url2)
    finally:
        os.chdir(cwd)
    procs = [p for p in procs if p]
    for url in urls:                  # wait until each answers HTTP at all
        for _ in range(240):
            try:
                urllib.request.urlopen(url + "/docs", timeout=2)
                break
            except Exception as e:
                if getattr(e, "code", None):
                    break
                time.sleep(0.5)
        else:
            for p in procs:           # don't leave half a server set behind
                p.terminate()
            raise RuntimeError(f"server at {url} did not start; see {log_dir / 'servers.log'}")
    return gpu, procs


async def _run(args) -> int:
    for f in args.plugin:             # e.g. registers stub: LLMs for scripted runs
        runpy.run_path(str(f), run_name="kb_plugin")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    kdir = Path(args.knowledge_dir or out / "knowledge").resolve()
    store = ImplStore(kdir / "impl")
    olog = OutcomeLog(kdir / "outcomes.jsonl")
    claims = ClaimStore(kdir / "claims.json")
    updater = (SemanticUpdater(olog, claims, model=args.reflect_model, every_n=args.reflect_every,
                               logger=logger, trace_dir=kdir / "reflect_traces")
               if args.reflect_model else None)
    procs, evaluators = [], {}

    def evaluator_for(target: str):
        """One evaluator per slot target (CUDA and OpenCL problems can share a run)."""
        if target in evaluators:
            return evaluators[target]
        if target == "opencl":
            if args.evaluator == "board":
                ev = AdrenoEvaluator(args.board_host, remote_dir=args.remote_dir or "/tmp/kernelblaster_multislot",
                                     ssh_key=args.ssh_key, work_root=out / "work")
            elif args.evaluator == "server":
                from kernelblaster.config import GPUType
                ev = OpenCLServerEvaluator(GPUType(args.gpu), out / "work")
            elif args.evaluator == "local":
                ev = OpenCLLocalEvaluator(work_root=out / "work", max_parallel=args.compile_jobs)
            else:
                raise SystemExit(f"--evaluator {args.evaluator} does not run OpenCL problems (local|server|board)")
        elif args.evaluator == "server":
            gpu, ps = _start_servers(args.gpu, out / "servers")
            procs.extend(ps)
            ev = ServerEvaluator(gpu, out / "work")
        elif args.evaluator == "ssh":
            ev = SSHEvaluator(args.ssh_host, arch=arch_for("cuda"), remote_dir=args.remote_dir, ssh_key=args.ssh_key,
                              cuda_visible_devices=args.cuda_visible_devices, remote_path=args.remote_cuda_bin,
                              work_root=out / "work", max_parallel=args.compile_jobs)
        elif args.evaluator == "local":
            ev = LocalEvaluator(arch_for("cuda"), work_root=out / "work", max_parallel=args.compile_jobs)
        else:
            raise SystemExit(f"--evaluator {args.evaluator} runs OpenCL problems only")
        evaluators[target] = ev
        return ev

    def arch_for(target: str) -> str:
        if args.arch:
            return args.arch
        from kernelblaster.agents.multislot_opencl import DEFAULT_ARCH
        return DEFAULT_ARCH if target == "opencl" else "75"

    dirs = [d for p in args.problems for d in find_problems(Path(p))]
    targets = set()
    summary = []
    try:
        for d in dirs:
            p = SlotProblem(d)
            targets.add(p.target)
            t0 = time.time()
            agent = MultiSlotAgent(p, evaluator_for(p.target), model=args.model, model_fix=args.fix_model,
                                   out_dir=out / p.name, arch=arch_for(p.target), impl_store=store,
                                   outcome_log=olog, claims=claims, semantic_updater=updater,
                                   rollouts=args.rollouts, steps=args.steps,
                                   max_fix_attempts=args.max_fix, seed=args.seed, logger=logger, ncu=args.ncu)
            try:
                res = await agent.run()
            except Exception as e:
                logger.error(f"{p.name}: {e!r}")
                summary.append({"problem": p.name, "error": repr(e)})
                continue
            res["seconds"] = round(time.time() - t0, 1)
            summary.append({k: res[k] for k in ("problem", "baseline_partition_us", "final_partition_us",
                                                 "speedup", "final_ok", "llm_calls", "llm_usage", "seconds")})
            logger.info(f"{p.name}: {res['baseline_partition_us']:.1f} -> {res['final_partition_us']} us "
                        f"(speedup {res['speedup']}) in {res['seconds']}s, {res['llm_calls']} LLM calls")
    finally:
        for pr in procs:
            pr.terminate()
        for ev in evaluators.values():
            if hasattr(ev, "close"):
                await ev.close()
    refl = dict(updater.total_usage) if updater is not None else {}
    if refl:
        summary.append({"reflection_usage": refl})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    tot = sum((s.get("llm_usage") or {}).get("cost_usd", 0.0) for s in summary) + float(refl.get("cost_usd", 0.0))
    logger.info(f"LLM cost reported by providers (agent + reflection): ${tot:.4f}")
    if args.export_winners:
        for t in sorted(targets) or ["cuda"]:
            dst = Path(args.export_winners) / t if len(targets) > 1 else Path(args.export_winners)
            arch = arch_for(t)
            p = store.export_winners(dst, target=t, arch=f"sm_{arch}" if t == "cuda" else arch)
            logger.info(f"{t} winners -> {p}")
    print(json.dumps(summary, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problems", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.getenv("MODEL_CODEGEN_HARD", "claude-code:sonnet"))
    ap.add_argument("--fix-model", default=None)
    ap.add_argument("--reflect-model", default=os.getenv("KB_REFLECT_MODEL"))
    ap.add_argument("--reflect-every", type=int, default=8)
    ap.add_argument("--knowledge-dir", default=os.getenv("KB_KNOWLEDGE_DIR"))
    ap.add_argument("--arch", default=None,
                    help="CUDA: SM number, e.g. 75 / 89 (default 75); OpenCL: device, e.g. adreno650 (default) / titanrtx")
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--max-fix", type=int, default=2)
    ap.add_argument("--compile-jobs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evaluator", choices=["local", "server", "ssh", "board"], default="local",
                    help="local nvcc|g++ + run | KB compile/exec servers | build here, run on --ssh-host (CUDA) | "
                         "build + run on the Adreno board over ssh (OpenCL)")
    ap.add_argument("--board-host", default=None, help="user@board for --evaluator board "
                    "(default $KERNELBLASTER_ADRENO_BOARD_HOST)")
    ap.add_argument("--gpu", default="titanrtx", help="GPUType for --evaluator server")
    ap.add_argument("--ssh-host", help="user@host with the GPU (--evaluator ssh)")
    ap.add_argument("--ssh-key", help="ssh identity file")
    ap.add_argument("--remote-dir", help="writable dir on the ssh host / board for the binaries "
                    "(board default /tmp/kernelblaster_multislot, removed at the end)")
    ap.add_argument("--cuda-visible-devices", help="GPU(s) to use on the ssh host")
    ap.add_argument("--remote-cuda-bin", default="/usr/local/cuda/bin", help="ncu/CUDA bin dir on the ssh host")
    ap.add_argument("--ncu", choices=["off", "best", "all"], default="off",
                    help="NCU per slot: baseline + new slot bests (best) or every verified candidate (all)")
    ap.add_argument("--export-winners", default=None, help="dir for winners.json (ModelBlaster ingest)")
    ap.add_argument("--plugin", action="append", default=[],
                    help="python file run before the problems (e.g. it registers stub: models); repeatable")
    sys.exit(asyncio.run(_run(ap.parse_args())))


if __name__ == "__main__":
    main()
