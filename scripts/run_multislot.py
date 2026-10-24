"""Run the multi-slot optimiser on ModelBlaster-exported partition problems.

    python scripts/run_multislot.py PROBLEM_DIR_OR_ROOT [...] --out OUT \
        --model claude-code:sonnet [--reflect-model claude-code:haiku] \
        [--knowledge-dir DIR] [--arch 75] [--rollouts 4 --steps 6] \
        [--evaluator local|server --gpu titanrtx]

PROBLEM_DIR_OR_ROOT: a directory with manifest.json, or a root searched for
them (``pipeline/kb_export.py`` output). The knowledge directory holds the
implementation store (``impl/``), the outcome log and the semantic claims;
share it across runs so later problems start from earlier winners and
lessons. ``--model`` is any name ``generate_code`` routes: ``claude-code:<m>``
(Claude Code CLI, no key), ``stub:<name>``, or an API model.

Writes OUT/<problem>/result.json and best/slots/*.cu, OUT/summary.json, and
(with --export-winners) the store's winners for ModelBlaster's kb_ingest.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
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
from kernelblaster.knowledge.slots import LocalEvaluator, ServerEvaluator, SlotProblem, find_problems  # noqa: E402


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
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    kdir = Path(args.knowledge_dir or out / "knowledge").resolve()
    store = ImplStore(kdir / "impl")
    olog = OutcomeLog(kdir / "outcomes.jsonl")
    claims = ClaimStore(kdir / "claims.json")
    updater = (SemanticUpdater(olog, claims, model=args.reflect_model, every_n=args.reflect_every,
                               logger=logger, trace_dir=kdir / "reflect_traces")
               if args.reflect_model else None)
    procs = []
    if args.evaluator == "server":
        gpu, procs = _start_servers(args.gpu, out / "servers")
        evaluator = ServerEvaluator(gpu, out / "work")
    else:
        evaluator = LocalEvaluator(args.arch, work_root=out / "work", max_parallel=args.compile_jobs)
    dirs = [d for p in args.problems for d in find_problems(Path(p))]
    summary = []
    try:
        for d in dirs:
            p = SlotProblem(d)
            t0 = time.time()
            agent = MultiSlotAgent(p, evaluator, model=args.model, model_fix=args.fix_model,
                                   out_dir=out / p.name, arch=args.arch, impl_store=store,
                                   outcome_log=olog, claims=claims, semantic_updater=updater,
                                   rollouts=args.rollouts, steps=args.steps,
                                   max_fix_attempts=args.max_fix, seed=args.seed, logger=logger)
            try:
                res = await agent.run()
            except Exception as e:
                logger.error(f"{p.name}: {e!r}")
                summary.append({"problem": p.name, "error": repr(e)})
                continue
            res["seconds"] = round(time.time() - t0, 1)
            summary.append({k: res[k] for k in ("problem", "baseline_partition_us", "final_partition_us",
                                                 "speedup", "final_ok", "llm_calls", "seconds")})
            logger.info(f"{p.name}: {res['baseline_partition_us']:.1f} -> {res['final_partition_us']} us "
                        f"(speedup {res['speedup']}) in {res['seconds']}s, {res['llm_calls']} LLM calls")
    finally:
        for pr in procs:
            pr.terminate()
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if args.export_winners:
        p = store.export_winners(Path(args.export_winners), arch=f"sm_{args.arch}")
        logger.info(f"winners -> {p}")
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
    ap.add_argument("--arch", default="75", help="SM number, e.g. 75 / 89")
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--max-fix", type=int, default=2)
    ap.add_argument("--compile-jobs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evaluator", choices=["local", "server"], default="local")
    ap.add_argument("--gpu", default="titanrtx", help="GPUType for --evaluator server")
    ap.add_argument("--export-winners", default=None, help="dir for winners.json (ModelBlaster ingest)")
    sys.exit(asyncio.run(_run(ap.parse_args())))


if __name__ == "__main__":
    main()
