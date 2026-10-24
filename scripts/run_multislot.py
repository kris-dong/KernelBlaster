"""Run the multi-slot optimiser on ModelBlaster-exported partition problems.

    python scripts/run_multislot.py PROBLEM_DIR_OR_ROOT [...] --out OUT \
        --model claude-code:sonnet [--reflect-model claude-code:haiku] \
        [--knowledge-dir DIR] [--arch 75] [--rollouts 4 --steps 6] \
        [--evaluator local|server|ssh|board ...] [--ncu off|best|all] \
        [--parallel-problems N] [--gpus auto] [--ssh-host H1 --ssh-host H2 --cuda-visible-devices auto]

PROBLEM_DIR_OR_ROOT: a directory with manifest.json, or a root searched for
them (``pipeline/kb_export.py`` output). The knowledge directory holds the
implementation store (``impl/``), the outcome log and the semantic claims;
share it across runs so later problems start from earlier winners and
lessons. ``--model`` is any name ``generate_code`` routes: ``claude-code:<m>``
(Claude Code CLI, no key), ``stub:<name>`` (registered by a ``--plugin``
file), or an API model.

Model heterogeneity (``agents/model_router.py``): ``--model-simple`` /
``--model-hard`` / ``--fix-model`` (or MODEL_CODEGEN_SIMPLE / MODEL_CODEGEN_HARD /
MODEL_FIX, as in the legacy optimized-RL flow, or ``--models-config`` JSON) give
each role its model; ``--hetero-policy`` says when codegen escalates to the hard
model, e.g. ``category,stall:3,share:0.5,explore,fix-escalate``.

Writes OUT/<problem>/result.json and best/slots/*.cu, OUT/summary.json, and
(with --export-winners) the store's winners for ModelBlaster's kb_ingest.
OUT/telemetry.jsonl has the span and LLM-call events (docs/telemetry.md;
``scripts/telemetry_report.py OUT``); KB_TELEMETRY=0 turns telemetry off.

Load balancing (docs/fast_paths.md, "Load balancing"): problems run
``--parallel-problems`` at a time (default: the GPUs in the pool) over one
shared store / outcome log / claims; every timed run and NCU pass leases one
GPU. ``--gpus auto|all|0,2`` picks local GPUs (auto: the idle ones);
``--evaluator ssh`` takes ``--ssh-host`` repeatedly, with ``--cuda-visible-devices
auto|ids`` once for all hosts or once per host; ``--compile-jobs`` defaults to
an adaptive limit from the free CPUs and memory.

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

from kernelblaster import telemetry as tel  # noqa: E402
from kernelblaster.agents.model_router import ModelRouter  # noqa: E402
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
    tel.configure(out / "telemetry.jsonl")
    with tel.span("kb.run", collect=True, laps=True) as sp:
        rc = await _run_inner(args, sp)
    tel.close()
    return rc


async def _run_inner(args, run_span) -> int:
    tel.lap("setup.knowledge")        # telemetry phases of the run (laps of kb.run)
    for f in args.plugin:             # e.g. registers stub: LLMs for scripted runs
        runpy.run_path(str(f), run_name="kb_plugin")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    kdir = Path(args.knowledge_dir or out / "knowledge").resolve()
    router = ModelRouter.build(args.model, simple=args.model_simple, hard=args.model_hard, fix=args.fix_model,
                               reflect=args.reflect_model, upkeep=args.upkeep_model, policy=args.hetero_policy,
                               config=args.models_config)
    args.reflect_model, args.upkeep_model = router.models["reflect"], router.models["upkeep"]
    logger.info(f"models: {router.describe()}")
    store = ImplStore(kdir / "impl")
    kv = None
    if args.knowledge == "v2":
        # knowledge.db (SQLite) replaces outcomes.jsonl + claims.json; see docs/knowledge_architecture.md
        from kernelblaster.knowledge.v2 import KnowledgeV2
        from kernelblaster.knowledge.v2.reflect_v2 import SemanticUpdaterV2
        kv = KnowledgeV2(kdir, upkeep_model=args.upkeep_model, logger=logger, upkeep_every=args.upkeep_every,
                         explore_frac=args.explore_frac)
        for gh in args.graph_hints or [x for x in os.getenv("KB_GRAPH_HINTS", "").split(os.pathsep) if x]:
            kv.sync_graph_hints(gh, target_key=f"cuda:sm_{args.arch or '75'}")   # MB decisions so far
        olog, claims = kv.outcomes, kv.claims
        updater = (SemanticUpdaterV2(kv, model=args.reflect_model, every_n=args.reflect_every, logger=logger,
                                     trace_dir=kdir / "reflect_traces") if args.reflect_model else None)
    else:
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
            hosts, cvd = _ssh_hosts(args)
            ev = SSHEvaluator(hosts, arch=arch_for("cuda"), remote_dir=args.remote_dir, ssh_key=args.ssh_key,
                              cuda_visible_devices=cvd, remote_path=args.remote_cuda_bin,
                              work_root=out / "work", max_parallel=args.compile_jobs)
        elif args.evaluator == "local":
            ev = LocalEvaluator(arch_for("cuda"), work_root=out / "work", max_parallel=args.compile_jobs,
                                gpus=args.gpus)
        else:
            raise SystemExit(f"--evaluator {args.evaluator} runs OpenCL problems only")
        evaluators[target] = ev
        return ev

    def arch_for(target: str) -> str:
        if args.arch:
            return args.arch
        from kernelblaster.agents.multislot_opencl import DEFAULT_ARCH
        return DEFAULT_ARCH if target == "opencl" else "75"

    tel.lap("setup.problems")
    dirs = [d for p in args.problems for d in find_problems(Path(p))]
    problems = [SlotProblem(d) for d in dirs]
    names = [p.name for p in problems]
    if len(set(names)) != len(names):           # OUT/<problem> must stay separate per problem
        raise SystemExit(f"problem names collide: {sorted(n for n in set(names) if names.count(n) > 1)}")
    targets = {p.target for p in problems}
    results: list = [None] * len(problems)
    t_all = time.time()
    tel.lap("problems")
    try:
        n_par = args.parallel_problems or await _default_parallel([evaluator_for(t) for t in sorted(targets)])
        n_par = max(1, min(n_par, len(problems) or 1))
        logger.info(f"{len(problems)} problem(s), {n_par} at a time")
        gate = asyncio.Semaphore(n_par)

        async def one(i: int, p: SlotProblem):
            async with gate:
                t0 = time.time()
                agent = MultiSlotAgent(p, evaluator_for(p.target), model=args.model, router=router,
                                       out_dir=out / p.name, arch=arch_for(p.target), impl_store=store,
                                       outcome_log=olog, claims=claims, semantic_updater=updater,
                                       rollouts=args.rollouts, steps=args.steps,
                                       max_fix_attempts=args.max_fix, seed=args.seed, logger=logger, ncu=args.ncu,
                                       knowledge_v2=kv)
                try:
                    res = await agent.run()
                except Exception as e:
                    logger.error(f"{p.name}: {e!r}")
                    results[i] = {"problem": p.name, "error": repr(e)}
                    return
                res["seconds"] = round(time.time() - t0, 1)
                results[i] = {k: res[k] for k in ("problem", "baseline_partition_us", "final_partition_us",
                                                  "speedup", "final_ok", "llm_calls", "llm_usage", "seconds")}
                if res.get("telemetry"):
                    results[i]["telemetry"] = {"wall_s": res["telemetry"]["wall_s"],
                                               "llm": res["telemetry"]["llm"]["total"],
                                               **res["telemetry"]["derived"]}
                logger.info(f"{p.name}: {res['baseline_partition_us']:.1f} -> {res['final_partition_us']} us "
                            f"(speedup {res['speedup']}) in {res['seconds']}s, {res['llm_calls']} LLM calls")

        await asyncio.gather(*[one(i, p) for i, p in enumerate(problems)])
        tel.lap("finish")
        if n_par > 1:
            # a problem's final forced pass may have overlapped another's; catch up on the rest
            if updater is not None:
                try:
                    await updater.maybe_reflect(force=True)
                except Exception as e:
                    logger.warning(f"final semantic update failed: {e!r}")
            elif kv is not None and len(kv.outcomes) > int(kv.db.get_meta("upkeep_cursor", 0) or 0):
                try:
                    await kv.maybe_upkeep(force=True)
                except Exception as e:
                    logger.warning(f"final knowledge-v2 upkeep failed: {e!r}")
    finally:
        for pr in procs:
            pr.terminate()
        for ev in evaluators.values():
            if hasattr(ev, "close"):
                await ev.close()
    summary = [r for r in results if r is not None]
    lb = {"parallel_problems": n_par if problems else 0, "seconds": round(time.time() - t_all, 1)}
    for t, ev in evaluators.items():
        pool = getattr(ev, "_pool", None)
        jobs = getattr(ev, "_jobs", None)
        lb[t] = {"evaluator": type(ev).__name__,
                 **({"gpus": pool.stats(), "peak_busy_gpus": pool.peak_busy} if pool is not None else {}),
                 **({"compile_jobs": jobs.describe(), "peak_compile_jobs": jobs.peak} if jobs is not None else {})}
    summary.append({"load_balance": lb})
    refl = dict(updater.total_usage) if updater is not None else {}
    if refl:
        summary.append({"reflection_usage": refl})
    if kv is not None:
        summary.append({"knowledge_v2": {"upkeep_model": args.upkeep_model, "upkeep_usage": dict(kv.usage),
                                         **{k: v for k, v in kv.summary().items() if k != "upkeep_usage"}}})
    summary.append({"models": router.describe()})
    tsum = tel.full_summary(run_span)          # the whole run so far (problems + catch-up passes)
    if tsum is not None:
        summary.append({"telemetry": tsum})
        d = tsum["derived"]
        logger.info("telemetry: " + ", ".join(f"{k} {v:.0f}s ({d['time_by_category_pct'].get(k, 0):.0f}%)"
                                              for k, v in d["time_by_category_s"].items() if v >= 0.5)
                    + f"; tokens in {tsum['llm']['total']['input_tokens']} out {tsum['llm']['total']['output_tokens']}"
                    + f", truncated {tsum['llm']['total']['truncated']}; events in {out / 'telemetry.jsonl'}")
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


def _ssh_hosts(args) -> tuple[list[str], dict]:
    """--ssh-host (repeatable) + --cuda-visible-devices (one for every host, or one per host)."""
    hosts = args.ssh_host or []
    if not hosts:
        raise SystemExit("--evaluator ssh needs --ssh-host")
    if not args.remote_dir:
        raise SystemExit("--evaluator ssh needs --remote-dir")
    cvd = args.cuda_visible_devices or ["auto"]
    if len(cvd) == 1:
        cvd = cvd * len(hosts)
    if len(cvd) != len(hosts):
        raise SystemExit(f"--cuda-visible-devices: give one value, or one per --ssh-host ({len(hosts)})")
    return hosts, dict(zip(hosts, cvd))


async def _default_parallel(evs) -> int:
    """Problems at a time by default: the GPUs the evaluators can run on."""
    n = 1
    for ev in evs:
        if hasattr(ev, "pool"):
            n = max(n, len(await ev.pool()))
        elif type(ev).__name__ == "ServerEvaluator":
            from kernelblaster.servers.utils.resources import local_devices
            ids = os.getenv("KERNELBLASTER_GPU_SERVER_GPU_IDS", "").strip()
            n = max(n, len(ids.split(",")) if ids else len(await asyncio.to_thread(local_devices, None)))
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problems", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.getenv("MODEL_CODEGEN_HARD", "claude-code:sonnet"))
    ap.add_argument("--fix-model", default=None, help="fix role (default: --model-simple; env MODEL_FIX)")
    ap.add_argument("--model-simple", default=None,
                    help="codegen for mechanical techniques (default --model; env MODEL_CODEGEN_SIMPLE)")
    ap.add_argument("--model-hard", default=None,
                    help="codegen for structural techniques / escalations (default --model; env MODEL_CODEGEN_HARD)")
    ap.add_argument("--hetero-policy", default=None,
                    help="when codegen uses --model-hard: category,stall:N,share:X,explore,fix-escalate,always,never "
                         "(default category,fix-escalate; env KB_HETERO_POLICY)")
    ap.add_argument("--models-config", default=None,
                    help='JSON {"models": {role: model}, "policy": "..."}; roles codegen_simple, codegen_hard, '
                         "fix, reflect, upkeep; flags win")
    ap.add_argument("--reflect-model", default=os.getenv("KB_REFLECT_MODEL"))
    ap.add_argument("--reflect-every", type=int, default=8)
    ap.add_argument("--knowledge-dir", default=os.getenv("KB_KNOWLEDGE_DIR"))
    ap.add_argument("--arch", default=None,
                    help="CUDA: SM number, e.g. 75 / 89 (default 75); OpenCL: device, e.g. adreno650 (default) / titanrtx")
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--max-fix", type=int, default=2)
    ap.add_argument("--compile-jobs", type=int, default=None,
                    help="concurrent nvcc processes (default: adaptive to free CPUs and memory; "
                         "env KERNELBLASTER_MULTISLOT_COMPILE_JOBS)")
    ap.add_argument("--parallel-problems", type=int, default=None,
                    help="problems optimised at a time over the shared GPU pool (default: number of GPUs)")
    ap.add_argument("--gpus", default="auto",
                    help="local GPUs: auto (idle ones; KERNELBLASTER_ALLOW_SHARED_GPU=1 adds busy ones), all, "
                         "or nvidia-smi indices / UUIDs, e.g. 0,2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evaluator", choices=["local", "server", "ssh", "board"], default="local",
                    help="local nvcc|g++ + run | KB compile/exec servers | build here, run on --ssh-host (CUDA) | "
                         "build + run on the Adreno board over ssh (OpenCL)")
    ap.add_argument("--board-host", default=None, help="user@board for --evaluator board "
                    "(default $KERNELBLASTER_ADRENO_BOARD_HOST)")
    ap.add_argument("--gpu", default="titanrtx", help="GPUType for --evaluator server")
    ap.add_argument("--ssh-host", action="append", default=[],
                    help="user@host with GPUs (--evaluator ssh); repeatable, runs spread over every host's GPUs")
    ap.add_argument("--ssh-key", help="ssh identity file")
    ap.add_argument("--remote-dir", help="writable dir on the ssh host / board for the binaries "
                    "(board default /tmp/kernelblaster_multislot, removed at the end)")
    ap.add_argument("--cuda-visible-devices", action="append", default=[],
                    help="GPUs on the ssh host(s): auto (default: idle ones), all, or ids, e.g. 2,3; "
                         "once for every host or once per --ssh-host")
    ap.add_argument("--remote-cuda-bin", default="/usr/local/cuda/bin", help="ncu/CUDA bin dir on the ssh host")
    ap.add_argument("--ncu", choices=["off", "best", "all"], default="off",
                    help="NCU per slot: baseline + new slot bests (best) or every verified candidate (all)")
    ap.add_argument("--knowledge", choices=["v1", "v2"], default="v1",
                    help="v1: claims.json + technique bandit; v2: knowledge.db (states, ontology, hypotheses)")
    ap.add_argument("--upkeep-model", default=None,
                    help="v2: model for hypothesis generation + state discovery (unset: deterministic upkeep only)")
    ap.add_argument("--upkeep-every", type=int, default=8, help="v2: new outcomes between upkeep passes")
    ap.add_argument("--explore-frac", type=float, default=0.15, help="v2: share of steps for hypotheses")
    ap.add_argument("--graph-hints", action="append", default=[],
                    help="v2: ModelBlaster graph_hints.json whose accepted/rejected decisions are synced into the "
                         "graph hypotheses at the start of the run (repeatable; env KB_GRAPH_HINTS, ':'-separated)")
    ap.add_argument("--export-winners", default=None, help="dir for winners.json (ModelBlaster ingest)")
    ap.add_argument("--plugin", action="append", default=[],
                    help="python file run before the problems (e.g. it registers stub: models); repeatable")
    sys.exit(asyncio.run(_run(ap.parse_args())))


if __name__ == "__main__":
    main()
