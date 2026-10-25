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
OUT/trace/ has the full trace -- every prompt, response, candidate source,
build, driver and NCU output, decision (docs/trace.md; ``scripts/trace_view.py
OUT``); ``--trace meta|off`` (env KB_TRACE) keeps hashes only / nothing.

Load balancing (docs/fast_paths.md, "Load balancing"): problems run
``--parallel-problems`` at a time (default: the GPUs in the pool) over one
shared store / outcome log / claims; every timed run and NCU pass leases one
GPU. ``--gpus auto|all|0,2`` picks local GPUs (auto: the idle ones);
``--evaluator ssh`` takes ``--ssh-host`` repeatedly, with ``--cuda-visible-devices
auto|ids`` once for all hosts or once per host; ``--compile-jobs`` defaults to
an adaptive limit from the free CPUs and memory.

Resume (docs/multislot_optimization.md, "Resume"): OUT keeps a journal per
problem (OUT/<problem>/journal) and OUT/run.json; rerunning the same command
after a kill continues where it stopped (finished problems are skipped,
in-flight steps reuse their journaled LLM responses). A different
configuration is refused (exit 3) unless --force-restart. SIGTERM / SIGINT stop
new steps, let in-flight ones finish (--stop-grace), flush and exit 75; a second
signal exits 76 at once.

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
import fcntl
import json
import os
import runpy
import signal
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from kernelblaster import telemetry as tel  # noqa: E402
from kernelblaster import trace  # noqa: E402
from kernelblaster.agents import journal as J  # noqa: E402
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


def run_config(args, router) -> dict:
    """What a resumed run must share with the journal (operational flags -- evaluator,
    hosts, GPUs, parallelism -- may change between segments)."""
    return {"models": router.describe(), "rollouts": args.rollouts, "steps": args.steps, "max_fix": args.max_fix,
            "seed": args.seed, "reward": args.reward, "e2e_accept_pct": args.e2e_accept_pct,
            "knowledge": args.knowledge, "ncu": args.ncu, "arch": args.arch,
            "knowledge_dir": str(Path(args.knowledge_dir).resolve()) if args.knowledge_dir else None}


class Stopper:
    """SIGTERM / SIGINT: the first stops new steps, lets in-flight steps finish for
    ``grace_s`` (then cancels them), flushes the journals and closes the evaluators
    (exit ``EXIT_INTERRUPTED``); a second exits at once (``EXIT_ABORTED``)."""

    def __init__(self, grace_s: float):
        self.event = asyncio.Event()
        self.grace_s = grace_s
        self.signal: Optional[int] = None
        self.task: Optional[asyncio.Task] = None
        self.cancelled = False

    def install(self, loop, task) -> None:
        self.task = task
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            if signal.getsignal(sig) is signal.SIG_IGN:     # inherited ignore (nohup, a shell's `&`): kept
                continue
            try:
                loop.add_signal_handler(sig, self._on, sig)
            except (NotImplementedError, RuntimeError, ValueError):
                pass
        if os.getenv("KB_STOP_WITH_PARENT", "").lower() in ("1", "true", "yes", "on"):
            self._watch = loop.create_task(self._watch_parent(os.getppid()))

    async def _watch_parent(self, ppid: int) -> None:
        """KB_STOP_WITH_PARENT=1 (ModelBlaster's kb_graph_opt): the parent died -> stop as on SIGTERM."""
        while not self.event.is_set():
            await asyncio.sleep(2.0)
            if os.getppid() != ppid:
                logger.warning(f"parent process {ppid} is gone")
                self._on(signal.SIGTERM)
                return

    def _on(self, sig) -> None:
        if self.event.is_set():
            logger.error(f"second signal {signal.Signals(sig).name}: exiting now (journals are consistent)")
            try:
                tel.close()
            finally:
                os._exit(J.EXIT_ABORTED)
        self.signal = sig
        self.event.set()
        logger.warning(f"{signal.Signals(sig).name}: no new steps; in-flight steps get {self.grace_s:.0f}s, then "
                       f"the run stops (resumable: rerun the same command); a second signal exits at once")
        asyncio.get_running_loop().call_later(self.grace_s, self._cancel)

    def _cancel(self) -> None:
        if self.task is not None and not self.task.done():
            logger.warning("grace period over: abandoning in-flight steps")
            self.cancelled = True
            self.task.cancel()


async def _run(args) -> int:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stopper = Stopper(args.stop_grace)
    args._stopper = stopper
    # one process per out dir: a resumed run waits for an orphan of the killed one to stop
    lock_fd = os.open(str(out / ".lock"), os.O_RDWR | os.O_CREAT, 0o666)
    t_end = time.monotonic() + args.stop_grace + 120
    while True:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() > t_end:
                logger.error(f"{out} is in use by another run_multislot.py process ({out / '.lock'})")
                return J.EXIT_LOCKED
            await asyncio.sleep(0.5)
    seg = len((J.read_json(out / "run.json") or {}).get("segments", [])) if not args.force_restart else 0
    tel.configure(out / "telemetry.jsonl", run_id=f"{tel.run_id()}.s{seg}" if seg else None)
    # the trace store (docs/trace.md): prompts, responses, candidates, builds, driver / NCU output;
    # a resumed segment appends to the same store
    trace.configure(out / "trace", level=args.trace, run_id=tel.run_id())
    task = asyncio.current_task()
    stopper.install(asyncio.get_running_loop(), task)
    with tel.span("kb.run", collect=True, laps=True, segment=seg) as sp:
        try:
            rc = await _run_inner(args, sp)
        except asyncio.CancelledError:
            if not stopper.event.is_set():
                raise
            rc = J.EXIT_INTERRUPTED
        finally:
            ts = trace.close()
            if ts:
                logger.info(f"trace: {ts['records']} records, {ts['blobs']} blobs "
                            f"({ts['bytes_stored'] / 1e6:.1f} MB stored of {ts['bytes_in'] / 1e6:.1f} MB, "
                            f"{ts['blob_hits']} dedup hits, {ts['seconds']:.2f}s) in {out / 'trace'}")
    tel.close()
    return rc


async def _run_inner(args, run_span) -> int:
    tel.lap("setup.knowledge")        # telemetry phases of the run (laps of kb.run)
    plugin_evaluator = None
    for f in args.plugin:             # e.g. registers stub: LLMs for scripted runs
        g = runpy.run_path(str(f), run_name="kb_plugin")
        plugin_evaluator = g.get("kb_evaluator", plugin_evaluator)   # --evaluator plugin
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
        if args.evaluator == "plugin":
            if plugin_evaluator is None:
                raise SystemExit("--evaluator plugin needs a --plugin file defining kb_evaluator(target, args, out)")
            ev = plugin_evaluator(target, args, out)
        elif target == "opencl":
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
                              work_root=out / "work", max_parallel=args.compile_jobs, runtime=args.runtime,
                              gpu_model=args.gpu_model, remote_manifest=out / "remote.jsonl")
        elif args.evaluator == "local":
            ev = LocalEvaluator(arch_for("cuda"), work_root=out / "work", max_parallel=args.compile_jobs,
                                gpus=args.gpus, runtime=args.runtime, gpu_model=args.gpu_model)
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
    # ---- resume (docs/multislot_optimization.md, "Resume"): an out dir with a journal
    # continues where it stopped when the configuration and the problems match
    stopper = getattr(args, "_stopper", None) or Stopper(args.stop_grace)
    run_j = J.RunJournal(out)
    config = run_config(args, router)
    phashes = {p.name: J.problem_hash(p) for p in problems}
    journals = {p.name: J.ProblemJournal(out / p.name / "journal") for p in problems}
    if args.force_restart:
        if run_j.exists or any(j.exists for j in journals.values()):
            logger.warning(f"--force-restart: discarding the journals in {out}")
        run_j.clear()
        for name, j in journals.items():
            j.clear()
            for f in ("result.json", "proposals.json"):
                (out / name / f).unlink(missing_ok=True)
    else:
        why = run_j.check(config, phashes)
        for p in problems:
            st = journals[p.name].state
            if st and st.get("problem_hash") not in (None, phashes[p.name]):
                why.append(f"problem {p.name}: journal hash {st.get('problem_hash')} != {phashes[p.name]}")
            if st and not run_j.exists and (st.get("steps"), st.get("rollouts")) != (args.steps, args.rollouts):
                why.append(f"problem {p.name}: journal has {st.get('rollouts')}x{st.get('steps')} rollouts x steps")
        if why:
            logger.error(f"{out} holds a run with another configuration; refusing to resume (--force-restart "
                         f"discards it, or use another --out):\n  " + "\n  ".join(why))
            return J.EXIT_CONFIG_MISMATCH
    resuming = run_j.exists or any(j.exists for j in journals.values())
    if args.resume and not resuming:
        logger.info(f"--resume: no journal in {out}; starting fresh")
    segment = run_j.begin(config, phashes, sys.argv)
    if resuming:
        done = [n for n, j in journals.items() if j.phase == "done" and (out / n / "result.json").exists()]
        logger.info(f"resuming {out} (segment {segment}): {len(done)} of {len(problems)} problem(s) done")
        tel.emit({"t": "resume", "ts": time.time(), "segment": segment, "problems_done": len(done)})
        # keep the timings on the GPU model the killed segment measured on
        if args.gpu_model is None:
            models = {(j.state or {}).get("gpu", {}).get("name") for j in journals.values()
                      if (j.state or {}).get("gpu")} - {None}
            if len(models) == 1:
                args.gpu_model = models.pop()
                logger.info(f"resume: pinning --gpu-model {args.gpu_model!r} (the journal's GPU)")
    prev_refl = run_j.usage_before("reflection", segment)
    prev_upk = run_j.usage_before("upkeep", segment)

    def on_checkpoint(_agent):
        """Run-level usage (reflection / upkeep) persisted with every problem checkpoint."""
        if updater is not None:
            run_j.set_usage("reflection", dict(updater.total_usage), segment)
        if kv is not None:
            run_j.set_usage("upkeep", dict(kv.usage), segment)
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
                if stopper.event.is_set():                 # stopping: problems not started stay as they are
                    results[i] = {"problem": p.name, "interrupted": True, "phase": journals[p.name].phase}
                    return
                jr = journals[p.name]
                if jr.phase == "done" and (out / p.name / "result.json").exists():
                    res = json.loads((out / p.name / "result.json").read_text())
                    results[i] = _summary_row(res)
                    results[i]["resume"] = "done in an earlier segment"
                    return
                ev = evaluator_for(p.target)
                if hasattr(ev, "cleanup_previous") and resuming:     # remote leftovers of the killed segment
                    await ev.cleanup_previous()
                agent = MultiSlotAgent(p, ev, model=args.model, router=router, journal=jr,
                                       stop_event=stopper.event, on_checkpoint=on_checkpoint,
                                       out_dir=out / p.name, arch=arch_for(p.target), impl_store=store,
                                       outcome_log=olog, claims=claims, semantic_updater=updater,
                                       rollouts=args.rollouts, steps=args.steps,
                                       max_fix_attempts=args.max_fix, seed=args.seed, logger=logger, ncu=args.ncu,
                                       knowledge_v2=kv, reward=args.reward, e2e_accept_pct=args.e2e_accept_pct)
                try:
                    res = await agent.run()
                except J.RunInterrupted as e:
                    logger.warning(str(e))
                    results[i] = {"problem": p.name, "interrupted": True, "steps_done": len(agent.steps_log)}
                    return
                except asyncio.CancelledError:
                    agent._checkpoint()                    # completed steps are in already; flush the rest
                    raise
                except Exception as e:
                    logger.error(f"{p.name}: {e!r}")
                    results[i] = {"problem": p.name, "error": repr(e)}
                    return
                res["seconds"] = round(time.time() - t0, 1)
                results[i] = _summary_row(res)
                logger.info(f"{p.name}: {res['baseline_partition_us']:.1f} -> {res['final_partition_us']} us "
                            f"(speedup {res['speedup']}) in {res['seconds']}s, {res['llm_calls']} LLM calls")

        await asyncio.gather(*[one(i, p) for i, p in enumerate(problems)])
        tel.lap("finish")
        if stopper.event.is_set():
            raise J.RunInterrupted("stopped on a signal")
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
    except (J.RunInterrupted, asyncio.CancelledError) as e:
        interrupted = e
    else:
        interrupted = None
    finally:
        for pr in procs:
            pr.terminate()
        for ev in evaluators.values():
            if hasattr(ev, "close"):
                try:
                    await ev.close()
                except Exception as e:
                    logger.warning(f"closing {type(ev).__name__} failed: {e!r}")
        on_checkpoint(None)
    if interrupted is not None:
        n_done = sum(1 for r in results if r and not r.get("interrupted") and not r.get("error"))
        run_j.end(segment, "interrupted", signal=stopper.signal, problems_done=n_done)
        logger.warning(f"interrupted: {n_done} of {len(problems)} problem(s) finished; journals in {out} -- rerun "
                       f"the same command to resume (exit {J.EXIT_INTERRUPTED})")
        return J.EXIT_INTERRUPTED
    summary = [r for r in results if r is not None]
    lb = {"parallel_problems": n_par if problems else 0, "seconds": round(time.time() - t_all, 1)}
    for t, ev in evaluators.items():
        pool = getattr(ev, "_pool", None)
        jobs = getattr(ev, "_jobs", None)
        lb[t] = {"evaluator": type(ev).__name__,
                 **({"runtime": ev.host_stats()} if hasattr(ev, "host_stats") else {}),
                 **({"gpus": pool.stats(), "peak_busy_gpus": pool.peak_busy} if pool is not None else {}),
                 **({"compile_jobs": jobs.describe(), "peak_compile_jobs": jobs.peak} if jobs is not None else {})}
    summary.append({"load_balance": lb})
    refl = dict(updater.total_usage) if updater is not None else {}
    if updater is not None:                    # + earlier segments of a resumed run
        for k, v in prev_refl.items():
            refl[k] = refl.get(k, 0) + v
    if refl:
        summary.append({"reflection_usage": refl})
    if kv is not None:
        upk = dict(kv.usage)
        for k, v in prev_upk.items():
            upk[k] = upk.get(k, 0) + v
        summary.append({"knowledge_v2": {"upkeep_model": args.upkeep_model, "upkeep_usage": upk,
                                         **{k: v for k, v in kv.summary().items() if k != "upkeep_usage"}}})
    summary.append({"models": router.describe()})
    summary.append({"resume": {"segment": segment, "segments": segment + 1, "resumed": resuming,
                               "llm_replayed": sum((r.get("resume_info") or {}).get("llm_replayed", 0)
                                                   for r in results if r)}})
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
    run_j.end(segment, "done")
    print(json.dumps(summary, indent=1))
    return 0


def _summary_row(res: dict) -> dict:
    row = {k: res.get(k) for k in ("problem", "baseline_partition_us", "final_partition_us", "speedup", "final_ok",
                                   "llm_calls", "llm_usage", "seconds", "reward", "baseline_e2e_us", "final_e2e_us",
                                   "e2e_speedup")}
    if res.get("resume"):
        row["resume_info"] = {k: v for k, v in res["resume"].items() if k != "run_segments"}
    if res.get("telemetry"):
        row["telemetry"] = {"wall_s": res["telemetry"]["wall_s"], "llm": res["telemetry"]["llm"]["total"],
                            **res["telemetry"]["derived"]}
    return row


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
    ap.add_argument("--gpu-model", default=None,
                    help="local/ssh evaluators: only GPUs of this model, e.g. 'NVIDIA L40S' (none of them: an error); "
                         "every result.json records the GPU it was timed on ('gpu')")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reward", choices=["e2e", "slot"], default=os.getenv("KERNELBLASTER_MULTISLOT_REWARD", "e2e"),
                    help="what a step is rewarded and accepted on: the deployed end-to-end partition time (default) "
                         "or the changed slot's time")
    ap.add_argument("--e2e-accept-pct", type=float, default=None,
                    help="end-to-end gain (%% of the partition) a step needs to join the composite (default 1.0; "
                         "env KERNELBLASTER_MULTISLOT_E2E_ACCEPT_PCT)")
    ap.add_argument("--runtime", choices=["auto", "host", "process"],
                    default=os.getenv("KERNELBLASTER_MULTISLOT_RUNTIME", "auto"),
                    help="local/ssh evaluators: persistent driver host per problem and GPU (auto: where the driver "
                         "has it) or one driver process per run")
    ap.add_argument("--resume", action="store_true",
                    help="continue the run in --out (also the default whenever --out holds a journal with the same "
                         "configuration; a different configuration is refused)")
    ap.add_argument("--force-restart", action="store_true",
                    help="discard the journals / results in --out and start over")
    ap.add_argument("--stop-grace", type=float, default=float(os.getenv("KB_STOP_GRACE_S", "120")),
                    help="seconds in-flight steps get to finish after SIGTERM / SIGINT (then they are abandoned; "
                         "a resumed run redoes them from their journaled LLM responses)")
    ap.add_argument("--evaluator", choices=["local", "server", "ssh", "board", "plugin"], default="local",
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
    ap.add_argument("--trace", choices=["off", "meta", "full"], default=trace.core._env_level("full"),
                    help="trace store OUT/trace (docs/trace.md): full = prompts, responses, sources, build / driver "
                         "/ NCU output (deduplicated, gzip); meta = hashes and sizes only; off (env KB_TRACE)")
    ap.add_argument("--plugin", action="append", default=[],
                    help="python file run before the problems (e.g. it registers stub: models); repeatable")
    sys.exit(asyncio.run(_run(ap.parse_args())))


if __name__ == "__main__":
    main()
