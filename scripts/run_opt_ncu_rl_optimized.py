"""Standalone runner for the optimized RL CUDA-optimisation flow.

Companion to ``src/kernelblaster/agents/opt_ncu_rl_optimized.py`` and
``src/kernelblaster/agents/database_optimized.py``. Mirrors the shape of
``scripts/run_kgen_opencl.py``: starts servers (or reuses existing ones), walks
a problem set, dispatches one ``OptimizedRLNCUAgent.run()`` per problem, and
writes outputs under ``out/<experiment-name>/<model>/<level>/<problem>/``.

Defaults are chosen so a clean invocation does the right thing on a host with
the ``kernelblaster`` container already running and CUDA compile + GPU servers
on ``localhost:7001`` / ``localhost:7002`` (or whatever ``COMPILE_SERVER_URL``
/ ``GPU_SERVER_URL_<GPU>`` are pointed at by the wrapper)::

    python scripts/run_opt_ncu_rl_optimized.py \
        --subset L1 --problem-numbers 1-5 \
        --experiment-name opt_ncu_rl_v1

Model heterogeneity envelope (override via env vars):

    MODEL_PLAN              cheap model for state analysis + plan generation
    MODEL_CODEGEN_SIMPLE    cheap model for memory-pattern / occupancy edits
    MODEL_CODEGEN_HARD      premium model for tiling / fusion / tensor-core
    MODEL_FIX               fix-attempt model (default: MODEL_CODEGEN_SIMPLE)

When ``MODEL_*`` env vars are unset the agent uses ``--model`` for everything,
which lets you A/B compare against the legacy flow at the same model rate.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

# SSL bootstrap so the OpenAI / Anthropic clients work behind corporate proxies
# (mirrors run_kgen_opencl.py).
if not os.environ.get("SSL_CERT_FILE") and not os.environ.get("REQUESTS_CA_BUNDLE"):
    try:
        import certifi
        _cert = certifi.where()
        if _cert and Path(_cert).exists():
            os.environ["SSL_CERT_FILE"] = _cert
            os.environ.setdefault("REQUESTS_CA_BUNDLE", _cert)
    except Exception:
        pass

from src.kernelblaster.agents.cost_tracker import CostTracker
from src.kernelblaster.agents.database_optimized import OptimizedOptimizationDatabase
from src.kernelblaster.agents.feedback import FeedbackConfig
from src.kernelblaster.agents.opt_ncu_rl_optimized import OptimizedRLNCUAgent
from src.kernelblaster.agents.progress_writer import ProgressWriter
from src.kernelblaster.agents.utils import generate_code_retry
from src.kernelblaster.config import GPUType, config


# ---------------------------------------------------------------------------
# Model smoke test — fail fast before any expensive work begins.
# ---------------------------------------------------------------------------


_SMOKE_PROMPT = (
    "Reply with EXACTLY this token sequence and nothing else: SMOKE_OK_42. "
    "Do not add quotes, punctuation, code blocks, or commentary."
)


def _smoke_probe_thread_body(
    *,
    model: str,
    timeout_s: int,
    out: dict,  # mutated under a thread-local key (model)
    tracker: Optional["CostTracker"] = None,
) -> None:
    """Body of a daemon thread that runs one smoke probe end-to-end.

    Each thread spins up its own event loop with ``asyncio.run`` so that the
    blocking I/O in the underlying SDK (Anthropic / Bedrock auth handshake on
    first call) is isolated from the main runner's loop. The outer
    ``thread.join(timeout=...)`` is what gives us a hard, cancellation-proof
    deadline.
    """
    import asyncio as _a
    try:
        async def _do_call():
            return await _a.wait_for(
                generate_code_retry(
                    messages=[{"role": "user", "content": _SMOKE_PROMPT}],
                    model=model,
                    logger=logger,
                    max_retries=1,
                ),
                timeout=timeout_s,
            )
        resp = _a.run(_do_call())
        text = (resp.generations[0] if resp.generations else "").strip()
        usage = getattr(resp, "usage", None) or {}
        if tracker is not None:
            try:
                tracker.record(model=model, usage=usage, role="smoke_test", logger=logger)
            except Exception:
                pass
        if "SMOKE_OK_42" in text:
            out[model] = (True, f"strict OK ({len(text)} chars, usage={usage})")
        elif len(text) >= 5 and any(c.isalnum() for c in text):
            out[model] = (True, f"loose OK (preview={text[:40]!r})")
        else:
            out[model] = (False, f"empty / unparseable response: {text[:80]!r}")
    except _a.TimeoutError:
        out[model] = (False, f"timeout after {timeout_s}s (inner)")
    except BaseException as e:  # noqa: BLE001 — surface anything provider-side
        out[model] = (False, f"exception: {type(e).__name__}: {e}")


def _sync_run_smoke_tests(
    models: dict[str, str],
    *,
    timeout_s: int = 15,
    overall_timeout_s: Optional[int] = None,
    tracker: Optional["CostTracker"] = None,
) -> bool:
    """Threading-based smoke runner — bulletproof against blocking SDK calls.

    Spawns one daemon thread per unique model. Each thread runs its own
    ``asyncio.run`` with an inner per-model timeout. The main thread waits
    for each via ``thread.join(timeout=...)``; if the join expires the thread
    is abandoned (it's daemon, so the process can still exit cleanly). This
    avoids the well-known issue where ``asyncio.wait_for`` blocks on
    ``CancelledError`` propagation through synchronous boto3/httpx code.
    """
    import threading
    import time as _time

    seen: dict[str, list[str]] = {}
    for role, name in models.items():
        if not name:
            continue
        seen.setdefault(name, []).append(role)

    if not seen:
        logger.warning("Smoke test skipped: no models configured.")
        return True

    if overall_timeout_s is None:
        overall_timeout_s = max(60, timeout_s * 2)

    model_list = ", ".join(seen.keys())
    logger.info(
        f"Smoke-testing {len(seen)} unique model(s) IN PARALLEL "
        f"(per-model {timeout_s}s, batch {overall_timeout_s}s, thread-isolated): "
        f"{model_list}"
    )
    for name, roles in seen.items():
        logger.info(f"  → probing {name}  (roles: {', '.join(roles)})")

    out: dict[str, tuple[bool, str]] = {}
    threads: list[tuple[threading.Thread, str]] = []
    for name, _roles in seen.items():
        # daemon=True → thread does NOT block process exit even if it hangs.
        t = threading.Thread(
            target=_smoke_probe_thread_body,
            kwargs={
                "model": name,
                "timeout_s": timeout_s,
                "out": out,
                "tracker": tracker,
            },
            name=f"smoke-{name[:24]}",
            daemon=True,
        )
        t.start()
        threads.append((t, name))

    deadline = _time.monotonic() + overall_timeout_s
    for t, name in threads:
        remaining = max(0.0, deadline - _time.monotonic())
        t.join(timeout=remaining)
        if t.is_alive():
            out.setdefault(
                name,
                (False, f"abandoned: thread still alive at {overall_timeout_s}s batch deadline"),
            )

    all_ok = True
    for name, roles in seen.items():
        ok, info = out.get(name, (False, "no result captured"))
        marker = "OK " if ok else "FAIL"
        logger.info(f"  [{marker}] {name}  (roles: {', '.join(roles)})  — {info}")
        if not ok:
            all_ok = False
    return all_ok


async def run_smoke_tests(
    models: dict[str, str],
    *,
    timeout_s: int = 15,
    tracker: Optional["CostTracker"] = None,
    overall_timeout_s: Optional[int] = None,
) -> bool:
    """Async wrapper — delegates to the thread-based ``_sync_run_smoke_tests``.

    Pushes the smoke test off the main event loop entirely so the rest of
    the runner (cost tracker, progress writer) can keep ticking.
    """
    return await asyncio.to_thread(
        _sync_run_smoke_tests,
        models,
        timeout_s=timeout_s,
        overall_timeout_s=overall_timeout_s,
        tracker=tracker,
    )


async def smoke_test_model(
    model: str,
    *,
    timeout_s: int = 15,
    tracker: Optional["CostTracker"] = None,
) -> tuple[bool, str]:
    """Single-model smoke probe — kept for back-compat / direct callers.

    Delegates to the thread-based implementation so callers also get the
    cancellation-proof deadline behaviour.
    """
    out: dict[str, tuple[bool, str]] = {}
    import threading, time as _time
    t = threading.Thread(
        target=_smoke_probe_thread_body,
        kwargs={"model": model, "timeout_s": timeout_s, "out": out, "tracker": tracker},
        name=f"smoke-{model[:24]}",
        daemon=True,
    )
    t.start()
    deadline = _time.monotonic() + max(timeout_s + 5, 30)
    t.join(timeout=max(0.0, deadline - _time.monotonic()))
    if t.is_alive():
        return False, f"abandoned: thread still alive past {timeout_s + 5}s"
    return out.get(model, (False, "no result captured"))


# ---------------------------------------------------------------------------
# Problem discovery — reuses the kernelbench-cuda layout.
# ---------------------------------------------------------------------------


_SUBSET_TO_DIR = {
    "L1": "level1",
    "L2": "level2",
    "L3": "level3",
    "level1": "level1",
    "level2": "level2",
    "level3": "level3",
    "sol-level1": "sol-level1",
    "sol-level2": "sol-level2",
}

# Kernel + driver candidates, tried in order. Matches the pre-refactor
# permissive scan (this script's inputs are batch kgen outputs whose
# filenames drift — ``final_cuda.cu`` = canonical kgen, ``init.cu`` =
# pre-optimisation seed, ``kernel.cu`` = legacy alias; ditto for the
# driver). This is script-local behaviour — ``KernelBenchCUDASource``
# stays strict at ``init.cu``+``driver.cpp``.
_KERNEL_CANDIDATES = ("final_cuda.cu", "init.cu", "kernel.cu")
_DRIVER_CANDIDATES = ("driver.cpp", "test_driver.cpp")


def collect_problems(args) -> list["Problem"]:
    """Walk ``data/kernelbench-cuda/<sub>/`` and yield Problem objects.

    Migrated in Item-2-followup cleanup Phase 2a. Yields
    ``data.sources.Problem`` instances with role-keyed artifacts —
    ``problem.artifact("kernel")`` (the .cu to optimise, may be
    ``final_cuda.cu``/``init.cu``/``kernel.cu``) and
    ``problem.artifact("driver")`` (the test driver, ``driver.cpp`` or
    ``test_driver.cpp``). The permissive candidate scan stays here
    because RL-optimized batch runs produce ``final_cuda.cu``, which
    ``KernelBenchCUDASource`` deliberately doesn't accept (strict
    ``init.cu``-only for the curated tree).
    """
    from data.sources import Problem, parse_problem_numbers

    sub = _SUBSET_TO_DIR.get(args.subset, args.subset)
    base = ROOT_DIR / "data" / "kernelbench-cuda" / sub
    if not base.is_dir():
        raise SystemExit(f"Subset directory not found: {base}")

    _nums = parse_problem_numbers(args.problem_numbers)
    wanted: set[int] | None = set(_nums) if _nums is not None else None

    is_sol = sub.startswith("sol-")
    source_name = "sol-execbench-cuda" if is_sol else "kernelbench-cuda"

    out: list[Problem] = []
    for prob_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        try:
            num = int(prob_dir.name.split("_", 1)[0])
        except ValueError:
            continue
        if wanted is not None and num not in wanted:
            continue
        if args.start and num < args.start:
            continue
        if args.end and num > args.end:
            continue

        kernel = _first_existing(prob_dir, _KERNEL_CANDIDATES)
        driver = _first_existing(prob_dir, _DRIVER_CANDIDATES)
        if kernel is None or driver is None:
            logger.debug(f"Skipping {prob_dir.name}: missing init.cu or driver.cpp")
            continue

        out.append(Problem(
            id=f"{source_name}:{sub}/{prob_dir.name}",
            source=source_name,
            tier=sub,
            problem_num=num,
            problem_name=prob_dir.name,
            curated_artifacts={"kernel": kernel, "driver": driver},
            backends_supported=frozenset({"cuda"}),
        ))
    out.sort(key=lambda p: p.id)
    return out


def _first_existing(directory, candidates):
    for name in candidates:
        p = directory / name
        if p.is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# Agent driver.
# ---------------------------------------------------------------------------

# Per-problem artifacts (init.cu, success_*/failure_*/no_baseline_* markers,
# trajectory dirs) live under <out_root>/<problem_id>/<AGENT_NAME>/. This
# constant is the *single source of truth* — the FeedbackAgent constructor
# (feedback.py:91) appends ``agent_name`` to ``base_folder``, and the resume
# filter below has to mirror that path layout to find the markers.
AGENT_NAME = "opt_ncu_rl_optimized"


async def _run_one(
    *,
    problem: "Problem",
    out_root: Path,
    model: str,
    gpu: GPUType,
    database_path: Path,
    num_iterations: int,
    max_steps: int,
    seed_from_init: int,
    bandit_c: float,
    prune_patience: int,
    max_fix: int,
    timeout_s: int,
    cost_tracker: Optional[CostTracker] = None,
    progress_writer: Optional[ProgressWriter] = None,
) -> bool:
    # Item-2 cleanup Phase 2a: ``problem`` is now a ``data.sources.Problem``.
    # ``filesystem_id`` strips the ``<source>:`` prefix so the output-folder
    # layout matches the pre-migration ``problem.id`` shape.
    problem_id = problem.id
    folder = out_root / problem.filesystem_id
    folder.mkdir(parents=True, exist_ok=True)

    job_logger = logger.bind(problem_id=problem_id)
    log_file = folder / "run.log"
    handler_id = job_logger.add(
        log_file,
        level=config.LOG_LEVEL,
        backtrace=True,
        diagnose=True,
        format=config.CUSTOM_LOGGER_FORMAT,
        filter=lambda r, pid=problem_id: r["extra"].get("problem_id") == pid,
    )

    fb_config = FeedbackConfig(
        agent_name=AGENT_NAME,
        base_folder=folder,
        logger=job_logger,
        init_user_prompt="",
        model=model,
        gpu=gpu,
        test_code_fp=problem.artifact("driver"),
        max_attempts=1,
    )

    try:
        agent = OptimizedRLNCUAgent(
            fb_config=fb_config,
            code_to_optimize_fp=problem.artifact("kernel"),
            database_path=database_path,
            max_rollout_steps=max_steps,
            num_rl_iterations=num_iterations,
            seed_from_init_count=seed_from_init,
            bandit_exploration=bandit_c,
            prune_patience=prune_patience,
            max_fix_attempts=max_fix,
            cost_tracker=cost_tracker,
            problem_id=problem_id,
            progress_writer=progress_writer,
        )

        # Mark this problem as started in the live progress feed (file-only).
        if progress_writer is not None:
            progress_writer.problem_started(problem_id)

        ok = False
        no_baseline = False
        result_path: Optional[Path] = None
        timed_out = False
        try:
            result_path = await asyncio.wait_for(agent.run(), timeout=timeout_s * 60)
            no_baseline = "no_baseline" in result_path.name
            ok = "success" in result_path.name or no_baseline
            if no_baseline:
                job_logger.warning(
                    f"OPT-RL kept best rollout for {problem.id} (no baseline to compare): {result_path}"
                )
            elif ok:
                job_logger.info(f"OPT-RL succeeded for {problem.id}: {result_path}")
            else:
                job_logger.warning(f"OPT-RL did not improve {problem.id}: {result_path}")
        except asyncio.TimeoutError:
            job_logger.error(f"OPT-RL timed out for {problem.id}")
            timed_out = True

        if progress_writer is not None:
            final_cycles = None
            init_cycles = getattr(agent, "initial_cycles", None) if "agent" in locals() else None
            best_cycles = (
                int(getattr(agent, "best_cycles", 0))
                if "agent" in locals() and getattr(agent, "best_cycles", float("inf")) != float("inf")
                else None
            )
            if best_cycles is not None:
                final_cycles = best_cycles
            if timed_out:
                status_override = "timeout"
            elif no_baseline:
                status_override = "no_baseline"
            else:
                status_override = None
            progress_writer.problem_finished(
                problem.id,
                success=ok,
                final_cycles=final_cycles,
                init_cycles=init_cycles,
                status_override=status_override,
            )
        return ok
    finally:
        job_logger.remove(handler_id)


async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the optimised RL CUDA optimisation flow over a problem set."
    )
    parser.add_argument("--model", type=str, default=config.MODEL,
                        help="Default model. MODEL_PLAN / MODEL_CODEGEN_* env vars override per-call routing.")
    parser.add_argument("--gpu", type=str, default="l40s",
                        choices=[g.value for g in GPUType])
    parser.add_argument("--subset", type=str, default="sol-level2")
    parser.add_argument("--problem-numbers", type=str, default=None,
                        help="Comma/range list, e.g. '1,3,5-9'.")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--experiment-name", type=str, default="opt_ncu_rl_optimized")
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help=(
            "Subdirectory under <experiment-name>/ for this run's outputs. "
            "Replaces the legacy '<model-name>' slot, which is meaningless when "
            "different roles use different models. If omitted, a deterministic "
            "fingerprint is computed from the four MODEL_* env vars + --model "
            "(e.g. 'plan-haiku45__simple-sonnet46__hard-opus47'). "
            "Set to e.g. 'baseline' or 'expt_2026-04-26_v3' for human-readable runs."
        ),
    )
    parser.add_argument("--num-iterations", type=int, default=50,
                        help="Parallel rollouts per problem.")
    parser.add_argument("--max-steps", type=int, default=5,
                        help="Steps per rollout.")
    parser.add_argument("--seed-from-init", type=int, default=10,
                        help="First N rollouts seed from init.cu; later ones reseed from top-K.")
    parser.add_argument("--bandit-c", type=float, default=1.4,
                        help="UCB1 exploration constant.")
    parser.add_argument("--prune-patience", type=int, default=2)
    parser.add_argument("--max-fix-attempts", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="How many problems to optimise in parallel.")
    parser.add_argument("--timeout", type=int, default=240,
                        help="Per-problem timeout in minutes.")
    parser.add_argument("--gpu-server-url", type=str, default=None,
                        help="If set, KERNELBLASTER_GPU_SERVER_URL_* will be exported.")
    parser.add_argument("--compile-server-url", type=str, default=None,
                        help="If set, exported as COMPILE_SERVER_URL.")
    # Note: the smoke test now lives in scripts/smoke_test_models.py and is
    # invoked by scripts/run_opt_ncu_rl.sh BEFORE this script. Keeping
    # --skip-smoke-test as an accepted-but-ignored flag for backwards
    # compatibility with anyone still passing it from old wrappers.
    parser.add_argument(
        "--skip-smoke-test", action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--smoke-test-timeout", type=int, default=15,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--smoke-test-batch-timeout", type=int, default=60,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--smoke-test-soft-fail", action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--cost-live-interval", type=float, default=30.0,
                        help="Seconds between live cost summaries (0 disables periodic logging).")
    parser.add_argument("--no-cost-tracking", action="store_true",
                        help="Disable LLM cost / token tracking entirely.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the existing run directory. Skips problems that already "
            "have ``success_rl_optimization.cu`` in their output folder and "
            "preserves the previous progress.jsonl event log. By default, "
            "previously-failed problems are RETRIED (failures are often "
            "transient, e.g. rate limit / timeout). Pair with --resume-skip-failed "
            "to also skip those."
        ),
    )
    parser.add_argument(
        "--resume-skip-failed",
        action="store_true",
        help=(
            "When --resume is set, also skip problems with a "
            "``failure_rl_optimization.cu`` artifact (treat past failure as final)."
        ),
    )
    args = parser.parse_args()

    # Server URL plumbing — same env-var convention as the legacy flow.
    if args.compile_server_url:
        os.environ["COMPILE_SERVER_URL"] = args.compile_server_url
    if args.gpu_server_url:
        # The CUDA flow keys the GPU URL by GPU type, e.g. GPU_SERVER_URL_L40S.
        os.environ[f"GPU_SERVER_URL_{args.gpu.upper()}"] = args.gpu_server_url

    if args.model != config.MODEL:
        config.MODEL = args.model

    gpu_type = GPUType(args.gpu)

    def _short_model_name(m: str) -> str:
        """Trim provider prefixes + version dates to a compact slug."""
        s = (
            m.replace("llmgateway/", "")
            .replace("eos/", "")
            .replace("chipnemo/", "")
            .replace("azure/", "")
            .replace("anthropic/", "")
            .replace("openai/", "")
            .replace("/", "-")
            .lower()
        )
        # Strip trailing date stamps like "-2025-08-07".
        import re as _re
        s = _re.sub(r"-\d{4}-\d{2}-\d{2}$", "", s)
        # Compact: claude-sonnet-4-6 → sonnet46, claude-haiku-4-5 → haiku45,
        # gpt-5-mini → gpt5mini, etc.
        s = s.replace("claude-", "").replace("-", "")
        return s

    if args.run_tag:
        run_tag = args.run_tag
    else:
        # Auto-fingerprint from the four roles. Skip duplicates so
        # uniform-model runs collapse to a single slug.
        plan = _short_model_name(os.getenv("MODEL_PLAN") or config.MODEL)
        simple = _short_model_name(os.getenv("MODEL_CODEGEN_SIMPLE") or config.MODEL)
        hard = _short_model_name(os.getenv("MODEL_CODEGEN_HARD") or config.MODEL)
        fix = _short_model_name(
            os.getenv("MODEL_FIX") or os.getenv("MODEL_CODEGEN_SIMPLE") or config.MODEL
        )
        if plan == simple == hard == fix:
            run_tag = plan  # uniform run; one slug is enough
        else:
            run_tag = f"plan-{plan}__simple-{simple}__hard-{hard}__fix-{fix}"

    out_root = ROOT_DIR / "out" / "kernelbench-cuda" / args.experiment_name / run_tag

    out_root.mkdir(parents=True, exist_ok=True)
    log_file = out_root / "run.log"
    logger.configure(
        handlers=[
            dict(sink=sys.stderr, format=config.CUSTOM_LOGGER_FORMAT,
                 level=config.LOG_LEVEL, colorize=True, backtrace=True, diagnose=True),
            dict(sink=log_file, format=config.CUSTOM_LOGGER_FORMAT,
                 level=config.LOG_LEVEL, colorize=False, backtrace=True, diagnose=True),
        ],
        extra=dict(agent_name="opt_ncu_rl_optimized", problem_id=None),
    )

    # ── Output banner ────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info(f"Run tag:        {run_tag}")
    logger.info(f"Output root:    {out_root}")
    logger.info(f"  per-problem:  {out_root}/<level>/<problem_dir>/")
    logger.info(f"  run log:      {out_root}/run.log")
    logger.info("=" * 72)

    # ── Resolve effective per-role models ────────────────────────────
    role_models = {
        "default":         config.MODEL,
        "MODEL_PLAN":      os.getenv("MODEL_PLAN") or config.MODEL,
        "MODEL_CODEGEN_SIMPLE": os.getenv("MODEL_CODEGEN_SIMPLE") or config.MODEL,
        "MODEL_CODEGEN_HARD":   os.getenv("MODEL_CODEGEN_HARD") or config.MODEL,
        "MODEL_FIX":       os.getenv("MODEL_FIX")
                           or os.getenv("MODEL_CODEGEN_SIMPLE")
                           or config.MODEL,
    }
    logger.info("Tiered models:")
    for role, name in role_models.items():
        logger.info(f"  {role:<22} = {name}")

    # ── Cost tracker (file-only live output, no logger spam) ─────────
    cost_tracker: Optional[CostTracker] = None
    cost_snapshot_path = out_root / "cost_live.json"
    cost_history_path = out_root / "cost_live.jsonl"
    if not args.no_cost_tracking:
        cost_tracker = CostTracker()
        # On --resume, fold the prior snapshot's totals/aggregates back into
        # the live tracker BEFORE start_file_dump fires. Otherwise the next
        # tick would overwrite cost_live.json with zero-totals from the new
        # CostTracker instance. cost_live.jsonl is append-only and already
        # preserved across resumes.
        if args.resume:
            cost_tracker.load_snapshot(cost_snapshot_path, logger=logger)
        if args.cost_live_interval > 0:
            cost_tracker.start_file_dump(
                snapshot_path=cost_snapshot_path,
                history_path=cost_history_path,
                interval_s=args.cost_live_interval,
            )
            logger.info(
                f"Live cost  → {cost_snapshot_path}  (snapshot, refreshed every "
                f"{args.cost_live_interval:.0f}s)"
            )
            logger.info(
                f"Cost time-series → {cost_history_path}  (JSONL, append-only)"
            )
    else:
        logger.warning("Cost tracking disabled via --no-cost-tracking.")

    # ── Progress writer (file-only live output, no logger spam) ──────
    progress_writer: Optional[ProgressWriter] = ProgressWriter(out_root, resume=args.resume)
    progress_snapshot_path = progress_writer.snapshot_path
    progress_events_path = progress_writer.events_path
    logger.info(
        f"Live progress → {progress_snapshot_path}  (snapshot, current state of all problems)"
    )
    logger.info(
        f"Progress events → {progress_events_path}  (JSONL, append-only)"
    )

    # NOTE: smoke testing has been moved out of this process — see
    # scripts/smoke_test_models.py and scripts/run_opt_ncu_rl.sh. The wrapper
    # invokes the standalone smoke under ``timeout --signal=KILL`` BEFORE
    # this runner starts, so blocking-IO hangs in the SDK can never reach
    # this point. The legacy --smoke-test-* flags are silently accepted
    # (argparse.SUPPRESS) for back-compat but no longer wired.

    problems = collect_problems(args)
    if not problems:
        logger.error("No problems found. Check --subset / --problem-numbers / --start--end.")
        return

    # ── Resume filter ────────────────────────────────────────────────
    if args.resume:
        from data.sources import Problem
        kept: list[Problem] = []
        skipped_success = 0
        skipped_failed = 0
        for p in problems:
            # Markers live under <out_root>/<problem-id>/<AGENT_NAME>/, not
            # directly under <problem-id>/ — FeedbackAgent appends agent_name
            # to base_folder. Without this subdir, every prior run looked
            # like a fresh problem and resume re-ran everything.
            pdir = out_root / p.filesystem_id / AGENT_NAME
            success_marker = pdir / "success_rl_optimization.cu"
            no_baseline_marker = pdir / "no_baseline_rl_optimization.cu"
            failure_marker = pdir / "failure_rl_optimization.cu"
            if success_marker.exists():
                logger.info(
                    f"RESUME  skipping {p.id}  (already succeeded: {success_marker.name})"
                )
                skipped_success += 1
                continue
            if no_baseline_marker.exists():
                # Treat no-baseline as 'kept' — the artifact is usable; re-running
                # without first repairing init.cu won't change the outcome.
                logger.info(
                    f"RESUME  skipping {p.id}  (no_baseline kernel already kept: "
                    f"{no_baseline_marker.name})"
                )
                skipped_success += 1
                continue
            if args.resume_skip_failed and failure_marker.exists():
                logger.info(
                    f"RESUME  skipping {p.id}  (previously failed; --resume-skip-failed set)"
                )
                skipped_failed += 1
                continue
            kept.append(p)
        logger.info(
            f"RESUME  kept {len(kept)} of {len(problems)} problems  "
            f"(skipped: {skipped_success} success, {skipped_failed} failed)"
        )
        problems = kept
        if not problems:
            logger.info("All matching problems already complete; nothing to do.")
            # Still write the final cost summary in case smoke-test calls happened.
            if cost_tracker is not None:
                cost_tracker.stop_live_logging()
                cost_tracker.write_summary_json(out_root / "cost_summary.json")
            return
    logger.info(f"Optimising {len(problems)} problems")

    # Shared database — accumulates rewards across problems for warm UCB stats.
    database_path = ROOT_DIR / "data" / "kernelblaster" / "optimization_database.json"
    if not database_path.exists():
        logger.warning(f"Database file not found at {database_path}; agent will start cold.")

    semaphore = asyncio.Semaphore(args.concurrency)
    succeeded = 0
    failed = 0

    async def _bound(problem):
        nonlocal succeeded, failed
        async with semaphore:
            ok = await _run_one(
                problem=problem,
                out_root=out_root,
                model=config.MODEL,
                gpu=gpu_type,
                database_path=database_path,
                num_iterations=args.num_iterations,
                max_steps=args.max_steps,
                seed_from_init=args.seed_from_init,
                bandit_c=args.bandit_c,
                prune_patience=args.prune_patience,
                max_fix=args.max_fix_attempts,
                timeout_s=args.timeout,
                cost_tracker=cost_tracker,
                progress_writer=progress_writer,
            )
            if ok:
                succeeded += 1
            else:
                failed += 1
            # One-line console nudge per problem completion (parsable, not spammy).
            if cost_tracker is not None:
                t = cost_tracker.totals()
                logger.info(
                    f"DONE  {problem.id}  "
                    f"status={'success' if ok else 'failed'}  "
                    f"calls={t['calls']}  cost=${t['cost_usd']:.3f}  "
                    f"({t['elapsed_s']:.0f}s elapsed)"
                )
            else:
                logger.info(f"DONE  {problem.id}  status={'success' if ok else 'failed'}")

    tasks = [asyncio.create_task(_bound(p), name=f"bound:{p.id}") for p in problems]
    try:
        await asyncio.gather(*tasks)
    finally:
        if cost_tracker is not None:
            cost_tracker.stop_live_logging()
            # Final compact log line so the summary still ends up in run.log.
            t = cost_tracker.totals()
            logger.info(
                f"FINAL  calls={t['calls']}  in={t['input_tokens']:,}  "
                f"out={t['output_tokens']:,}  cost=${t['cost_usd']:.3f}  "
                f"({t['elapsed_s']:.0f}s elapsed)"
            )
            # Full breakdown stays in JSON only (no logger spam).
            cost_summary_path = out_root / "cost_summary.json"
            cost_tracker.write_summary_json(cost_summary_path)
            logger.info(f"Cost summary JSON written to: {cost_summary_path}")

    logger.info(f"Done: {succeeded} succeeded, {failed} failed of {len(problems)}")


def _signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, exiting…")
    sys.exit(0)


def main() -> None:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
