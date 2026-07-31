#!/usr/bin/env python3
"""Standalone runner for the RISC-V + Zephyr + spike optimized RL flow (P5.13).

RISC-V analogue of ``scripts/run_opt_ncu_rl_optimized.py``. Minimum
viable end-to-end: pre-stages the modelblaster problem (extract →
generate_skeleton → generate_kernels with BACKEND=reference), starts
the framework's compile server + spike exec server, instantiates
:class:`OptimizedRLRiscvAgent`, and runs one KernelBench problem.

Typical invocation::

    source /path/to/zephyr-chipyard-sw/scripts/set_envvars_sdk.sh
    export PATH=/path/to/spike/bin:/path/to/zephyr/env/bin:$PATH
    export AWS_BEARER_TOKEN_BEDROCK=...          # for Bedrock LLM
    export MODEL_PLAN=us.meta.llama4-scout-17b-instruct-v1:0
    export MODEL_CODEGEN_SIMPLE=us.meta.llama4-scout-17b-instruct-v1:0
    export MODEL_CODEGEN_HARD=us.meta.llama4-maverick-17b-instruct-v1:0
    export MODEL_FIX=us.meta.llama4-scout-17b-instruct-v1:0

    python scripts/run_opt_riscv_rl_optimized.py \
        --modelblaster-root /path/to/zephyr-chipyard-sw \
        --bench-file /path/to/kb/level1/019_ReLU.py \
        --target scalar \
        --num-iterations 2 --max-steps 2 --seed-from-init 2

The runner is deliberately compact — no cost dashboards, no progress-
writer plumbing, no cross-problem aggregation. Ship it, run one
problem, iterate.
"""
from __future__ import annotations

import argparse
import asyncio
import atexit
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ``aiohttp`` imports ``ssl`` at import time; guard the CA bundle up front.
if not os.environ.get("SSL_CERT_FILE") and not os.environ.get("REQUESTS_CA_BUNDLE"):
    try:
        import certifi  # type: ignore
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except Exception:
        _default = Path("/etc/ssl/certs/ca-certificates.crt")
        if _default.exists():
            os.environ["SSL_CERT_FILE"] = str(_default)
if os.environ.get("SSL_CERT_FILE") and not os.environ.get("REQUESTS_CA_BUNDLE"):
    os.environ["REQUESTS_CA_BUNDLE"] = os.environ["SSL_CERT_FILE"]

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from loguru import logger  # noqa: E402

from src.kernelblaster.config import config, GPUType  # noqa: E402
from src.kernelblaster.agents.feedback import FeedbackConfig  # noqa: E402
from src.kernelblaster.agents.opt_riscv_rl_optimized import OptimizedRLRiscvAgent  # noqa: E402


BRIDGE_SCRIPT = REPO_ROOT / "scripts" / "riscv" / "bridge_build.sh"


# ---------------------------------------------------------------------------
# Modelblaster pre-stage
# ---------------------------------------------------------------------------

def stage_problem(
    modelblaster_root: Path,
    bench_file: Path,
    target: str,
    quant: str,
    force_extract: bool,
) -> tuple[Path, Path]:
    """Run modelblaster ``BACKEND=reference`` once to produce the
    baseline harness + kernels.c. Returns
    ``(stage_dir, io_npz_path)``. Idempotent — skips if the io.npz +
    generated/<target>/kernels.c both already exist unless
    ``force_extract`` is True.
    """
    stem = bench_file.stem
    # Same sanitisation extract_graph applies.
    import re as _re
    sanitised = _re.sub(r"[^A-Za-z0-9_]", "_", stem)
    sanitised = _re.sub(r"__+", "_", sanitised).strip("_")
    kb_name = f"kb_{sanitised}"

    stage_dir = modelblaster_root / "modelblaster" / "examples" / "kernelbench" / kb_name / quant
    io_npz = stage_dir / "generated" / "io.npz"
    kernels_c = stage_dir / "generated" / target / "kernels.c"

    if not force_extract and io_npz.exists() and kernels_c.exists():
        logger.info(f"Modelblaster pre-stage present at {stage_dir} — skipping (use --force-extract to rerun)")
        return stage_dir, io_npz

    logger.info(f"Running modelblaster pre-stage (BACKEND=reference) for {kb_name}...")
    env = os.environ.copy()
    env["BENCH_FILE"] = str(bench_file.resolve())
    env["TARGET"] = target
    env["QUANT"] = quant
    env["RUNNER"] = "spike"
    env["BACKEND"] = "reference"
    env["FORCE_EXTRACT"] = "1" if force_extract else "0"
    # Point the heavy per-bench data dir at /scratch so the repo partition doesn't fill.
    env.setdefault("MB_KB_DATA_ROOT", f"/scratch/{os.environ.get('USER', 'kris')}/kb_data")
    Path(env["MB_KB_DATA_ROOT"]).mkdir(parents=True, exist_ok=True)
    env.setdefault("TMPDIR", f"/scratch/{os.environ.get('USER', 'kris')}/kb_tmp")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    # PYTHONPATH so modelblaster.pipeline.* resolves.
    env["PYTHONPATH"] = f"{modelblaster_root}:{env.get('PYTHONPATH', '')}"

    result = subprocess.run(
        ["bash", "modelblaster/examples/kernelbench/run_one.sh"],
        cwd=str(modelblaster_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-30:] + result.stderr.splitlines()[-30:])
        raise RuntimeError(f"modelblaster pre-stage failed:\n{tail}")
    logger.info(f"Modelblaster pre-stage done. Baseline elf: {stage_dir}/build/{target}/zephyr/zephyr.elf")

    if not (io_npz.exists() and kernels_c.exists()):
        raise RuntimeError(
            f"Pre-stage completed but expected artifacts missing:\n"
            f"  {io_npz}  exists={io_npz.exists()}\n"
            f"  {kernels_c}  exists={kernels_c.exists()}"
        )
    return stage_dir, io_npz


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

_SPAWNED_PROCS: list[subprocess.Popen] = []


def _cleanup_procs() -> None:
    for p in _SPAWNED_PROCS:
        try:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
        except Exception:
            pass


atexit.register(_cleanup_procs)


def _wait_healthy(url: str, timeout_s: float = 30.0) -> bool:
    import urllib.request
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
                if resp.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def start_compile_server(port: int, artifacts_dir: Path, board_host: str = "") -> str:
    """Start the framework's unified compile server. ZephyrCompileStrategy
    is auto-registered on import so ``?backend=riscv`` dispatches correctly.
    """
    cmd = [
        sys.executable, "-m", "src.kernelblaster.servers.compile_server",
        "--host", "127.0.0.1", "--port", str(port),
        "--num-workers", "2",
        "--artifacts-dir", str(artifacts_dir),
    ]
    if board_host:
        cmd += ["--board-host", board_host]
    logger.info(f"Starting compile server: {shlex.join(cmd)}")
    p = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
    _SPAWNED_PROCS.append(p)
    url = f"http://127.0.0.1:{port}"
    if not _wait_healthy(url):
        raise RuntimeError(f"compile server at {url} did not become healthy in 30s")
    logger.info(f"Compile server healthy at {url}")
    return url


def start_spike_exec_server(
    port: int,
    modelblaster_root: Path,
    spike_binary: Optional[str] = None,
) -> str:
    """Start the framework's exec server with the spike strategy."""
    cmd = [
        sys.executable, "-m", "src.kernelblaster.servers.exec_server",
        "--host", "127.0.0.1", "--port", str(port),
        "--strategy", "spike",
        "--modelblaster-root", str(modelblaster_root),
        "--num-workers", "1",
    ]
    if spike_binary:
        cmd += ["--spike-binary", spike_binary]
    logger.info(f"Starting spike exec server: {shlex.join(cmd)}")
    p = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
    _SPAWNED_PROCS.append(p)
    url = f"http://127.0.0.1:{port}"
    if not _wait_healthy(url):
        raise RuntimeError(f"spike exec server at {url} did not become healthy in 30s")
    logger.info(f"Spike exec server healthy at {url}")
    return url


def start_firesim_exec_server(
    port: int,
    modelblaster_root: Path,
    args: argparse.Namespace,
) -> str:
    """Start the framework's exec server with the firesim strategy.

    Mirrors :func:`start_spike_exec_server` but wires through the
    firesim-runner-specific flags: chipyard root, env.sh, queue on/off,
    priority, timeout. Values default to env-var fallback inside the
    strategy, so a minimal CLI still works when FIRESIM_ROOT etc. are
    already exported.
    """
    cmd = [
        sys.executable, "-m", "src.kernelblaster.servers.exec_server",
        "--host", "127.0.0.1", "--port", str(port),
        "--strategy", "firesim",
        "--modelblaster-root", str(modelblaster_root),
        "--num-workers", "1",
    ]
    if args.firesim_root:
        cmd += ["--firesim-root", args.firesim_root]
    if args.firesim_env:
        cmd += ["--firesim-env", args.firesim_env]
    if args.no_firesim_queue:
        cmd += ["--no-firesim-queue"]
    if args.firesim_queue_root:
        cmd += ["--firesim-queue-root", args.firesim_queue_root]
    if args.firesim_queue_bin:
        cmd += ["--firesim-queue-bin", args.firesim_queue_bin]
    if args.firesim_queue_priority is not None:
        cmd += ["--firesim-queue-priority", str(args.firesim_queue_priority)]
    if args.firesim_queue_timeout is not None:
        cmd += ["--firesim-queue-timeout", str(args.firesim_queue_timeout)]
    if args.firesim_default_timeout is not None:
        cmd += ["--firesim-default-timeout", str(args.firesim_default_timeout)]
    if args.firesim_python_bin:
        cmd += ["--firesim-python-bin", args.firesim_python_bin]

    logger.info(f"Starting firesim exec server: {shlex.join(cmd)}")
    p = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
    _SPAWNED_PROCS.append(p)
    url = f"http://127.0.0.1:{port}"
    if not _wait_healthy(url):
        raise RuntimeError(f"firesim exec server at {url} did not become healthy in 30s")
    logger.info(f"FireSim exec server healthy at {url}")
    return url


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main(args: argparse.Namespace) -> int:
    modelblaster_root = Path(args.modelblaster_root).resolve()
    bench_file = Path(args.bench_file).resolve()

    if not modelblaster_root.exists():
        raise SystemExit(f"--modelblaster-root {modelblaster_root} does not exist")
    if not bench_file.exists():
        raise SystemExit(f"--bench-file {bench_file} does not exist")

    if not BRIDGE_SCRIPT.exists():
        raise SystemExit(
            f"bridge script missing at {BRIDGE_SCRIPT}. Ship P5.12 first."
        )

    # 1. Modelblaster pre-stage.
    stage_dir, io_npz = stage_problem(
        modelblaster_root=modelblaster_root,
        bench_file=bench_file,
        target=args.target,
        quant=args.quant,
        force_extract=args.force_extract,
    )
    kernels_c = stage_dir / "generated" / args.target / "kernels.c"
    orig_kernels_c = kernels_c.with_suffix(".c.orig")
    # Reset kernels.c to the reference version if a prior RL run left an override.
    if orig_kernels_c.exists():
        shutil.copy(orig_kernels_c, kernels_c)

    # 2. Env for the compile-server subprocess (it will spawn the bridge).
    build_cmd_template = (
        f"bash {BRIDGE_SCRIPT} "
        f"{{job_name}} {{source_file}} {{output_path}} {{board}}"
    )
    os.environ["KERNELBLASTER_ZEPHYR_BUILD_CMD"] = build_cmd_template
    os.environ["KERNELBLASTER_MODELBLASTER_STAGE_DIR"] = str(stage_dir)
    os.environ["KERNELBLASTER_MODELBLASTER_TARGET"] = args.target
    os.environ["KERNELBLASTER_MODELBLASTER_ROOT"] = str(modelblaster_root)
    # Ensure PYTHONPATH lets modelblaster.pipeline.* resolve inside the bridge's cmake calls.
    os.environ["PYTHONPATH"] = f"{modelblaster_root}:{os.environ.get('PYTHONPATH', '')}"

    # 3. Start compile + exec servers.
    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    compile_url = start_compile_server(args.compile_port, artifacts_dir)
    if args.strategy == "firesim":
        exec_url = start_firesim_exec_server(
            args.exec_port, modelblaster_root, args,
        )
    else:
        spike_binary = args.spike_binary or shutil.which("spike")
        if not spike_binary:
            raise SystemExit("spike binary not on PATH; pass --spike-binary")
        exec_url = start_spike_exec_server(
            args.exec_port, modelblaster_root, spike_binary=spike_binary,
        )

    # 4. Wire the framework's URL config to these servers.
    os.environ["COMPILE_SERVER_URL"] = compile_url
    # When --strategy=firesim, override --gpu to match unless the user
    # explicitly picked a RISC-V FPGA target — the agent's compile
    # server dispatches on GPUType.zephyr_board, and spike vs. firesim
    # need different boards (spike_riscv64 vs. chipyard_riscv64/...).
    if args.strategy == "firesim" and args.gpu == "riscv_spike":
        args.gpu = "riscv_fpga_zephyr"
        logger.info(
            "Overriding --gpu to riscv_fpga_zephyr for --strategy=firesim"
        )
    os.environ[f"GPU_SERVER_URL_{args.gpu.upper()}"] = exec_url
    os.environ["KERNELBLASTER_RISCV_COMPILE_SERVER_URL"] = compile_url
    config.COMPILE_SERVER_URL = compile_url
    gpu_type = GPUType(args.gpu)
    config.GPU_SERVER_URLS[gpu_type] = exec_url

    # 5. Configure the RL run.
    experiment_name = args.experiment_name or f"riscv_smoke_{int(time.time())}"
    if args.out_root:
        base_out = Path(args.out_root).resolve()
    else:
        # Prefer REPO_ROOT/out when writable; fall back to
        # ``/scratch/<user>/kb_out`` which is normally user-writable.
        candidate = REPO_ROOT / "out" / "riscv_spike"
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            base_out = candidate
        except PermissionError:
            base_out = Path(f"/scratch/{os.environ.get('USER', 'kris')}/kb_out/riscv_spike")
    base_out.mkdir(parents=True, exist_ok=True)
    problem_dir = base_out / experiment_name / f"{stage_dir.parent.name}"
    problem_dir.mkdir(parents=True, exist_ok=True)

    # Sink loguru into a run log alongside the console.
    logger.add(str(problem_dir / "run.log"), level="INFO")

    # 6. Configure the DB path + seed with RISC-V-flavored states.
    #
    # The DB's markdown parser expects a very specific shape (``#### State: ...``
    # + ``**Primary Bottleneck**:`` + ``- **<technique>**: N% ...``); rather
    # than reformat the seed markdown, we build the persistent JSON directly
    # from :meth:`RiscvZephyrBackend.get_default_optimizations` + the
    # backend's :attr:`technique_map`. This gives the DB a RISC-V-vocab
    # state catalog on first init instead of falling through to the CUDA
    # default JSON.
    database_path = problem_dir / "riscv_optimization_database.md"
    persist_json = database_path.with_suffix(".json")
    if not persist_json.exists():
        from src.kernelblaster.backends import RiscvZephyrBackend as _RB
        _seed_backend = _RB(gpu=gpu_type)
        seed_strategies: dict = {}
        for bottleneck, entries in _seed_backend.get_default_optimizations().items():
            state_name = f"{bottleneck}_riscv"
            seed_strategies[state_name] = {
                "primary_bottleneck": bottleneck,
                "secondary_characteristics": [],
                "optimizations": [
                    {
                        "technique": tech_id,
                        "predicted_improvement": pct,
                        "description": _seed_backend.technique_map.get(tech_id, ""),
                        "category": (
                            "memory" if bottleneck == "memory_bound"
                            else "compute" if bottleneck == "compute_bound"
                            else "latency" if bottleneck == "latency_bound"
                            else "hybrid"
                        ),
                        "actual_improvement": None,
                        "confidence_score": 0.5,
                        "usage_count": 0,
                        "predicted_speedup": 1.0 + pct / 100.0,
                        "actual_speedup": None,
                        "initial_elapsed_cycles": None,
                        "last_updated": None,
                    }
                    for tech_id, pct in entries
                ],
            }
        # ``known_states`` is a dict of {name: StateProfile fields} —
        # let the DB populate its structural defaults there. What we
        # care about is ``optimization_strategies`` (the technique
        # catalog).
        seed_json = {
            "known_states": {},
            "optimization_strategies": seed_strategies,
            "composite_optimizations": {},
            "discovered_states": {},
        }
        import json as _json
        persist_json.parent.mkdir(parents=True, exist_ok=True)
        persist_json.write_text(_json.dumps(seed_json, indent=2))
        logger.info(
            f"Seeded RISC-V DB with {len(seed_strategies)} states / "
            f"{sum(len(s['optimizations']) for s in seed_strategies.values())} "
            f"technique entries → {persist_json}"
        )

    # 7. FeedbackConfig — RISC-V has no distinct driver.cpp; we point
    # ``test_code_fp`` at the reference kernels.c so the LLM sees the
    # kernel signature verbatim. The base class only uses this for its
    # ``self.test_code`` string field.
    fb_config = FeedbackConfig(
        agent_name="opt_riscv_rl_optimized",
        base_folder=problem_dir,
        logger=logger,
        init_user_prompt="",
        model=args.model,
        gpu=gpu_type,
        test_code_fp=kernels_c,
        max_attempts=1,
    )

    # 8. Instantiate agent + run.
    agent = OptimizedRLRiscvAgent(
        fb_config=fb_config,
        code_to_optimize_fp=kernels_c,
        database_path=database_path,
        gpu=gpu_type,
        max_rollout_steps=args.max_steps,
        num_rl_iterations=args.num_iterations,
        seed_from_init_count=args.seed_from_init,
        bandit_exploration=args.bandit_c,
        prune_patience=args.prune_patience,
        max_fix_attempts=args.max_fix_attempts,
        io_npz_path=io_npz,
        spike_args_str=args.spike_extra_args or "",
        use_exec_batching=not args.no_exec_batching,
    )

    logger.info("=" * 72)
    logger.info(f"Model routing:")
    logger.info(f"  MODEL_PLAN            = {agent.model_plan}")
    logger.info(f"  MODEL_CODEGEN_SIMPLE  = {agent.model_codegen_simple}")
    logger.info(f"  MODEL_CODEGEN_HARD    = {agent.model_codegen_hard}")
    logger.info(f"  MODEL_FIX             = {agent.model_fix}")
    logger.info(f"  Backend               = {agent.backend.name} ({agent.gpu.value})")
    logger.info(f"  Baseline elf          = {stage_dir}/build/{args.target}/zephyr/zephyr.elf")
    logger.info(f"  Output                = {problem_dir}")
    logger.info("=" * 72)

    started_at = time.time()
    try:
        await asyncio.wait_for(agent.initialize(), timeout=args.timeout_min * 60)
        result_path = await asyncio.wait_for(agent.run(), timeout=args.timeout_min * 60)
    except asyncio.TimeoutError:
        logger.error(f"RL run timed out after {args.timeout_min} min")
        return 2

    elapsed = time.time() - started_at
    logger.info("=" * 72)
    logger.info(f"RESULT path: {result_path}")
    logger.info(f"Initial cycles:  {agent.initial_cycles}")
    logger.info(f"Best cycles:     {agent.best_cycles}")
    if agent.initial_cycles and agent.best_cycles != float("inf"):
        pct = (agent.initial_cycles - agent.best_cycles) / agent.initial_cycles * 100
        logger.info(f"Improvement:     {pct:+.2f}%")
    logger.info(f"Total trajectories completed: {agent.total_trajectories}")
    logger.info(f"Elapsed:         {elapsed:.1f}s")
    logger.info("=" * 72)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone RISC-V (Zephyr + spike) RL runner for one KernelBench problem.",
    )
    p.add_argument("--modelblaster-root", required=True,
                   help="Path to zephyr-chipyard-sw (the repo containing modelblaster/).")
    p.add_argument("--bench-file", required=True,
                   help="Absolute path to a KernelBench level1 .py file.")
    p.add_argument("--target", default="scalar",
                   help="modelblaster backend target: scalar | rvv | rvv_f16 | gemmini | ...")
    p.add_argument("--quant", default="fp32",
                   choices=["fp32", "fp16", "int8"])
    p.add_argument("--force-extract", action="store_true",
                   help="Re-run modelblaster pre-stage even if artifacts exist.")
    p.add_argument("--gpu", default="riscv_spike",
                   help="GPUType value. Should stay riscv_spike for local spike runs.")
    p.add_argument("--model", default=config.MODEL,
                   help="Default model (MODEL_PLAN etc. env vars override per-call).")

    # RL knobs — mirror the CUDA runner's defaults.
    p.add_argument("--num-iterations", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--seed-from-init", type=int, default=10)
    p.add_argument("--bandit-c", type=float, default=1.4)
    p.add_argument("--prune-patience", type=int, default=2)
    p.add_argument("--max-fix-attempts", type=int, default=2)
    p.add_argument(
        "--no-exec-batching",
        action="store_true",
        help="Disable client-side /gpu/batch coalescing. Default is "
             "enabled — coalesces concurrent-rollout exec calls into "
             "one batched HTTP round-trip. Tune size/wait via env "
             "KERNELBLASTER_EXEC_BATCH_SIZE / _MAX_WAIT_MS.",
    )
    p.add_argument("--timeout-min", type=int, default=60,
                   help="Overall RL-loop timeout (minutes).")

    # Server + exec-strategy knobs.
    p.add_argument("--strategy", choices=["spike", "firesim"], default="spike",
                   help="Exec strategy: 'spike' (functional simulator, "
                        "default) or 'firesim' (FPGA via "
                        "modelblaster.validation.firesim_runner). Selects "
                        "the exec server strategy and the RL agent's "
                        "GPUType inference.")
    p.add_argument("--compile-port", type=int, default=22401)
    p.add_argument("--exec-port", type=int, default=22402)
    p.add_argument("--artifacts-dir", default="/tmp/kb_riscv_artifacts")
    p.add_argument("--spike-binary", default=None,
                   help="Path to spike; falls back to PATH lookup.")
    p.add_argument("--spike-extra-args", default=None,
                   help="Comma-list of extra --spike-arg values (e.g. 'isa=rv64gc').")

    # FireSim strategy knobs — only consulted when --strategy=firesim.
    p.add_argument("--firesim-root", default=None,
                   help="<chipyard>/sims/firesim. Overrides FIRESIM_ROOT.")
    p.add_argument("--firesim-env", default=None,
                   help="chipyard env.sh. Overrides FIRESIM_ENV.")
    p.add_argument("--no-firesim-queue", action="store_true",
                   help="Skip on-host firesim queue; drive firesim direct.")
    p.add_argument("--firesim-queue-root", default=None,
                   help="FIRESIM_QUEUE_ROOT (e.g. /scratch/dima/firesim_queue).")
    p.add_argument("--firesim-queue-bin", default=None,
                   help="firesim-queue CLI path.")
    p.add_argument("--firesim-queue-priority", type=int, default=5,
                   help="FIRESIM_QUEUE_PRIORITY.")
    p.add_argument("--firesim-queue-timeout", type=int, default=None,
                   help="FIRESIM_QUEUE_TIMEOUT — daemon-side wall cap. "
                        "Absent = let workload run to natural completion.")
    p.add_argument("--firesim-default-timeout", type=int, default=900,
                   help="Fallback subprocess timeout (seconds).")
    p.add_argument("--firesim-python-bin", default=None,
                   help="Python interpreter for firesim_runner (defaults "
                        "to sys.executable).")

    p.add_argument("--experiment-name", default=None)
    p.add_argument("--out-root", default=None,
                   help="Where per-problem outputs land. Defaults to "
                        "REPO_ROOT/out/riscv_spike when writable, else "
                        "/scratch/<user>/kb_out/riscv_spike.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Loguru default format — same shape as run_opt_ncu_rl_optimized.
    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))

    def _sigint(_sig, _frame):
        logger.warning("SIGINT — killing spawned servers…")
        _cleanup_procs()
        sys.exit(130)

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    rc = asyncio.run(async_main(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
