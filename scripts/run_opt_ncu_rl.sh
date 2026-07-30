#!/bin/bash
# Wrapper for scripts/run_opt_ncu_rl_optimized.py.
#
# Sets sensible defaults for the optimized RL CUDA flow's environment
# (model dispatch, server URLs, ncu timeout) and invokes the python runner.
# Every variable can be overridden by exporting it before invoking this
# script, or by passing it inline:
#
#   PROBLEM_NUMBERS=1-5 SUBSET=sol-level2 ./scripts/run_opt_ncu_rl.sh
#
#   MODEL_CODEGEN_HARD=claude-opus-4-7 NUM_ITERATIONS=20 \
#       ./scripts/run_opt_ncu_rl.sh
#
# Resume previous runs:
#   ./scripts/run_opt_ncu_rl.sh --resume               # skip succeeded, retry failed
#   ./scripts/run_opt_ncu_rl.sh --resume-skip-failed   # skip both succeeded AND failed
#   (or set RESUME=1 / RESUME_SKIP_FAILED=1 in the env)
#
# Any other positional args are forwarded to the python runner verbatim.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# ── CLI flag parsing (before defaults so --help/--resume short-circuit) ──
# Recognised wrapper-level flags are consumed here. Anything else is
# forwarded verbatim to the python runner. Env vars (RESUME=1 etc.) still
# work as before; CLI flags simply set the same vars.

# Pre-seed RESUME/RESUME_SKIP_FAILED from env (they may have been exported)
# so CLI flags can be additive.
RESUME="${RESUME:-0}"
RESUME_SKIP_FAILED="${RESUME_SKIP_FAILED:-0}"

print_usage() {
    cat <<'USAGE'
Usage: scripts/run_opt_ncu_rl.sh [wrapper-flags] [-- <runner args>]

Convenience wrapper for scripts/run_opt_ncu_rl_optimized.py.

Wrapper flags:
  --resume                Skip problems with success_rl_optimization.cu or
                          no_baseline_rl_optimization.cu in their run dir;
                          retry previously-failed problems (failures are
                          often transient — rate limits, timeouts).
  --resume-skip-failed    Implies --resume; ALSO skip problems with a prior
                          failure_rl_optimization.cu artifact.
  -h, --help              Show this message and exit.

Everything after `--` (or any other positional) is forwarded to the
python runner — useful for e.g. --skip-smoke-test, --num-iterations, etc.

Common env-var overrides (see top of script for the full list):
  PROBLEM_NUMBERS=1-5     Subset of problem numbers to run
  SUBSET=sol-level2       Dataset subset
  CONCURRENCY=5           How many problems to optimise in parallel
  NUM_ITERATIONS=20       Rollouts per problem
  TIMEOUT_MIN=240         Per-problem timeout (minutes)
  RUN_TAG=mytag           Subdir under <experiment-name>/ for outputs
USAGE
}

POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --resume)
            RESUME=1
            shift
            ;;
        --resume-skip-failed)
            RESUME=1
            RESUME_SKIP_FAILED=1
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        --)
            shift
            while [ $# -gt 0 ]; do POSITIONAL+=("$1"); shift; done
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
# Restore positionals safely under set -u (empty array would otherwise error).
set -- ${POSITIONAL[@]+"${POSITIONAL[@]}"}

# ── Defaults (override by exporting before invoking) ────────────────
SUBSET="${SUBSET:-sol-level2}"
#
# Default = the *must-serialize pool* + 65 (rerun): large-memory MoE problems
# that need a dedicated GPU. Use a pinned uncontested GPU (see comment block
# below on KERNELBLASTER_GPU_SERVER_GPU_IDS); CONCURRENCY can stay at default
# since with the pinned GPU there is no cross-user contention.
#
# ── Default = never-RL'd pool (10 problems) ──
# These are in the dataset (drivers verified under scripts/kgen_step_cuda.py
# full) but have NEVER had an opt_ncu_rl run. Highest expected value for new
# successful optimizations.
#   2   decoder_layer_full_block                          full transformer block
#   4   fused_residual_rms_mlp                            residual + RMSNorm + SwiGLU
#   5   swiglu_mlp_backward                               MLP backward
#   9   decoder_layer_with_residual_connections           transformer block
#   12  moe_expert_batched_execution_with_capacity_factor MoE batched
#   13  expert_weighted_aggregation_with_shared_expert    MoE aggregation
#   14  audio_encoder_varlen_attention_with_chunking_bwd  varlen attn bwd
#   16  moe_expert_mlp_with_load_balancing                MoE load-balanced
#   19  decoder_layer_fused_attention_mlp                 transformer fused
#   20  decoder_layer_pre_post_norm_residual              transformer dual-norm
#
# Note: 002/004/019/020 are full transformer blocks at large bf16 sizes; they
# may hit the same fp32-cancellation regime that bit 063 if the kernel chain
# accumulates drift. If RL fails on those, treat as a kernel-quality issue
# rather than a driver issue (drivers were patched and verified earlier).
#
# **GPU pinning is the right mechanism here, not CONCURRENCY=1.** Pick the
# least-contested GPU from `nvidia-smi` and pin via:
#   export KERNELBLASTER_GPU_SERVER_GPU_IDS=2     # whichever GPU has most free memory
# Cross-problem CONCURRENCY can stay >1; what matters is that the pinned GPU
# is NOT shared with another user (chooper's process triggered all the prior
# OOMs by occupying GPU 0).
PROBLEM_NUMBERS="${PROBLEM_NUMBERS:-2,4,5,9,12,13,14,16,19,20}"
#
# ── Previous batches (already attempted) ──
#
# Newly-generated kgens (this session) — all succeeded in RL:
#   35,43,44,57,66,67,80,81 → success_rl_optimization.cu present.
#
# Must-serialize MoE pool — all timed out under TIMEOUT_MIN=240; bump
# TIMEOUT_MIN=480+ and rerun on a pinned GPU:
#   PROBLEM_NUMBERS=47,48,65,82  TIMEOUT_MIN=480 ./scripts/run_opt_ncu_rl.sh
#   47  moe_training_token_repeat_and_expert_computation  (NE=256, ~26 GiB)
#   48  moe_expert_inference_batched_dispatch             (NE=256, ~6 GiB)
#   65  sparse_expert_dispatch_and_combine                (NE=128, ~3 GiB)
#   82  moe_layer_complete_forward_with_residual          (NE=160, ~7.5 GiB)
#
# RL produced no improvement (codegen fixation, do not rerun without fix):
#   1   fused_vision_multihead_attention_bwd  vectorized_memory_access fixation
#
# ── Parallel-safe pool (already run; rerun via override below) ──
# Set:
#   PROBLEM_NUMBERS=27,28,30,31,42,49,61,64,65,68,73  ./scripts/run_opt_ncu_rl.sh
# These are the 11 small/medium kgens added this session — already RL-verified
# in the previous batch with strong improvements (see git log / progress.json).
#
# Override e.g. `PROBLEM_NUMBERS=3,6-8,10,11,15,17,18,21,27,28,30,31,42,49,61,64,65,68,73`
# to also include the prior-run RL-successes (3, 6, 7, 8, 10, 11, 15, 17, 18, 21).
#
# Previously failed in RL flow (LLM-quality / compile-fixation issues):
#   1   fused_vision_multihead_attention_bwd     RL-failed last run: all 10
#                                                trajectories had compile fails
#                                                on vectorized_memory_access
#                                                codegen. Driver verifies fine.
#   24  moe_expert_parallel_execution            same compile-fixation pattern.
#
# OOM-prone in prior contested-GPU run (may work on uncontested GPU now):
#   5   swiglu_mlp_backward                      ~3 GiB
#   9   decoder_layer_with_residual_connections  ~3 GiB
#   13  expert_weighted_aggreg (NE=512)          ~5-7 GiB
#   14  audio_encoder_varlen_attn_bwd            ~5 GiB
#   16  moe_expert_mlp_load_balancing            ~3 GiB
#
# Drivers patched but verify still failing (need round-3 driver fix):
#   2   decoder_layer_full_block          fp32+fp64-ref, ~6 GiB
#   4   fused_residual_rms_mlp            H=16384 I=53248, ~15-20 GiB
#   19  decoder_layer_fused_attn_mlp      fp32+fp64-ref, ~6 GiB
#
# Special-case (need kgen-time fix or different validation path):
#   12  moe_expert_batched_execution_with_capacity_factor
#                                         no_baseline last run
#   20  decoder_layer_pre_post_norm_residual
#                                         curated source declares __global__
#                                         kernels but uses libtorch instead;
#                                         needs kgen-time fix.
#
# kgens still pending (not in data dir; need fresh kgen run):
#   25, 26, 29, 80, 81 (MoE backward / large MoE / drift retries)
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opt_ncu_rl_optimized}"
RUN_TAG="${RUN_TAG:-}"

NUM_ITERATIONS="${NUM_ITERATIONS:-10}"
MAX_STEPS="${MAX_STEPS:-5}"
SEED_FROM_INIT="${SEED_FROM_INIT:-10}"
BANDIT_C="${BANDIT_C:-1.4}"
PRUNE_PATIENCE="${PRUNE_PATIENCE:-2}"
MAX_FIX_ATTEMPTS="${MAX_FIX_ATTEMPTS:-3}"
CONCURRENCY="${CONCURRENCY:-2}"
TIMEOUT_MIN="${TIMEOUT_MIN:-240}"

GPU_TYPE="${GPU_TYPE:-l40s}"

# ── Model IDs — Bedrock-flavoured ────────────────────────────────────
# The environment uses Anthropic via Bedrock (auth_mode=bearer-token), which
# requires the ``anthropic.<model>-v[N](:M)?`` form — bare ``claude-*``
# strings hang on the smoke test because ``_normalize_bedrock_model_id``
# won't rewrite them.
#
# Bedrock model availability is per-account: a model active for one account
# may be Legacy for another. To list the IDs ACTIVE on your account:
#
#   aws bedrock list-foundation-models --region us-east-1 \
#       --by-provider anthropic \
#       --query 'modelSummaries[?modelLifecycle.status==`ACTIVE`].[modelId]' \
#       --output table
#
# Defaults below target the current generation (Opus 4.7 / Sonnet 4.6 /
# Haiku 4.5). If your account flags any of these as Legacy with a 404,
# either fall back to ``anthropic.claude-sonnet-4-5-20250929-v1:0`` (proven
# and recent) or substitute with the date-stamped form returned by the AWS
# command above (e.g. ``anthropic.claude-opus-4-7-20260115-v1:0``).
# IDs copied directly from this account's Bedrock model catalog (so the
# v-suffix shape varies — some have date+v1:0, some have plain -v1, and
# some have neither). Override per-role via the env vars below.
DEFAULT_MODEL="${DEFAULT_MODEL:-anthropic.claude-opus-4-6-v1}"

# Tiered model dispatch. Set any of these to ``${DEFAULT_MODEL}`` to disable
# heterogeneity.
export MODEL_PLAN="${MODEL_PLAN:-anthropic.claude-haiku-4-5-20251001-v1:0}"
export MODEL_CODEGEN_SIMPLE="${MODEL_CODEGEN_SIMPLE:-anthropic.claude-sonnet-4-6}"
export MODEL_CODEGEN_HARD="${MODEL_CODEGEN_HARD:-anthropic.claude-opus-4-6-v1}"
export MODEL_FIX="${MODEL_FIX:-anthropic.claude-haiku-4-5-20251001-v1:0}"

# LLM output budget. Bedrock's default of 4096 is too tight for full-kernel
# rewrites of complex L2 problems (decoder layers, fused MLP, etc.) and
# truncates the response mid-```cpp block, surfacing as
# "Error: The code should be contained within ```cpp and ``` tags."
# 16384 gives generous headroom for Opus/Sonnet rewrites.
export BEDROCK_MAX_TOKENS="${BEDROCK_MAX_TOKENS:-16384}"
export ANTHROPIC_MAX_TOKENS="${ANTHROPIC_MAX_TOKENS:-16384}"

# Server URLs — default to fresh ports in the 7000 range to avoid clashing
# with stale/leftover server processes from prior flows on 2001/2002/2003/2004.
# Override via COMPILE_SERVER_URL / GPU_SERVER_URL_<GPU_TYPE> if you have
# servers already running on different ports you want to reuse.
export COMPILE_SERVER_URL="${COMPILE_SERVER_URL:-http://localhost:7001}"
GPU_TYPE_UPPER="$(echo "$GPU_TYPE" | tr '[:lower:]' '[:upper:]')"
gpu_url_var="GPU_SERVER_URL_${GPU_TYPE_UPPER}"
if [ -z "${!gpu_url_var:-}" ]; then
    export "${gpu_url_var}=http://localhost:7002"
fi

# NCU profile timeout (per call). Long for deep kernels with slow CPU refs.
export KERNELBLASTER_NCU_TIMEOUT_S="${KERNELBLASTER_NCU_TIMEOUT_S:-600}"

# GPU server worker count. Each worker pins to one GPU (round-robin over
# KERNELBLASTER_GPU_SERVER_GPU_IDS, or GPUs 0..N-1 if not set). Default 1 was
# a hard serialization bottleneck — with CONCURRENCY=5 problems all funneled
# through one worker. Bump to match the visible GPU count (host has 4× L40S).
# Override per-machine via KERNELBLASTER_GPU_SERVER_NUM_WORKERS in the env or
# pin to a specific subset via KERNELBLASTER_GPU_SERVER_GPU_IDS="0,1,2,3".
export KERNELBLASTER_GPU_SERVER_NUM_WORKERS="${KERNELBLASTER_GPU_SERVER_NUM_WORKERS:-4}"

# Compile server already defaults to (physical CPU cores - 1) which is fine,
# but expose the knob explicitly so it shows up in the banner.
COMPILE_SERVER_NUM_WORKERS="${KERNELBLASTER_COMPILE_SERVER_NUM_WORKERS:-}"
if [ -n "$COMPILE_SERVER_NUM_WORKERS" ]; then
    export KERNELBLASTER_COMPILE_SERVER_NUM_WORKERS="$COMPILE_SERVER_NUM_WORKERS"
fi

# Cost tracking knobs.
COST_LIVE_INTERVAL="${COST_LIVE_INTERVAL:-5}"

# Resume controls are set above (CLI flags --resume / --resume-skip-failed
# or env vars RESUME=1 / RESUME_SKIP_FAILED=1). Re-asserted here as a no-op
# safety net in case future edits move the flag-parsing block.
RESUME="${RESUME:-0}"
RESUME_SKIP_FAILED="${RESUME_SKIP_FAILED:-0}"

# Smoke-test controls.
#   SKIP_SMOKE_TEST=1            bypass the model-reachability check entirely
#   SMOKE_TEST_SOFT_FAIL=1       log smoke failures but proceed (don't abort)
#   SMOKE_TEST_TIMEOUT=15        per-model timeout in seconds
#   SMOKE_TEST_BATCH_TIMEOUT=60  hard ceiling on the whole smoke batch
SKIP_SMOKE_TEST="${SKIP_SMOKE_TEST:-0}"
SMOKE_TEST_SOFT_FAIL="${SMOKE_TEST_SOFT_FAIL:-0}"
SMOKE_TEST_TIMEOUT="${SMOKE_TEST_TIMEOUT:-15}"
SMOKE_TEST_BATCH_TIMEOUT="${SMOKE_TEST_BATCH_TIMEOUT:-60}"

# ── Banner ───────────────────────────────────────────────────────────
echo "================================================================"
echo "  KernelBlaster optimized RL flow — wrapper"
echo "================================================================"
echo "  Subset:           $SUBSET"
echo "  Problem numbers:  $PROBLEM_NUMBERS"
echo "  Experiment name:  $EXPERIMENT_NAME"
[ -n "$RUN_TAG" ] && echo "  Run tag:          $RUN_TAG"
echo "  GPU type:         $GPU_TYPE"
echo "  GPU workers:      $KERNELBLASTER_GPU_SERVER_NUM_WORKERS (KERNELBLASTER_GPU_SERVER_NUM_WORKERS)"
echo "  Concurrency:      $CONCURRENCY problem(s) in parallel"
echo "  Per problem:      $NUM_ITERATIONS rollouts × $MAX_STEPS steps"
echo "  Per-problem TO:   $TIMEOUT_MIN min"
echo "  Cost interval:    every ${COST_LIVE_INTERVAL}s to file"
case "$RESUME" in
    1|true|yes|on) echo "  Resume:           ON (skip succeeded problems)";;
    *)             echo "  Resume:           off (full re-run)";;
esac
case "$RESUME_SKIP_FAILED" in
    1|true|yes|on) echo "  Resume policy:    ALSO skip previously-failed problems";;
esac
case "$SKIP_SMOKE_TEST" in
    1|true|yes|on) echo "  Smoke test:       SKIPPED";;
    *)
        line="  Smoke test:       on (per-model ${SMOKE_TEST_TIMEOUT}s, batch ${SMOKE_TEST_BATCH_TIMEOUT}s)"
        case "$SMOKE_TEST_SOFT_FAIL" in
            1|true|yes|on) line="$line — soft-fail";;
        esac
        echo "$line"
        ;;
esac
echo "----------------------------------------------------------------"
echo "  Models:"
echo "    default:           $DEFAULT_MODEL"
echo "    MODEL_PLAN:        $MODEL_PLAN"
echo "    MODEL_CODEGEN_SIMPLE: $MODEL_CODEGEN_SIMPLE"
echo "    MODEL_CODEGEN_HARD:   $MODEL_CODEGEN_HARD"
echo "    MODEL_FIX:         $MODEL_FIX"
echo "----------------------------------------------------------------"
echo "  Server URLs:"
echo "    COMPILE:  $COMPILE_SERVER_URL"
echo "    GPU:      ${!gpu_url_var}"
echo "    NCU TO:   ${KERNELBLASTER_NCU_TIMEOUT_S}s"
echo "================================================================"

# ── Container sanity check (best-effort, doesn't block the run) ─────
if command -v docker >/dev/null 2>&1; then
    if docker ps --format '{{.Names}}' | grep -qx 'kernelblaster' 2>/dev/null; then
        echo "  ✓ kernelblaster container is running"
    else
        echo "  ⚠ kernelblaster container not detected — make sure compile (2001) and"
        echo "    gpu (2002) servers are reachable, or override the *_SERVER_URL vars."
    fi
fi

# ── Compile + GPU servers ────────────────────────────────────────────
# Mirrors scripts/run_single_kernelblaster_sol_fix.sh: spawn the GPU server
# via scripts/start_gpu_server.py and the compile server directly via
# ``python -m src.kernelblaster.servers.compile``, both as background
# processes inheriting this shell. Trap-based cleanup tears them down on
# Ctrl-C / normal exit so we don't leak server processes between runs.
#
# Reuse semantics:
#   - If the configured URL already responds to /health, we don't spawn a
#     duplicate (keeps repeat runs fast and avoids "address in use").
#   - Otherwise we spawn and wait up to 20s for /health.
COMPILE_PORT=$(echo "$COMPILE_SERVER_URL" | sed -E 's|.*:([0-9]+).*|\1|')
GPU_PORT=$(echo "${!gpu_url_var}" | sed -E 's|.*:([0-9]+).*|\1|')

server_health() {
    curl -sf --max-time 3 "$1/health" >/dev/null 2>&1
}
wait_for_health() {
    local url="$1" name="$2" max_wait="${3:-20}"
    local i=0
    while [ "$i" -lt "$max_wait" ]; do
        server_health "$url" && return 0
        sleep 1
        i=$((i + 1))
    done
    return 1
}

mkdir -p out
COMPILE_LOG="${COMPILE_LOG:-out/compile_server.log}"
GPU_LOG="${GPU_LOG:-out/gpu_server.log}"
GPU_INFO_FILE="${GPU_INFO_FILE:-out/gpu_server_info.txt}"

COMPILE_SERVER_PID=""
GPU_SERVER_PID=""        # PID of the actual server process
GPU_STARTER_PID=""       # PID of scripts/start_gpu_server.py (its child is the server)
SERVERS_SPAWNED_BY_US=0

cleanup_servers() {
    if [ "$SERVERS_SPAWNED_BY_US" -ne 1 ]; then
        return  # we didn't start them; don't kill them
    fi
    echo ""
    echo "Cleaning up servers spawned by this script…"
    if [ -n "$COMPILE_SERVER_PID" ] && kill -0 "$COMPILE_SERVER_PID" 2>/dev/null; then
        echo "  → compile server (PID $COMPILE_SERVER_PID)"
        kill -TERM "$COMPILE_SERVER_PID" 2>/dev/null
        sleep 1
        kill -0 "$COMPILE_SERVER_PID" 2>/dev/null && kill -KILL "$COMPILE_SERVER_PID" 2>/dev/null
    fi
    if [ -n "$GPU_SERVER_PID" ] && kill -0 "$GPU_SERVER_PID" 2>/dev/null; then
        echo "  → gpu server (PID $GPU_SERVER_PID)"
        kill -TERM "$GPU_SERVER_PID" 2>/dev/null
        sleep 1
        kill -0 "$GPU_SERVER_PID" 2>/dev/null && kill -KILL "$GPU_SERVER_PID" 2>/dev/null
    fi
    if [ -n "$GPU_STARTER_PID" ] && kill -0 "$GPU_STARTER_PID" 2>/dev/null; then
        echo "  → gpu starter (PID $GPU_STARTER_PID)"
        kill -TERM "$GPU_STARTER_PID" 2>/dev/null
        sleep 1
        kill -0 "$GPU_STARTER_PID" 2>/dev/null && kill -KILL "$GPU_STARTER_PID" 2>/dev/null
    fi
    rm -f "$GPU_INFO_FILE"
}
trap 'cleanup_servers; exit 130' INT TERM
trap cleanup_servers EXIT

echo "----------------------------------------------------------------"
echo "  Checking compile + gpu servers…"

# ─── compile server ───────────────────────────────────────────────
if server_health "$COMPILE_SERVER_URL"; then
    echo "  ✓ compile already running at $COMPILE_SERVER_URL (reusing)"
else
    echo "  → starting compile server on port $COMPILE_PORT (log: $COMPILE_LOG)"
    nohup python -m src.kernelblaster.servers.compile_server --port "$COMPILE_PORT" \
        > "$COMPILE_LOG" 2>&1 &
    COMPILE_SERVER_PID=$!
    SERVERS_SPAWNED_BY_US=1
    if wait_for_health "$COMPILE_SERVER_URL" "compile" 20; then
        echo "    ✓ compile up at $COMPILE_SERVER_URL  (PID $COMPILE_SERVER_PID)"
    else
        echo "    ✗ compile failed to come up in 20s. Last 20 lines of log:"
        tail -n 20 "$COMPILE_LOG" 2>&1 | sed 's/^/      /'
        cleanup_servers
        exit 2
    fi
fi

# ─── gpu server ───────────────────────────────────────────────────
if server_health "${!gpu_url_var}"; then
    echo "  ✓ gpu already running at ${!gpu_url_var} (reusing)"
else
    echo "  → starting gpu server on port $GPU_PORT (log: $GPU_LOG)"
    rm -f "$GPU_INFO_FILE"
    nohup python scripts/start_gpu_server.py \
        --port "$GPU_PORT" \
        --log-file "$GPU_LOG" \
        --info-file "$GPU_INFO_FILE" \
        > /dev/null 2>&1 &
    GPU_STARTER_PID=$!
    SERVERS_SPAWNED_BY_US=1
    # start_gpu_server.py writes the actual server PID on the second line
    # of $GPU_INFO_FILE once the child process is up.
    if wait_for_health "${!gpu_url_var}" "gpu" 20; then
        if [ -f "$GPU_INFO_FILE" ]; then
            GPU_SERVER_PID="$(tail -n 1 "$GPU_INFO_FILE")"
        fi
        echo "    ✓ gpu up at ${!gpu_url_var}  (PID ${GPU_SERVER_PID:-?}, starter $GPU_STARTER_PID)"
    else
        echo "    ✗ gpu failed to come up in 20s. Last 20 lines of log:"
        tail -n 20 "$GPU_LOG" 2>&1 | sed 's/^/      /'
        cleanup_servers
        exit 2
    fi
fi

# ── Smoke test (separate subprocess; killable via shell ``timeout``) ─
case "$SKIP_SMOKE_TEST" in
    1|true|yes|on)
        echo "  Smoke test:       SKIPPED via SKIP_SMOKE_TEST"
        ;;
    *)
        echo "----------------------------------------------------------------"
        echo "  Smoke test (per-model ${SMOKE_TEST_TIMEOUT}s, batch ${SMOKE_TEST_BATCH_TIMEOUT}s)..."
        # Pass the four MODEL_* env vars through the python subprocess via
        # --from-env. ``timeout --signal=KILL`` ensures the whole process
        # tree dies even if the SDK is in uncancellable blocking I/O.
        smoke_rc=0
        timeout --signal=KILL "$SMOKE_TEST_BATCH_TIMEOUT" \
            python scripts/smoke_test_models.py --from-env \
                --timeout "$SMOKE_TEST_TIMEOUT" || smoke_rc=$?
        case "$smoke_rc" in
            0)
                echo "  Smoke test:       PASS"
                ;;
            124|137)
                # 124 = GNU timeout fired; 137 = SIGKILL (timeout --signal=KILL)
                echo "  Smoke test:       BATCH TIMEOUT (rc=$smoke_rc) — at least"
                echo "                    one model never returned within"
                echo "                    ${SMOKE_TEST_BATCH_TIMEOUT}s; subprocess killed."
                case "$SMOKE_TEST_SOFT_FAIL" in
                    1|true|yes|on) echo "                    SMOKE_TEST_SOFT_FAIL=1 — proceeding anyway." ;;
                    *)             echo "                    Aborting. Set SMOKE_TEST_SOFT_FAIL=1 or SKIP_SMOKE_TEST=1 to bypass."; exit 1 ;;
                esac
                ;;
            *)
                echo "  Smoke test:       FAILED (rc=$smoke_rc)"
                case "$SMOKE_TEST_SOFT_FAIL" in
                    1|true|yes|on) echo "                    SMOKE_TEST_SOFT_FAIL=1 — proceeding anyway." ;;
                    *)             echo "                    Aborting. Set SMOKE_TEST_SOFT_FAIL=1 or SKIP_SMOKE_TEST=1 to bypass."; exit "$smoke_rc" ;;
                esac
                ;;
        esac
        echo "----------------------------------------------------------------"
        ;;
esac

# ── Compose runner args ─────────────────────────────────────────────
RUNNER_ARGS=(
    --model "$DEFAULT_MODEL"
    --gpu "$GPU_TYPE"
    --subset "$SUBSET"
    --problem-numbers "$PROBLEM_NUMBERS"
    --experiment-name "$EXPERIMENT_NAME"
    --num-iterations "$NUM_ITERATIONS"
    --max-steps "$MAX_STEPS"
    --seed-from-init "$SEED_FROM_INIT"
    --bandit-c "$BANDIT_C"
    --prune-patience "$PRUNE_PATIENCE"
    --max-fix-attempts "$MAX_FIX_ATTEMPTS"
    --concurrency "$CONCURRENCY"
    --timeout "$TIMEOUT_MIN"
    --cost-live-interval "$COST_LIVE_INTERVAL"
)
if [ -n "$RUN_TAG" ]; then
    RUNNER_ARGS+=( --run-tag "$RUN_TAG" )
fi
case "$RESUME" in
    1|true|yes|on) RUNNER_ARGS+=( --resume ) ;;
esac
case "$RESUME_SKIP_FAILED" in
    1|true|yes|on) RUNNER_ARGS+=( --resume-skip-failed ) ;;
esac
case "$SKIP_SMOKE_TEST" in
    1|true|yes|on) RUNNER_ARGS+=( --skip-smoke-test ) ;;
esac
case "$SMOKE_TEST_SOFT_FAIL" in
    1|true|yes|on) RUNNER_ARGS+=( --smoke-test-soft-fail ) ;;
esac
RUNNER_ARGS+=( --smoke-test-timeout "$SMOKE_TEST_TIMEOUT" )
RUNNER_ARGS+=( --smoke-test-batch-timeout "$SMOKE_TEST_BATCH_TIMEOUT" )
# Forward any extra positional args (e.g. --skip-smoke-test).
if [ "$#" -gt 0 ]; then
    RUNNER_ARGS+=( "$@" )
fi

echo "  Invoking: python scripts/run_opt_ncu_rl_optimized.py ${RUNNER_ARGS[*]}"
echo "================================================================"

# NOT ``exec`` — that would replace the bash process and skip the EXIT trap,
# leaving the compile/gpu servers running after the runner exits. Run the
# python normally so cleanup_servers fires via the trap when the runner
# finishes (success, failure, or Ctrl-C).
RUN_RC=0
python scripts/run_opt_ncu_rl_optimized.py "${RUNNER_ARGS[@]}" || RUN_RC=$?

# cleanup_servers is invoked by the EXIT trap; nothing more to do here.
exit "$RUN_RC"
