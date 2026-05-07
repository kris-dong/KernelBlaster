#!/bin/bash

# OpenCL / Qualcomm Adreno optimization script for KernelBlaster
# Runs the RL-based OpenCL kernel optimization flow targeting Adreno GPUs.
# Must be run inside the Qualcomm docker container (docker_qualcomm/).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

echo "=========================================="
echo "KernelBlaster OpenCL / Adreno Optimization"
echo "=========================================="

# Defaults
DATASET="${DATASET:-kernelbench-opencl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opencl_rl}"
# MODEL="${MODEL:-gpt-5-mini-2025-08-07}"
GPU_TYPE="${GPU_TYPE:-adreno650}"
BOARD_HOST="${BOARD_HOST:-root@10.44.120.201}"
# Default ports moved to 6013/6014 to avoid colliding with the long-running
# kernelblaster-qualcomm-test container which publishes 6003/6004 to the host
# (and ``kernelblaster`` is on host network mode, so binding 6003/6004 fails
# with EADDRINUSE). Override per-run via env if you need different ports.
OPENCL_COMPILE_PORT="${OPENCL_COMPILE_PORT:-6013}"
OPENCL_GPU_PORT="${OPENCL_GPU_PORT:-6014}"
SUBSET="${SUBSET:-L1}"
# PROBLEM_NUMBERS="${PROBLEM_NUMBERS:-2,3,4,6}"
# PROBLEM_NUMBERS="${PROBLEM_NUMBERS:-5,7,8,9,10,11,12,13,14,15,16,17,18,19,20}"
# PROBLEM_NUMBERS="${PROBLEM_NUMBERS:-5,7,8,9,11,14,15,16,17,18,19,20}"
# PROBLEM_NUMBERS="${PROBLEM_NUMBERS:-5,7,8,9,11,16,17,18,19,20}"
# Tier A: pointwise activations newly kgen'd via Sonnet (2026-04-30) and copied
# into data/benchmark-opencl/L1/. Problems 21-32 cover Sigmoid/Tanh/Softmax/
# LogSoftmax/Swish/GELU/SELU/HardSigmoid/Softplus/Softsign/ELU/HardTanh.
# PROBLEM_NUMBERS="${PROBLEM_NUMBERS:-21,22,23,24,25,26,27,28,29,30,31,32}"
# Tier A recovery (2026-04-30): 21-25 finished cleanly the first run; 26 was
# interrupted and 27-32 never started when the disk filled. Delete the
# interrupted 26_GELU_ run dir + retry the remaining 7 problems.
# PROBLEM_NUMBERS="${PROBLEM_NUMBERS:-26,27,28,29,30,31,32}"
# Tier B + C: 20 newly-kgen'd seeds (Sonnet, 2026-04-30) — reductions, norms,
# losses, parametric norms, argmin/argmax. All copied into
# data/benchmark-opencl/L1/ and ready for opt-rl optimization.
#   33-36, 40       parametric norms (BatchNorm, InstanceNorm, GroupNorm,
#                                     RMSNorm, LayerNorm)
#   37-39           vector norms (FrobeniusNorm, L1Norm, L2Norm)
#   47-50           reductions over a dim (Sum, Mean, Max, Product)
#   51-53           index/min reductions (Argmax, Argmin, Min)
#   88              MinGPTNewGelu (pointwise)
#   94, 95, 96, 100 losses (MSELoss, CrossEntropyLoss, HuberLoss, HingeLoss)
# PROBLEM_NUMBERS="${PROBLEM_NUMBERS:-33,34,35,36,37,38,39,40,47,48,49,50,51,52,53,88,94,95,96,100}"
# Tier B+C recovery (2026-04-30): 33, 34, 100 succeeded the first run.
# 35-37 failed and 38 was interrupted when the Adreno board's tmpfs (/tmp,
# 3.8 G RAM-backed) filled with stale compile artifacts. 39, 40, 47-53, 88,
# 94-96 never started before the host runner was killed. After clearing
# the board's /tmp + deleting the failed run dirs, retry the 17 remaining.
# PROBLEM_NUMBERS="${PROBLEM_NUMBERS:-35,36,37,38,39,40,47,48,49,50,51,52,53,88,94,95,96}"
# Second recovery (2026-04-30): 35 succeeded; 36-96 (16 problems) all failed
# because the board's /tmp filled again partway through (each problem leaves
# a ~128 MB reference_output.bin on /tmp/kernelblaster_gpu/<uuid>/ that never
# gets cleaned up — 3.8 G tmpfs saturates after ~30 problems). Cleared board
# /tmp + deleted the 16 failed run dirs; retry just those 16.
# PROBLEM_NUMBERS="${PROBLEM_NUMBERS:-36,37,38,39,40,47,48,49,50,51,52,53,88,94,95,96}"
# Launch-geometry rewrite (2026-04-30): three problems with the
# "one-work-item-per-row" structural bug were rewritten to use cooperative
# workgroup reduction (one workgroup per row, 256 threads cooperating in
# __local memory). Pre-RL hand-validated kernel times on Adreno 650:
#     23_Softmax    4.10ms -> 0.121ms (34x; beats QNN gpu_kernel by 2.7x)
#     24_LogSoftmax 3.51ms -> 0.119ms (29x; beats QNN by 2.7x)
#     38_L1Norm     4.85ms -> 0.084ms (58x; beats QNN by 42x)
# RL output dirs deleted; rerunning RL on these as fresh seeds. The seeds
# now use a magic-comment + reqd_work_group_size convention (parsed by the
# host driver) so the bandit can experiment on top of a properly parallel
# baseline rather than a serially-bottlenecked one.
# Outstanding: 33_BatchNorm and 40_LayerNorm have the same structural bug
# but their hand-rewrites need more debug (33 inefficient memory pattern;
# 40 verification mismatch). Held back from this batch.
PROBLEM_NUMBERS="${PROBLEM_NUMBERS:-23,24,38}"
RL_ITERATIONS="${RL_ITERATIONS:-10}"
RL_ROLLOUT_STEPS="${RL_ROLLOUT_STEPS:-10}"
TIMEOUT="${TIMEOUT:-480}"
KGEN_OPENCL="${KGEN_OPENCL:-0}"

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Model: $MODEL"
echo "  GPU: $GPU_TYPE"
echo "  Board: $BOARD_HOST"
echo "  Compile server port: $OPENCL_COMPILE_PORT"
echo "  GPU server port: $OPENCL_GPU_PORT"
echo "  Subset: $SUBSET"
echo "  Problems: $PROBLEM_NUMBERS"
echo "  RL iterations: $RL_ITERATIONS"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Kgen OpenCL: $KGEN_OPENCL"
echo ""

if [ ! -d "src" ]; then
    echo "Error: src directory not found. Run this from the KernelBlaster repo root."
    exit 1
fi

# Verify SSH connectivity to the board.
# Avoid BatchMode here: in some environments, interactive ssh works while
# BatchMode preflight fails (e.g., host-key acceptance or agent/passphrase flow).
echo "Checking SSH connectivity to $BOARD_HOST..."
if ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$BOARD_HOST" "echo ok" >/dev/null 2>&1; then
    echo "Error: Cannot SSH to $BOARD_HOST. Check network and SSH keys."
    exit 1
fi
echo "SSH connectivity OK."
echo ""

# Export env vars consumed by the OpenCL agent
export KERNELBLASTER_ADRENO_BOARD_HOST="$BOARD_HOST"

mkdir -p "out/${DATASET}/${EXPERIMENT_NAME}"

# Signal handler for cleanup
OPENCL_COMPILE_PID=""
OPENCL_GPU_PID=""

cleanup_all() {
    echo ""
    echo "=========================================="
    echo "Cleaning up..."
    echo "=========================================="

    if command -v lsof >/dev/null 2>&1; then
        for port in $OPENCL_COMPILE_PORT $OPENCL_GPU_PORT; do
            PID=$(lsof -ti:$port 2>/dev/null)
            if [ -n "$PID" ]; then
                echo "Terminating server on port $port (PID $PID)..."
                kill -TERM $PID 2>/dev/null
                sleep 1
                kill -0 $PID 2>/dev/null && kill -KILL $PID 2>/dev/null
            fi
        done
    fi

    echo "Cleanup completed."
}

trap cleanup_all SIGINT SIGTERM

RL_EXPERIMENT_NAME="${RL_EXPERIMENT_NAME:-$EXPERIMENT_NAME}"

echo "Starting OpenCL RL optimization..."
echo ""

KGEN_FLAG=""
if [ "$KGEN_OPENCL" = "1" ] || [ "$KGEN_OPENCL" = "true" ]; then
    KGEN_FLAG="--kgen-opencl"
fi

python scripts/run_RL.py \
  --experiment-name "$RL_EXPERIMENT_NAME" \
  --dataset "$DATASET" \
  --opencl-perf \
  --use-rl \
  --rl-iterations "$RL_ITERATIONS" \
  --rl-rollout-steps "$RL_ROLLOUT_STEPS" \
  --rl-buffer-size 100 \
  --rl-update-frequency 3 \
  --concurrency 1 \
  --problem-numbers "$PROBLEM_NUMBERS" \
  --subset "$SUBSET" \
  --timeout "$TIMEOUT" \
  --gpu "$GPU_TYPE" \
  --compiler-port "$OPENCL_COMPILE_PORT" \
  --gpu-port "$OPENCL_GPU_PORT" \
  --board-host "$BOARD_HOST" \
  --retry \
  --no-resume \
  --no-baseline-optimization \
  $KGEN_FLAG

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "run_RL.py exited with status $EXIT_CODE but continuing to cleanup."
fi

echo "run_RL.py finished; proceeding to cleanup."
cleanup_all

if [ $EXIT_CODE -eq 0 ]; then
    exit 0
else
    exit 1
fi
