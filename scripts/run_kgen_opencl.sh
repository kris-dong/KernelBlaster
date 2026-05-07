#!/bin/bash

# Standalone OpenCL kgen: translate PyTorch reference.py → driver.c + kernel.cl
# Runs only the translation + two-step validation, NOT the RL optimization.
#
# Usage:
#   PROBLEM_NUMBERS=1-5 ./scripts/run_kgen_opencl.sh
#   SUBSET=L1 PROBLEM_NUMBERS=1 ./scripts/run_kgen_opencl.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

echo "=========================================="
echo "KernelBlaster OpenCL Kgen (translation)"
echo "=========================================="

# Defaults
GPU_TYPE="${GPU_TYPE:-adreno650}"
BOARD_HOST="${BOARD_HOST:-root@192.0.2.201}"
OPENCL_COMPILE_PORT="${OPENCL_COMPILE_PORT:-6003}"
OPENCL_GPU_PORT="${OPENCL_GPU_PORT:-6004}"
SUBSET="${SUBSET:-L1}"
PROBLEM_NUMBERS="${PROBLEM_NUMBERS:-2-10}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-kgen_opencl}"
TIMEOUT="${TIMEOUT:-240}"
CONCURRENCY="${CONCURRENCY:-10}"
PRECISION="${PRECISION:-fp16}"

echo "Configuration:"
echo "  GPU: $GPU_TYPE"
echo "  Board: $BOARD_HOST"
echo "  Compile server port: $OPENCL_COMPILE_PORT"
echo "  GPU server port: $OPENCL_GPU_PORT"
echo "  Subset: $SUBSET"
echo "  Problems: $PROBLEM_NUMBERS"
echo "  Max attempts: $MAX_ATTEMPTS"
echo "  Precision: $PRECISION"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Timeout: ${TIMEOUT}min"
echo "  Concurrency: $CONCURRENCY"
echo ""

if [ ! -d "src" ]; then
    echo "Error: src directory not found. Run this from the KernelBlaster repo root."
    exit 1
fi

# Verify SSH connectivity to the board
echo "Checking SSH connectivity to $BOARD_HOST..."
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$BOARD_HOST" "echo ok" >/dev/null 2>&1; then
    echo "Error: Cannot SSH to $BOARD_HOST. Check network and SSH keys."
    exit 1
fi
echo "SSH connectivity OK."
echo ""

export KERNELBLASTER_ADRENO_BOARD_HOST="$BOARD_HOST"

cleanup_all() {
    echo ""
    echo "Cleaning up..."
    if command -v lsof >/dev/null 2>&1; then
        for port in $OPENCL_COMPILE_PORT $OPENCL_GPU_PORT; do
            PID=$(lsof -ti:$port 2>/dev/null)
            if [ -n "$PID" ]; then
                kill -TERM $PID 2>/dev/null
                sleep 1
                kill -0 $PID 2>/dev/null && kill -KILL $PID 2>/dev/null
            fi
        done
    fi
    echo "Cleanup completed."
}

trap cleanup_all SIGINT SIGTERM

echo "Starting OpenCL kgen..."
echo ""

python scripts/run_kgen_opencl.py \
  --experiment-name "$EXPERIMENT_NAME" \
  --subset "$SUBSET" \
  --problem-numbers "$PROBLEM_NUMBERS" \
  --precision "$PRECISION" \
  --max-attempts "$MAX_ATTEMPTS" \
  --gpu "$GPU_TYPE" \
  --compiler-port "$OPENCL_COMPILE_PORT" \
  --gpu-port "$OPENCL_GPU_PORT" \
  --board-host "$BOARD_HOST" \
  --timeout "$TIMEOUT" \
  --concurrency "$CONCURRENCY"

EXIT_CODE=$?

cleanup_all

exit $EXIT_CODE
