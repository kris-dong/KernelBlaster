#!/bin/bash
# Test the OpenCL compile server + Adreno GPU server flow using the
# benchmark-opencl/L1/001_Square_matrix_multiplication example.
#
# This script runs inside the docker container. It:
#   1. Starts the OpenCL compile server (port 6003)
#   2. Starts the Adreno GPU server (port 6004)
#   3. Submits a compilation request via the /compile_opencl endpoint
#   4. Submits the compiled binary for execution via /gpu/binary
#   5. Verifies correctness output ("passed")
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BOARD_HOST="${BOARD_HOST:-root@192.0.2.201}"
COMPILE_PORT=6003
GPU_PORT=6004
ARTIFACTS_DIR="/tmp/kernelblaster_test_opencl"

PROBLEM_DIR="${REPO_ROOT}/data/benchmark-opencl/L1/001_Square_matrix_multiplication"
DRIVER_C="${PROBLEM_DIR}/driver.c"
KERNEL_CL="${PROBLEM_DIR}/kernel.cl"

echo "=== OpenCL Server Flow Test ==="
echo "Board host: ${BOARD_HOST}"
echo "Compile server port: ${COMPILE_PORT}"
echo "GPU server port: ${GPU_PORT}"
echo "Problem: ${PROBLEM_DIR}"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "--- Cleaning up ---"
    if [ -n "$COMPILE_PID" ] && kill -0 "$COMPILE_PID" 2>/dev/null; then
        echo "Stopping compile server (PID $COMPILE_PID)"
        kill "$COMPILE_PID" 2>/dev/null || true
        wait "$COMPILE_PID" 2>/dev/null || true
    fi
    if [ -n "$GPU_PID" ] && kill -0 "$GPU_PID" 2>/dev/null; then
        echo "Stopping GPU server (PID $GPU_PID)"
        kill "$GPU_PID" 2>/dev/null || true
        wait "$GPU_PID" 2>/dev/null || true
    fi
    rm -rf "${ARTIFACTS_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

# Create artifacts dir
mkdir -p "${ARTIFACTS_DIR}"

# Verify files exist
if [ ! -f "$DRIVER_C" ]; then
    echo "ERROR: Driver file not found: $DRIVER_C"
    exit 1
fi
if [ ! -f "$KERNEL_CL" ]; then
    echo "ERROR: Kernel file not found: $KERNEL_CL"
    exit 1
fi

echo "--- Step 1: Starting OpenCL compile server ---"
cd "$REPO_ROOT"
python -m src.kernelblaster.servers.compile_opencl \
    --port "$COMPILE_PORT" \
    --num-workers 2 \
    --board-host "$BOARD_HOST" \
    --artifacts-dir "$ARTIFACTS_DIR" \
    &
COMPILE_PID=$!
echo "Compile server PID: $COMPILE_PID"

echo "--- Step 2: Starting Adreno GPU server ---"
python -m src.kernelblaster.servers.gpu_adreno \
    --port "$GPU_PORT" \
    --num-workers 1 \
    --board-host "$BOARD_HOST" \
    &
GPU_PID=$!
echo "GPU server PID: $GPU_PID"

# Wait for servers to be ready
echo ""
echo "Waiting for servers to start..."
for i in $(seq 1 30); do
    COMPILE_READY=false
    GPU_READY=false

    if curl -s "http://localhost:${COMPILE_PORT}/health" > /dev/null 2>&1; then
        COMPILE_READY=true
    fi
    if curl -s "http://localhost:${GPU_PORT}/health" > /dev/null 2>&1; then
        GPU_READY=true
    fi

    if [ "$COMPILE_READY" = true ] && [ "$GPU_READY" = true ]; then
        break
    fi
    sleep 1
done

if [ "$COMPILE_READY" != true ]; then
    echo "ERROR: Compile server failed to start on port $COMPILE_PORT"
    exit 1
fi
if [ "$GPU_READY" != true ]; then
    echo "ERROR: GPU server failed to start on port $GPU_PORT"
    exit 1
fi

echo "Both servers are healthy."
echo ""

echo "--- Step 3: Submitting compilation request ---"
COMPILE_RESPONSE=$(curl -s "http://localhost:${COMPILE_PORT}/compile_opencl?job_name=test_matmul_001&main_file=${DRIVER_C}&kernel_file=${KERNEL_CL}&opencl_version=opencl_2.0&remote=1")
echo "Compile response: $COMPILE_RESPONSE"

COMPILE_SUCCESS=$(echo "$COMPILE_RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin).get('success', False))")
if [ "$COMPILE_SUCCESS" != "True" ]; then
    COMPILE_MSG=$(echo "$COMPILE_RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin).get('message', 'unknown error'))")
    echo "ERROR: Compilation failed: $COMPILE_MSG"
    exit 1
fi

OUTPUT_PATH=$(echo "$COMPILE_RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin).get('output_path', ''))")
echo "Compiled binary (local): $OUTPUT_PATH"
echo ""

echo "--- Step 4: Submitting GPU execution request ---"
# Upload binary + kernel.cl for execution on the board
KERNEL_FILES_JSON=$(python3 -c "import json; print(json.dumps(['${KERNEL_CL}']))")
echo "kernel_files JSON: $KERNEL_FILES_JSON"
GPU_RESPONSE=$(curl -s --max-time 600 -X POST "http://localhost:${GPU_PORT}/gpu/binary" \
    -F "binary=@${OUTPUT_PATH}" \
    -F "args=--profile" \
    -F "n_runs=1" \
    -F "timeout=600" \
    -F "profile=true" \
    -F "kernel_files=${KERNEL_FILES_JSON}")
echo "GPU response: $GPU_RESPONSE"
echo ""

GPU_SUCCESS=$(echo "$GPU_RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin).get('success', False))")
GPU_STDOUT=$(echo "$GPU_RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin).get('stdout', ''))")

echo "--- Step 5: Checking results ---"
echo "GPU stdout: $GPU_STDOUT"

if echo "$GPU_STDOUT" | grep -qi "passed"; then
    echo ""
    echo "=== TEST PASSED: Kernel executed correctly on Adreno GPU ==="
    exit 0
else
    echo ""
    echo "=== TEST FAILED: Kernel did not produce 'passed' output ==="
    exit 1
fi
